#ifndef NPULIB_API_DATA_QUEUE_H_
#define NPULIB_API_DATA_QUEUE_H_

#include "uop_kernel.h"

namespace vta {
/*!
 * \brief Base class of all queues to send and recv serial data.
 */
template <class T>
class BaseQueue {
 public:
  ~BaseQueue() {
    if (fpga_buff_ != nullptr) {
      VTAMemFree(fpga_buff_);
    }
  }
  /*! \return Content of DRAM buffer. */
  char* dram_buffer() const {
    return dram_buffer_;
  }
  /*! \return Physical address of DRAM. */
  vta_phy_addr_t dram_phy_addr() const {
    CHECK(fpga_buff_phy_);
    return fpga_buff_phy_;
  }
  /*! \return Whether there is pending information. */
  bool pending() const {
    return sram_begin_ != sram_end_;
  }
  /*! \brief Initialize the space of the buffer. */
  void InitSpace(uint32_t elem_bytes, uint32_t max_bytes, bool coherent, bool always_cache) {
    coherent_ = coherent;
    always_cache_ = always_cache;
    elem_bytes_ = elem_bytes;
    // Allocate buffer ahead of time
    fpga_buff_ = static_cast<char*>(VTAMemAlloc(
        max_bytes, coherent_ || always_cache_));
    CHECK(fpga_buff_ != nullptr);
    fpga_buff_phy_ = VTAMemGetPhyAddr(fpga_buff_);
  }
  /*!
   * \brief Reset the pointer of the buffer.
   *  Set SRAM pointer to be the current end.
   */
  virtual void Reset() {
    dram_buffer_.clear();
    sram_begin_ = sram_end_;
  }

 protected:
  // Cache coherence access (shared memory only)
  bool coherent_{false};
  // Make the buffer cacheable
  bool always_cache_{false};
  // Element bytes
  uint32_t elem_bytes_{0};
  // Begin location of current SRAM read in FIFO mode
  uint32_t sram_begin_{0};
  // End location of current SRAM write in FIFO mode
  uint32_t sram_end_{0};
  // The buffer in DRAM
  std::vector<T> dram_buffer_;
  // FPGA accessible buffer
  void* fpga_buff_{NULL};
  // Physical address of the FPGA buffer
  vta_phy_addr_t fpga_buff_phy_{0};
};

/*!
 * \brief Micro op buffer that manages the micro op cache.
 */
template<int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class UopQueue : public BaseQueue<VTAUop> {
 public:
  void InitSpace() {
    BaseQueue::InitSpace(kElemBytes, kMaxBytes, kCoherent, kAlwaysCache);
  }
  // Push data to the queue
  template<typename FAutoSync>
  void Push(UopKernel* kernel, FAutoSync fautosync) {
    // if the micro-op is cached in VTA SRAM, skip
    if (kernel->cached()) return;
    // check if we've exceeded the size of the allocated FPGA readable buffer
    size_t num_op = kernel->size();
    if (dram_buffer_.size() + num_op > kMaxElems) {
      fautosync();
      CHECK(dram_buffer_.size() <= kMaxElems);
    }
    // Cannot have a micro-op kernel larger than SRAM buffer
    CHECK(num_op <= kMaxNumUop);
    uint32_t uop_begin = 0;
    if (sram_end_ + num_op > kMaxNumUop) {
      // Need to evict
      cache_idx_ = 0;
      sram_begin_ = 0;
      sram_end_ = num_op;
    } else {
      uop_begin = sram_end_;
      sram_end_ += num_op;
    }
    // Simple eviction policy
    uint32_t evict_begin = cache_idx_;
    for (; cache_idx_ < cache_.size(); ++cache_idx_) {
      if (cache_[cache_idx_]->sram_begin_ >= sram_end_) break;
      // Mark the kernel as "invalid"
      cache_[cache_idx_]->sram_begin_ = 0;
      cache_[cache_idx_]->sram_end_ = 0;
    }
    // Increase size of buffer
    kernel->sram_begin_ = uop_begin;
    kernel->sram_end_ = sram_end_;
    CHECK(kernel->cached());
    cache_.insert(cache_.begin() + cache_idx_, kernel);
    cache_.erase(cache_.begin() + evict_begin, cache_.begin() + cache_idx_);
    cache_idx_ = evict_begin + 1;
  }
  // Flush micro op load instruction
  void FlushUopLoad(VTAMemInsn* insn) {
    if (sram_begin_ != sram_end_) {
      // Derive offset in FPGA-readable buffer
      int32_t offset = 0;
      for (uint32_t i = 0; i < cache_idx_ - 1; ++i) {
        offset += cache_[i]->size() * kElemBytes;
      }
      insn->memory_type = VTA_MEM_ID_UOP;
      insn->sram_base = sram_begin_;
      // Update cache idx to physical address map
      insn->dram_base = (fpga_buff_phy_ + offset) / kElemBytes;
      insn->y_size = 1;
      insn->x_size = (sram_end_ - sram_begin_);
      insn->x_stride = (sram_end_ - sram_begin_);
      insn->y_pad_0 = 0;
      insn->y_pad_1 = 0;
      insn->x_pad_0 = 0;
      insn->x_pad_1 = 0;
      // Reset indices
      sram_begin_ = sram_end_;
    }
  }
  /*! \brief clear cache and reset base queue buffer.*/
  void Reset() {
    cache_.clear();
    cache_idx_ = 0;
    BaseQueue<VTAUop>::Reset();
  }
  void AutoReadBarrier() {
    ReadBarrier();
  }
  /*! \brief Writer barrier to make sure that data written by CPU is visible to VTA. */
  void ReadBarrier() {
    CHECK(fpga_buff_ != nullptr);
    CHECK(fpga_buff_phy_);
    // Iterate over caches; allocate buffer in FPGA-readable memory
    uint32_t buff_size = 0;
    for (uint32_t i = 0; i < cache_.size(); ++i) {
      buff_size += cache_[i]->size() * kElemBytes;
    }
    CHECK(buff_size <= kMaxBytes);
    // Move kernel contents to FPGA readable buffer
    uint32_t offset = 0;
    for (uint32_t i = 0; i < cache_.size(); ++i) {
      uint32_t ksize = cache_[i]->size() * kElemBytes;
      VTAMemCopyFromHost(static_cast<char*>(fpga_buff_) + offset,
                         cache_[i]->data(),
                         ksize);
      // Update offset
      offset += ksize;
    }
    // Flush if we're using a shared memory system
    // and if interface is non-coherent
    if (!coherent_ && always_cache_) {
      VTAFlushCache(fpga_buff_,
                    fpga_buff_phy_,
                    offset);
    }
  }

 private:
  // Cache pointer
  uint32_t cache_idx_{0};
  // Cached ring, sorted by sram_begin
  std::vector<UopKernel*> cache_;
  // Constants
  static constexpr int kElemBytes = sizeof(VTAUop);
  static constexpr int kMaxNumUop = VTA_UOP_BUFF_DEPTH;
  static constexpr int kMaxElems = kMaxBytes / kElemBytes;
};



// Instruction Queue
template<int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class InsnQueue : public BaseQueue<VTAGenericInsn> {
 public:
  /*! \brief Initialize the space. */
  void InitSpace() {
    BaseQueue::InitSpace(kElemBytes, kMaxBytes, kCoherent, kAlwaysCache);
    // Initialize the stage
    std::fill(pending_pop_prev_, pending_pop_prev_ + 4, 0);
    std::fill(pending_pop_next_, pending_pop_next_ + 4, 0);
  }
  /*! \return The data pointer. */
  VTAGenericInsn* data() {
    return dram_buffer_.data();
  }
  /*! \return Number of instructions. */
  uint32_t count() {
    return dram_buffer_.size();
  }
  // Insert dependency push of load
  void DepPop(int from, int to) {
    // NOTE: This instruction executes on queue[to]
    if (from < to) {
      if (pending_pop_prev_[to]) {
        this->CommitPendingPop(to);
      }
      pending_pop_prev_[to] = 1;
    } else {
      if (pending_pop_next_[to]) {
        this->CommitPendingPop(to);
      }
      pending_pop_next_[to] = 1;
    }
    // Impossible condition
    CHECK(from != kLoadStage || to != kStoreStage);
    CHECK(from != kStoreStage || to != kLoadStage);
  }
  // Insert dependency push of load
  void DepPush(int from, int to) {
    // NOTE: this instruction executes on queue[from]
    this->CommitPendingPop(from);
    if (!dram_buffer_.empty()) {
      VTAMemInsn* mptr = reinterpret_cast<VTAMemInsn*>(&dram_buffer_.back());
      if (GetPipelineStage(mptr) == from) {
        if (from < to && !mptr->push_next_dep) {
          // push(LD->C) or push(C->ST)
          mptr->push_next_dep = true; return;
        } else if (from > to && !mptr->push_prev_dep) {
          // push(C->LD) or push(ST->C)
          mptr->push_prev_dep = true; return;
        }
      }
    }
    if (from < to) {
      // Push next dep
      PushNoop(from, false, true, false, false);
    } else {
      // Push prev dep
      PushNoop(from, true, false, false, false);
    }
  }
  // Create a new instruction for a GEMM stage
  VTAGemInsn* CreateGemInsn() {
    return reinterpret_cast<VTAGemInsn*>(
        Create(kComputeStage));
  }
  // Create a new instruction for a ALU stage
  VTAAluInsn* CreateAluInsn() {
    return reinterpret_cast<VTAAluInsn*>(
        Create(kComputeStage));
  }
  // Create a new instruction for a memory stage
  VTAMemInsn* CreateMemInsn(int memory_type) {
    return reinterpret_cast<VTAMemInsn*>(
        Create(GetMemPipelineStage(memory_type)));
  }
  // create a new instruction for a store stage
  VTAMemInsn* CreateStoreInsn() {
    return reinterpret_cast<VTAMemInsn*>(
        Create(kStoreStage));
  }
  // Rewrite instruction stream to force serial execution
  void RewriteForceSerial() {
    int insn_count = count();
    VTAMemInsn* mem_ptr = reinterpret_cast<VTAMemInsn*>(data());
    VTAMemInsn* mem_last_store_ptr = nullptr;
    VTAMemInsn* mem_last_ptr = nullptr;
    for (int i = 1; i < insn_count; ++i) {
      PipelineStage prev = GetPipelineStageAll(mem_ptr + i - 1);
      PipelineStage now = GetPipelineStageAll(mem_ptr + i);
      if (prev == kLoadStage && now == kComputeStage) {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = true;
        mem_ptr[i].pop_prev_dep = true;
        mem_ptr[i].pop_next_dep = false;
      } else if (prev == kComputeStage && now == kLoadStage) {
        mem_ptr[i - 1].push_prev_dep = true;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = true;
      } else if (prev == kStoreStage && now == kComputeStage) {
        mem_ptr[i - 1].push_prev_dep = true;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = true;
      } else if (prev == kComputeStage && now == kStoreStage) {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = true;
        mem_ptr[i].pop_prev_dep = true;
        mem_ptr[i].pop_next_dep = false;
      } else {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = false;
      }
      if (now == kStoreStage) {
        mem_last_store_ptr = &mem_ptr[i];
      }
      mem_last_ptr = &mem_ptr[i];
    }
    // set dependency to make sure all core instruction get excuted
    // before last FINISH instruction
    if (mem_last_store_ptr && mem_last_ptr == mem_last_store_ptr) {
      mem_last_store_ptr->push_prev_dep = true;
      if (!pending_pop_next_[kComputeStage]) {
        DepPop(kStoreStage, kComputeStage);
      }
      CommitPendingPop(kComputeStage);
    } else {
        pending_pop_next_[kComputeStage] = 0;
    }
    DepPush(kComputeStage, kLoadStage);
    DepPop(kLoadStage, kComputeStage);
    if (!pending_pop_next_[kLoadStage]) {
      DepPop(kComputeStage, kLoadStage);
    }
    CommitPendingPop(kLoadStage);
    DepPush(kLoadStage, kComputeStage);
    CommitPendingPop(kComputeStage);
  }
  // Helper function: Get Opcode string
  const char* getOpcodeString(int opcode, bool use_imm) {
      // The string name
      if (opcode == VTA_ALU_OPCODE_MIN) {
          if (use_imm) {
              return "min imm";
          } else {
              return "min";
          }
      } else if (opcode == VTA_ALU_OPCODE_MAX) {
          if (use_imm) {
              return "max imm";
          } else {
              return "max";
          }
      } else if (opcode == VTA_ALU_OPCODE_ADD) {
          if (use_imm) {
              return "add imm";
          } else {
              return "add";
          }
      } else if (opcode == VTA_ALU_OPCODE_MUL) {
          if (use_imm) {
              return "mul imm";
          } else {
              return "mul";
          }
      } else if (opcode == VTA_ALU_OPCODE_SHR) {
          return "shr";
      }

      return "unknown op";
  }
  // Dump instructions in the queue
  void DumpInsn() {
    // Keep tabs on dependence queues
    int l2g_queue = 0;
    int g2l_queue = 0;
    int s2g_queue = 0;
    int g2s_queue = 0;
    // Converter
    union VTAInsn c;
    // Iterate over all instructions
    int insn_count = count();
    const VTAGenericInsn* insn = data();
    printf("There are %u instructions\n", insn_count);
    for (int i = 0; i < insn_count; ++i) {
      // Fetch instruction and decode opcode
      c.generic = insn[i];
      printf("INSTRUCTION %u: ", i);
      if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.x_size == 0) {
          if (c.mem.opcode == VTA_OPCODE_STORE) {
            printf("NOP-STORE-STAGE\n");
          } else if (GetMemPipelineStage(c.mem.memory_type) == kComputeStage) {
            printf("NOP-COMPUTE-STAGE\n");
          } else {
            printf("NOP-MEMORY-STAGE\n");
          }
          printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                 static_cast<int>(c.mem.pop_prev_dep),
                 static_cast<int>(c.mem.pop_next_dep),
                 static_cast<int>(c.mem.push_prev_dep),
                 static_cast<int>(c.mem.push_next_dep));
          // Count status in queues
          if (c.mem.opcode == VTA_OPCODE_STORE) {
            CHECK(c.mem.pop_next_dep == false);
            CHECK(c.mem.push_next_dep == false);
            if (c.mem.pop_prev_dep) g2s_queue--;
            if (c.mem.push_prev_dep) s2g_queue++;
          } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                     (c.mem.memory_type == VTA_MEM_ID_INP ||
                      c.mem.memory_type == VTA_MEM_ID_WGT) ) {
            CHECK(c.mem.pop_prev_dep == false);
            CHECK(c.mem.push_prev_dep == false);
            if (c.mem.pop_next_dep) g2l_queue--;
            if (c.mem.push_next_dep) l2g_queue++;
          } else {
            if (c.mem.pop_prev_dep) l2g_queue--;
            if (c.mem.push_prev_dep) g2l_queue++;
            if (c.mem.pop_next_dep) s2g_queue--;
            if (c.mem.push_next_dep) g2s_queue++;
          }
          printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
          printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
          continue;
        }
        // Print instruction field information
        if (c.mem.opcode == VTA_OPCODE_LOAD) {
          printf("LOAD ");
          if (c.mem.memory_type == VTA_MEM_ID_UOP) printf("UOP\n");
          if (c.mem.memory_type == VTA_MEM_ID_WGT) printf("WGT\n");
          if (c.mem.memory_type == VTA_MEM_ID_INP) printf("INP\n");
          if (c.mem.memory_type == VTA_MEM_ID_ACC) printf("ACC\n");
          if (c.mem.memory_type == VTA_MEM_ID_MOVE) printf("MOV\n");
        }
        if (c.mem.opcode == VTA_OPCODE_STORE) {
          printf("STORE:\n");
        }
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep),
               static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep),
               static_cast<int>(c.mem.push_next_dep));
        printf("\tDRAM: 0x%08x, SRAM:0x%04x\n",
               static_cast<int>(c.mem.dram_base),
               static_cast<int>(c.mem.sram_base));
        printf("\ty: size=%d, pad=[%d, %d]\n",
               static_cast<int>(c.mem.y_size),
               static_cast<int>(c.mem.y_pad_0),
               static_cast<int>(c.mem.y_pad_1));
        printf("\tx: size=%d, stride=%d, pad=[%d, %d]\n",
               static_cast<int>(c.mem.x_size),
               static_cast<int>(c.mem.x_stride),
               static_cast<int>(c.mem.x_pad_0),
               static_cast<int>(c.mem.x_pad_1));
      } else if (c.mem.opcode == VTA_OPCODE_GEMM) {
        // Print instruction field information
        printf("GEMM\n");

        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep),
               static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep),
               static_cast<int>(c.mem.push_next_dep));
        printf("\treset_out: %d\n", static_cast<int>(c.gemm.reset_reg));
        printf("\trange (%d, %d)\n",
               static_cast<int>(c.gemm.uop_bgn),
               static_cast<int>(c.gemm.uop_end));
        printf("\touter loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
               static_cast<int>(c.gemm.iter_out),
               static_cast<int>(c.gemm.wgt_factor_out),
               static_cast<int>(c.gemm.src_factor_out),
               static_cast<int>(c.gemm.dst_factor_out));
        printf("\tinner loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
               static_cast<int>(c.gemm.iter_in),
               static_cast<int>(c.gemm.wgt_factor_in),
               static_cast<int>(c.gemm.src_factor_in),
               static_cast<int>(c.gemm.dst_factor_in));
      } else if (c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        printf("ALU - %s\n", getOpcodeString(c.alu.alu_opcode, c.alu.use_imm));
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep),
               static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep),
               static_cast<int>(c.mem.push_next_dep));
        printf("\treset_out: %d\n", static_cast<int>(c.alu.reset_reg));
        printf("\trange (%d, %d)\n",
               static_cast<int>(c.alu.uop_bgn),
               static_cast<int>(c.alu.uop_end));
        printf("\touter loop - iter: %d, dst: %d, src: %d\n",
               static_cast<int>(c.alu.iter_out),
               static_cast<int>(c.alu.dst_factor_out),
               static_cast<int>(c.alu.src_factor_out));
        printf("\tinner loop - iter: %d, dst: %d, src: %d\n",
               static_cast<int>(c.alu.iter_in),
               static_cast<int>(c.alu.dst_factor_in),
               static_cast<int>(c.alu.src_factor_in));
      } else if (c.mem.opcode == VTA_OPCODE_FINISH) {
        printf("FINISH\n");
      }

      // Count status in queues
      if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.opcode == VTA_OPCODE_STORE) {
            CHECK(c.mem.pop_next_dep == false);
            CHECK(c.mem.push_next_dep == false);
            if (c.mem.pop_prev_dep) g2s_queue--;
            if (c.mem.push_prev_dep) s2g_queue++;
        } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                   (c.mem.memory_type == VTA_MEM_ID_INP ||
                    c.mem.memory_type == VTA_MEM_ID_WGT) ) {
            CHECK(c.mem.pop_prev_dep == false);
            CHECK(c.mem.push_prev_dep == false);
            if (c.mem.pop_next_dep) g2l_queue--;
            if (c.mem.push_next_dep) l2g_queue++;
        } else {
            if (c.mem.pop_prev_dep) l2g_queue--;
            if (c.mem.push_prev_dep) g2l_queue++;
            if (c.mem.pop_next_dep) s2g_queue--;
            if (c.mem.push_next_dep) g2s_queue++;
        }
      } else if (c.mem.opcode == VTA_OPCODE_GEMM ||
                 c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        if (c.gemm.pop_prev_dep) l2g_queue--;
        if (c.gemm.push_prev_dep) g2l_queue++;
        if (c.gemm.pop_next_dep) s2g_queue--;
        if (c.gemm.push_next_dep) g2s_queue++;
      }
      printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
      printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
    }
  }
  // Commit all pending pop of corresponding stage
  void CommitPendingPop(int stage) {
    // Handle the LD<->compute queue
    // NOTE: pop executes on target(stage)
    CHECK(stage > 0 && stage < 4);
    if (pending_pop_prev_[stage] ||
        pending_pop_next_[stage]) {
      PushNoop(stage, false, false,
               pending_pop_prev_[stage],
               pending_pop_next_[stage]);
      pending_pop_prev_[stage] = 0;
      pending_pop_next_[stage] = 0;
    }
  }
  void CommitPending() {
    for (int i = kLoadStage; i <= kStoreStage; ++i) {
      CommitPendingPop(i);
    }
  }
  bool PendingPop() {
    for (int i = kLoadStage; i <= kStoreStage; ++i) {
      if (pending_pop_prev_[i]) return true;
      if (pending_pop_next_[i]) return true;
    }
    return false;
  }
  void AutoReadBarrier() {
    ReadBarrier();
  }
  /*! \brief Writer barrier to make sure that data written by CPU is visible to VTA. */
  void ReadBarrier() {
    CHECK(fpga_buff_ != nullptr);
    CHECK(fpga_buff_phy_);
    uint32_t buff_size = dram_buffer_.size() * elem_bytes_;
    CHECK(buff_size <= kMaxBytes);
    //printf("InstQ ReadBarrier\n");
    // Copy contents of DRAM buffer to FPGA buff
    VTAMemCopyFromHost(fpga_buff_,
                       dram_buffer_.data(),
                       buff_size);
    // Flush if we're using a shared memory system
    // and if interface is non-coherent
    if (!coherent_ && always_cache_) {
      VTAFlushCache(fpga_buff_,
                    fpga_buff_phy_,
                    buff_size);
    }
  }

 protected:
  /*! \return Add new instruction to the buffer. */
  VTAGenericInsn* NextInsn() {
    VTAGenericInsn insn;
    dram_buffer_.push_back(insn);
    return &dram_buffer_.back();
  }
  // Create a new instruction for a given stage
  VTAGenericInsn* Create(PipelineStage stage) {
    VTAGenericInsn* gptr = NextInsn();
    VTAMemInsn* mptr = reinterpret_cast<VTAMemInsn*>(gptr);
    mptr->pop_prev_dep = pending_pop_prev_[stage];
    mptr->pop_next_dep = pending_pop_next_[stage];
    mptr->push_prev_dep = false;
    mptr->push_next_dep = false;
    pending_pop_prev_[stage] = 0;
    pending_pop_next_[stage] = 0;
    return gptr;
  }
  // Get stage of the memory
  static PipelineStage GetMemPipelineStage(int memory_type) {
    if (memory_type == VTA_MEM_ID_ACC) return kComputeStage;
    if (memory_type == VTA_MEM_ID_UOP) return kComputeStage;
    if (memory_type == VTA_MEM_ID_MOVE) return kComputeStage;
    return kLoadStage;
  }
  // Get stage of the computation
  static PipelineStage GetPipelineStage(VTAMemInsn* insn) {
    if (insn->opcode == VTA_OPCODE_GEMM) return kComputeStage;
    if (insn->opcode == VTA_OPCODE_ALU) return kComputeStage;
    if (insn->opcode == VTA_OPCODE_LOAD) {
      if (insn->x_size == 0) return kNoneStage;
      if (insn->memory_type == VTA_MEM_ID_ACC) return kComputeStage;
      if (insn->memory_type == VTA_MEM_ID_UOP) return kComputeStage;
      if (insn->memory_type == VTA_MEM_ID_MOVE) return kComputeStage;
      return kLoadStage;
    }
    if (insn->opcode == VTA_OPCODE_STORE) {
      // FIXME: Right now memory_type is a 2-bit field which means that
      //        VTA_MEM_ID_OUT will appear as 0. For now we'll refrain from
      //        checking the memory_type to avoid an CHECK error...
      return kStoreStage;
    }
    LOG(FATAL) << "not reached";
    return kNoneStage;
  }

  // Get stage of memory and computation
  static PipelineStage GetPipelineStageAll(VTAMemInsn* insn) {
      PipelineStage stage = GetPipelineStage(insn);
      if (stage != kNoneStage) return stage;
      return GetMemPipelineStage(insn->memory_type);
  }

  // Push no-op
  void PushNoop(int stage,
                bool push_prev_dep, bool push_next_dep,
                bool pop_prev_dep, bool pop_next_dep) {
    VTAMemInsn* insn = reinterpret_cast<VTAMemInsn*>(NextInsn());
    insn->opcode = (stage == kStoreStage ? VTA_OPCODE_STORE : VTA_OPCODE_LOAD);
    insn->push_prev_dep = push_prev_dep;
    insn->push_next_dep = push_next_dep;
    insn->pop_prev_dep = pop_prev_dep;
    insn->pop_next_dep = pop_next_dep;
    insn->sram_base = 0;
    insn->dram_base = 0;
    insn->y_size = 0;
    insn->x_size = 0;
    insn->x_stride = 0;
    insn->y_pad_0 = 0;
    insn->y_pad_1 = 0;
    insn->x_pad_0 = 0;
    insn->x_pad_1 = 0;
    insn->memory_type = (stage == kLoadStage ? VTA_MEM_ID_INP : VTA_MEM_ID_UOP);
  }

 private:
  // Pending pop of each isntruction queue, qid=0 is not used
  int pending_pop_prev_[4];
  int pending_pop_next_[4];
  static constexpr int kElemBytes = sizeof(VTAGenericInsn);
  static constexpr int kMaxElems = kMaxBytes / kElemBytes;
};



}

#endif
