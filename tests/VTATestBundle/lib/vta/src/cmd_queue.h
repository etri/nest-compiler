#ifndef NPULIB_API_COMMAND_QUEUE_H_
#define NPULIB_API_COMMAND_QUEUE_H_


#include <vta/driver.h>
#include "uop_kernel.h"

#include <memory>

namespace vta {

/*!
 * \brief The command queue object that handles the request.
 */
class CommandQueue {
 public:
  CommandQueue() {
    this->InitSpace();
  }
  void InitSpace() {
    uop_queue_.InitSpace();
    insn_queue_.InitSpace();
    device_ = VTADeviceAlloc();
    CHECK(device_ != nullptr);
  }

  ~CommandQueue() {
    VTADeviceFree(device_);
  }

  uint32_t GetElemBytes(uint32_t memory_id) {
    uint32_t elem_bytes = 0;
    switch (memory_id) {
      case VTA_MEM_ID_UOP:
          elem_bytes = VTA_UOP_ELEM_BYTES;
          break;
      case VTA_MEM_ID_INP:
          elem_bytes = VTA_INP_ELEM_BYTES;
          break;
      case VTA_MEM_ID_WGT:
          elem_bytes = VTA_WGT_ELEM_BYTES;
          break;
      case VTA_MEM_ID_ACC:
          elem_bytes = VTA_ACC_ELEM_BYTES;
          break;
      case VTA_MEM_ID_OUT:
          elem_bytes = VTA_INP_ELEM_BYTES;
          break;
      case VTA_MEM_ID_MOVE:
          elem_bytes = VTA_INP_ELEM_BYTES;
      default:
          LOG(FATAL) << "Memory id not recognized:" << memory_id;
          break;
    }
    /*
     * elements size should not larger than VTA_PAGE_BYTES.
     *
     */
    CHECK_GE(VTA_PAGE_BYTES, elem_bytes);
    return elem_bytes;
  }

  void LoadBuffer2D(void* src_dram_addr,
                    uint32_t src_elem_offset,
                    uint32_t x_size,
                    uint32_t y_size,
                    uint32_t x_stride,
                    uint32_t x_pad_before,
                    uint32_t y_pad_before,
                    uint32_t x_pad_after,
                    uint32_t y_pad_after,
                    uint32_t dst_sram_index,
                    uint32_t dst_memory_type) {
    VTAMemInsn* insn = insn_queue_.CreateMemInsn(dst_memory_type);
    insn->opcode = VTA_OPCODE_LOAD;
    insn->memory_type = dst_memory_type;
    insn->sram_base = dst_sram_index;
    DataBuffer* src = DataBuffer::FromHandle(src_dram_addr);
    if(dst_memory_type!=VTA_MEM_ID_MOVE) {
      insn->dram_base = src->phy_addr() / GetElemBytes(dst_memory_type) + src_elem_offset;
    }
    else {
      insn->dram_base = *(uint32_t *) src_dram_addr;
    }
    insn->y_size = y_size;
    insn->x_size = x_size;
    insn->x_stride = x_stride;
    insn->y_pad_0 = y_pad_before;
    insn->y_pad_1 = y_pad_after;
    insn->x_pad_0 = x_pad_before;
    insn->x_pad_1 = x_pad_after;
    this->CheckInsnOverFlow();
  }

  void StoreBuffer2D(uint32_t src_sram_index,
                     uint32_t src_memory_type,
                     void* dst_dram_addr,
                     uint32_t dst_elem_offset,
                     uint32_t x_size,
                     uint32_t y_size,
                     uint32_t x_stride) {
    VTAMemInsn* insn = insn_queue_.CreateStoreInsn();
    insn->opcode = VTA_OPCODE_STORE;
    insn->memory_type = src_memory_type;
    insn->sram_base = src_sram_index;
    DataBuffer* dst = DataBuffer::FromHandle(dst_dram_addr);
    insn->dram_base = dst->phy_addr() / GetElemBytes(src_memory_type) + dst_elem_offset;
    insn->y_size = y_size;
    insn->x_size = x_size;
    insn->x_stride = x_stride;
    insn->y_pad_0 = 0;
    insn->y_pad_1 = 0;
    insn->x_pad_0 = 0;
    insn->x_pad_1 = 0;
    this->CheckInsnOverFlow();

    //printf("store2d dram_addr = %p, dst->phy_addr=%p (drambase=%p)\n",dst_dram_addr, (void*)dst->phy_addr(), (void*)insn->dram_base );

  }

  void DepPush(int from_qid, int to_qid) {
   // printf("dep_push(%d %d)\n",from_qid, to_qid);
    insn_queue_.DepPush(from_qid, to_qid);
  }

  void DepPop(int from_qid, int to_qid) {
  //  printf("dep_pop(%d %d)\n",from_qid, to_qid);    
    insn_queue_.DepPop(from_qid, to_qid);
  }

  void ReadBarrier(void* buffer, uint32_t elem_bits, uint32_t start, uint32_t extent) {
    if (!(debug_flag_ & VTA_DEBUG_SKIP_READ_BARRIER)) {
      //printf("CmdQ ReadBarrier\n");
      uint32_t elem_bytes = (elem_bits + 8 - 1) / 8;
      DataBuffer::FromHandle(buffer)->FlushCache(
          elem_bytes * start, elem_bytes * extent);
    }
  }

  void WriteBarrier(void* buffer, uint32_t elem_bits, uint32_t start, uint32_t extent) {
    if (!(debug_flag_ & VTA_DEBUG_SKIP_WRITE_BARRIER)) {
      //printf("CmdQ WriteBarrier\n");
      uint32_t elem_bytes = (elem_bits + 8 - 1) / 8;
      DataBuffer::FromHandle(buffer)->InvalidateCache(
          elem_bytes * start, elem_bytes * extent);
    }
  }

  void Synchronize(uint32_t wait_cycles) {
    //printf("sync\n");    
    // Insert dependences to force serialization
    if (debug_flag_ & VTA_DEBUG_FORCE_SERIAL) {
      insn_queue_.RewriteForceSerial();
    } else {
      // This will issue finish after last store finishes
      insn_queue_.DepPush(kStoreStage, kComputeStage);
      insn_queue_.DepPush(kLoadStage, kComputeStage);
      insn_queue_.DepPop(kStoreStage, kComputeStage);
      insn_queue_.DepPop(kLoadStage, kComputeStage);
      insn_queue_.CommitPendingPop(kComputeStage);
    }
    // NOTE: FINISH cannot contain pop
    VTAGemInsn* insn = insn_queue_.CreateGemInsn();
    insn->opcode = VTA_OPCODE_FINISH;
    CHECK(!insn_queue_.PendingPop());
    // Check if there are no instruction to execute at all
    if (insn_queue_.count() == 0) return;
    // Synchronization for the queues
    uop_queue_.AutoReadBarrier();
    insn_queue_.AutoReadBarrier();
    // Dump instructions if debug enabled

    if (debug_flag_ & VTA_DEBUG_DUMP_INSN) {
      insn_queue_.DumpInsn();
    }

    //printf("[Synchronize] check last instruction FINISH");
    // Make sure that the last instruction is a finish instruction
    CHECK(reinterpret_cast<VTAMemInsn*>(
        insn_queue_.data())[insn_queue_.count()-1].opcode == VTA_OPCODE_FINISH);
    //printf(" -- DONE\n");

    //printf("[Synchronize] check exceed physical memory limit");
    // Make sure that we don't exceed contiguous physical memory limits
    CHECK(insn_queue_.count() * sizeof(VTAGenericInsn) < VTA_MAX_XFER);
    //printf(" -- DONE\n");

    //printf("[Synchronize] device run. device=%p", device_);
    int timeout = VTADeviceRun(
        device_,
        insn_queue_.dram_phy_addr(),
        insn_queue_.count(),
        wait_cycles);
    //printf("[Synchronize] device run. -- DONE\n");

    //printf("[Synchronize] check timeout");
    CHECK_EQ(timeout, 0);
    //printf(" -- DONE\n");
    
    //printf("[Synchronize] reset uop queue");
    // Reset buffers
    uop_queue_.Reset();
    //printf(" -- DONE\n");

    //printf("[Synchronize] reset uop queue");
    insn_queue_.Reset();
    //printf(" -- DONE\n");    
  }

  // Get record kernel
  UopKernel* record_kernel() const {
    //printf("rk\n");
    CHECK(record_kernel_ != nullptr);
    return record_kernel_;
  }

  // Set debug flag
  void SetDebugFlag(int debug_flag) {
    debug_flag_ = debug_flag;
  }

  void PushGEMMOp(void** uop_handle,
                  int (*finit)(void*),
                  void* signature,
                  int nbytes,
                  bool from_outmem=false) {
    UopKernelMap** uptr = reinterpret_cast<UopKernelMap**>(uop_handle);
    
    if (uptr[0] == nullptr) {
      uptr[0] = new UopKernelMap();
    }
    UopKernel** kptr = uptr[0]->Get(signature, nbytes);
    if (kptr[0] == nullptr) {
      record_kernel_ = new UopKernel(static_cast<char*>(signature), nbytes);
      CHECK_EQ((*finit)(signature), 0);
      kptr[0] = static_cast<UopKernel*>(record_kernel_);

      if (debug_flag_ & VTA_DEBUG_DUMP_UOP) {
          printf("dump kernel\n");
        record_kernel_->Dump();
      }
      record_kernel_ = nullptr;
    }
    this->PushGEMMOp(static_cast<UopKernel*>(kptr[0]), from_outmem);
    this->CheckInsnOverFlow();
  }

  void PushALUUop(void** uop_handle,
                  int (*finit)(void*),
                  void* signature,
                  int nbytes) {
    UopKernelMap** uptr = reinterpret_cast<UopKernelMap**>(uop_handle);
    //printf("UKM:%p\n",uptr[0]);
    if (uptr[0] == nullptr) {
      uptr[0] = new UopKernelMap();
    }
    UopKernel** kptr = uptr[0]->Get(signature, nbytes);
    if (kptr[0] == nullptr) {
      record_kernel_ = new UopKernel(static_cast<char*>(signature), nbytes);
      CHECK_EQ((*finit)(signature), 0);
      kptr[0] = static_cast<UopKernel*>(record_kernel_);
      if (debug_flag_ & VTA_DEBUG_DUMP_UOP) {
        record_kernel_->Dump();
      }
      record_kernel_ = nullptr;
    }
    this->PushALUUop(static_cast<UopKernel*>(kptr[0]));
    this->CheckInsnOverFlow();
  }

  static std::shared_ptr<CommandQueue>& ThreadLocal() {
    static std::shared_ptr<CommandQueue> inst =
        std::make_shared<CommandQueue>();
    if (inst == nullptr) {
      inst = std::make_shared<CommandQueue>();
    }
    return inst;
  }

  static void Shutdown() {
    ThreadLocal().reset();
  }

 private:
  // Push GEMM uop to the command buffer
  void PushGEMMOp(UopKernel* kernel, bool from_outmem=false) {
    uop_queue_.Push(kernel,
                    [this]() { this->AutoSync(); });
    if (uop_queue_.pending()) {
      VTAMemInsn* insn = insn_queue_.CreateMemInsn(VTA_MEM_ID_UOP);
      insn->opcode = VTA_OPCODE_LOAD;
      uop_queue_.FlushUopLoad(insn);
    }
    VTAGemInsn* insn = insn_queue_.CreateGemInsn();
    insn->opcode = VTA_OPCODE_GEMM;
    insn->reset_reg = kernel->reset_out_;
    insn->uop_bgn = kernel->sram_begin_;
    insn->uop_end = kernel->sram_end_;
    insn->gemm_from_output = from_outmem;
    const std::vector<UopKernel::LoopEntry> &loop = kernel->loop();
    if (loop.size() > 0) {
      insn->iter_out = loop[0].extent;
      insn->wgt_factor_out = loop[0].wgt_factor;
      insn->src_factor_out = loop[0].src_factor;
      insn->dst_factor_out = loop[0].dst_factor;
    } else {
      insn->iter_out = 1;
      insn->wgt_factor_out = 0;
      insn->src_factor_out = 0;
      insn->dst_factor_out = 0;
    }
    if (loop.size() > 1) {
      insn->iter_in = loop[1].extent;
      insn->wgt_factor_in = loop[1].wgt_factor;
      insn->src_factor_in = loop[1].src_factor;
      insn->dst_factor_in = loop[1].dst_factor;
    } else {
      insn->iter_in = 1;
      insn->wgt_factor_in = 0;
      insn->src_factor_in = 0;
      insn->dst_factor_in = 0;
    }
  }

  // Push ALU uop to the command buffer
  void PushALUUop(UopKernel* kernel) {
    uop_queue_.Push(kernel,
                    [this]() { this->AutoSync(); });
    if (uop_queue_.pending()) {
      VTAMemInsn* insn = insn_queue_.CreateMemInsn(VTA_MEM_ID_UOP);
      insn->opcode = VTA_OPCODE_LOAD;
      uop_queue_.FlushUopLoad(insn);
    }
    VTAAluInsn* insn = insn_queue_.CreateAluInsn();
    insn->opcode = VTA_OPCODE_ALU;
    insn->reset_reg = kernel->reset_out_;
    insn->uop_bgn = kernel->sram_begin_;
    insn->uop_end = kernel->sram_end_;
    insn->alu_opcode = kernel->opcode_;
    insn->use_imm = kernel->use_imm_;
    insn->imm = kernel->imm_val_;
    const std::vector<UopKernel::LoopEntry> &loop = kernel->loop();
    if (loop.size() == 0) {
      insn->iter_out = 1;
      insn->dst_factor_out = 0;
      insn->src_factor_out = 0;
      insn->iter_in = 1;
      insn->dst_factor_in = 0;
      insn->src_factor_in = 0;
    } else if (loop.size() == 1) {
      insn->iter_out = 1;
      insn->dst_factor_out = 0;
      insn->src_factor_out = 0;
      insn->iter_in = loop[0].extent;
      insn->dst_factor_in = loop[0].dst_factor;
      insn->src_factor_in = loop[0].src_factor;
    } else {
      insn->iter_out = loop[0].extent;
      insn->dst_factor_out = loop[0].dst_factor;
      insn->src_factor_out = loop[0].src_factor;
      insn->iter_in = loop[1].extent;
      insn->dst_factor_in = loop[1].dst_factor;
      insn->src_factor_in = loop[1].src_factor;
    }
  }

  void CheckInsnOverFlow() {
    // At each API call, we can at most commit:
    // one pending store, one pending load, and one uop
    if ((insn_queue_.count() + 4) * sizeof(VTAGenericInsn) >= VTA_MAX_XFER) {
      this->AutoSync();
    }
  }
  // Auto sync when instruction overflow
  void AutoSync() {
    this->Synchronize(1 << 31);
  }

  // Internal debug flag
  int debug_flag_{ VTA_DEBUG_DUMP_INSN }; // org = 0
  // The kernel we are currently recording
  UopKernel* record_kernel_{nullptr};
  // Micro op queue
  UopQueue<VTA_MAX_XFER, kBufferCoherent, kAlwaysCache> uop_queue_;
  // instruction queue
  InsnQueue<VTA_MAX_XFER, kBufferCoherent, kAlwaysCache> insn_queue_;
  // Device handle
  VTADeviceHandle device_{nullptr};
};

}

#endif
