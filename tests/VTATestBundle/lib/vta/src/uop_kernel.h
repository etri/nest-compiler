#ifndef NPULIB_API_UOP_KERNEL_H_
#define NPULIB_API_UOP_KERNEL_H_


#include <vta/hw_spec.h>
#include <vector>

namespace vta {
class UopKernel {
 public:
  /*! \brief Loop information. */
  struct LoopEntry {
    uint32_t extent;
    uint32_t dst_factor;
    uint32_t src_factor;
    uint32_t wgt_factor;
  };
  /*!
   * \brief Construct UopKernel with signature.
   * \param signature The pointer to signature.
   * \param nbytes Number of bytes.
   */
  UopKernel(const char* signature, int nbytes)
      : signature_(signature, signature + nbytes) {
  }
  /*!
   * \brief Verify if the signature is correct.
   * \param signature Signature ptr.
   * \param nbytes Number of bytes.
   */
  bool MatchSignature(void* signature, int nbytes) const {
    if (static_cast<size_t>(nbytes) != signature_.size()) return false;
    return memcmp(signature, signature_.data(), nbytes) == 0;
  }
  /*! \return Whether the kernel is cached in SRAM. */
  bool cached() const {
    return sram_begin_ != sram_end_;
  }
  /*! \return The length of the micro op sequence. */
  size_t size() const {
    return seq_.size();
  }
  /*! \return The micro-op data. */
  const VTAUop* data() const {
    return seq_.data();
  }
  /*! \return The loop structure. */
  const std::vector<LoopEntry>& loop() const {
    return loop_;
  }
  /*!
   * \brief Declare loop start.
   * \param extent The loop extent.
   * \param dst_factor Loop factor of accum index.
   * \param src_factor Loop factor of input index
   * \param wgt_factor Loop factor of weight index.
   */
  void PushLoopBegin(uint32_t extent,
                     uint32_t dst_factor,
                     uint32_t src_factor,
                     uint32_t wgt_factor) {
    LoopEntry le;
    le.extent = extent;
    le.dst_factor = dst_factor;
    le.src_factor = src_factor;
    le.wgt_factor = wgt_factor;
    CHECK_EQ(seq_.size(), 0U);
    CHECK_LT(loop_.size(), 2U);
    loop_.push_back(le);
    ++loop_ptr_;
  }
  /*!
   * \brief Declare loop end.
   */
  void PushLoopEnd() {
    --loop_ptr_;
  }
  /*!
   * \brief Push micro op into kernel.
   * \param mode Set to GEMM mode if set to 0, ALU mode is set to 1.
   * \param reset_out Resets the accum to 0.
   * \param dst_index The accum memory index.
   * \param src_index The input memory (gemm) / accum memory (alu) index.
   * \param wgt_index The weight memory index.
   * \param opcode The ALU opcode.
   * \param use_imm Use immediate in ALU mode if set to true.
   * \param imm_val Immediate value in ALU mode.
   */
  void Push(uint32_t mode,
            uint32_t reset_out,
            uint32_t dst_index,
            uint32_t src_index,
            uint32_t wgt_index,
            uint32_t opcode,
            uint32_t use_imm,
            int32_t imm_val) {
    // The loop nest structure
    VerifyDep(dst_index);
    VTAUop op;
    op.dst_idx = dst_index;
    op.src_idx = src_index;
    op.wgt_idx = wgt_index;
    seq_.push_back(op);
    // Ensure that mode is consistent if set
    if (mode_ == 0xFFFFFFFF) {
      mode_ = mode;
    } else {
      CHECK(mode_ == mode);
    }
    // Set reset_out field if unset
    if (reset_out_ == 0xFFFFFFFF) {
      reset_out_ = reset_out;
    } else {
      CHECK(reset_out_ == reset_out);
    }
    // Check kernel op and imm/imm_val in ALU mode
    if (mode == 1) {
      if (opcode_ == 0xFFFFFFFF) {
        opcode_ = opcode;
        use_imm_ = use_imm;
        imm_val_ = imm_val;
      } else {
        CHECK(opcode_ == opcode);
        CHECK(use_imm_ == use_imm);
        CHECK(imm_val_ == imm_val);
      }
    }
  }
  /*! \brief Dump kernel micro ops to stdout. */
  void Dump() {
    uint32_t size = seq_.size();
    printf("There are %u uops\n", size);
    for (uint32_t i = 0; i < size; ++i) {
      printf("[%04u]\t acc=%u, inp=%u, wgt=%u\n",
             i,
             seq_[i].dst_idx,
             seq_[i].src_idx,
             seq_[i].wgt_idx);
    }
    printf("\n");
  }

 public:
  // The kernel's mode, opcode, immediate setting and value
  uint32_t mode_{0xFFFFFFFF};  // UOP type: 0xFFFFFFFF - unset, 0 - GEMM, 1 - ALU
  uint32_t opcode_{0xFFFFFFFF};
  uint32_t reset_out_{0xFFFFFFFF};
  bool use_imm_{false};
  int16_t imm_val_{0};

 private:
  // Verify that we don't write to the same acc_mem index two cycles in a row
  void VerifyDep(uint32_t dst_index) {
    size_t step = std::min(static_cast<size_t>(2U), seq_.size());
    for (size_t i = seq_.size() - step; i < seq_.size(); ++i) {
      CHECK(seq_[i].dst_idx != dst_index);
    }
  }
  // The uop buffer
  template<int, bool, bool>
  friend class UopQueue;
  friend class CommandQueue;
  // SRAM location if begin != end
  uint32_t sram_begin_{0};
  uint32_t sram_end_{0};
  // The signature used for verification
  std::vector<char> signature_;
  // Internal sequence
  std::vector<VTAUop> seq_;
  // The loop nest structure specific to ALU instructions
  std::vector<LoopEntry> loop_;
  // The loop pointer
  size_t loop_ptr_{0};
};

// Internal kernel structure
class UopKernelMap {
 public:
  // Simple hash map
  UopKernel** Get(void* signature,
                  int nbytes) {
    //printf("kmap_size=%d",kmap_.size() );
    uint32_t key = 0;
    CHECK(nbytes == 0 || nbytes == sizeof(int));
    if (nbytes == sizeof(int)) {
      memcpy(&key, signature, sizeof(int));
      key = key + 1;
    }
    CHECK_LT(key, 100);
    if (kmap_.size() <= key) {
      kmap_.resize(key + 1, nullptr);
    }
    //printf("UK get - key:%d, sig = %p, n=%d ... kmapsize=%d \n ", key, signature,nbytes,kmap_.size());
   // for(int i=0;i<km)
    return &(kmap_[key]);
  }

 private:
  std::vector<UopKernel*> kmap_;
};


}

#endif
