#include <cassert>
#include <cstring>
#include <vector>
#include <memory>

#include "vta/data_buf.h"
#include "dat_queue.h"
#include "cmd_queue.h"


void* VTABufferAlloc(size_t size);

void VTABufferFree(void* buffer) ;

void VTABufferCopy(const void* from,
                   size_t from_offset,
                   void* to,
                   size_t to_offset,
                   size_t size,
                   int kind_mask);

VTACommandHandle VTATLSCommandHandle() ;

void VTARuntimeShutdown();
void VTASetDebugMode(VTACommandHandle cmd, int debug_flag) ;

void* VTABufferCPUPtr(VTACommandHandle cmd, void* buffer) ;

void VTAWriteBarrier(VTACommandHandle cmd,
                     void* buffer,
                     uint32_t elem_bits,
                     uint32_t start,
                     uint32_t extent) ;
void VTAReadBarrier(VTACommandHandle cmd,
                    void* buffer,
                    uint32_t elem_bits,
                    uint32_t start,
                    uint32_t extent) ;
void VTALoadBuffer2D(VTACommandHandle cmd,
                     void* src_dram_addr,
                     uint32_t src_elem_offset,
                     uint32_t x_size,
                     uint32_t y_size,
                     uint32_t x_stride,
                     uint32_t x_pad_before,
                     uint32_t y_pad_before,
                     uint32_t x_pad_after,
                     uint32_t y_pad_after,
                     uint32_t dst_sram_index,
                     uint32_t dst_memory_type) ;

void VTAStoreBuffer2D(VTACommandHandle cmd,
                      uint32_t src_sram_index,
                      uint32_t src_memory_type,
                      void* dst_dram_addr,
                      uint32_t dst_elem_offset,
                      uint32_t x_size,
                      uint32_t y_size,
                      uint32_t x_stride) ;

void VTAUopPush(uint32_t mode,
                uint32_t reset_out,
                uint32_t dst_index,
                uint32_t src_index,
                uint32_t wgt_index,
                uint32_t opcode,
                uint32_t use_imm,
                int32_t imm_val) ;

void VTAUopLoopBegin(uint32_t extent,
                     uint32_t dst_factor,
                     uint32_t src_factor,
                     uint32_t wgt_factor) ;

void VTAUopLoopEnd() ;

int VTAPushGEMMOp(void** uop_handle,
                  int (*finit)(void*),
                  void* signature,
                  int nbytes,
                  bool from_outmem
                  ) ;
int VTAPushALUOp(void** uop_handle,
                 int (*finit)(void*),
                 void* signature,
                 int nbytes) ;

int VTADepPush(VTACommandHandle cmd, int from_qid, int to_qid) ;

int VTADepPop(VTACommandHandle cmd, int from_qid, int to_qid) ;
void VTASynchronize(VTACommandHandle cmd, uint32_t wait_cycles);