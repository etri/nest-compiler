#include <cassert>
#include <cstring>
#include <vector>
#include <memory>

#include <vta/driver.h>
#include <vta/hw_spec.h>
#include <vta/runtime.h>

#include "vta/data_buf.h"
#include "dat_queue.h"
#include "cmd_queue.h"
#include "uop_kernel.h"
#include "api.h"

void* VTABufferAlloc(size_t size) {
  return vta::DataBuffer::Alloc(size);
}

void VTABufferFree(void* buffer) {
  vta::DataBuffer::Free(vta::DataBuffer::FromHandle(buffer));
}

void VTABufferCopy(const void* from,
                   size_t from_offset,
                   void* to,
                   size_t to_offset,
                   size_t size,
                   int kind_mask) {
  vta::DataBuffer* from_buffer = nullptr;
  vta::DataBuffer* to_buffer = nullptr;

  //printf("from:%p (off:%d), to:%p (off:%d) size=%d, mask=%d\n", from,from_offset,to,to_offset,size,kind_mask);

  if (kind_mask & 2) {
    from_buffer = vta::DataBuffer::FromHandle(from);
    from = from_buffer->virt_addr();
  }
  if (kind_mask & 1) {
    to_buffer = vta::DataBuffer::FromHandle(to);
    to = to_buffer->virt_addr();
  }

  if (from_buffer) {
    // This is an FPGA to host mem transfer
    from_buffer->InvalidateCache(from_offset, size);
    from_buffer->MemCopyToHost(static_cast<char*>(to) + to_offset,
                                   static_cast<const char*>(from) + from_offset,
                                   size);
  } else if (to_buffer) {
    // This is a host to FPGA mem transfer
    to_buffer->MemCopyFromHost(static_cast<char*>(to) + to_offset,
                               static_cast<const char*>(from) + from_offset,
                               size);
    to_buffer->FlushCache(to_offset, size);
  }
}

VTACommandHandle VTATLSCommandHandle() {
  return vta::CommandQueue::ThreadLocal().get();
}

void VTARuntimeShutdown() {
  vta::CommandQueue::Shutdown();
}

void VTASetDebugMode(VTACommandHandle cmd, int debug_flag) {
  static_cast<vta::CommandQueue*>(cmd)->
      SetDebugFlag(debug_flag);
}

void* VTABufferCPUPtr(VTACommandHandle cmd, void* buffer) {
  return vta::DataBuffer::FromHandle(buffer)->virt_addr();
}

void VTAWriteBarrier(VTACommandHandle cmd,
                     void* buffer,
                     uint32_t elem_bits,
                     uint32_t start,
                     uint32_t extent) {
  static_cast<vta::CommandQueue*>(cmd)->
      WriteBarrier(buffer, elem_bits, start, extent);
}

void VTAReadBarrier(VTACommandHandle cmd,
                    void* buffer,
                    uint32_t elem_bits,
                    uint32_t start,
                    uint32_t extent) {
  static_cast<vta::CommandQueue*>(cmd)->
      ReadBarrier(buffer, elem_bits, start, extent);
}

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
                     uint32_t dst_memory_type) {
  static_cast<vta::CommandQueue*>(cmd)->
      LoadBuffer2D(src_dram_addr, src_elem_offset,
                   x_size, y_size, x_stride,
                   x_pad_before, y_pad_before,
                   x_pad_after, y_pad_after,
                   dst_sram_index, dst_memory_type);
}

void VTAStoreBuffer2D(VTACommandHandle cmd,
                      uint32_t src_sram_index,
                      uint32_t src_memory_type,
                      void* dst_dram_addr,
                      uint32_t dst_elem_offset,
                      uint32_t x_size,
                      uint32_t y_size,
                      uint32_t x_stride) {
  static_cast<vta::CommandQueue*>(cmd)->
      StoreBuffer2D(src_sram_index, src_memory_type,
                    dst_dram_addr, dst_elem_offset,
                    x_size, y_size, x_stride);
}

void VTAUopPush(uint32_t mode,
                uint32_t reset_out,
                uint32_t dst_index,
                uint32_t src_index,
                uint32_t wgt_index,
                uint32_t opcode,
                uint32_t use_imm,
                int32_t imm_val) {
    //printf("VTAuoppush");
  vta::CommandQueue::ThreadLocal()->record_kernel()
      ->Push(mode, reset_out, dst_index, src_index,
             wgt_index, opcode, use_imm, imm_val);
}

void VTAUopLoopBegin(uint32_t extent,
                     uint32_t dst_factor,
                     uint32_t src_factor,
                     uint32_t wgt_factor) {
                       //printf("VTAuopLoopB");
  vta::CommandQueue::ThreadLocal()->record_kernel()
      ->PushLoopBegin(extent, dst_factor, src_factor, wgt_factor);
}

void VTAUopLoopEnd() {
  //printf("VTAuopLoopE");
  vta::CommandQueue::ThreadLocal()->record_kernel()
      ->PushLoopEnd();
}

int VTAPushGEMMOp(void** uop_handle,
                  int (*finit)(void*),
                  void* signature,
                  int nbytes,
                  bool from_outmem) {
	//printf("VTAPushGEMMOp %p, %p, %p, %d\n",uop_handle, finit, signature, nbytes);
	
  vta::CommandQueue::ThreadLocal()->
      PushGEMMOp(uop_handle, finit, signature, nbytes, from_outmem);
  return 0;
}

int VTAPushALUOp(void** uop_handle,
                 int (*finit)(void*),
                 void* signature,
                 int nbytes) {
  vta::CommandQueue::ThreadLocal()->
      PushALUUop(uop_handle, finit, signature, nbytes);
  return 0;
}

int VTADepPush(VTACommandHandle cmd, int from_qid, int to_qid) {
  static_cast<vta::CommandQueue*>(cmd)->
      DepPush(from_qid, to_qid);
  return 0;
}

int VTADepPop(VTACommandHandle cmd, int from_qid, int to_qid) {
  static_cast<vta::CommandQueue*>(cmd)->
      DepPop(from_qid, to_qid);
  return 0;
}

void VTASynchronize(VTACommandHandle cmd, uint32_t wait_cycles) {
  static_cast<vta::CommandQueue*>(cmd)->
      Synchronize(wait_cycles);
}
