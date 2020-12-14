#ifndef NPULIB_API_DATA_BUF_H_
#define NPULIB_API_DATA_BUF_H_


#include "common.h"
#include <vta/driver.h>
#include <dmlc/logging.h>
#include "data_buf.h"
namespace vta {

/*!
 * \brief Data buffer represents data on CMA.
 */
struct DataBuffer {
  /*! \return Virtual address of the data. */
  void* virt_addr() const {
    return data_;
  }
  /*! \return Physical address of the data. */
  vta_phy_addr_t phy_addr() const {
    return phy_addr_;
  }
  /*!
   * \brief Invalidate the cache of given location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void InvalidateCache(size_t offset, size_t size) {
    if (!kBufferCoherent && kAlwaysCache) {
      VTAInvalidateCache(reinterpret_cast<char *>(data_) + offset,
                         phy_addr_ + offset,
                         size);
    }
  }
  /*!
   * \brief Invalidate the cache of certain location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void FlushCache(size_t offset, size_t size) {
    if (!kBufferCoherent && kAlwaysCache) {
      VTAFlushCache(reinterpret_cast<char *>(data_) + offset,
                    phy_addr_ + offset,
                    size);
    }
  }
  /*!
   * \brief Performs a copy operation from host memory to buffer allocated with VTAMemAlloc.
   * \param dst The desination buffer in FPGA-accessible memory. Has to be allocated with VTAMemAlloc().
   * \param src The source buffer in host memory.
   * \param size Size of the region in Bytes.
   */
  void MemCopyFromHost(void* dst, const void* src, size_t size) {
    VTAMemCopyFromHost(dst, src, size);
  }
  /*!
   * \brief Performs a copy operation from buffer allocated with VTAMemAlloc to host memory.
   * \param dst The desination buffer in host memory.
   * \param src The source buffer in FPGA-accessible memory. Has to be allocated with VTAMemAlloc().
   * \param size Size of the region in Bytes.
   */
  void MemCopyToHost(void* dst, const void* src, size_t size) {
    VTAMemCopyToHost(dst, src, size);
  }
  /*!
   * \brief Allocate a buffer of a given size.
   * \param size The size of the buffer.
   */
  static DataBuffer* Alloc(size_t size) {
    void* data = VTAMemAlloc(size, kAlwaysCache);
    CHECK(data != nullptr);
    DataBuffer* buffer = new DataBuffer();
    buffer->data_ = data;
    buffer->phy_addr_ = VTAMemGetPhyAddr(data);
    return buffer;
  }
  /*!
   * \brief Free the data buffer.
   * \param buffer The buffer to be freed.
   */
  static void Free(DataBuffer* buffer) {
    VTAMemFree(buffer->data_);
    delete buffer;
  }
  /*!
   * \brief Create data buffer header from buffer ptr.
   * \param buffer The buffer pointer.
   * \return The corresponding data buffer header.
   */
  static DataBuffer* FromHandle(const void* buffer) {
    return const_cast<DataBuffer*>(
        reinterpret_cast<const DataBuffer*>(buffer));
  }

 private:
  /*! \brief The internal data. */
  void* data_;
  /*! \brief The physical address of the buffer, excluding header. */
  vta_phy_addr_t phy_addr_;
};

}

#endif