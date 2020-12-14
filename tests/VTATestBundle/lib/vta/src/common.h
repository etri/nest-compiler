#ifndef NPULIB_COMMON_H
#define NPULIB_COMMON_H




namespace vta
{

/*! \brief Enable coherent access of data buffers between VTA and CPU */
static const bool kBufferCoherent = VTA_COHERENT_ACCESSES;
/*! \brief Always cache buffers (otherwise, write back to DRAM from CPU) */
static const bool kAlwaysCache = true;

// used for queue
enum PipelineStage : int {
  kNoneStage = 0,
  kLoadStage = 1,
  kComputeStage = 2,
  kStoreStage = 3
};


}

#endif
