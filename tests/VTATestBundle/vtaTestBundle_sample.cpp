
#include <iostream>
#include <ctime>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cassert>

#include "vta/runtime.h"
#include "vtaTestBundle.h"


#define RESET_IOCTL _IOWR('X', 101, unsigned long)

using namespace std;

#define MIN_W_SIZE 7
#define MIN_H_SIZE 7
#define OUT_TILE_SIZE_W 14
#define OUT_TILE_SIZE_H 14




void xlnk_reset() {

  int xlnkfd = open("/dev/xlnk", O_RDWR | O_CLOEXEC);
  if (xlnkfd < 0) {
    printf("Reset failed - could not open device: %d\n", xlnkfd);
    return;
  }
  if (ioctl(xlnkfd, RESET_IOCTL, 0) < 0) {
    printf("Reset failed - IOCTL failed: %d\n", errno);
  }
  close(xlnkfd);
}
#define BITSTREAM_PATH "1x16_i8w8a32_15_15_18_17.bin"
#define BS_FPGA_MAN "/sys/class/fpga_manager/fpga0/firmware"
#define BS_FPGA_MAN_FLAGS "/sys/class/fpga_manager/fpga0/flags"
void writeBitStream()
{
  fstream fs_flag;
  fs_flag.open(BS_FPGA_MAN_FLAGS,std::fstream::out);
  fs_flag << "1";  //  // flag 1 = download bitstream fully

  fstream fs_bitstream;
  fs_bitstream.open(BS_FPGA_MAN,std::fstream::out);
  fs_bitstream << BITSTREAM_PATH;
}

struct ResetAcc {
  uint32_t base_;
  uint32_t size_;
};

int reset_acc(void *param) {
  ResetAcc *p = (ResetAcc *) (param);
  VTAUopLoopBegin(p->size_, 1, 0, 0);
  VTAUopPush(0, 1, p->base_, 0, 0, 0, 0, 0);
  VTAUopLoopEnd();
  return 0;
}

struct gemmop_mxn {
  uint32_t tile_size_h_;
  uint32_t tile_size_w_;
  uint32_t src_factor_;
  uint32_t kernel_h_;
  uint32_t kernel_w_;
  uint32_t stride_h_;
  uint32_t stride_w_;
};
int gemmop_mxn(void *param) {

  struct gemmop_mxn *ctx = (struct gemmop_mxn *) (param);
  VTAUopLoopBegin(ctx->tile_size_h_, ctx->tile_size_w_, ctx->src_factor_ * ctx->stride_h_, 0);
  VTAUopLoopBegin(ctx->kernel_h_, 0, ctx->src_factor_, ctx->kernel_w_);

  for (int dx = 0; dx < ctx->kernel_w_; dx++) {
    for (int j = 0; j < ctx->tile_size_w_; j++) {
      VTAUopPush(0, 0, j, j*ctx->stride_w_ + dx, dx, 0, 0, 0);
    }
  }
  VTAUopLoopEnd();
  VTAUopLoopEnd();

  return 0;
}

// alu func
struct ShiftContext {
  uint32_t base_;
  uint32_t size_;
  uint8_t shift_;
};
int alu_shr(void *param) {
  ShiftContext *p = (ShiftContext *) (param);
  VTAUopLoopBegin(p->size_, 1, 1, 0);
  VTAUopPush(1, 0, p->base_, 0, 0, 3, 1, p->shift_);
  VTAUopLoopEnd();
  return 0;
}

struct AddContext {
  uint32_t dst_;
  uint32_t src_;
  uint32_t size_;
};
int alu_add(void *param) {
  AddContext *p = (AddContext *) (param);
  VTAUopLoopBegin(p->size_, 1, 0, 0);
  VTAUopPush(1, 0, p->dst_, p->src_, 0, 2, 0, 0);
  VTAUopLoopEnd();
  return 0;
}
int alu_max(void *param) {
  ResetAcc *p = (ResetAcc *) (param);
  VTAUopLoopBegin(p->size_, 1, 1, 0);
  VTAUopPush(1, 0, p->base_, 0, 0, 1, 1, -128);
  VTAUopLoopEnd();
  return 0;
}

int alu_relu(void *param) {
  ResetAcc *p = (ResetAcc *) (param);
  VTAUopLoopBegin(p->size_, 1, 1, 0);
  VTAUopPush(1, 0, p->base_, 0, 0, 1, 1, 0);
  VTAUopLoopEnd();
  return 0;
}

int alu_min(void *param) {
  ResetAcc *p = (ResetAcc *) (param);
  VTAUopLoopBegin(p->size_, 1, 1, 0);
  VTAUopPush(1, 0, p->base_, 0, 0, 0, 1, 127);
  VTAUopLoopEnd();
  return 0;
}


void conv_kernel(uint32_t sizeOutTileH,
                 uint32_t sizeOutTileW,
                 uint32_t pad_size,
                 uint32_t stride_size,
                 uint32_t x_pad_before,
                 uint32_t x_pad_after,
                 uint32_t y_pad_before,
                 uint32_t y_pad_after,
                 uint32_t outputWOffset,
                 uint32_t outputHOffset,
                 uint32_t sizeInTileH,
                 uint32_t sizeInTileW,
                 uint32_t numInputChannel,
                 uint32_t inputHStride,
                 uint32_t inputWStride,
                 void *vta_input_buf,
                 uint32_t KN,
                 uint32_t KH,
                 uint32_t KW,
                 void *vta_filter_buf,
                 uint32_t out_h,
                 uint32_t out_w,
                 void *vta_output_buf,
                 bool relu,
                 uint8_t shift,
                 bool bias,
                 void *vta_bias_buf

) {

  for (uint32_t filterIdx = 0; filterIdx < KN / 16; filterIdx++) {
    VTACommandHandle vtaCmdH{nullptr};
    vtaCmdH = VTATLSCommandHandle();
    VTASetDebugMode(vtaCmdH, VTA_DEBUG_DUMP_INSN | VTA_DEBUG_DUMP_UOP);

// for very short width input
    uint32_t w_dummy = 0;
    uint32_t outDummyH = 0;
    uint32_t inDummyH = 0;
    uint32_t outDummyW = 0;
    uint32_t inDummyW = 0;

    uint32_t resizedOutTileH = (sizeInTileH-KH) / stride_size + 1;
    uint32_t resizedOutTileW = (sizeInTileW-KW) / stride_size + 1;
    uint32_t resizedInTileH =  (resizedOutTileH - 1) * stride_size + KH;
    uint32_t resizedInTileW =  (resizedOutTileW - 1) * stride_size + KW;

    if(resizedOutTileH < MIN_H_SIZE)
    {
      outDummyH = MIN_H_SIZE - resizedOutTileH;
      resizedOutTileH = MIN_H_SIZE;
      resizedInTileH = (resizedOutTileH - 1) * stride_size + KH;
      inDummyH = resizedInTileH - sizeInTileH;
    }

    if(resizedOutTileW < MIN_W_SIZE)
    {
      outDummyW = MIN_W_SIZE - resizedOutTileW;
      resizedOutTileW = MIN_W_SIZE;
      resizedInTileW = (resizedOutTileW - 1) * stride_size + KW;
      inDummyW = resizedInTileW - sizeInTileW;

    }

// acc buffer reset
    void *uopHandle_reset[2];
    uopHandle_reset[0] = nullptr;

    ResetAcc reset;
    reset.base_ = 0;
    reset.size_ = resizedOutTileH * resizedOutTileW;
    VTAPushGEMMOp(uopHandle_reset, &reset_acc, &reset, 0);

    uint32_t inputWOffset = (int) (outputWOffset * stride_size - pad_size ) < 0 ? 0 : outputWOffset * stride_size - pad_size;
    uint32_t inputHOffset = (int) (outputHOffset * stride_size - pad_size ) < 0 ? 0 : outputHOffset * stride_size - pad_size;

    for (uint32_t channelIdx = 0; channelIdx < numInputChannel / 16; channelIdx++) {
      uint32_t sram_offset = channelIdx * inputHStride * inputWStride + inputHOffset * inputWStride + inputWOffset;
      uint32_t filter_offset = filterIdx * (numInputChannel/16) * KH * KW + channelIdx * KH * KW;
      uint32_t y_size = (inputHStride - inputHOffset + y_pad_before) < resizedInTileH ?
                        inputHStride - inputHOffset : resizedInTileH - y_pad_before;
      uint32_t x_size = (inputWStride - inputWOffset + x_pad_before) < resizedInTileW ?
                        inputWStride - inputWOffset : resizedInTileW - x_pad_before;

      VTADepPush(vtaCmdH, 2, 1);
      VTADepPop(vtaCmdH, 2, 1);
      VTALoadBuffer2D(vtaCmdH, vta_input_buf,
                      sram_offset,
                      x_size,
                      y_size,
                      inputWStride,
                      x_pad_before,
                      y_pad_before,
                      resizedInTileW - x_pad_before - x_size,
                      resizedInTileH - y_pad_before - y_size,
                      0,        //index
                      2);    //type

      VTALoadBuffer2D(vtaCmdH, vta_filter_buf,
                      filter_offset,
                      KH * KW,
                      1,
                      KH * KW,
                      0, 0, 0, 0, 0, 1);

      VTADepPush(vtaCmdH, 1, 2);
      VTADepPop(vtaCmdH, 1, 2);

      void *uopHandle_gemm[2];
      uopHandle_gemm[0] = nullptr;

      struct gemmop_mxn gemm_ctx;
      gemm_ctx.tile_size_h_ = resizedOutTileH;
      gemm_ctx.tile_size_w_ = resizedOutTileW;
      gemm_ctx.src_factor_ = resizedInTileW;
      gemm_ctx.kernel_h_ = KH;
      gemm_ctx.kernel_w_ = KW;
      gemm_ctx.stride_h_ = stride_size;
      gemm_ctx.stride_w_ = stride_size;

      VTAPushGEMMOp(uopHandle_gemm, &gemmop_mxn, &gemm_ctx, 0);



    }

    if(bias){

      uint32_t sram_offset = resizedOutTileH * resizedOutTileW;
      VTALoadBuffer2D(vtaCmdH, (int32_t*)vta_bias_buf,
                      filterIdx,
                      16,
                      1,
                      16,
                      0,
                      0,
                      0,
                      0,
                      sram_offset,        //index
                      3);    //type



      AddContext add_ctx;
      add_ctx.dst_ = 0;
      add_ctx.size_ = resizedOutTileH * resizedOutTileW;
      add_ctx.src_ = resizedOutTileH * resizedOutTileW;
      void *uopHandle_add[2];
      uopHandle_add[0] = nullptr;
      VTAPushALUOp(uopHandle_add, &alu_add, &add_ctx, 0);


    }



    ShiftContext shift_ctx;
    shift_ctx.base_ = 0;
    shift_ctx.size_ = resizedOutTileH * resizedOutTileW;
    shift_ctx.shift_ = shift;

    void *uopHandle_shr[2];
    uopHandle_shr[0] = nullptr;
    VTAPushALUOp(uopHandle_shr, &alu_shr, &shift_ctx, 0);


    void *uopHandle_max[2];
    uopHandle_max[0] = nullptr;
    if(!relu) {
      VTAPushALUOp(uopHandle_max, &alu_max, &reset, 0);
    }
    else{
      VTAPushALUOp(uopHandle_max, &alu_relu, &reset, 0);
    }
    void *uopHandle_min[2];
    uopHandle_min[0] = nullptr;
    VTAPushALUOp(uopHandle_min, &alu_min, &reset, 0);


    VTADepPush(vtaCmdH, 2, 3);
    VTADepPop(vtaCmdH, 2, 3);

    if (outDummyW == 0) {
      VTAStoreBuffer2D(vtaCmdH,
                       0,
                       4,
                       vta_output_buf,
                       filterIdx * out_w * out_h + outputHOffset * out_w + outputWOffset,
                       sizeOutTileW + w_dummy,
                       sizeOutTileH,
                       out_w);
    } else {
      for (int hIdx = 0; hIdx < sizeOutTileH; hIdx++) {
        VTAStoreBuffer2D(vtaCmdH,
                         0 + hIdx * resizedOutTileW,
                         4,
                         vta_output_buf,
                         filterIdx * out_w * out_h+(outputHOffset + hIdx) * out_w + outputWOffset,
                         sizeOutTileW,
                         1,
                         out_w);
      }
    }
    VTASynchronize(vtaCmdH, 1 << 31);
  }
}





int convolution(int8_t* input, int8_t *kernel, int8_t *output, int N, int H, int W, int C, int KN, int KH, int KW, int pad_size,
                int stride_size, bool doRelu, bool doBias, uint8_t shift, int out_h, int out_w) {
  int is_diff = 0;
  srand(time(NULL));

  //N, H, W, C
  //N == 1
  //group == 1
  struct timeval tv_s, tv_e;



  int8_t *IN_NHWC = input;

  //transpose input NHWC -> NCHW
  int8_t *IN_NCHW = (int8_t *)malloc(N*C*H*W);
  for (int n = 0; n < N; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int c = 0; c < C; c++) {
          *(IN_NCHW + n*C*H*W + c*H*W + h*W + w)=*(IN_NHWC + n*H*W*C + h*W*C + w*C + c);
        }
      }
    }
  }

  // reshape input NCHW -> N{C//16}16HW
  int8_t *in_NC_16HW = IN_NCHW;

  //transpose input N{C//16}16HW -> N{C//16}HW16
  int8_t *in_NC_HW16 = (int8_t *)malloc(N * (C / 16) * H * W * 16);
  for (int n = 0; n < N; n++) {
    for (int c_ = 0; c_ < C / 16; c_++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          for (int c = 0; c < 16; c++) {
            *(in_NC_HW16 + n*(C/16)*H*W*16 + c_*H*W*16 + h*W*16 + w*16 + c) =
                *(in_NC_16HW + n*(C/16)*16*H*W + c_*16*H*W + c*H*W + h*W + w);
          }
        }
      }
    }
  }





  // filter NHWC
  int8_t *filter_NHWC = kernel;

  // reshape filter NHWC -> {N//16}16HW{C//16}16
  assert(KN % 16 == 0);

  int8_t* filter_N_16HWC_16 = filter_NHWC;

  // transpose filter {N//16}16HW{C//16}16 =>  {N//16}{C//16}HW1616
  int8_t* filter_N_C_HW1616 = (int8_t*) malloc ((KN / 16) *(C / 16)*KH*KW*16*16);
  for (int n_ = 0; n_ < KN / 16; n_++) {
    for (int c_ = 0; c_ < C / 16; c_++) {
      for (int h = 0; h < KH; h++) {
        for (int w = 0; w < KW; w++) {
          for (int n = 0; n < 16; n++) {
            for (int c = 0; c < 16; c++) {
              *(filter_N_C_HW1616 + n_*(C/16)*KH*KW*16*16 + c_ *KH*KW*16*16 + h*KW*16*16 + w*16*16 + n*16 + c) =
                  *(filter_N_16HWC_16 + n_*16*KH*KW*C + n*KH*KW*C + h*KW*C + w*C + c_*16 + c);
            }
          }
        }
      }
    }
  }
  int32_t *bias;
  if(doBias) {
    auto locBias = KN * KW * KH * C;
    assert(locBias % 4 == 0);
    locBias /= 4;

    bias = (int32_t *) kernel + locBias;
  }
  int8_t *out_VTA = (int8_t *) malloc(out_h * out_w * KN);



  // VTA DRAM alloc
  // vta's input buffer, as the same size of in_NC_HW16[N][C/16][H][W][16];
  void *vta_input_buf = VTABufferAlloc(N * C * H * W);
  // vta's filter buffer, as the same size of filter_N_C_HW1616[KN/16][C/16][KH][KW][16][16];
  void *vta_filter_buf = VTABufferAlloc(KN * KH * KW * C);
  void *vta_output_buf = VTABufferAlloc(out_h * out_w * KN);
  void *vta_bias_buf;
  if(doBias) {
    vta_bias_buf = VTABufferAlloc(KN * 4);
  }
  // VTA Buffer copy
  VTABufferCopy(in_NC_HW16, 0, vta_input_buf, 0, N * C * H * W, 1);
  VTABufferCopy(filter_N_C_HW1616, 0, vta_filter_buf, 0, KN * KH * KW * C, 1);

  if(doBias) {
    VTABufferCopy(bias, 0, vta_bias_buf, 0, KN * 4, 1);
  }

  uint32_t nTileW, nTileH;
  uint32_t sizeOutTileW = OUT_TILE_SIZE_W;
  uint32_t sizeOutTileH = OUT_TILE_SIZE_H;

  uint32_t sizeInTileW = (sizeOutTileW - 1) * stride_size + KW;
  uint32_t sizeInTileH = (sizeOutTileH - 1) * stride_size + KH;

  int isRemainTileW = (out_w % sizeOutTileW) ? 1 : 0;
  int isRemainTileH = (out_h % sizeOutTileH) ? 1 : 0;
  nTileW = out_w / sizeOutTileW + isRemainTileW;
  nTileH = out_h / sizeOutTileH + isRemainTileH;




  for (int tileH = 0; tileH < nTileH; tileH++) {
    for (int tileW = 0; tileW < nTileW; tileW++) {

      uint32_t x_pad_before = tileW == 0 ? pad_size : 0;
      uint32_t x_pad_after = tileW == (nTileW - 1) ? pad_size : 0;
      uint32_t y_pad_before = tileH == 0 ? pad_size : 0;
      uint32_t y_pad_after = tileH == (nTileH - 1) ? pad_size : 0;
      uint32_t outputWOffset = tileW * sizeOutTileW;
      uint32_t outputHOffset = tileH * sizeOutTileH;

      uint32_t resizedOutTileH = sizeOutTileH;
      uint32_t resizedInTileH = sizeInTileH;
      if (tileH == nTileH-1) {
        resizedOutTileH = out_h - tileH * sizeOutTileH;
        resizedInTileH = (resizedOutTileH -1) * stride_size + KH;
      }

      if ((tileW != nTileW - 1)) {
        conv_kernel(resizedOutTileH,
                    sizeOutTileW,
                    pad_size,
                    stride_size,
                    x_pad_before,
                    x_pad_after,
                    y_pad_before,
                    y_pad_after,
                    outputWOffset,
                    outputHOffset,
                    resizedInTileH,
                    sizeInTileW,
                    C,
                    H,
                    W,
                    vta_input_buf,
                    KN,
                    KH,
                    KW,
                    vta_filter_buf,
                    out_h,
                    out_w,
                    vta_output_buf,
                    doRelu,
                    shift,
                    doBias,
                    vta_bias_buf
        );
      } else {
        uint32_t sizeOutTileBoundaryW = out_w - tileW * sizeOutTileW;
        uint32_t sizeInTileBoundaryW = (sizeOutTileBoundaryW - 1) * stride_size + KW ;

        conv_kernel(resizedOutTileH,
                    sizeOutTileBoundaryW,
                    pad_size,
                    stride_size,
                    x_pad_before,
                    x_pad_after,
                    y_pad_before,
                    y_pad_after,
                    outputWOffset,
                    outputHOffset,
                    resizedInTileH,
                    sizeInTileBoundaryW,
                    C,
                    H,
                    W,
                    vta_input_buf,
                    KN,
                    KH,
                    KW,
                    vta_filter_buf,
                    out_h,
                    out_w,
                    vta_output_buf,
                    doRelu,
                    shift,
                    doBias,
                    vta_bias_buf
        );

      }
    }
  }
  VTABufferCopy(vta_output_buf, 0, out_VTA, 0, out_h * out_w * KN, 2);

  // end case
  VTABufferFree(vta_input_buf);
  VTABufferFree(vta_output_buf);
  VTABufferFree(vta_filter_buf);


  VTARuntimeShutdown();








  for (int b = 0; b < N; b++) {
    for (int f = 0; f < KN; f++) {
      for (int i = 0; i < out_h; i++) {    //h
        for (int j = 0; j < out_w; j++) {    //w

          int out_ch = f / 16;
          int oo = f % 16;
          auto value =*(out_VTA + (out_ch * out_h * out_w * 16 + i * out_w * 16 + j * 16 + oo));
          *(output + b * out_h * out_w * KN + i * out_w * KN + j * KN + f) = value;


        }
      }
    }
  }

  free(out_VTA);

  free(IN_NCHW);
  free(in_NC_HW16);
  free(filter_N_C_HW1616);



}



SymbolTableEntry symbolTableEntry[2]={{"input",0,3136,'1'},{"results",3136,3136,'1'}};
BundleConfig vtaTestBundle_config={4736, 6272, 0, 64, 2, symbolTableEntry};
int vtaTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  writeBitStream();
  int8_t *conv_res = (int8_t *)malloc(3136);
  convolution(mutableWeight + 0, constantWeight + 0, conv_res, 1, 14, 14, 16, 16, 3, 3, 1, 1, 0, 1, 8, 14, 14);
  convolution(conv_res, constantWeight + 2368, mutableWeight + 3136 , 1, 14, 14, 16, 16, 3, 3, 1, 1, 0,1,  8, 14, 14);
  free(conv_res);  return 0;
}

