

#include "vta/runtime.h"
#include "VTABundle.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include "vtaConvTestBundle.h"
SymbolTableEntry symbolTableEntry_vtaConvTestBundle[2]={{"inputP",0,3136,'1'},{"outP",3136,3136,'1'}};
BundleConfig vtaConvTestBundle_config = {4736, 6272, 0, 64, 2, symbolTableEntry_vtaConvTestBundle};
int8_t* filterP;
int8_t* biasP;
int8_t* filter2P;
int8_t* bias2P;
extern VTACommandHandle vtaCmdH;

void vtaConvTestMainEntry_load_module(uint8_t *constantWeight){
  filterP = (int8_t *)VTABufferAlloc(2304);
  VTABufferCopy((int8_t *)(constantWeight + 0), 0, filterP, 0, 2304, 1);
  biasP = (int8_t *)VTABufferAlloc(64);
  VTABufferCopy((int8_t *)(constantWeight + 2304), 0, biasP, 0, 64, 1);
  filter2P = (int8_t *)VTABufferAlloc(2304);
  VTABufferCopy((int8_t *)(constantWeight + 2368), 0, filter2P, 0, 2304, 1);
  bias2P = (int8_t *)VTABufferAlloc(64);
  VTABufferCopy((int8_t *)(constantWeight + 4672), 0, bias2P, 0, 64, 1);
}

void vtaConvTestMainEntry_destroy_module(){
  VTABufferFree(filterP);
  VTABufferFree(biasP);
  VTABufferFree(filter2P);
  VTABufferFree(bias2P);
}
int vtaConvTestMainEntry(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations){

  //Run allocactivation : conv_res
  int8_t *conv_res = (int8_t *)malloc(3136);

  //Run convolution : conv
  int8_t* inputP = (int8_t*)mutableWeight + 0;
  int8_t* conv_input_transpose = (int8_t *)VTABufferAlloc(3136);
  transpose_nhwc2vtaio(inputP, (int8_t* )VTABufferGetVirtAddr(conv_input_transpose), 1, 14, 14, 16);
  int8_t* conv_output_bef_transpose = (int8_t *)VTABufferAlloc(3136);
  convolution_wo_tr(conv_input_transpose, filterP, (int32_t *)biasP, conv_output_bef_transpose, 1, 14, 14, 16, 16, 3, 3, 1, 1, 0, 1, 8, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(conv_output_bef_transpose), conv_res, 1, 14, 14, 16 );
  VTABufferFree(conv_input_transpose);
  VTABufferFree(conv_output_bef_transpose);

  //Run convolution : conv2
  int8_t* outP = (int8_t*)mutableWeight + 3136;
  int8_t* conv2_input_transpose = (int8_t *)VTABufferAlloc(3136);
  transpose_nhwc2vtaio(conv_res, (int8_t* )VTABufferGetVirtAddr(conv2_input_transpose), 1, 14, 14, 16);
  int8_t* conv2_output_bef_transpose = (int8_t *)VTABufferAlloc(3136);
  convolution_wo_tr(conv2_input_transpose, filter2P, (int32_t *)bias2P, conv2_output_bef_transpose, 1, 14, 14, 16, 16, 3, 3, 1, 1, 0, 1, 8, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(conv2_output_bef_transpose), outP, 1, 14, 14, 16 );
  VTABufferFree(conv2_input_transpose);
  VTABufferFree(conv2_output_bef_transpose);

  //Run deallocactivation : dealloc_conv_res
  free(conv_res);
  return 0;
}
