

#include "vta/runtime.h"
#include "VTABundle.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include "caffe2_resnet50.h"
SymbolTableEntry symbolTableEntry_caffe2_resnet50[2]={{"gpu_0_data_0",0,150528,'1'},{"gpu_0_softmax_1",602112,1000,'1'}};
BundleConfig caffe2_resnet50_config = {25613152, 606112, 0, 64, 2, symbolTableEntry_caffe2_resnet50};
int8_t* gpu_0_conv1_w_0__1;
int8_t* conv_bias__53;
int8_t* gpu_0_res2_0_branch2a_w_0__1;
int8_t* conv_bias__54;
int8_t* gpu_0_res2_0_branch2b_w_0__1;
int8_t* conv_bias__55;
int8_t* gpu_0_res2_0_branch2c_w_0__1;
int8_t* conv_bias__56;
int8_t* gpu_0_res2_0_branch1_w_0__1;
int8_t* conv_bias__57;
int8_t* gpu_0_res2_1_branch2a_w_0__1;
int8_t* conv_bias__58;
int8_t* gpu_0_res2_1_branch2b_w_0__1;
int8_t* conv_bias__59;
int8_t* gpu_0_res2_1_branch2c_w_0__1;
int8_t* conv_bias__60;
int8_t* gpu_0_res2_2_branch2a_w_0__1;
int8_t* conv_bias__61;
int8_t* gpu_0_res2_2_branch2b_w_0__1;
int8_t* conv_bias;
int8_t* gpu_0_res2_2_branch2c_w_0__1;
int8_t* conv_bias__62;
int8_t* gpu_0_res3_0_branch2a_w_0__1;
int8_t* conv_bias__63;
int8_t* gpu_0_res3_0_branch2b_w_0__1;
int8_t* conv_bias__64;
int8_t* gpu_0_res3_0_branch2c_w_0__1;
int8_t* conv_bias__65;
int8_t* gpu_0_res3_0_branch1_w_0__1;
int8_t* conv_bias__66;
int8_t* gpu_0_res3_1_branch2a_w_0__1;
int8_t* conv_bias__67;
int8_t* gpu_0_res3_1_branch2b_w_0__1;
int8_t* conv_bias__68;
int8_t* gpu_0_res3_1_branch2c_w_0__1;
int8_t* conv_bias__69;
int8_t* gpu_0_res3_2_branch2a_w_0__1;
int8_t* conv_bias__70;
int8_t* gpu_0_res3_2_branch2b_w_0__1;
int8_t* conv_bias__71;
int8_t* gpu_0_res3_2_branch2c_w_0__1;
int8_t* conv_bias__72;
int8_t* gpu_0_res3_3_branch2a_w_0__1;
int8_t* conv_bias__73;
int8_t* gpu_0_res3_3_branch2b_w_0__1;
int8_t* conv_bias__11;
int8_t* gpu_0_res3_3_branch2c_w_0__1;
int8_t* conv_bias__74;
int8_t* gpu_0_res4_0_branch2a_w_0__1;
int8_t* conv_bias__75;
int8_t* gpu_0_res4_0_branch2b_w_0__1;
int8_t* conv_bias__76;
int8_t* gpu_0_res4_0_branch2c_w_0__1;
int8_t* conv_bias__77;
int8_t* gpu_0_res4_0_branch1_w_0__1;
int8_t* conv_bias__78;
int8_t* gpu_0_res4_1_branch2a_w_0__1;
int8_t* conv_bias__79;
int8_t* gpu_0_res4_1_branch2b_w_0__1;
int8_t* conv_bias__80;
int8_t* gpu_0_res4_1_branch2c_w_0__1;
int8_t* conv_bias__81;
int8_t* gpu_0_res4_2_branch2a_w_0__1;
int8_t* conv_bias__82;
int8_t* gpu_0_res4_2_branch2b_w_0__1;
int8_t* conv_bias__83;
int8_t* gpu_0_res4_2_branch2c_w_0__1;
int8_t* conv_bias__84;
int8_t* gpu_0_res4_3_branch2a_w_0__1;
int8_t* conv_bias__85;
int8_t* gpu_0_res4_3_branch2b_w_0__1;
int8_t* conv_bias__86;
int8_t* gpu_0_res4_3_branch2c_w_0__1;
int8_t* conv_bias__87;
int8_t* gpu_0_res4_4_branch2a_w_0__1;
int8_t* conv_bias__88;
int8_t* gpu_0_res4_4_branch2b_w_0__1;
int8_t* conv_bias__89;
int8_t* gpu_0_res4_4_branch2c_w_0__1;
int8_t* conv_bias__90;
int8_t* gpu_0_res4_5_branch2a_w_0__1;
int8_t* conv_bias__91;
int8_t* gpu_0_res4_5_branch2b_w_0__1;
int8_t* conv_bias__3;
int8_t* gpu_0_res4_5_branch2c_w_0__1;
int8_t* conv_bias__26;
int8_t* gpu_0_res5_0_branch2a_w_0__1;
int8_t* conv_bias__92;
int8_t* gpu_0_res5_0_branch2b_w_0__1;
int8_t* conv_bias__93;
int8_t* gpu_0_res5_0_branch2c_w_0__1;
int8_t* conv_bias__94;
int8_t* gpu_0_res5_0_branch1_w_0__1;
int8_t* conv_bias__95;
int8_t* gpu_0_res5_1_branch2a_w_0__1;
int8_t* conv_bias__96;
int8_t* gpu_0_res5_1_branch2b_w_0__1;
int8_t* conv_bias__97;
int8_t* gpu_0_res5_1_branch2c_w_0__1;
int8_t* conv_bias__98;
int8_t* gpu_0_res5_2_branch2a_w_0__1;
int8_t* conv_bias__99;
int8_t* gpu_0_res5_2_branch2b_w_0__1;
int8_t* conv_bias__13;
int8_t* gpu_0_res5_2_branch2c_w_0__1;
int8_t* conv_bias__45;
int8_t* gpu_0_pred_w_0__1;
int8_t* gpu_0_pred_b_0;
extern VTACommandHandle vtaCmdH;

void caffe2_resnet50_load_module(uint8_t *constantWeight){
  gpu_0_conv1_w_0__1 = (int8_t *)VTABufferAlloc(9408);
  VTABufferCopy((int8_t *)(constantWeight + 0), 0, gpu_0_conv1_w_0__1, 0, 9408, 1);
  conv_bias__53 = (int8_t *)VTABufferAlloc(256);
  VTABufferCopy((int8_t *)(constantWeight + 9408), 0, conv_bias__53, 0, 256, 1);
  gpu_0_res2_0_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(4096);
  VTABufferCopy((int8_t *)(constantWeight + 9664), 0, gpu_0_res2_0_branch2a_w_0__1, 0, 4096, 1);
  conv_bias__54 = (int8_t *)VTABufferAlloc(256);
  VTABufferCopy((int8_t *)(constantWeight + 13760), 0, conv_bias__54, 0, 256, 1);
  gpu_0_res2_0_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(36864);
  VTABufferCopy((int8_t *)(constantWeight + 14016), 0, gpu_0_res2_0_branch2b_w_0__1, 0, 36864, 1);
  conv_bias__55 = (int8_t *)VTABufferAlloc(256);
  VTABufferCopy((int8_t *)(constantWeight + 50880), 0, conv_bias__55, 0, 256, 1);
  gpu_0_res2_0_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(16384);
  VTABufferCopy((int8_t *)(constantWeight + 51136), 0, gpu_0_res2_0_branch2c_w_0__1, 0, 16384, 1);
  conv_bias__56 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 67520), 0, conv_bias__56, 0, 1024, 1);
  gpu_0_res2_0_branch1_w_0__1 = (int8_t *)VTABufferAlloc(16384);
  VTABufferCopy((int8_t *)(constantWeight + 68544), 0, gpu_0_res2_0_branch1_w_0__1, 0, 16384, 1);
  conv_bias__57 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 84928), 0, conv_bias__57, 0, 1024, 1);
  gpu_0_res2_1_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(16384);
  VTABufferCopy((int8_t *)(constantWeight + 85952), 0, gpu_0_res2_1_branch2a_w_0__1, 0, 16384, 1);
  conv_bias__58 = (int8_t *)VTABufferAlloc(256);
  VTABufferCopy((int8_t *)(constantWeight + 102336), 0, conv_bias__58, 0, 256, 1);
  gpu_0_res2_1_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(36864);
  VTABufferCopy((int8_t *)(constantWeight + 102592), 0, gpu_0_res2_1_branch2b_w_0__1, 0, 36864, 1);
  conv_bias__59 = (int8_t *)VTABufferAlloc(256);
  VTABufferCopy((int8_t *)(constantWeight + 139456), 0, conv_bias__59, 0, 256, 1);
  gpu_0_res2_1_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(16384);
  VTABufferCopy((int8_t *)(constantWeight + 139712), 0, gpu_0_res2_1_branch2c_w_0__1, 0, 16384, 1);
  conv_bias__60 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 156096), 0, conv_bias__60, 0, 1024, 1);
  gpu_0_res2_2_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(16384);
  VTABufferCopy((int8_t *)(constantWeight + 157120), 0, gpu_0_res2_2_branch2a_w_0__1, 0, 16384, 1);
  conv_bias__61 = (int8_t *)VTABufferAlloc(256);
  VTABufferCopy((int8_t *)(constantWeight + 173504), 0, conv_bias__61, 0, 256, 1);
  gpu_0_res2_2_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(36864);
  VTABufferCopy((int8_t *)(constantWeight + 173760), 0, gpu_0_res2_2_branch2b_w_0__1, 0, 36864, 1);
  conv_bias = (int8_t *)VTABufferAlloc(256);
  VTABufferCopy((int8_t *)(constantWeight + 210624), 0, conv_bias, 0, 256, 1);
  gpu_0_res2_2_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(16384);
  VTABufferCopy((int8_t *)(constantWeight + 210880), 0, gpu_0_res2_2_branch2c_w_0__1, 0, 16384, 1);
  conv_bias__62 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 227264), 0, conv_bias__62, 0, 1024, 1);
  gpu_0_res3_0_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(32768);
  VTABufferCopy((int8_t *)(constantWeight + 228288), 0, gpu_0_res3_0_branch2a_w_0__1, 0, 32768, 1);
  conv_bias__63 = (int8_t *)VTABufferAlloc(512);
  VTABufferCopy((int8_t *)(constantWeight + 261056), 0, conv_bias__63, 0, 512, 1);
  gpu_0_res3_0_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(147456);
  VTABufferCopy((int8_t *)(constantWeight + 261568), 0, gpu_0_res3_0_branch2b_w_0__1, 0, 147456, 1);
  conv_bias__64 = (int8_t *)VTABufferAlloc(512);
  VTABufferCopy((int8_t *)(constantWeight + 409024), 0, conv_bias__64, 0, 512, 1);
  gpu_0_res3_0_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(65536);
  VTABufferCopy((int8_t *)(constantWeight + 409536), 0, gpu_0_res3_0_branch2c_w_0__1, 0, 65536, 1);
  conv_bias__65 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 475072), 0, conv_bias__65, 0, 2048, 1);
  gpu_0_res3_0_branch1_w_0__1 = (int8_t *)VTABufferAlloc(131072);
  VTABufferCopy((int8_t *)(constantWeight + 477120), 0, gpu_0_res3_0_branch1_w_0__1, 0, 131072, 1);
  conv_bias__66 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 608192), 0, conv_bias__66, 0, 2048, 1);
  gpu_0_res3_1_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(65536);
  VTABufferCopy((int8_t *)(constantWeight + 610240), 0, gpu_0_res3_1_branch2a_w_0__1, 0, 65536, 1);
  conv_bias__67 = (int8_t *)VTABufferAlloc(512);
  VTABufferCopy((int8_t *)(constantWeight + 675776), 0, conv_bias__67, 0, 512, 1);
  gpu_0_res3_1_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(147456);
  VTABufferCopy((int8_t *)(constantWeight + 676288), 0, gpu_0_res3_1_branch2b_w_0__1, 0, 147456, 1);
  conv_bias__68 = (int8_t *)VTABufferAlloc(512);
  VTABufferCopy((int8_t *)(constantWeight + 823744), 0, conv_bias__68, 0, 512, 1);
  gpu_0_res3_1_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(65536);
  VTABufferCopy((int8_t *)(constantWeight + 824256), 0, gpu_0_res3_1_branch2c_w_0__1, 0, 65536, 1);
  conv_bias__69 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 889792), 0, conv_bias__69, 0, 2048, 1);
  gpu_0_res3_2_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(65536);
  VTABufferCopy((int8_t *)(constantWeight + 891840), 0, gpu_0_res3_2_branch2a_w_0__1, 0, 65536, 1);
  conv_bias__70 = (int8_t *)VTABufferAlloc(512);
  VTABufferCopy((int8_t *)(constantWeight + 957376), 0, conv_bias__70, 0, 512, 1);
  gpu_0_res3_2_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(147456);
  VTABufferCopy((int8_t *)(constantWeight + 957888), 0, gpu_0_res3_2_branch2b_w_0__1, 0, 147456, 1);
  conv_bias__71 = (int8_t *)VTABufferAlloc(512);
  VTABufferCopy((int8_t *)(constantWeight + 1105344), 0, conv_bias__71, 0, 512, 1);
  gpu_0_res3_2_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(65536);
  VTABufferCopy((int8_t *)(constantWeight + 1105856), 0, gpu_0_res3_2_branch2c_w_0__1, 0, 65536, 1);
  conv_bias__72 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 1171392), 0, conv_bias__72, 0, 2048, 1);
  gpu_0_res3_3_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(65536);
  VTABufferCopy((int8_t *)(constantWeight + 1173440), 0, gpu_0_res3_3_branch2a_w_0__1, 0, 65536, 1);
  conv_bias__73 = (int8_t *)VTABufferAlloc(512);
  VTABufferCopy((int8_t *)(constantWeight + 1238976), 0, conv_bias__73, 0, 512, 1);
  gpu_0_res3_3_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(147456);
  VTABufferCopy((int8_t *)(constantWeight + 1239488), 0, gpu_0_res3_3_branch2b_w_0__1, 0, 147456, 1);
  conv_bias__11 = (int8_t *)VTABufferAlloc(512);
  VTABufferCopy((int8_t *)(constantWeight + 1386944), 0, conv_bias__11, 0, 512, 1);
  gpu_0_res3_3_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(65536);
  VTABufferCopy((int8_t *)(constantWeight + 1387456), 0, gpu_0_res3_3_branch2c_w_0__1, 0, 65536, 1);
  conv_bias__74 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 1452992), 0, conv_bias__74, 0, 2048, 1);
  gpu_0_res4_0_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(131072);
  VTABufferCopy((int8_t *)(constantWeight + 1455040), 0, gpu_0_res4_0_branch2a_w_0__1, 0, 131072, 1);
  conv_bias__75 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 1586112), 0, conv_bias__75, 0, 1024, 1);
  gpu_0_res4_0_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(589824);
  VTABufferCopy((int8_t *)(constantWeight + 1587136), 0, gpu_0_res4_0_branch2b_w_0__1, 0, 589824, 1);
  conv_bias__76 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 2176960), 0, conv_bias__76, 0, 1024, 1);
  gpu_0_res4_0_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 2177984), 0, gpu_0_res4_0_branch2c_w_0__1, 0, 262144, 1);
  conv_bias__77 = (int8_t *)VTABufferAlloc(4096);
  VTABufferCopy((int8_t *)(constantWeight + 2440128), 0, conv_bias__77, 0, 4096, 1);
  gpu_0_res4_0_branch1_w_0__1 = (int8_t *)VTABufferAlloc(524288);
  VTABufferCopy((int8_t *)(constantWeight + 2444224), 0, gpu_0_res4_0_branch1_w_0__1, 0, 524288, 1);
  conv_bias__78 = (int8_t *)VTABufferAlloc(4096);
  VTABufferCopy((int8_t *)(constantWeight + 2968512), 0, conv_bias__78, 0, 4096, 1);
  gpu_0_res4_1_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 2972608), 0, gpu_0_res4_1_branch2a_w_0__1, 0, 262144, 1);
  conv_bias__79 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 3234752), 0, conv_bias__79, 0, 1024, 1);
  gpu_0_res4_1_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(589824);
  VTABufferCopy((int8_t *)(constantWeight + 3235776), 0, gpu_0_res4_1_branch2b_w_0__1, 0, 589824, 1);
  conv_bias__80 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 3825600), 0, conv_bias__80, 0, 1024, 1);
  gpu_0_res4_1_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 3826624), 0, gpu_0_res4_1_branch2c_w_0__1, 0, 262144, 1);
  conv_bias__81 = (int8_t *)VTABufferAlloc(4096);
  VTABufferCopy((int8_t *)(constantWeight + 4088768), 0, conv_bias__81, 0, 4096, 1);
  gpu_0_res4_2_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 4092864), 0, gpu_0_res4_2_branch2a_w_0__1, 0, 262144, 1);
  conv_bias__82 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 4355008), 0, conv_bias__82, 0, 1024, 1);
  gpu_0_res4_2_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(589824);
  VTABufferCopy((int8_t *)(constantWeight + 4356032), 0, gpu_0_res4_2_branch2b_w_0__1, 0, 589824, 1);
  conv_bias__83 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 4945856), 0, conv_bias__83, 0, 1024, 1);
  gpu_0_res4_2_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 4946880), 0, gpu_0_res4_2_branch2c_w_0__1, 0, 262144, 1);
  conv_bias__84 = (int8_t *)VTABufferAlloc(4096);
  VTABufferCopy((int8_t *)(constantWeight + 5209024), 0, conv_bias__84, 0, 4096, 1);
  gpu_0_res4_3_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 5213120), 0, gpu_0_res4_3_branch2a_w_0__1, 0, 262144, 1);
  conv_bias__85 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 5475264), 0, conv_bias__85, 0, 1024, 1);
  gpu_0_res4_3_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(589824);
  VTABufferCopy((int8_t *)(constantWeight + 5476288), 0, gpu_0_res4_3_branch2b_w_0__1, 0, 589824, 1);
  conv_bias__86 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 6066112), 0, conv_bias__86, 0, 1024, 1);
  gpu_0_res4_3_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 6067136), 0, gpu_0_res4_3_branch2c_w_0__1, 0, 262144, 1);
  conv_bias__87 = (int8_t *)VTABufferAlloc(4096);
  VTABufferCopy((int8_t *)(constantWeight + 6329280), 0, conv_bias__87, 0, 4096, 1);
  gpu_0_res4_4_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 6333376), 0, gpu_0_res4_4_branch2a_w_0__1, 0, 262144, 1);
  conv_bias__88 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 6595520), 0, conv_bias__88, 0, 1024, 1);
  gpu_0_res4_4_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(589824);
  VTABufferCopy((int8_t *)(constantWeight + 6596544), 0, gpu_0_res4_4_branch2b_w_0__1, 0, 589824, 1);
  conv_bias__89 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 7186368), 0, conv_bias__89, 0, 1024, 1);
  gpu_0_res4_4_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 7187392), 0, gpu_0_res4_4_branch2c_w_0__1, 0, 262144, 1);
  conv_bias__90 = (int8_t *)VTABufferAlloc(4096);
  VTABufferCopy((int8_t *)(constantWeight + 7449536), 0, conv_bias__90, 0, 4096, 1);
  gpu_0_res4_5_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 7453632), 0, gpu_0_res4_5_branch2a_w_0__1, 0, 262144, 1);
  conv_bias__91 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 7715776), 0, conv_bias__91, 0, 1024, 1);
  gpu_0_res4_5_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(589824);
  VTABufferCopy((int8_t *)(constantWeight + 7716800), 0, gpu_0_res4_5_branch2b_w_0__1, 0, 589824, 1);
  conv_bias__3 = (int8_t *)VTABufferAlloc(1024);
  VTABufferCopy((int8_t *)(constantWeight + 8306624), 0, conv_bias__3, 0, 1024, 1);
  gpu_0_res4_5_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(262144);
  VTABufferCopy((int8_t *)(constantWeight + 8307648), 0, gpu_0_res4_5_branch2c_w_0__1, 0, 262144, 1);
  conv_bias__26 = (int8_t *)VTABufferAlloc(4096);
  VTABufferCopy((int8_t *)(constantWeight + 8569792), 0, conv_bias__26, 0, 4096, 1);
  gpu_0_res5_0_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(524288);
  VTABufferCopy((int8_t *)(constantWeight + 8573888), 0, gpu_0_res5_0_branch2a_w_0__1, 0, 524288, 1);
  conv_bias__92 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 9098176), 0, conv_bias__92, 0, 2048, 1);
  gpu_0_res5_0_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(2359296);
  VTABufferCopy((int8_t *)(constantWeight + 9100224), 0, gpu_0_res5_0_branch2b_w_0__1, 0, 2359296, 1);
  conv_bias__93 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 11459520), 0, conv_bias__93, 0, 2048, 1);
  gpu_0_res5_0_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(1048576);
  VTABufferCopy((int8_t *)(constantWeight + 11461568), 0, gpu_0_res5_0_branch2c_w_0__1, 0, 1048576, 1);
  conv_bias__94 = (int8_t *)VTABufferAlloc(8192);
  VTABufferCopy((int8_t *)(constantWeight + 12510144), 0, conv_bias__94, 0, 8192, 1);
  gpu_0_res5_0_branch1_w_0__1 = (int8_t *)VTABufferAlloc(2097152);
  VTABufferCopy((int8_t *)(constantWeight + 12518336), 0, gpu_0_res5_0_branch1_w_0__1, 0, 2097152, 1);
  conv_bias__95 = (int8_t *)VTABufferAlloc(8192);
  VTABufferCopy((int8_t *)(constantWeight + 14615488), 0, conv_bias__95, 0, 8192, 1);
  gpu_0_res5_1_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(1048576);
  VTABufferCopy((int8_t *)(constantWeight + 14623680), 0, gpu_0_res5_1_branch2a_w_0__1, 0, 1048576, 1);
  conv_bias__96 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 15672256), 0, conv_bias__96, 0, 2048, 1);
  gpu_0_res5_1_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(2359296);
  VTABufferCopy((int8_t *)(constantWeight + 15674304), 0, gpu_0_res5_1_branch2b_w_0__1, 0, 2359296, 1);
  conv_bias__97 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 18033600), 0, conv_bias__97, 0, 2048, 1);
  gpu_0_res5_1_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(1048576);
  VTABufferCopy((int8_t *)(constantWeight + 18035648), 0, gpu_0_res5_1_branch2c_w_0__1, 0, 1048576, 1);
  conv_bias__98 = (int8_t *)VTABufferAlloc(8192);
  VTABufferCopy((int8_t *)(constantWeight + 19084224), 0, conv_bias__98, 0, 8192, 1);
  gpu_0_res5_2_branch2a_w_0__1 = (int8_t *)VTABufferAlloc(1048576);
  VTABufferCopy((int8_t *)(constantWeight + 19092416), 0, gpu_0_res5_2_branch2a_w_0__1, 0, 1048576, 1);
  conv_bias__99 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 20140992), 0, conv_bias__99, 0, 2048, 1);
  gpu_0_res5_2_branch2b_w_0__1 = (int8_t *)VTABufferAlloc(2359296);
  VTABufferCopy((int8_t *)(constantWeight + 20143040), 0, gpu_0_res5_2_branch2b_w_0__1, 0, 2359296, 1);
  conv_bias__13 = (int8_t *)VTABufferAlloc(2048);
  VTABufferCopy((int8_t *)(constantWeight + 22502336), 0, conv_bias__13, 0, 2048, 1);
  gpu_0_res5_2_branch2c_w_0__1 = (int8_t *)VTABufferAlloc(1048576);
  VTABufferCopy((int8_t *)(constantWeight + 22504384), 0, gpu_0_res5_2_branch2c_w_0__1, 0, 1048576, 1);
  conv_bias__45 = (int8_t *)VTABufferAlloc(8192);
  VTABufferCopy((int8_t *)(constantWeight + 23552960), 0, conv_bias__45, 0, 8192, 1);
  gpu_0_pred_w_0__1 = (int8_t *)VTABufferAlloc(2048000);
  VTABufferCopy((int8_t *)(constantWeight + 23561152), 0, gpu_0_pred_w_0__1, 0, 2048000, 1);
  gpu_0_pred_b_0 = (int8_t *)VTABufferAlloc(4000);
  VTABufferCopy((int8_t *)(constantWeight + 25609152), 0, gpu_0_pred_b_0, 0, 4000, 1);
}

void caffe2_resnet50_destroy_module(){
  VTABufferFree(gpu_0_conv1_w_0__1);
  VTABufferFree(conv_bias__53);
  VTABufferFree(gpu_0_res2_0_branch2a_w_0__1);
  VTABufferFree(conv_bias__54);
  VTABufferFree(gpu_0_res2_0_branch2b_w_0__1);
  VTABufferFree(conv_bias__55);
  VTABufferFree(gpu_0_res2_0_branch2c_w_0__1);
  VTABufferFree(conv_bias__56);
  VTABufferFree(gpu_0_res2_0_branch1_w_0__1);
  VTABufferFree(conv_bias__57);
  VTABufferFree(gpu_0_res2_1_branch2a_w_0__1);
  VTABufferFree(conv_bias__58);
  VTABufferFree(gpu_0_res2_1_branch2b_w_0__1);
  VTABufferFree(conv_bias__59);
  VTABufferFree(gpu_0_res2_1_branch2c_w_0__1);
  VTABufferFree(conv_bias__60);
  VTABufferFree(gpu_0_res2_2_branch2a_w_0__1);
  VTABufferFree(conv_bias__61);
  VTABufferFree(gpu_0_res2_2_branch2b_w_0__1);
  VTABufferFree(conv_bias);
  VTABufferFree(gpu_0_res2_2_branch2c_w_0__1);
  VTABufferFree(conv_bias__62);
  VTABufferFree(gpu_0_res3_0_branch2a_w_0__1);
  VTABufferFree(conv_bias__63);
  VTABufferFree(gpu_0_res3_0_branch2b_w_0__1);
  VTABufferFree(conv_bias__64);
  VTABufferFree(gpu_0_res3_0_branch2c_w_0__1);
  VTABufferFree(conv_bias__65);
  VTABufferFree(gpu_0_res3_0_branch1_w_0__1);
  VTABufferFree(conv_bias__66);
  VTABufferFree(gpu_0_res3_1_branch2a_w_0__1);
  VTABufferFree(conv_bias__67);
  VTABufferFree(gpu_0_res3_1_branch2b_w_0__1);
  VTABufferFree(conv_bias__68);
  VTABufferFree(gpu_0_res3_1_branch2c_w_0__1);
  VTABufferFree(conv_bias__69);
  VTABufferFree(gpu_0_res3_2_branch2a_w_0__1);
  VTABufferFree(conv_bias__70);
  VTABufferFree(gpu_0_res3_2_branch2b_w_0__1);
  VTABufferFree(conv_bias__71);
  VTABufferFree(gpu_0_res3_2_branch2c_w_0__1);
  VTABufferFree(conv_bias__72);
  VTABufferFree(gpu_0_res3_3_branch2a_w_0__1);
  VTABufferFree(conv_bias__73);
  VTABufferFree(gpu_0_res3_3_branch2b_w_0__1);
  VTABufferFree(conv_bias__11);
  VTABufferFree(gpu_0_res3_3_branch2c_w_0__1);
  VTABufferFree(conv_bias__74);
  VTABufferFree(gpu_0_res4_0_branch2a_w_0__1);
  VTABufferFree(conv_bias__75);
  VTABufferFree(gpu_0_res4_0_branch2b_w_0__1);
  VTABufferFree(conv_bias__76);
  VTABufferFree(gpu_0_res4_0_branch2c_w_0__1);
  VTABufferFree(conv_bias__77);
  VTABufferFree(gpu_0_res4_0_branch1_w_0__1);
  VTABufferFree(conv_bias__78);
  VTABufferFree(gpu_0_res4_1_branch2a_w_0__1);
  VTABufferFree(conv_bias__79);
  VTABufferFree(gpu_0_res4_1_branch2b_w_0__1);
  VTABufferFree(conv_bias__80);
  VTABufferFree(gpu_0_res4_1_branch2c_w_0__1);
  VTABufferFree(conv_bias__81);
  VTABufferFree(gpu_0_res4_2_branch2a_w_0__1);
  VTABufferFree(conv_bias__82);
  VTABufferFree(gpu_0_res4_2_branch2b_w_0__1);
  VTABufferFree(conv_bias__83);
  VTABufferFree(gpu_0_res4_2_branch2c_w_0__1);
  VTABufferFree(conv_bias__84);
  VTABufferFree(gpu_0_res4_3_branch2a_w_0__1);
  VTABufferFree(conv_bias__85);
  VTABufferFree(gpu_0_res4_3_branch2b_w_0__1);
  VTABufferFree(conv_bias__86);
  VTABufferFree(gpu_0_res4_3_branch2c_w_0__1);
  VTABufferFree(conv_bias__87);
  VTABufferFree(gpu_0_res4_4_branch2a_w_0__1);
  VTABufferFree(conv_bias__88);
  VTABufferFree(gpu_0_res4_4_branch2b_w_0__1);
  VTABufferFree(conv_bias__89);
  VTABufferFree(gpu_0_res4_4_branch2c_w_0__1);
  VTABufferFree(conv_bias__90);
  VTABufferFree(gpu_0_res4_5_branch2a_w_0__1);
  VTABufferFree(conv_bias__91);
  VTABufferFree(gpu_0_res4_5_branch2b_w_0__1);
  VTABufferFree(conv_bias__3);
  VTABufferFree(gpu_0_res4_5_branch2c_w_0__1);
  VTABufferFree(conv_bias__26);
  VTABufferFree(gpu_0_res5_0_branch2a_w_0__1);
  VTABufferFree(conv_bias__92);
  VTABufferFree(gpu_0_res5_0_branch2b_w_0__1);
  VTABufferFree(conv_bias__93);
  VTABufferFree(gpu_0_res5_0_branch2c_w_0__1);
  VTABufferFree(conv_bias__94);
  VTABufferFree(gpu_0_res5_0_branch1_w_0__1);
  VTABufferFree(conv_bias__95);
  VTABufferFree(gpu_0_res5_1_branch2a_w_0__1);
  VTABufferFree(conv_bias__96);
  VTABufferFree(gpu_0_res5_1_branch2b_w_0__1);
  VTABufferFree(conv_bias__97);
  VTABufferFree(gpu_0_res5_1_branch2c_w_0__1);
  VTABufferFree(conv_bias__98);
  VTABufferFree(gpu_0_res5_2_branch2a_w_0__1);
  VTABufferFree(conv_bias__99);
  VTABufferFree(gpu_0_res5_2_branch2b_w_0__1);
  VTABufferFree(conv_bias__13);
  VTABufferFree(gpu_0_res5_2_branch2c_w_0__1);
  VTABufferFree(conv_bias__45);
  VTABufferFree(gpu_0_pred_w_0__1);
  VTABufferFree(gpu_0_pred_b_0);
}
int caffe2_resnet50(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations){

  //Run allocactivation : Conv_gpu_0_conv1_1__1_quantize_res
  int8_t *Conv_gpu_0_conv1_1__1_quantize_res = (int8_t *)malloc(150528);

  //Run quantize : Conv_gpu_0_conv1_1__1_quantize
  int8_t* gpu_0_data_0 = (int8_t*)mutableWeight + 0;
  quantize(gpu_0_data_0, Conv_gpu_0_conv1_1__1_quantize_res, 150528, 1/32.000000, 0 );

  //Run allocactivation : Conv_gpu_0_conv1_1__10_res
  int8_t *Conv_gpu_0_conv1_1__10_res = (int8_t *)malloc(150528);

  //Run transpose : Conv_gpu_0_conv1_1__10
  transpose(Conv_gpu_0_conv1_1__1_quantize_res, Conv_gpu_0_conv1_1__10_res, 1, 3, 224, 224, 1, 224, 224, 3, 0, 2, 3, 1 );

  //Run deallocactivation : dealloc_Conv_gpu_0_conv1_1__1_quantize_res
  free(Conv_gpu_0_conv1_1__1_quantize_res);

  //Run allocactivation : Conv_gpu_0_conv1_1__2_res
  int8_t *Conv_gpu_0_conv1_1__2_res = (int8_t *)malloc(802816);

  //Run convolution : Conv_gpu_0_conv1_1__2
  nonvtaconvolution(Conv_gpu_0_conv1_1__10_res, 1.0/32.000000, 0, (int8_t *)VTABufferGetVirtAddr(gpu_0_conv1_w_0__1), 1.0/64.000000, 0, (int8_t *)VTABufferGetVirtAddr(conv_bias__53), 1.0/2048.000000, 0, Conv_gpu_0_conv1_1__2_res, 1.0/8.000000, 0, 1, 224, 224, 3, 64, 7, 7, 3, 2, 1, 1, 0, 1, 112, 112 );

  //Run deallocactivation : dealloc_Conv_gpu_0_conv1_1__10_res
  free(Conv_gpu_0_conv1_1__10_res);

  //Run relu : Relu_gpu_0_res_conv1_bn_2__1
  relu(Conv_gpu_0_conv1_1__2_res, Conv_gpu_0_conv1_1__2_res, 802816 );

  //Run allocactivation : MaxPool_gpu_0_pool1_1__1_res
  int8_t *MaxPool_gpu_0_pool1_1__1_res = (int8_t *)malloc(200704);

  //Run maxpool : MaxPool_gpu_0_pool1_1__2
  maxpool(Conv_gpu_0_conv1_1__2_res, MaxPool_gpu_0_pool1_1__1_res, 1, 112, 112, 64, 3, 3, 1, 2 );

  //Run deallocactivation : dealloc_Conv_gpu_0_conv1_1__2_res
  free(Conv_gpu_0_conv1_1__2_res);

  //Run allocactivation : Conv_gpu_0_res2_0_branch2a_1__2_res
  int8_t *Conv_gpu_0_res2_0_branch2a_1__2_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res2_0_branch2a_1__2
  int8_t* Conv_gpu_0_res2_0_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(MaxPool_gpu_0_pool1_1__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_0_branch2a_1__2_input_transpose), 1, 56, 56, 64);
  int8_t* Conv_gpu_0_res2_0_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res2_0_branch2a_1__2_input_transpose, gpu_0_res2_0_branch2a_w_0__1, (int32_t *)conv_bias__54, Conv_gpu_0_res2_0_branch2a_1__2_output_bef_transpose, 1, 56, 56, 64, 64, 1, 1, 0, 1, 0, 1, 7, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_0_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res2_0_branch2a_1__2_res, 1, 56, 56, 64 );
  VTABufferFree(Conv_gpu_0_res2_0_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_0_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res2_0_branch2a_bn_2__1
  relu(Conv_gpu_0_res2_0_branch2a_1__2_res, Conv_gpu_0_res2_0_branch2a_1__2_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res2_0_branch2b_1__2_res
  int8_t *Conv_gpu_0_res2_0_branch2b_1__2_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res2_0_branch2b_1__2
  int8_t* Conv_gpu_0_res2_0_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Conv_gpu_0_res2_0_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_0_branch2b_1__2_input_transpose), 1, 56, 56, 64);
  int8_t* Conv_gpu_0_res2_0_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res2_0_branch2b_1__2_input_transpose, gpu_0_res2_0_branch2b_w_0__1, (int32_t *)conv_bias__55, Conv_gpu_0_res2_0_branch2b_1__2_output_bef_transpose, 1, 56, 56, 64, 64, 3, 3, 1, 1, 0, 1, 8, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_0_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res2_0_branch2b_1__2_res, 1, 56, 56, 64 );
  VTABufferFree(Conv_gpu_0_res2_0_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_0_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res2_0_branch2a_1__2_res
  free(Conv_gpu_0_res2_0_branch2a_1__2_res);

  //Run relu : Relu_gpu_0_res2_0_branch2b_bn_2__1
  relu(Conv_gpu_0_res2_0_branch2b_1__2_res, Conv_gpu_0_res2_0_branch2b_1__2_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res2_0_branch2c_1__2_res
  int8_t *Conv_gpu_0_res2_0_branch2c_1__2_res = (int8_t *)malloc(802816);

  //Run convolution : Conv_gpu_0_res2_0_branch2c_1__2
  int8_t* Conv_gpu_0_res2_0_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Conv_gpu_0_res2_0_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_0_branch2c_1__2_input_transpose), 1, 56, 56, 64);
  int8_t* Conv_gpu_0_res2_0_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(802816);
  convolution_wo_tr(Conv_gpu_0_res2_0_branch2c_1__2_input_transpose, gpu_0_res2_0_branch2c_w_0__1, (int32_t *)conv_bias__56, Conv_gpu_0_res2_0_branch2c_1__2_output_bef_transpose, 1, 56, 56, 64, 256, 1, 1, 0, 1, 0, 1, 6, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_0_branch2c_1__2_output_bef_transpose), Conv_gpu_0_res2_0_branch2c_1__2_res, 1, 56, 56, 256 );
  VTABufferFree(Conv_gpu_0_res2_0_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_0_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res2_0_branch2b_1__2_res
  free(Conv_gpu_0_res2_0_branch2b_1__2_res);

  //Run allocactivation : Conv_gpu_0_res2_0_branch1_1__2_res
  int8_t *Conv_gpu_0_res2_0_branch1_1__2_res = (int8_t *)malloc(802816);

  //Run convolution : Conv_gpu_0_res2_0_branch1_1__2
  int8_t* Conv_gpu_0_res2_0_branch1_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(MaxPool_gpu_0_pool1_1__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_0_branch1_1__2_input_transpose), 1, 56, 56, 64);
  int8_t* Conv_gpu_0_res2_0_branch1_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(802816);
  convolution_wo_tr(Conv_gpu_0_res2_0_branch1_1__2_input_transpose, gpu_0_res2_0_branch1_w_0__1, (int32_t *)conv_bias__57, Conv_gpu_0_res2_0_branch1_1__2_output_bef_transpose, 1, 56, 56, 64, 256, 1, 1, 0, 1, 0, 1, 7, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_0_branch1_1__2_output_bef_transpose), Conv_gpu_0_res2_0_branch1_1__2_res, 1, 56, 56, 256 );
  VTABufferFree(Conv_gpu_0_res2_0_branch1_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_0_branch1_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_MaxPool_gpu_0_pool1_1__1_res
  free(MaxPool_gpu_0_pool1_1__1_res);

  //Run elementadd : Sum_gpu_0_res2_0_branch2c_bn_2__1
  elemadd(Conv_gpu_0_res2_0_branch2c_1__2_res, 1.0/8.000000, 0, Conv_gpu_0_res2_0_branch1_1__2_res, 1.0/4.000000, 0, Conv_gpu_0_res2_0_branch1_1__2_res, 1.0/4.000000, 0, 802816 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res2_0_branch2c_1__2_res
  free(Conv_gpu_0_res2_0_branch2c_1__2_res);

  //Run allocactivation : Relu_gpu_0_res2_0_branch2c_bn_3__1_res
  int8_t *Relu_gpu_0_res2_0_branch2c_bn_3__1_res = (int8_t *)malloc(802816);

  //Run relu : Relu_gpu_0_res2_0_branch2c_bn_3__1
  relu(Conv_gpu_0_res2_0_branch1_1__2_res, Relu_gpu_0_res2_0_branch2c_bn_3__1_res, 802816, 0.250000, 0.125000 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res2_0_branch1_1__2_res
  free(Conv_gpu_0_res2_0_branch1_1__2_res);

  //Run allocactivation : Conv_gpu_0_res2_1_branch2a_1__2_res
  int8_t *Conv_gpu_0_res2_1_branch2a_1__2_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res2_1_branch2a_1__2
  int8_t* Conv_gpu_0_res2_1_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(802816);
  transpose_nhwc2vtaio(Relu_gpu_0_res2_0_branch2c_bn_3__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_1_branch2a_1__2_input_transpose), 1, 56, 56, 256);
  int8_t* Conv_gpu_0_res2_1_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res2_1_branch2a_1__2_input_transpose, gpu_0_res2_1_branch2a_w_0__1, (int32_t *)conv_bias__58, Conv_gpu_0_res2_1_branch2a_1__2_output_bef_transpose, 1, 56, 56, 256, 64, 1, 1, 0, 1, 0, 1, 8, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_1_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res2_1_branch2a_1__2_res, 1, 56, 56, 64 );
  VTABufferFree(Conv_gpu_0_res2_1_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_1_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res2_1_branch2a_bn_2__1
  relu(Conv_gpu_0_res2_1_branch2a_1__2_res, Conv_gpu_0_res2_1_branch2a_1__2_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res2_1_branch2b_1__2_res
  int8_t *Conv_gpu_0_res2_1_branch2b_1__2_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res2_1_branch2b_1__2
  int8_t* Conv_gpu_0_res2_1_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Conv_gpu_0_res2_1_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_1_branch2b_1__2_input_transpose), 1, 56, 56, 64);
  int8_t* Conv_gpu_0_res2_1_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res2_1_branch2b_1__2_input_transpose, gpu_0_res2_1_branch2b_w_0__1, (int32_t *)conv_bias__59, Conv_gpu_0_res2_1_branch2b_1__2_output_bef_transpose, 1, 56, 56, 64, 64, 3, 3, 1, 1, 0, 1, 7, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_1_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res2_1_branch2b_1__2_res, 1, 56, 56, 64 );
  VTABufferFree(Conv_gpu_0_res2_1_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_1_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res2_1_branch2a_1__2_res
  free(Conv_gpu_0_res2_1_branch2a_1__2_res);

  //Run relu : Relu_gpu_0_res2_1_branch2b_bn_2__1
  relu(Conv_gpu_0_res2_1_branch2b_1__2_res, Conv_gpu_0_res2_1_branch2b_1__2_res, 200704 );

  //Run allocactivation : Sum_gpu_0_res2_1_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res2_1_branch2c_bn_2__1_res = (int8_t *)malloc(802816);

  //Run convolution : Conv_gpu_0_res2_1_branch2c_1__2
  int8_t* Conv_gpu_0_res2_1_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Conv_gpu_0_res2_1_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_1_branch2c_1__2_input_transpose), 1, 56, 56, 64);
  int8_t* Conv_gpu_0_res2_1_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(802816);
  convolution_wo_tr(Conv_gpu_0_res2_1_branch2c_1__2_input_transpose, gpu_0_res2_1_branch2c_w_0__1, (int32_t *)conv_bias__60, Conv_gpu_0_res2_1_branch2c_1__2_output_bef_transpose, 1, 56, 56, 64, 256, 1, 1, 0, 1, 0, 1, 7, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_1_branch2c_1__2_output_bef_transpose), Sum_gpu_0_res2_1_branch2c_bn_2__1_res, 1, 56, 56, 256 );
  VTABufferFree(Conv_gpu_0_res2_1_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_1_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res2_1_branch2b_1__2_res
  free(Conv_gpu_0_res2_1_branch2b_1__2_res);

  //Run elementadd : Sum_gpu_0_res2_1_branch2c_bn_2__1
  elemadd(Sum_gpu_0_res2_1_branch2c_bn_2__1_res, 1.0/8.000000, 0, Relu_gpu_0_res2_0_branch2c_bn_3__1_res, 1.0/8.000000, 0, Sum_gpu_0_res2_1_branch2c_bn_2__1_res, 1.0/8.000000, 0, 802816 );

  //Run deallocactivation : dealloc_Relu_gpu_0_res2_0_branch2c_bn_3__1_res
  free(Relu_gpu_0_res2_0_branch2c_bn_3__1_res);

  //Run relu : Relu_gpu_0_res2_1_branch2c_bn_3__1
  relu(Sum_gpu_0_res2_1_branch2c_bn_2__1_res, Sum_gpu_0_res2_1_branch2c_bn_2__1_res, 802816 );

  //Run allocactivation : Conv_gpu_0_res2_2_branch2a_1__2_res
  int8_t *Conv_gpu_0_res2_2_branch2a_1__2_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res2_2_branch2a_1__2
  int8_t* Conv_gpu_0_res2_2_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(802816);
  transpose_nhwc2vtaio(Sum_gpu_0_res2_1_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_2_branch2a_1__2_input_transpose), 1, 56, 56, 256);
  int8_t* Conv_gpu_0_res2_2_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res2_2_branch2a_1__2_input_transpose, gpu_0_res2_2_branch2a_w_0__1, (int32_t *)conv_bias__61, Conv_gpu_0_res2_2_branch2a_1__2_output_bef_transpose, 1, 56, 56, 256, 64, 1, 1, 0, 1, 0, 1, 7, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_2_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res2_2_branch2a_1__2_res, 1, 56, 56, 64 );
  VTABufferFree(Conv_gpu_0_res2_2_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_2_branch2a_1__2_output_bef_transpose);

  //Run allocactivation : Relu_gpu_0_res2_2_branch2a_bn_2__1_res
  int8_t *Relu_gpu_0_res2_2_branch2a_bn_2__1_res = (int8_t *)malloc(200704);

  //Run relu : Relu_gpu_0_res2_2_branch2a_bn_2__1
  relu(Conv_gpu_0_res2_2_branch2a_1__2_res, Relu_gpu_0_res2_2_branch2a_bn_2__1_res, 200704, 0.125000, 0.062500 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res2_2_branch2a_1__2_res
  free(Conv_gpu_0_res2_2_branch2a_1__2_res);

  //Run allocactivation : Conv_gpu_0_res2_2_branch2b_1__2_res
  int8_t *Conv_gpu_0_res2_2_branch2b_1__2_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res2_2_branch2b_1__2
  int8_t* Conv_gpu_0_res2_2_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Relu_gpu_0_res2_2_branch2a_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_2_branch2b_1__2_input_transpose), 1, 56, 56, 64);
  int8_t* Conv_gpu_0_res2_2_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res2_2_branch2b_1__2_input_transpose, gpu_0_res2_2_branch2b_w_0__1, (int32_t *)conv_bias, Conv_gpu_0_res2_2_branch2b_1__2_output_bef_transpose, 1, 56, 56, 64, 64, 3, 3, 1, 1, 0, 1, 9, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_2_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res2_2_branch2b_1__2_res, 1, 56, 56, 64 );
  VTABufferFree(Conv_gpu_0_res2_2_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_2_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Relu_gpu_0_res2_2_branch2a_bn_2__1_res
  free(Relu_gpu_0_res2_2_branch2a_bn_2__1_res);

  //Run relu : Relu_gpu_0_res2_2_branch2b_bn_2__1
  relu(Conv_gpu_0_res2_2_branch2b_1__2_res, Conv_gpu_0_res2_2_branch2b_1__2_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res2_2_branch2c_1__2_res
  int8_t *Conv_gpu_0_res2_2_branch2c_1__2_res = (int8_t *)malloc(802816);

  //Run convolution : Conv_gpu_0_res2_2_branch2c_1__2
  int8_t* Conv_gpu_0_res2_2_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Conv_gpu_0_res2_2_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_2_branch2c_1__2_input_transpose), 1, 56, 56, 64);
  int8_t* Conv_gpu_0_res2_2_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(802816);
  convolution_wo_tr(Conv_gpu_0_res2_2_branch2c_1__2_input_transpose, gpu_0_res2_2_branch2c_w_0__1, (int32_t *)conv_bias__62, Conv_gpu_0_res2_2_branch2c_1__2_output_bef_transpose, 1, 56, 56, 64, 256, 1, 1, 0, 1, 0, 1, 6, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res2_2_branch2c_1__2_output_bef_transpose), Conv_gpu_0_res2_2_branch2c_1__2_res, 1, 56, 56, 256 );
  VTABufferFree(Conv_gpu_0_res2_2_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res2_2_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res2_2_branch2b_1__2_res
  free(Conv_gpu_0_res2_2_branch2b_1__2_res);

  //Run allocactivation : Sum_gpu_0_res2_2_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res2_2_branch2c_bn_2__1_res = (int8_t *)malloc(802816);

  //Run elementadd : Sum_gpu_0_res2_2_branch2c_bn_2__1
  elemadd(Conv_gpu_0_res2_2_branch2c_1__2_res, 1.0/8.000000, 0, Sum_gpu_0_res2_1_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res2_2_branch2c_bn_2__1_res, 1.0/4.000000, 0, 802816 );

  //Run deallocactivation : dealloc_Sum_gpu_0_res2_1_branch2c_bn_2__1_res
  free(Sum_gpu_0_res2_1_branch2c_bn_2__1_res);

  //Run deallocactivation : dealloc_Conv_gpu_0_res2_2_branch2c_1__2_res
  free(Conv_gpu_0_res2_2_branch2c_1__2_res);

  //Run relu : Relu_gpu_0_res2_2_branch2c_bn_3__1
  relu(Sum_gpu_0_res2_2_branch2c_bn_2__1_res, Sum_gpu_0_res2_2_branch2c_bn_2__1_res, 802816 );

  //Run allocactivation : Conv_gpu_0_res3_0_branch2a_1__2_res
  int8_t *Conv_gpu_0_res3_0_branch2a_1__2_res = (int8_t *)malloc(401408);

  //Run convolution : Conv_gpu_0_res3_0_branch2a_1__2
  int8_t* Conv_gpu_0_res3_0_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(802816);
  transpose_nhwc2vtaio(Sum_gpu_0_res2_2_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_0_branch2a_1__2_input_transpose), 1, 56, 56, 256);
  int8_t* Conv_gpu_0_res3_0_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(401408);
  convolution_wo_tr(Conv_gpu_0_res3_0_branch2a_1__2_input_transpose, gpu_0_res3_0_branch2a_w_0__1, (int32_t *)conv_bias__63, Conv_gpu_0_res3_0_branch2a_1__2_output_bef_transpose, 1, 56, 56, 256, 128, 1, 1, 0, 1, 0, 1, 6, 56, 56, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_0_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res3_0_branch2a_1__2_res, 1, 56, 56, 128 );
  VTABufferFree(Conv_gpu_0_res3_0_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_0_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res3_0_branch2a_bn_2__1
  relu(Conv_gpu_0_res3_0_branch2a_1__2_res, Conv_gpu_0_res3_0_branch2a_1__2_res, 401408 );

  //Run allocactivation : Conv_gpu_0_res3_0_branch2b_1__2_res
  int8_t *Conv_gpu_0_res3_0_branch2b_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res3_0_branch2b_1__2
  int8_t* Conv_gpu_0_res3_0_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(401408);
  transpose_nhwc2vtaio(Conv_gpu_0_res3_0_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_0_branch2b_1__2_input_transpose), 1, 56, 56, 128);
  int8_t* Conv_gpu_0_res3_0_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res3_0_branch2b_1__2_input_transpose, gpu_0_res3_0_branch2b_w_0__1, (int32_t *)conv_bias__64, Conv_gpu_0_res3_0_branch2b_1__2_output_bef_transpose, 1, 56, 56, 128, 128, 3, 3, 1, 2, 0, 1, 8, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_0_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res3_0_branch2b_1__2_res, 1, 28, 28, 128 );
  VTABufferFree(Conv_gpu_0_res3_0_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_0_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_0_branch2a_1__2_res
  free(Conv_gpu_0_res3_0_branch2a_1__2_res);

  //Run relu : Relu_gpu_0_res3_0_branch2b_bn_2__1
  relu(Conv_gpu_0_res3_0_branch2b_1__2_res, Conv_gpu_0_res3_0_branch2b_1__2_res, 100352 );

  //Run allocactivation : Sum_gpu_0_res3_2_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res3_2_branch2c_bn_2__1_res = (int8_t *)malloc(401408);

  //Run convolution : Conv_gpu_0_res3_0_branch2c_1__2
  int8_t* Conv_gpu_0_res3_0_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Conv_gpu_0_res3_0_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_0_branch2c_1__2_input_transpose), 1, 28, 28, 128);
  int8_t* Conv_gpu_0_res3_0_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(401408);
  convolution_wo_tr(Conv_gpu_0_res3_0_branch2c_1__2_input_transpose, gpu_0_res3_0_branch2c_w_0__1, (int32_t *)conv_bias__65, Conv_gpu_0_res3_0_branch2c_1__2_output_bef_transpose, 1, 28, 28, 128, 512, 1, 1, 0, 1, 0, 1, 7, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_0_branch2c_1__2_output_bef_transpose), Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 1, 28, 28, 512 );
  VTABufferFree(Conv_gpu_0_res3_0_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_0_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_0_branch2b_1__2_res
  free(Conv_gpu_0_res3_0_branch2b_1__2_res);

  //Run allocactivation : Conv_gpu_0_res3_0_branch1_1__2_res
  int8_t *Conv_gpu_0_res3_0_branch1_1__2_res = (int8_t *)malloc(401408);

  //Run convolution : Conv_gpu_0_res3_0_branch1_1__2
  int8_t* Conv_gpu_0_res3_0_branch1_1__2_input_transpose = (int8_t *)VTABufferAlloc(802816);
  transpose_nhwc2vtaio(Sum_gpu_0_res2_2_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_0_branch1_1__2_input_transpose), 1, 56, 56, 256);
  int8_t* Conv_gpu_0_res3_0_branch1_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(401408);
  convolution_wo_tr(Conv_gpu_0_res3_0_branch1_1__2_input_transpose, gpu_0_res3_0_branch1_w_0__1, (int32_t *)conv_bias__66, Conv_gpu_0_res3_0_branch1_1__2_output_bef_transpose, 1, 56, 56, 256, 512, 1, 1, 0, 2, 0, 1, 5, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_0_branch1_1__2_output_bef_transpose), Conv_gpu_0_res3_0_branch1_1__2_res, 1, 28, 28, 512 );
  VTABufferFree(Conv_gpu_0_res3_0_branch1_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_0_branch1_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Sum_gpu_0_res2_2_branch2c_bn_2__1_res
  free(Sum_gpu_0_res2_2_branch2c_bn_2__1_res);

  //Run elementadd : Sum_gpu_0_res3_0_branch2c_bn_2__1
  elemadd(Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 1.0/4.000000, 0, Conv_gpu_0_res3_0_branch1_1__2_res, 1.0/8.000000, 0, Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 1.0/4.000000, 0, 401408 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_0_branch1_1__2_res
  free(Conv_gpu_0_res3_0_branch1_1__2_res);

  //Run relu : Relu_gpu_0_res3_0_branch2c_bn_3__1
  relu(Sum_gpu_0_res3_2_branch2c_bn_2__1_res, Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 401408 );

  //Run allocactivation : Conv_gpu_0_res3_1_branch2a_1__2_res
  int8_t *Conv_gpu_0_res3_1_branch2a_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res3_1_branch2a_1__2
  int8_t* Conv_gpu_0_res3_1_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(401408);
  transpose_nhwc2vtaio(Sum_gpu_0_res3_2_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_1_branch2a_1__2_input_transpose), 1, 28, 28, 512);
  int8_t* Conv_gpu_0_res3_1_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res3_1_branch2a_1__2_input_transpose, gpu_0_res3_1_branch2a_w_0__1, (int32_t *)conv_bias__67, Conv_gpu_0_res3_1_branch2a_1__2_output_bef_transpose, 1, 28, 28, 512, 128, 1, 1, 0, 1, 0, 1, 7, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_1_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res3_1_branch2a_1__2_res, 1, 28, 28, 128 );
  VTABufferFree(Conv_gpu_0_res3_1_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_1_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res3_1_branch2a_bn_2__1
  relu(Conv_gpu_0_res3_1_branch2a_1__2_res, Conv_gpu_0_res3_1_branch2a_1__2_res, 100352 );

  //Run allocactivation : Conv_gpu_0_res3_1_branch2b_1__2_res
  int8_t *Conv_gpu_0_res3_1_branch2b_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res3_1_branch2b_1__2
  int8_t* Conv_gpu_0_res3_1_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Conv_gpu_0_res3_1_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_1_branch2b_1__2_input_transpose), 1, 28, 28, 128);
  int8_t* Conv_gpu_0_res3_1_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res3_1_branch2b_1__2_input_transpose, gpu_0_res3_1_branch2b_w_0__1, (int32_t *)conv_bias__68, Conv_gpu_0_res3_1_branch2b_1__2_output_bef_transpose, 1, 28, 28, 128, 128, 3, 3, 1, 1, 0, 1, 7, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_1_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res3_1_branch2b_1__2_res, 1, 28, 28, 128 );
  VTABufferFree(Conv_gpu_0_res3_1_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_1_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_1_branch2a_1__2_res
  free(Conv_gpu_0_res3_1_branch2a_1__2_res);

  //Run allocactivation : Relu_gpu_0_res3_1_branch2b_bn_2__1_res
  int8_t *Relu_gpu_0_res3_1_branch2b_bn_2__1_res = (int8_t *)malloc(100352);

  //Run relu : Relu_gpu_0_res3_1_branch2b_bn_2__1
  relu(Conv_gpu_0_res3_1_branch2b_1__2_res, Relu_gpu_0_res3_1_branch2b_bn_2__1_res, 100352, 0.125000, 0.062500 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_1_branch2b_1__2_res
  free(Conv_gpu_0_res3_1_branch2b_1__2_res);

  //Run allocactivation : Conv_gpu_0_res3_1_branch2c_1__2_res
  int8_t *Conv_gpu_0_res3_1_branch2c_1__2_res = (int8_t *)malloc(401408);

  //Run convolution : Conv_gpu_0_res3_1_branch2c_1__2
  int8_t* Conv_gpu_0_res3_1_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Relu_gpu_0_res3_1_branch2b_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_1_branch2c_1__2_input_transpose), 1, 28, 28, 128);
  int8_t* Conv_gpu_0_res3_1_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(401408);
  convolution_wo_tr(Conv_gpu_0_res3_1_branch2c_1__2_input_transpose, gpu_0_res3_1_branch2c_w_0__1, (int32_t *)conv_bias__69, Conv_gpu_0_res3_1_branch2c_1__2_output_bef_transpose, 1, 28, 28, 128, 512, 1, 1, 0, 1, 0, 1, 7, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_1_branch2c_1__2_output_bef_transpose), Conv_gpu_0_res3_1_branch2c_1__2_res, 1, 28, 28, 512 );
  VTABufferFree(Conv_gpu_0_res3_1_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_1_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Relu_gpu_0_res3_1_branch2b_bn_2__1_res
  free(Relu_gpu_0_res3_1_branch2b_bn_2__1_res);

  //Run elementadd : Sum_gpu_0_res3_1_branch2c_bn_2__1
  elemadd(Conv_gpu_0_res3_1_branch2c_1__2_res, 1.0/8.000000, 0, Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 1.0/4.000000, 0, Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 1.0/4.000000, 0, 401408 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_1_branch2c_1__2_res
  free(Conv_gpu_0_res3_1_branch2c_1__2_res);

  //Run relu : Relu_gpu_0_res3_1_branch2c_bn_3__1
  relu(Sum_gpu_0_res3_2_branch2c_bn_2__1_res, Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 401408 );

  //Run allocactivation : Conv_gpu_0_res3_2_branch2a_1__2_res
  int8_t *Conv_gpu_0_res3_2_branch2a_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res3_2_branch2a_1__2
  int8_t* Conv_gpu_0_res3_2_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(401408);
  transpose_nhwc2vtaio(Sum_gpu_0_res3_2_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_2_branch2a_1__2_input_transpose), 1, 28, 28, 512);
  int8_t* Conv_gpu_0_res3_2_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res3_2_branch2a_1__2_input_transpose, gpu_0_res3_2_branch2a_w_0__1, (int32_t *)conv_bias__70, Conv_gpu_0_res3_2_branch2a_1__2_output_bef_transpose, 1, 28, 28, 512, 128, 1, 1, 0, 1, 0, 1, 6, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_2_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res3_2_branch2a_1__2_res, 1, 28, 28, 128 );
  VTABufferFree(Conv_gpu_0_res3_2_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_2_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res3_2_branch2a_bn_2__1
  relu(Conv_gpu_0_res3_2_branch2a_1__2_res, Conv_gpu_0_res3_2_branch2a_1__2_res, 100352 );

  //Run allocactivation : Conv_gpu_0_res3_2_branch2b_1__2_res
  int8_t *Conv_gpu_0_res3_2_branch2b_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res3_2_branch2b_1__2
  int8_t* Conv_gpu_0_res3_2_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Conv_gpu_0_res3_2_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_2_branch2b_1__2_input_transpose), 1, 28, 28, 128);
  int8_t* Conv_gpu_0_res3_2_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res3_2_branch2b_1__2_input_transpose, gpu_0_res3_2_branch2b_w_0__1, (int32_t *)conv_bias__71, Conv_gpu_0_res3_2_branch2b_1__2_output_bef_transpose, 1, 28, 28, 128, 128, 3, 3, 1, 1, 0, 1, 8, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_2_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res3_2_branch2b_1__2_res, 1, 28, 28, 128 );
  VTABufferFree(Conv_gpu_0_res3_2_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_2_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_2_branch2a_1__2_res
  free(Conv_gpu_0_res3_2_branch2a_1__2_res);

  //Run allocactivation : Relu_gpu_0_res3_2_branch2b_bn_2__1_res
  int8_t *Relu_gpu_0_res3_2_branch2b_bn_2__1_res = (int8_t *)malloc(100352);

  //Run relu : Relu_gpu_0_res3_2_branch2b_bn_2__1
  relu(Conv_gpu_0_res3_2_branch2b_1__2_res, Relu_gpu_0_res3_2_branch2b_bn_2__1_res, 100352, 0.125000, 0.062500 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_2_branch2b_1__2_res
  free(Conv_gpu_0_res3_2_branch2b_1__2_res);

  //Run allocactivation : Conv_gpu_0_res3_2_branch2c_1__2_res
  int8_t *Conv_gpu_0_res3_2_branch2c_1__2_res = (int8_t *)malloc(401408);

  //Run convolution : Conv_gpu_0_res3_2_branch2c_1__2
  int8_t* Conv_gpu_0_res3_2_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Relu_gpu_0_res3_2_branch2b_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_2_branch2c_1__2_input_transpose), 1, 28, 28, 128);
  int8_t* Conv_gpu_0_res3_2_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(401408);
  convolution_wo_tr(Conv_gpu_0_res3_2_branch2c_1__2_input_transpose, gpu_0_res3_2_branch2c_w_0__1, (int32_t *)conv_bias__72, Conv_gpu_0_res3_2_branch2c_1__2_output_bef_transpose, 1, 28, 28, 128, 512, 1, 1, 0, 1, 0, 1, 7, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_2_branch2c_1__2_output_bef_transpose), Conv_gpu_0_res3_2_branch2c_1__2_res, 1, 28, 28, 512 );
  VTABufferFree(Conv_gpu_0_res3_2_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_2_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Relu_gpu_0_res3_2_branch2b_bn_2__1_res
  free(Relu_gpu_0_res3_2_branch2b_bn_2__1_res);

  //Run elementadd : Sum_gpu_0_res3_2_branch2c_bn_2__1
  elemadd(Conv_gpu_0_res3_2_branch2c_1__2_res, 1.0/8.000000, 0, Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 1.0/4.000000, 0, Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 1.0/4.000000, 0, 401408 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_2_branch2c_1__2_res
  free(Conv_gpu_0_res3_2_branch2c_1__2_res);

  //Run relu : Relu_gpu_0_res3_2_branch2c_bn_3__1
  relu(Sum_gpu_0_res3_2_branch2c_bn_2__1_res, Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 401408 );

  //Run allocactivation : Conv_gpu_0_res3_3_branch2a_1__2_res
  int8_t *Conv_gpu_0_res3_3_branch2a_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res3_3_branch2a_1__2
  int8_t* Conv_gpu_0_res3_3_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(401408);
  transpose_nhwc2vtaio(Sum_gpu_0_res3_2_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_3_branch2a_1__2_input_transpose), 1, 28, 28, 512);
  int8_t* Conv_gpu_0_res3_3_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res3_3_branch2a_1__2_input_transpose, gpu_0_res3_3_branch2a_w_0__1, (int32_t *)conv_bias__73, Conv_gpu_0_res3_3_branch2a_1__2_output_bef_transpose, 1, 28, 28, 512, 128, 1, 1, 0, 1, 0, 1, 7, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_3_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res3_3_branch2a_1__2_res, 1, 28, 28, 128 );
  VTABufferFree(Conv_gpu_0_res3_3_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_3_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res3_3_branch2a_bn_2__1
  relu(Conv_gpu_0_res3_3_branch2a_1__2_res, Conv_gpu_0_res3_3_branch2a_1__2_res, 100352 );

  //Run allocactivation : Conv_gpu_0_res3_3_branch2b_1__2_res
  int8_t *Conv_gpu_0_res3_3_branch2b_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res3_3_branch2b_1__2
  int8_t* Conv_gpu_0_res3_3_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Conv_gpu_0_res3_3_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_3_branch2b_1__2_input_transpose), 1, 28, 28, 128);
  int8_t* Conv_gpu_0_res3_3_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res3_3_branch2b_1__2_input_transpose, gpu_0_res3_3_branch2b_w_0__1, (int32_t *)conv_bias__11, Conv_gpu_0_res3_3_branch2b_1__2_output_bef_transpose, 1, 28, 28, 128, 128, 3, 3, 1, 1, 0, 1, 8, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_3_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res3_3_branch2b_1__2_res, 1, 28, 28, 128 );
  VTABufferFree(Conv_gpu_0_res3_3_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_3_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_3_branch2a_1__2_res
  free(Conv_gpu_0_res3_3_branch2a_1__2_res);

  //Run relu : Relu_gpu_0_res3_3_branch2b_bn_2__1
  relu(Conv_gpu_0_res3_3_branch2b_1__2_res, Conv_gpu_0_res3_3_branch2b_1__2_res, 100352 );

  //Run allocactivation : Sum_gpu_0_res3_3_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res3_3_branch2c_bn_2__1_res = (int8_t *)malloc(401408);

  //Run convolution : Conv_gpu_0_res3_3_branch2c_1__2
  int8_t* Conv_gpu_0_res3_3_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Conv_gpu_0_res3_3_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_3_branch2c_1__2_input_transpose), 1, 28, 28, 128);
  int8_t* Conv_gpu_0_res3_3_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(401408);
  convolution_wo_tr(Conv_gpu_0_res3_3_branch2c_1__2_input_transpose, gpu_0_res3_3_branch2c_w_0__1, (int32_t *)conv_bias__74, Conv_gpu_0_res3_3_branch2c_1__2_output_bef_transpose, 1, 28, 28, 128, 512, 1, 1, 0, 1, 0, 1, 6, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res3_3_branch2c_1__2_output_bef_transpose), Sum_gpu_0_res3_3_branch2c_bn_2__1_res, 1, 28, 28, 512 );
  VTABufferFree(Conv_gpu_0_res3_3_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res3_3_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res3_3_branch2b_1__2_res
  free(Conv_gpu_0_res3_3_branch2b_1__2_res);

  //Run elementadd : Sum_gpu_0_res3_3_branch2c_bn_2__1
  elemadd(Sum_gpu_0_res3_3_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res3_2_branch2c_bn_2__1_res, 1.0/4.000000, 0, Sum_gpu_0_res3_3_branch2c_bn_2__1_res, 1.0/8.000000, 0, 401408 );

  //Run deallocactivation : dealloc_Sum_gpu_0_res3_2_branch2c_bn_2__1_res
  free(Sum_gpu_0_res3_2_branch2c_bn_2__1_res);

  //Run relu : Relu_gpu_0_res3_3_branch2c_bn_3__1
  relu(Sum_gpu_0_res3_3_branch2c_bn_2__1_res, Sum_gpu_0_res3_3_branch2c_bn_2__1_res, 401408 );

  //Run allocactivation : Conv_gpu_0_res4_0_branch2a_1__2_res
  int8_t *Conv_gpu_0_res4_0_branch2a_1__2_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res4_0_branch2a_1__2
  int8_t* Conv_gpu_0_res4_0_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(401408);
  transpose_nhwc2vtaio(Sum_gpu_0_res3_3_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_0_branch2a_1__2_input_transpose), 1, 28, 28, 512);
  int8_t* Conv_gpu_0_res4_0_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res4_0_branch2a_1__2_input_transpose, gpu_0_res4_0_branch2a_w_0__1, (int32_t *)conv_bias__75, Conv_gpu_0_res4_0_branch2a_1__2_output_bef_transpose, 1, 28, 28, 512, 256, 1, 1, 0, 1, 0, 1, 7, 28, 28, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_0_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res4_0_branch2a_1__2_res, 1, 28, 28, 256 );
  VTABufferFree(Conv_gpu_0_res4_0_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_0_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res4_0_branch2a_bn_2__1
  relu(Conv_gpu_0_res4_0_branch2a_1__2_res, Conv_gpu_0_res4_0_branch2a_1__2_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res4_0_branch2b_1__2_res
  int8_t *Conv_gpu_0_res4_0_branch2b_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_0_branch2b_1__2
  int8_t* Conv_gpu_0_res4_0_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_0_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_0_branch2b_1__2_input_transpose), 1, 28, 28, 256);
  int8_t* Conv_gpu_0_res4_0_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_0_branch2b_1__2_input_transpose, gpu_0_res4_0_branch2b_w_0__1, (int32_t *)conv_bias__76, Conv_gpu_0_res4_0_branch2b_1__2_output_bef_transpose, 1, 28, 28, 256, 256, 3, 3, 1, 2, 0, 1, 7, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_0_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res4_0_branch2b_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_0_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_0_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_0_branch2a_1__2_res
  free(Conv_gpu_0_res4_0_branch2a_1__2_res);

  //Run relu : Relu_gpu_0_res4_0_branch2b_bn_2__1
  relu(Conv_gpu_0_res4_0_branch2b_1__2_res, Conv_gpu_0_res4_0_branch2b_1__2_res, 50176 );

  //Run allocactivation : Sum_gpu_0_res4_0_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res4_0_branch2c_bn_2__1_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res4_0_branch2c_1__2
  int8_t* Conv_gpu_0_res4_0_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_0_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_0_branch2c_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_0_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res4_0_branch2c_1__2_input_transpose, gpu_0_res4_0_branch2c_w_0__1, (int32_t *)conv_bias__77, Conv_gpu_0_res4_0_branch2c_1__2_output_bef_transpose, 1, 14, 14, 256, 1024, 1, 1, 0, 1, 0, 1, 8, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_0_branch2c_1__2_output_bef_transpose), Sum_gpu_0_res4_0_branch2c_bn_2__1_res, 1, 14, 14, 1024 );
  VTABufferFree(Conv_gpu_0_res4_0_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_0_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_0_branch2b_1__2_res
  free(Conv_gpu_0_res4_0_branch2b_1__2_res);

  //Run allocactivation : Conv_gpu_0_res4_0_branch1_1__2_res
  int8_t *Conv_gpu_0_res4_0_branch1_1__2_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res4_0_branch1_1__2
  int8_t* Conv_gpu_0_res4_0_branch1_1__2_input_transpose = (int8_t *)VTABufferAlloc(401408);
  transpose_nhwc2vtaio(Sum_gpu_0_res3_3_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_0_branch1_1__2_input_transpose), 1, 28, 28, 512);
  int8_t* Conv_gpu_0_res4_0_branch1_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res4_0_branch1_1__2_input_transpose, gpu_0_res4_0_branch1_w_0__1, (int32_t *)conv_bias__78, Conv_gpu_0_res4_0_branch1_1__2_output_bef_transpose, 1, 28, 28, 512, 1024, 1, 1, 0, 2, 0, 1, 6, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_0_branch1_1__2_output_bef_transpose), Conv_gpu_0_res4_0_branch1_1__2_res, 1, 14, 14, 1024 );
  VTABufferFree(Conv_gpu_0_res4_0_branch1_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_0_branch1_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Sum_gpu_0_res3_3_branch2c_bn_2__1_res
  free(Sum_gpu_0_res3_3_branch2c_bn_2__1_res);

  //Run elementadd : Sum_gpu_0_res4_0_branch2c_bn_2__1
  elemadd(Sum_gpu_0_res4_0_branch2c_bn_2__1_res, 1.0/8.000000, 0, Conv_gpu_0_res4_0_branch1_1__2_res, 1.0/16.000000, 0, Sum_gpu_0_res4_0_branch2c_bn_2__1_res, 1.0/8.000000, 0, 200704 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_0_branch1_1__2_res
  free(Conv_gpu_0_res4_0_branch1_1__2_res);

  //Run relu : Relu_gpu_0_res4_0_branch2c_bn_3__1
  relu(Sum_gpu_0_res4_0_branch2c_bn_2__1_res, Sum_gpu_0_res4_0_branch2c_bn_2__1_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res4_1_branch2a_1__2_res
  int8_t *Conv_gpu_0_res4_1_branch2a_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_1_branch2a_1__2
  int8_t* Conv_gpu_0_res4_1_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Sum_gpu_0_res4_0_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_1_branch2a_1__2_input_transpose), 1, 14, 14, 1024);
  int8_t* Conv_gpu_0_res4_1_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_1_branch2a_1__2_input_transpose, gpu_0_res4_1_branch2a_w_0__1, (int32_t *)conv_bias__79, Conv_gpu_0_res4_1_branch2a_1__2_output_bef_transpose, 1, 14, 14, 1024, 256, 1, 1, 0, 1, 0, 1, 8, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_1_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res4_1_branch2a_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_1_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_1_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res4_1_branch2a_bn_2__1
  relu(Conv_gpu_0_res4_1_branch2a_1__2_res, Conv_gpu_0_res4_1_branch2a_1__2_res, 50176 );

  //Run allocactivation : Conv_gpu_0_res4_1_branch2b_1__2_res
  int8_t *Conv_gpu_0_res4_1_branch2b_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_1_branch2b_1__2
  int8_t* Conv_gpu_0_res4_1_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_1_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_1_branch2b_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_1_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_1_branch2b_1__2_input_transpose, gpu_0_res4_1_branch2b_w_0__1, (int32_t *)conv_bias__80, Conv_gpu_0_res4_1_branch2b_1__2_output_bef_transpose, 1, 14, 14, 256, 256, 3, 3, 1, 1, 0, 1, 7, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_1_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res4_1_branch2b_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_1_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_1_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_1_branch2a_1__2_res
  free(Conv_gpu_0_res4_1_branch2a_1__2_res);

  //Run relu : Relu_gpu_0_res4_1_branch2b_bn_2__1
  relu(Conv_gpu_0_res4_1_branch2b_1__2_res, Conv_gpu_0_res4_1_branch2b_1__2_res, 50176 );

  //Run allocactivation : Sum_gpu_0_res4_2_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res4_2_branch2c_bn_2__1_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res4_1_branch2c_1__2
  int8_t* Conv_gpu_0_res4_1_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_1_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_1_branch2c_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_1_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res4_1_branch2c_1__2_input_transpose, gpu_0_res4_1_branch2c_w_0__1, (int32_t *)conv_bias__81, Conv_gpu_0_res4_1_branch2c_1__2_output_bef_transpose, 1, 14, 14, 256, 1024, 1, 1, 0, 1, 0, 1, 7, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_1_branch2c_1__2_output_bef_transpose), Sum_gpu_0_res4_2_branch2c_bn_2__1_res, 1, 14, 14, 1024 );
  VTABufferFree(Conv_gpu_0_res4_1_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_1_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_1_branch2b_1__2_res
  free(Conv_gpu_0_res4_1_branch2b_1__2_res);

  //Run elementadd : Sum_gpu_0_res4_1_branch2c_bn_2__1
  elemadd(Sum_gpu_0_res4_2_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res4_0_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res4_2_branch2c_bn_2__1_res, 1.0/8.000000, 0, 200704 );

  //Run deallocactivation : dealloc_Sum_gpu_0_res4_0_branch2c_bn_2__1_res
  free(Sum_gpu_0_res4_0_branch2c_bn_2__1_res);

  //Run relu : Relu_gpu_0_res4_1_branch2c_bn_3__1
  relu(Sum_gpu_0_res4_2_branch2c_bn_2__1_res, Sum_gpu_0_res4_2_branch2c_bn_2__1_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res4_2_branch2a_1__2_res
  int8_t *Conv_gpu_0_res4_2_branch2a_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_2_branch2a_1__2
  int8_t* Conv_gpu_0_res4_2_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Sum_gpu_0_res4_2_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_2_branch2a_1__2_input_transpose), 1, 14, 14, 1024);
  int8_t* Conv_gpu_0_res4_2_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_2_branch2a_1__2_input_transpose, gpu_0_res4_2_branch2a_w_0__1, (int32_t *)conv_bias__82, Conv_gpu_0_res4_2_branch2a_1__2_output_bef_transpose, 1, 14, 14, 1024, 256, 1, 1, 0, 1, 0, 1, 7, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_2_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res4_2_branch2a_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_2_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_2_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res4_2_branch2a_bn_2__1
  relu(Conv_gpu_0_res4_2_branch2a_1__2_res, Conv_gpu_0_res4_2_branch2a_1__2_res, 50176 );

  //Run allocactivation : Conv_gpu_0_res4_2_branch2b_1__2_res
  int8_t *Conv_gpu_0_res4_2_branch2b_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_2_branch2b_1__2
  int8_t* Conv_gpu_0_res4_2_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_2_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_2_branch2b_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_2_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_2_branch2b_1__2_input_transpose, gpu_0_res4_2_branch2b_w_0__1, (int32_t *)conv_bias__83, Conv_gpu_0_res4_2_branch2b_1__2_output_bef_transpose, 1, 14, 14, 256, 256, 3, 3, 1, 1, 0, 1, 7, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_2_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res4_2_branch2b_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_2_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_2_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_2_branch2a_1__2_res
  free(Conv_gpu_0_res4_2_branch2a_1__2_res);

  //Run allocactivation : Relu_gpu_0_res4_2_branch2b_bn_2__1_res
  int8_t *Relu_gpu_0_res4_2_branch2b_bn_2__1_res = (int8_t *)malloc(50176);

  //Run relu : Relu_gpu_0_res4_2_branch2b_bn_2__1
  relu(Conv_gpu_0_res4_2_branch2b_1__2_res, Relu_gpu_0_res4_2_branch2b_bn_2__1_res, 50176, 0.125000, 0.062500 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_2_branch2b_1__2_res
  free(Conv_gpu_0_res4_2_branch2b_1__2_res);

  //Run allocactivation : Conv_gpu_0_res4_2_branch2c_1__2_res
  int8_t *Conv_gpu_0_res4_2_branch2c_1__2_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res4_2_branch2c_1__2
  int8_t* Conv_gpu_0_res4_2_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Relu_gpu_0_res4_2_branch2b_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_2_branch2c_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_2_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res4_2_branch2c_1__2_input_transpose, gpu_0_res4_2_branch2c_w_0__1, (int32_t *)conv_bias__84, Conv_gpu_0_res4_2_branch2c_1__2_output_bef_transpose, 1, 14, 14, 256, 1024, 1, 1, 0, 1, 0, 1, 6, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_2_branch2c_1__2_output_bef_transpose), Conv_gpu_0_res4_2_branch2c_1__2_res, 1, 14, 14, 1024 );
  VTABufferFree(Conv_gpu_0_res4_2_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_2_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Relu_gpu_0_res4_2_branch2b_bn_2__1_res
  free(Relu_gpu_0_res4_2_branch2b_bn_2__1_res);

  //Run elementadd : Sum_gpu_0_res4_2_branch2c_bn_2__1
  elemadd(Conv_gpu_0_res4_2_branch2c_1__2_res, 1.0/16.000000, 0, Sum_gpu_0_res4_2_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res4_2_branch2c_bn_2__1_res, 1.0/8.000000, 0, 200704 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_2_branch2c_1__2_res
  free(Conv_gpu_0_res4_2_branch2c_1__2_res);

  //Run relu : Relu_gpu_0_res4_2_branch2c_bn_3__1
  relu(Sum_gpu_0_res4_2_branch2c_bn_2__1_res, Sum_gpu_0_res4_2_branch2c_bn_2__1_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res4_3_branch2a_1__2_res
  int8_t *Conv_gpu_0_res4_3_branch2a_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_3_branch2a_1__2
  int8_t* Conv_gpu_0_res4_3_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Sum_gpu_0_res4_2_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_3_branch2a_1__2_input_transpose), 1, 14, 14, 1024);
  int8_t* Conv_gpu_0_res4_3_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_3_branch2a_1__2_input_transpose, gpu_0_res4_3_branch2a_w_0__1, (int32_t *)conv_bias__85, Conv_gpu_0_res4_3_branch2a_1__2_output_bef_transpose, 1, 14, 14, 1024, 256, 1, 1, 0, 1, 0, 1, 8, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_3_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res4_3_branch2a_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_3_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_3_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res4_3_branch2a_bn_2__1
  relu(Conv_gpu_0_res4_3_branch2a_1__2_res, Conv_gpu_0_res4_3_branch2a_1__2_res, 50176 );

  //Run allocactivation : Conv_gpu_0_res4_3_branch2b_1__2_res
  int8_t *Conv_gpu_0_res4_3_branch2b_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_3_branch2b_1__2
  int8_t* Conv_gpu_0_res4_3_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_3_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_3_branch2b_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_3_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_3_branch2b_1__2_input_transpose, gpu_0_res4_3_branch2b_w_0__1, (int32_t *)conv_bias__86, Conv_gpu_0_res4_3_branch2b_1__2_output_bef_transpose, 1, 14, 14, 256, 256, 3, 3, 1, 1, 0, 1, 8, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_3_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res4_3_branch2b_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_3_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_3_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_3_branch2a_1__2_res
  free(Conv_gpu_0_res4_3_branch2a_1__2_res);

  //Run allocactivation : Relu_gpu_0_res4_3_branch2b_bn_2__1_res
  int8_t *Relu_gpu_0_res4_3_branch2b_bn_2__1_res = (int8_t *)malloc(50176);

  //Run relu : Relu_gpu_0_res4_3_branch2b_bn_2__1
  relu(Conv_gpu_0_res4_3_branch2b_1__2_res, Relu_gpu_0_res4_3_branch2b_bn_2__1_res, 50176, 0.125000, 0.062500 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_3_branch2b_1__2_res
  free(Conv_gpu_0_res4_3_branch2b_1__2_res);

  //Run allocactivation : Sum_gpu_0_res4_3_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res4_3_branch2c_bn_2__1_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res4_3_branch2c_1__2
  int8_t* Conv_gpu_0_res4_3_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Relu_gpu_0_res4_3_branch2b_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_3_branch2c_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_3_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res4_3_branch2c_1__2_input_transpose, gpu_0_res4_3_branch2c_w_0__1, (int32_t *)conv_bias__87, Conv_gpu_0_res4_3_branch2c_1__2_output_bef_transpose, 1, 14, 14, 256, 1024, 1, 1, 0, 1, 0, 1, 7, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_3_branch2c_1__2_output_bef_transpose), Sum_gpu_0_res4_3_branch2c_bn_2__1_res, 1, 14, 14, 1024 );
  VTABufferFree(Conv_gpu_0_res4_3_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_3_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Relu_gpu_0_res4_3_branch2b_bn_2__1_res
  free(Relu_gpu_0_res4_3_branch2b_bn_2__1_res);

  //Run elementadd : Sum_gpu_0_res4_3_branch2c_bn_2__1
  elemadd(Sum_gpu_0_res4_3_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res4_2_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res4_3_branch2c_bn_2__1_res, 1.0/8.000000, 0, 200704 );

  //Run deallocactivation : dealloc_Sum_gpu_0_res4_2_branch2c_bn_2__1_res
  free(Sum_gpu_0_res4_2_branch2c_bn_2__1_res);

  //Run relu : Relu_gpu_0_res4_3_branch2c_bn_3__1
  relu(Sum_gpu_0_res4_3_branch2c_bn_2__1_res, Sum_gpu_0_res4_3_branch2c_bn_2__1_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res4_4_branch2a_1__2_res
  int8_t *Conv_gpu_0_res4_4_branch2a_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_4_branch2a_1__2
  int8_t* Conv_gpu_0_res4_4_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Sum_gpu_0_res4_3_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_4_branch2a_1__2_input_transpose), 1, 14, 14, 1024);
  int8_t* Conv_gpu_0_res4_4_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_4_branch2a_1__2_input_transpose, gpu_0_res4_4_branch2a_w_0__1, (int32_t *)conv_bias__88, Conv_gpu_0_res4_4_branch2a_1__2_output_bef_transpose, 1, 14, 14, 1024, 256, 1, 1, 0, 1, 0, 1, 8, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_4_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res4_4_branch2a_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_4_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_4_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res4_4_branch2a_bn_2__1
  relu(Conv_gpu_0_res4_4_branch2a_1__2_res, Conv_gpu_0_res4_4_branch2a_1__2_res, 50176 );

  //Run allocactivation : Conv_gpu_0_res4_4_branch2b_1__2_res
  int8_t *Conv_gpu_0_res4_4_branch2b_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_4_branch2b_1__2
  int8_t* Conv_gpu_0_res4_4_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_4_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_4_branch2b_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_4_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_4_branch2b_1__2_input_transpose, gpu_0_res4_4_branch2b_w_0__1, (int32_t *)conv_bias__89, Conv_gpu_0_res4_4_branch2b_1__2_output_bef_transpose, 1, 14, 14, 256, 256, 3, 3, 1, 1, 0, 1, 7, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_4_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res4_4_branch2b_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_4_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_4_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_4_branch2a_1__2_res
  free(Conv_gpu_0_res4_4_branch2a_1__2_res);

  //Run relu : Relu_gpu_0_res4_4_branch2b_bn_2__1
  relu(Conv_gpu_0_res4_4_branch2b_1__2_res, Conv_gpu_0_res4_4_branch2b_1__2_res, 50176 );

  //Run allocactivation : Sum_gpu_0_res4_4_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res4_4_branch2c_bn_2__1_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res4_4_branch2c_1__2
  int8_t* Conv_gpu_0_res4_4_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_4_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_4_branch2c_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_4_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res4_4_branch2c_1__2_input_transpose, gpu_0_res4_4_branch2c_w_0__1, (int32_t *)conv_bias__90, Conv_gpu_0_res4_4_branch2c_1__2_output_bef_transpose, 1, 14, 14, 256, 1024, 1, 1, 0, 1, 0, 1, 7, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_4_branch2c_1__2_output_bef_transpose), Sum_gpu_0_res4_4_branch2c_bn_2__1_res, 1, 14, 14, 1024 );
  VTABufferFree(Conv_gpu_0_res4_4_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_4_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_4_branch2b_1__2_res
  free(Conv_gpu_0_res4_4_branch2b_1__2_res);

  //Run elementadd : Sum_gpu_0_res4_4_branch2c_bn_2__1
  elemadd(Sum_gpu_0_res4_4_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res4_3_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res4_4_branch2c_bn_2__1_res, 1.0/8.000000, 0, 200704 );

  //Run deallocactivation : dealloc_Sum_gpu_0_res4_3_branch2c_bn_2__1_res
  free(Sum_gpu_0_res4_3_branch2c_bn_2__1_res);

  //Run relu : Relu_gpu_0_res4_4_branch2c_bn_3__1
  relu(Sum_gpu_0_res4_4_branch2c_bn_2__1_res, Sum_gpu_0_res4_4_branch2c_bn_2__1_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res4_5_branch2a_1__2_res
  int8_t *Conv_gpu_0_res4_5_branch2a_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_5_branch2a_1__2
  int8_t* Conv_gpu_0_res4_5_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Sum_gpu_0_res4_4_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_5_branch2a_1__2_input_transpose), 1, 14, 14, 1024);
  int8_t* Conv_gpu_0_res4_5_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_5_branch2a_1__2_input_transpose, gpu_0_res4_5_branch2a_w_0__1, (int32_t *)conv_bias__91, Conv_gpu_0_res4_5_branch2a_1__2_output_bef_transpose, 1, 14, 14, 1024, 256, 1, 1, 0, 1, 0, 1, 7, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_5_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res4_5_branch2a_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_5_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_5_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res4_5_branch2a_bn_2__1
  relu(Conv_gpu_0_res4_5_branch2a_1__2_res, Conv_gpu_0_res4_5_branch2a_1__2_res, 50176 );

  //Run allocactivation : Conv_gpu_0_res4_5_branch2b_1__2_res
  int8_t *Conv_gpu_0_res4_5_branch2b_1__2_res = (int8_t *)malloc(50176);

  //Run convolution : Conv_gpu_0_res4_5_branch2b_1__2
  int8_t* Conv_gpu_0_res4_5_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_5_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_5_branch2b_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_5_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(50176);
  convolution_wo_tr(Conv_gpu_0_res4_5_branch2b_1__2_input_transpose, gpu_0_res4_5_branch2b_w_0__1, (int32_t *)conv_bias__3, Conv_gpu_0_res4_5_branch2b_1__2_output_bef_transpose, 1, 14, 14, 256, 256, 3, 3, 1, 1, 0, 1, 8, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_5_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res4_5_branch2b_1__2_res, 1, 14, 14, 256 );
  VTABufferFree(Conv_gpu_0_res4_5_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_5_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_5_branch2a_1__2_res
  free(Conv_gpu_0_res4_5_branch2a_1__2_res);

  //Run relu : Relu_gpu_0_res4_5_branch2b_bn_2__1
  relu(Conv_gpu_0_res4_5_branch2b_1__2_res, Conv_gpu_0_res4_5_branch2b_1__2_res, 50176 );

  //Run allocactivation : Sum_gpu_0_res4_5_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res4_5_branch2c_bn_2__1_res = (int8_t *)malloc(200704);

  //Run convolution : Conv_gpu_0_res4_5_branch2c_1__2
  int8_t* Conv_gpu_0_res4_5_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(50176);
  transpose_nhwc2vtaio(Conv_gpu_0_res4_5_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_5_branch2c_1__2_input_transpose), 1, 14, 14, 256);
  int8_t* Conv_gpu_0_res4_5_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(200704);
  convolution_wo_tr(Conv_gpu_0_res4_5_branch2c_1__2_input_transpose, gpu_0_res4_5_branch2c_w_0__1, (int32_t *)conv_bias__26, Conv_gpu_0_res4_5_branch2c_1__2_output_bef_transpose, 1, 14, 14, 256, 1024, 1, 1, 0, 1, 0, 1, 6, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res4_5_branch2c_1__2_output_bef_transpose), Sum_gpu_0_res4_5_branch2c_bn_2__1_res, 1, 14, 14, 1024 );
  VTABufferFree(Conv_gpu_0_res4_5_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res4_5_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res4_5_branch2b_1__2_res
  free(Conv_gpu_0_res4_5_branch2b_1__2_res);

  //Run elementadd : Sum_gpu_0_res4_5_branch2c_bn_2__1
  elemadd(Sum_gpu_0_res4_5_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res4_4_branch2c_bn_2__1_res, 1.0/8.000000, 0, Sum_gpu_0_res4_5_branch2c_bn_2__1_res, 1.0/8.000000, 0, 200704 );

  //Run deallocactivation : dealloc_Sum_gpu_0_res4_4_branch2c_bn_2__1_res
  free(Sum_gpu_0_res4_4_branch2c_bn_2__1_res);

  //Run relu : Relu_gpu_0_res4_5_branch2c_bn_3__1
  relu(Sum_gpu_0_res4_5_branch2c_bn_2__1_res, Sum_gpu_0_res4_5_branch2c_bn_2__1_res, 200704 );

  //Run allocactivation : Conv_gpu_0_res5_0_branch2a_1__2_res
  int8_t *Conv_gpu_0_res5_0_branch2a_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res5_0_branch2a_1__2
  int8_t* Conv_gpu_0_res5_0_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Sum_gpu_0_res4_5_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_0_branch2a_1__2_input_transpose), 1, 14, 14, 1024);
  int8_t* Conv_gpu_0_res5_0_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res5_0_branch2a_1__2_input_transpose, gpu_0_res5_0_branch2a_w_0__1, (int32_t *)conv_bias__92, Conv_gpu_0_res5_0_branch2a_1__2_output_bef_transpose, 1, 14, 14, 1024, 512, 1, 1, 0, 1, 0, 1, 6, 14, 14, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_0_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res5_0_branch2a_1__2_res, 1, 14, 14, 512 );
  VTABufferFree(Conv_gpu_0_res5_0_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_0_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res5_0_branch2a_bn_2__1
  relu(Conv_gpu_0_res5_0_branch2a_1__2_res, Conv_gpu_0_res5_0_branch2a_1__2_res, 100352 );

  //Run allocactivation : Conv_gpu_0_res5_0_branch2b_1__2_res
  int8_t *Conv_gpu_0_res5_0_branch2b_1__2_res = (int8_t *)malloc(25088);

  //Run convolution : Conv_gpu_0_res5_0_branch2b_1__2
  int8_t* Conv_gpu_0_res5_0_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Conv_gpu_0_res5_0_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_0_branch2b_1__2_input_transpose), 1, 14, 14, 512);
  int8_t* Conv_gpu_0_res5_0_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(25088);
  convolution_wo_tr(Conv_gpu_0_res5_0_branch2b_1__2_input_transpose, gpu_0_res5_0_branch2b_w_0__1, (int32_t *)conv_bias__93, Conv_gpu_0_res5_0_branch2b_1__2_output_bef_transpose, 1, 14, 14, 512, 512, 3, 3, 1, 2, 0, 1, 9, 7, 7, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_0_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res5_0_branch2b_1__2_res, 1, 7, 7, 512 );
  VTABufferFree(Conv_gpu_0_res5_0_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_0_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_0_branch2a_1__2_res
  free(Conv_gpu_0_res5_0_branch2a_1__2_res);

  //Run allocactivation : Relu_gpu_0_res5_0_branch2b_bn_2__1_res
  int8_t *Relu_gpu_0_res5_0_branch2b_bn_2__1_res = (int8_t *)malloc(25088);

  //Run relu : Relu_gpu_0_res5_0_branch2b_bn_2__1
  relu(Conv_gpu_0_res5_0_branch2b_1__2_res, Relu_gpu_0_res5_0_branch2b_bn_2__1_res, 25088, 0.125000, 0.062500 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_0_branch2b_1__2_res
  free(Conv_gpu_0_res5_0_branch2b_1__2_res);

  //Run allocactivation : Conv_gpu_0_res5_0_branch2c_1__2_res
  int8_t *Conv_gpu_0_res5_0_branch2c_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res5_0_branch2c_1__2
  int8_t* Conv_gpu_0_res5_0_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(25088);
  transpose_nhwc2vtaio(Relu_gpu_0_res5_0_branch2b_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_0_branch2c_1__2_input_transpose), 1, 7, 7, 512);
  int8_t* Conv_gpu_0_res5_0_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res5_0_branch2c_1__2_input_transpose, gpu_0_res5_0_branch2c_w_0__1, (int32_t *)conv_bias__94, Conv_gpu_0_res5_0_branch2c_1__2_output_bef_transpose, 1, 7, 7, 512, 2048, 1, 1, 0, 1, 0, 1, 7, 7, 7, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_0_branch2c_1__2_output_bef_transpose), Conv_gpu_0_res5_0_branch2c_1__2_res, 1, 7, 7, 2048 );
  VTABufferFree(Conv_gpu_0_res5_0_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_0_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Relu_gpu_0_res5_0_branch2b_bn_2__1_res
  free(Relu_gpu_0_res5_0_branch2b_bn_2__1_res);

  //Run allocactivation : Conv_gpu_0_res5_0_branch1_1__2_res
  int8_t *Conv_gpu_0_res5_0_branch1_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res5_0_branch1_1__2
  int8_t* Conv_gpu_0_res5_0_branch1_1__2_input_transpose = (int8_t *)VTABufferAlloc(200704);
  transpose_nhwc2vtaio(Sum_gpu_0_res4_5_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_0_branch1_1__2_input_transpose), 1, 14, 14, 1024);
  int8_t* Conv_gpu_0_res5_0_branch1_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res5_0_branch1_1__2_input_transpose, gpu_0_res5_0_branch1_w_0__1, (int32_t *)conv_bias__95, Conv_gpu_0_res5_0_branch1_1__2_output_bef_transpose, 1, 14, 14, 1024, 2048, 1, 1, 0, 2, 0, 1, 6, 7, 7, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_0_branch1_1__2_output_bef_transpose), Conv_gpu_0_res5_0_branch1_1__2_res, 1, 7, 7, 2048 );
  VTABufferFree(Conv_gpu_0_res5_0_branch1_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_0_branch1_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Sum_gpu_0_res4_5_branch2c_bn_2__1_res
  free(Sum_gpu_0_res4_5_branch2c_bn_2__1_res);

  //Run allocactivation : Sum_gpu_0_res5_2_branch2c_bn_2__1_res
  int8_t *Sum_gpu_0_res5_2_branch2c_bn_2__1_res = (int8_t *)malloc(100352);

  //Run elementadd : Sum_gpu_0_res5_0_branch2c_bn_2__1
  elemadd(Conv_gpu_0_res5_0_branch2c_1__2_res, 1.0/4.000000, 0, Conv_gpu_0_res5_0_branch1_1__2_res, 1.0/4.000000, 0, Sum_gpu_0_res5_2_branch2c_bn_2__1_res, 1.0/2.000000, 0, 100352 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_0_branch2c_1__2_res
  free(Conv_gpu_0_res5_0_branch2c_1__2_res);

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_0_branch1_1__2_res
  free(Conv_gpu_0_res5_0_branch1_1__2_res);

  //Run relu : Relu_gpu_0_res5_0_branch2c_bn_3__1
  relu(Sum_gpu_0_res5_2_branch2c_bn_2__1_res, Sum_gpu_0_res5_2_branch2c_bn_2__1_res, 100352 );

  //Run allocactivation : Conv_gpu_0_res5_1_branch2a_1__2_res
  int8_t *Conv_gpu_0_res5_1_branch2a_1__2_res = (int8_t *)malloc(25088);

  //Run convolution : Conv_gpu_0_res5_1_branch2a_1__2
  int8_t* Conv_gpu_0_res5_1_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Sum_gpu_0_res5_2_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_1_branch2a_1__2_input_transpose), 1, 7, 7, 2048);
  int8_t* Conv_gpu_0_res5_1_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(25088);
  convolution_wo_tr(Conv_gpu_0_res5_1_branch2a_1__2_input_transpose, gpu_0_res5_1_branch2a_w_0__1, (int32_t *)conv_bias__96, Conv_gpu_0_res5_1_branch2a_1__2_output_bef_transpose, 1, 7, 7, 2048, 512, 1, 1, 0, 1, 0, 1, 7, 7, 7, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_1_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res5_1_branch2a_1__2_res, 1, 7, 7, 512 );
  VTABufferFree(Conv_gpu_0_res5_1_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_1_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res5_1_branch2a_bn_2__1
  relu(Conv_gpu_0_res5_1_branch2a_1__2_res, Conv_gpu_0_res5_1_branch2a_1__2_res, 25088 );

  //Run allocactivation : Conv_gpu_0_res5_1_branch2b_1__2_res
  int8_t *Conv_gpu_0_res5_1_branch2b_1__2_res = (int8_t *)malloc(25088);

  //Run convolution : Conv_gpu_0_res5_1_branch2b_1__2
  int8_t* Conv_gpu_0_res5_1_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(25088);
  transpose_nhwc2vtaio(Conv_gpu_0_res5_1_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_1_branch2b_1__2_input_transpose), 1, 7, 7, 512);
  int8_t* Conv_gpu_0_res5_1_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(25088);
  convolution_wo_tr(Conv_gpu_0_res5_1_branch2b_1__2_input_transpose, gpu_0_res5_1_branch2b_w_0__1, (int32_t *)conv_bias__97, Conv_gpu_0_res5_1_branch2b_1__2_output_bef_transpose, 1, 7, 7, 512, 512, 3, 3, 1, 1, 0, 1, 9, 7, 7, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_1_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res5_1_branch2b_1__2_res, 1, 7, 7, 512 );
  VTABufferFree(Conv_gpu_0_res5_1_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_1_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_1_branch2a_1__2_res
  free(Conv_gpu_0_res5_1_branch2a_1__2_res);

  //Run relu : Relu_gpu_0_res5_1_branch2b_bn_2__1
  relu(Conv_gpu_0_res5_1_branch2b_1__2_res, Conv_gpu_0_res5_1_branch2b_1__2_res, 25088 );

  //Run allocactivation : Conv_gpu_0_res5_1_branch2c_1__2_res
  int8_t *Conv_gpu_0_res5_1_branch2c_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res5_1_branch2c_1__2
  int8_t* Conv_gpu_0_res5_1_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(25088);
  transpose_nhwc2vtaio(Conv_gpu_0_res5_1_branch2b_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_1_branch2c_1__2_input_transpose), 1, 7, 7, 512);
  int8_t* Conv_gpu_0_res5_1_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res5_1_branch2c_1__2_input_transpose, gpu_0_res5_1_branch2c_w_0__1, (int32_t *)conv_bias__98, Conv_gpu_0_res5_1_branch2c_1__2_output_bef_transpose, 1, 7, 7, 512, 2048, 1, 1, 0, 1, 0, 1, 6, 7, 7, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_1_branch2c_1__2_output_bef_transpose), Conv_gpu_0_res5_1_branch2c_1__2_res, 1, 7, 7, 2048 );
  VTABufferFree(Conv_gpu_0_res5_1_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_1_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_1_branch2b_1__2_res
  free(Conv_gpu_0_res5_1_branch2b_1__2_res);

  //Run elementadd : Sum_gpu_0_res5_1_branch2c_bn_2__1
  elemadd(Conv_gpu_0_res5_1_branch2c_1__2_res, 1.0/8.000000, 0, Sum_gpu_0_res5_2_branch2c_bn_2__1_res, 1.0/2.000000, 0, Sum_gpu_0_res5_2_branch2c_bn_2__1_res, 1.0/2.000000, 0, 100352 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_1_branch2c_1__2_res
  free(Conv_gpu_0_res5_1_branch2c_1__2_res);

  //Run relu : Relu_gpu_0_res5_1_branch2c_bn_3__1
  relu(Sum_gpu_0_res5_2_branch2c_bn_2__1_res, Sum_gpu_0_res5_2_branch2c_bn_2__1_res, 100352 );

  //Run allocactivation : Conv_gpu_0_res5_2_branch2a_1__2_res
  int8_t *Conv_gpu_0_res5_2_branch2a_1__2_res = (int8_t *)malloc(25088);

  //Run convolution : Conv_gpu_0_res5_2_branch2a_1__2
  int8_t* Conv_gpu_0_res5_2_branch2a_1__2_input_transpose = (int8_t *)VTABufferAlloc(100352);
  transpose_nhwc2vtaio(Sum_gpu_0_res5_2_branch2c_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_2_branch2a_1__2_input_transpose), 1, 7, 7, 2048);
  int8_t* Conv_gpu_0_res5_2_branch2a_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(25088);
  convolution_wo_tr(Conv_gpu_0_res5_2_branch2a_1__2_input_transpose, gpu_0_res5_2_branch2a_w_0__1, (int32_t *)conv_bias__99, Conv_gpu_0_res5_2_branch2a_1__2_output_bef_transpose, 1, 7, 7, 2048, 512, 1, 1, 0, 1, 0, 1, 6, 7, 7, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_2_branch2a_1__2_output_bef_transpose), Conv_gpu_0_res5_2_branch2a_1__2_res, 1, 7, 7, 512 );
  VTABufferFree(Conv_gpu_0_res5_2_branch2a_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_2_branch2a_1__2_output_bef_transpose);

  //Run relu : Relu_gpu_0_res5_2_branch2a_bn_2__1
  relu(Conv_gpu_0_res5_2_branch2a_1__2_res, Conv_gpu_0_res5_2_branch2a_1__2_res, 25088 );

  //Run allocactivation : Conv_gpu_0_res5_2_branch2b_1__2_res
  int8_t *Conv_gpu_0_res5_2_branch2b_1__2_res = (int8_t *)malloc(25088);

  //Run convolution : Conv_gpu_0_res5_2_branch2b_1__2
  int8_t* Conv_gpu_0_res5_2_branch2b_1__2_input_transpose = (int8_t *)VTABufferAlloc(25088);
  transpose_nhwc2vtaio(Conv_gpu_0_res5_2_branch2a_1__2_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_2_branch2b_1__2_input_transpose), 1, 7, 7, 512);
  int8_t* Conv_gpu_0_res5_2_branch2b_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(25088);
  convolution_wo_tr(Conv_gpu_0_res5_2_branch2b_1__2_input_transpose, gpu_0_res5_2_branch2b_w_0__1, (int32_t *)conv_bias__13, Conv_gpu_0_res5_2_branch2b_1__2_output_bef_transpose, 1, 7, 7, 512, 512, 3, 3, 1, 1, 0, 1, 9, 7, 7, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_2_branch2b_1__2_output_bef_transpose), Conv_gpu_0_res5_2_branch2b_1__2_res, 1, 7, 7, 512 );
  VTABufferFree(Conv_gpu_0_res5_2_branch2b_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_2_branch2b_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_2_branch2a_1__2_res
  free(Conv_gpu_0_res5_2_branch2a_1__2_res);

  //Run allocactivation : Relu_gpu_0_res5_2_branch2b_bn_2__1_res
  int8_t *Relu_gpu_0_res5_2_branch2b_bn_2__1_res = (int8_t *)malloc(25088);

  //Run relu : Relu_gpu_0_res5_2_branch2b_bn_2__1
  relu(Conv_gpu_0_res5_2_branch2b_1__2_res, Relu_gpu_0_res5_2_branch2b_bn_2__1_res, 25088, 0.062500, 0.031250 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_2_branch2b_1__2_res
  free(Conv_gpu_0_res5_2_branch2b_1__2_res);

  //Run allocactivation : Conv_gpu_0_res5_2_branch2c_1__2_res
  int8_t *Conv_gpu_0_res5_2_branch2c_1__2_res = (int8_t *)malloc(100352);

  //Run convolution : Conv_gpu_0_res5_2_branch2c_1__2
  int8_t* Conv_gpu_0_res5_2_branch2c_1__2_input_transpose = (int8_t *)VTABufferAlloc(25088);
  transpose_nhwc2vtaio(Relu_gpu_0_res5_2_branch2b_bn_2__1_res, (int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_2_branch2c_1__2_input_transpose), 1, 7, 7, 512);
  int8_t* Conv_gpu_0_res5_2_branch2c_1__2_output_bef_transpose = (int8_t *)VTABufferAlloc(100352);
  convolution_wo_tr(Conv_gpu_0_res5_2_branch2c_1__2_input_transpose, gpu_0_res5_2_branch2c_w_0__1, (int32_t *)conv_bias__45, Conv_gpu_0_res5_2_branch2c_1__2_output_bef_transpose, 1, 7, 7, 512, 2048, 1, 1, 0, 1, 0, 1, 6, 7, 7, vtaCmdH);
  transpose_vtaio2nhwc((int8_t* )VTABufferGetVirtAddr(Conv_gpu_0_res5_2_branch2c_1__2_output_bef_transpose), Conv_gpu_0_res5_2_branch2c_1__2_res, 1, 7, 7, 2048 );
  VTABufferFree(Conv_gpu_0_res5_2_branch2c_1__2_input_transpose);
  VTABufferFree(Conv_gpu_0_res5_2_branch2c_1__2_output_bef_transpose);

  //Run deallocactivation : dealloc_Relu_gpu_0_res5_2_branch2b_bn_2__1_res
  free(Relu_gpu_0_res5_2_branch2b_bn_2__1_res);

  //Run elementadd : Sum_gpu_0_res5_2_branch2c_bn_2__1
  elemadd(Conv_gpu_0_res5_2_branch2c_1__2_res, 1.0/8.000000, 0, Sum_gpu_0_res5_2_branch2c_bn_2__1_res, 1.0/2.000000, 0, Sum_gpu_0_res5_2_branch2c_bn_2__1_res, 1.0/2.000000, 0, 100352 );

  //Run deallocactivation : dealloc_Conv_gpu_0_res5_2_branch2c_1__2_res
  free(Conv_gpu_0_res5_2_branch2c_1__2_res);

  //Run relu : Relu_gpu_0_res5_2_branch2c_bn_3__1
  relu(Sum_gpu_0_res5_2_branch2c_bn_2__1_res, Sum_gpu_0_res5_2_branch2c_bn_2__1_res, 100352 );

  //Run allocactivation : AveragePool_gpu_0_pool5_1__1_res
  int8_t *AveragePool_gpu_0_pool5_1__1_res = (int8_t *)malloc(2048);

  //Run avgpool : AveragePool_gpu_0_pool5_1__1
  avgpool(Sum_gpu_0_res5_2_branch2c_bn_2__1_res, 1.0/2.000000, 0, AveragePool_gpu_0_pool5_1__1_res, 1.0/8.000000, 0, 1, 7, 7, 2048, 1, 1, 1, 2048, 7, 7, 0, 1 );

  //Run deallocactivation : dealloc_Sum_gpu_0_res5_2_branch2c_bn_2__1_res
  free(Sum_gpu_0_res5_2_branch2c_bn_2__1_res);

  //Run tensorview : AveragePool_gpu_0_pool5_1__1_res__2
  int8_t* AveragePool_gpu_0_pool5_1__1_res__2 = AveragePool_gpu_0_pool5_1__1_res;

  //Run allocactivation : Gemm_gpu_0_pred_1__1_res
  int8_t *Gemm_gpu_0_pred_1__1_res = (int8_t *)malloc(1000);

  //Run fullyconnected : Gemm_gpu_0_pred_1__1
  fullyconnected(AveragePool_gpu_0_pool5_1__1_res__2, 1.0/8.000000, 0, (int8_t *)VTABufferGetVirtAddr(gpu_0_pred_w_0__1), 1.0/128.000000, 0, (int8_t *)VTABufferGetVirtAddr(gpu_0_pred_b_0), 1.0/1024.000000, 0, Gemm_gpu_0_pred_1__1_res, 1.0/8.000000, 0, 1, 2048, 2048, 1000, 1, 1000, 1 );

  //Run deallocactivation : dealloc_AveragePool_gpu_0_pool5_1__1_res
  free(AveragePool_gpu_0_pool5_1__1_res);

  //Run allocactivation : Gemm_gpu_0_pred_1__1_dequantize_res
  int8_t *Gemm_gpu_0_pred_1__1_dequantize_res = (int8_t *)malloc(4000);

  //Run dequantize : Gemm_gpu_0_pred_1__1_dequantize
  dequantize(Gemm_gpu_0_pred_1__1_res, Gemm_gpu_0_pred_1__1_dequantize_res, 1000, 1/8.000000, 0 );

  //Run deallocactivation : dealloc_Gemm_gpu_0_pred_1__1_res
  free(Gemm_gpu_0_pred_1__1_res);

  //Run softmax : Softmax_gpu_0_softmax_1
  int8_t* gpu_0_softmax_1 = (int8_t*)mutableWeight + 602112;
  softmax(Gemm_gpu_0_pred_1__1_dequantize_res, gpu_0_softmax_1, 1, 1000, 1, 1000 );

  //Run deallocactivation : dealloc_Gemm_gpu_0_pred_1__1_dequantize_res
  free(Gemm_gpu_0_pred_1__1_dequantize_res);
  return 0;
}