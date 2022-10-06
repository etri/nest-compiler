/**
 * Copyright (c) 2021, Etri.
 *
 * This program or software including the accompanying associated documentation
 * ("Software") is the proprietary software of Etri and/or its licensors.
 * Etri reserves all rights in and to the Software and all intellectual
 * property therein. The Software is distributed on an "AS IS" basis, without
 * warranties or conditions of any kind, unless required by applicable law or a
 * written agreement.
 *
 * @file: lib_nmp.cpp
 * @brief description: Define Wrappers for NMP's layer functions
 * @date: 11 17, 2021
 */

#include "nmp_dim_t.h"

extern "C" {

#include "lib_nmp.h"

// This is a variable used by Glow backends to determine the actual type used
// for size_t, dim_t and int variables when libjit was compiled.
size_t libjit_sizeTVar;
dim_t libjit_dimTVar;
int libjit_intVar;

/************
 * Halt layer
 * **********/

void _nmp_halt_layer() {
  nmp_halt_layer();
}

/************
 * Init layer
 * **********/

void _nmp_init_layer() {
  nmp_init_layer();
}

/************
 * Sync layer
 * **********/

void _nmp_sync_layer(int32_t num_tls, int32_t num_tlts, int32_t mask) {
  nmp_sync_layer(num_tls, num_tlts, mask);
}

/*******************
 * Activation Layer
 *******************/

int _nmp_activation_layer(
    uint16_t out_dtype,     // output data type
    uint32_t in_base,       // input base address (ext. memory)
    uint32_t out_base,      // output base address (ext. memory)
    uint32_t in_w,          // the number of inputs
    int16_t activation,     // activation type
    int16_t in_fxp,         // input fixed-point
    int16_t alpha_fxp,      // alpha fixed-point (valid for activation type
                            // NMP_ACT_PRELU)
    int16_t out_ls,         // output shift left
    int16_t dtypecast,      // typecast
    uint32_t part_thread_w, // the number of inputs to be processed in a tilelet
    uint32_t part_slice_w,  // the number of inputs to be processed at once in a
                            // tilelet
    uint16_t alpha,         // alpha value for activation type NMP_ACT_PRELU
    uint16_t ntlts_per_tle) // the nubmer of tilelets in a tile
{
  return nmp_activation_layer(
      out_dtype, in_base, out_base, in_w, activation, in_fxp, alpha_fxp, out_ls,
      dtypecast, part_thread_w, part_slice_w, alpha, ntlts_per_tle);
}

/*************************
 * Activation Relu6 Layer
 ************************/

int _nmp_activation_layer_relu6(
    uint16_t dfxp_mode,       // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,    // Layer Input base address
    uint32_t ofm_ddr_base,    // Layer Output base address
    uint32_t ifm_size,        // Layer ifm_c * ifm_h * ifm_w
    int16_t act_q_df,         // Quantize Param (act_df)
    int16_t act_q_wf,         // Quantize Param (act_wf)
    int16_t act_q_ls,         // Quantize Param (act_ls)
    int16_t act_q_loadmode,   // Quantized Input load type (act_load_mode)
    uint32_t cnfg_ifm_thread, // Mapper's config ifm size processed per tlt
    uint32_t cnfg_ifm_slice,  // Mapper's config sliding window
    uint16_t cnfg_num_tlt)    // Mapper's config number of tilelets per tle
{
  return nmp_activation_layer_relu6(
      dfxp_mode, ifm_ddr_base, ofm_ddr_base, ifm_size, act_q_df, act_q_wf,
      act_q_ls, act_q_loadmode, cnfg_ifm_thread, cnfg_ifm_slice, cnfg_num_tlt);
}

/****************
 * ELTWise Layer
 ****************/

int _nmp_eltwise_layer(
    uint16_t dfxp_mode,     // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm1_ddr_base, // Layer Input1 base address
    uint32_t ifm2_ddr_base, // Layer Input2 base address
    uint32_t ofm_ddr_base,  // Layer Output base address
    uint32_t ifm_size,      // Layer Input size = IFM_C * IFM_H * IFM_W
    int16_t eltwise_type,   // NMP_ELTWISE_SUM, NMP_ELTWISE_MUL, NMP_ELTWISE_MAX
    int16_t eltwise_q_df,   // Quantize Param (eltwise_df)
    int16_t eltwise_q_wf,   // Quantize Param (eltwise_wf)
    int16_t eltwise_q_ls,   // Quantize Param (eltwise_ls)
    int16_t eltwise_q_ifm1_loadmode, // Quantized Input load type
                                     // (eltwise_load_mode)
    int16_t eltwise_q_ifm2_loadmode, // Quantized Input load type
                                     // (eltwise_load_mode)
    uint32_t cnfg_ifm_thread,     // Mapper's config ifm size processed per tlt
    uint16_t cnfg_ifm_slice,      // Mapper's config sliding window
    uint16_t mapper_cnfg_num_tlt) // Mapper's config number of tilelets per tile
{
  return nmp_eltwise_layer(dfxp_mode, ifm1_ddr_base, ifm2_ddr_base,
                           ofm_ddr_base, ifm_size, eltwise_type, eltwise_q_df,
                           eltwise_q_wf, eltwise_q_ls, eltwise_q_ifm1_loadmode,
                           eltwise_q_ifm2_loadmode, cnfg_ifm_thread,
                           cnfg_ifm_slice, mapper_cnfg_num_tlt);
}

/***********************************************************
 * Convolution Layer istn ofm (ifm stationary ofm priority)
 **********************************************************/

// This IFM station OFM scheme is a speciallized implementation of the
// Convolution layer. It cannot be used for all kind of Convolutions, but
// provide the best performance for layer with the following characteristics:
// - small IFM_W & IFM_H
// - big IFM_C
// - big kernel size (IFM_C * KNL_H * KNL_W)
// - Small to medium number of OFM_C

// In this scheme, the IFM_W and OFM_C are not partitioned. There are only 2
// loops in the Convolution:
//  - IFM_H partitioning loop
//  - IFM_C partitioning loop
//  The mapper make sure that only satisfy Convolution layer is implemented
//  using this scheme.

int _nmp_conv_layer_istn_ofm(
    uint16_t dfxp_mode,     // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,  // Input base address
    uint32_t ofm_ddr_base,  // Output base address
    uint32_t wgts_ddr_base, // Param(weight) base address
    uint32_t bias_ddr_base, // Param(bias) base address
    uint16_t ifm_w,         // Input width
    uint16_t ifm_h,         // Input height
    uint16_t ifm_ch,        // Input channel
    uint16_t ofm_w,         // Output width
    uint16_t ofm_h,         // Output height
    uint16_t conv_kernel_w, // Convolution Param (kernel_width = kernel_ext_w |
                            // kernel_w)
    uint16_t conv_kernel_h, // Convolution Param (kernel_height = kernel_ext_h |
                            // kernel_h)
    uint16_t conv_num,      // Convolution Param (kernel_num)
    uint16_t
        conv_pad_w, // Convolution Param (pad_width = pad_right << 8 | pad_left)
    uint16_t conv_pad_h,     // Convolution Param (pad_heigth = pad_lower << 8 |
                             // pad_upper)
    uint16_t conv_stride,    // Convolution Param (stride)
    uint16_t conv_dilation,  // Convolution Param (dilation)
    uint16_t act_type,       // Activation type
    uint16_t prelu_slope,    // prelu slope param
    int16_t conv_q_df,       // Quantize Param (conv_df)
    int16_t conv_q_wf,       // Quantize Param (conv_wf)
    int16_t conv_q_bf,       // Quantize Param (conv_bf)
    int16_t conv_q_ls,       // Quantize Param (conv_ls)
    int16_t conv_q_loadmode, // Quantized Input load type (conv_load_mode)
    uint16_t mapper_cnfg_num_tlt, // Mapper's config number of tilelets per tile
    uint16_t mapper_cnfg_thread_np_y,  // Mapper's config thread partition along
                                       // the ofm_y direction
    uint16_t mapper_cnfg_thread_np_ch, // Mapper's config thread partition along
                                       // the ofm_ch direction per tle
    uint16_t mapper_cnfg_thread_conv_wgts_mb, // Mapper's config thread the
                                              // weight numbers per memblob
    uint16_t mapper_cnfg_slice_ofm_w,  // Mapper's config sliding window width
    uint16_t mapper_cnfg_slice_ofm_h,  // Mapper's config sliding window height
    uint16_t mapper_cnfg_slice_ifm_ch, // Mapper's config sliding window channel
    uint16_t mapper_cnfg_mb_size)      // Mapper's config memblob size of tlt
{
  return nmp_conv_layer_istn_ofm(
      dfxp_mode, ifm_ddr_base, ofm_ddr_base, wgts_ddr_base, bias_ddr_base,
      ifm_w, ifm_h, ifm_ch, ofm_w, ofm_h, conv_kernel_w, conv_kernel_h,
      conv_num, conv_pad_w, conv_pad_h, conv_stride, conv_dilation, act_type,
      prelu_slope, conv_q_df, conv_q_wf, conv_q_bf, conv_q_ls, conv_q_loadmode,
      mapper_cnfg_num_tlt, mapper_cnfg_thread_np_y, mapper_cnfg_thread_np_ch,
      mapper_cnfg_thread_conv_wgts_mb, mapper_cnfg_slice_ofm_w,
      mapper_cnfg_slice_ofm_h, mapper_cnfg_slice_ifm_ch, mapper_cnfg_mb_size);
}

/**************************************************************
 * Convolution Layer istn wgt (ifm stationary/ weight priority)
 **************************************************************/

//   - Weights per tlt should be enough to be in memblobs
//   - mapper_cnfg_slice_ofm_w should be same with ofm_w
//
// 1. Convolution layer Supports.
//   - any size for kernel width and height
//   - any size for kernel stride
//   - 2^n size for kernel dilation
//
// 2. This layer support the following type of Activation: ReLU, TanH, Sigmoid
//
// 3. Partition scheme
//   : Partition scheme is applied to output featuremap partition
//   : Every partition exception(remainder) can be taken by the functions.
//     you can use representative values which are frequently used in the
//     partition.
//   3.1. TLE partitioning (Using mapper_cnfg_thread_np_y,
//   mapper_cnfg_thread_np_ch)
//     : mapper_cnfg_thread_np_y * mapper_cnfg_thread_np_ch <=
//     mapper_cnfg_num_tle) a. Height partition (mapper_cnfg_thread_np_y)
//        : Each TLE takes a different group of job threads which is seperated
//        by mapper_cnfg_thread_np_y.
//     b. Channel partition (mapper_cnfg_thread_np_ch)
//        : Each TLE takes a different job thread which is seperated by
//        mapper_cnfg_thread_np_ch.
//   3.2. TLT partitioning (Atomatically using mapper_cnfg_num_tlt)
//     a. Channel partition (mapper_cnfg_num_tlt)
//        : Each TLT takes a different part of job thread which is seperated
//        along the output featuremap channels.
//          It will automatically assigned by mapper_cnfg_num_tlt.
//
//   Example)
//     * Convolution config
//       .ofm_h    = 128
//       .conv_num = 64
//     * Partition config
//       .mapper_cnfg_num_tle = 4  (TLE0 ~ TLE3)
//       .mapper_cnfg_num_tlt = 16 (TLT0 ~ TLT15)
//       .mapper_cnfg_thread_np_y  = 2
//       .mapper_cnfg_thread_np_ch = 2
//
//     - OFM partition
//       =========================
//       |                       |
//       |      TLE0 / TLE1      |
//       |                       |
//       ========================= OFM(1 ~ N/2)
//       |                       |
//       |      TLE2 / TLE3      |
//       |                       |
//       ========================= OFM((N/2+1) ~ N)
//
//     - OFM_H partition
//       =========================
//       |         TLE0          |
//       -.-.-.-.-.-.-.-.-.-.-.-.- ofm_h/2
//       |         TLE1          |
//       ------------------------- OFM1
//       |         TLE0          |
//       -.-.-.-.-.-.-.-.-.-.-.-.-
//       |         TLE1          |
//       ------------------------- OFM2
//                  ...
//       ------------------------- OFM (N/2-1)
//       |         TLE0          |
//       -.-.-.-.-.-.-.-.-.-.-.-.-
//       |         TLE1          |
//       ========================= OFM (N/2)
//

int _nmp_conv_layer_istn_wgt(
    uint16_t dfxp_mode,           // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,        // Input base address
    uint32_t ofm_ddr_base,        // Output base address
    uint32_t wgts_ddr_base,       // Param(weight) base address
    uint32_t bias_ddr_base,       // Param(bias) base address
    uint16_t ifm_w,               // Input width
    uint16_t ifm_h,               // Input height
    uint16_t ifm_ch,              // Input channel
    uint32_t ifm_size,            // Input featuremap size (1 ifm)
    uint16_t ofm_w,               // Output width
    uint16_t ofm_h,               // Output height
    uint32_t ofm_size,            // Output featuremap size (1 ofm)
    uint16_t conv_kx,             // Convolution Param (kernel_width)
    uint16_t conv_ky,             // Convolution Param (kernel_height)
    uint16_t conv_num,            // Convolution Param (kernel_num)
    uint16_t conv_weight_size_2d, // Convolution Param (kernel_size_w_x_h)
    uint16_t conv_weight_size_3d, // Convolution Param (kernel_size_w_x_h_x_ich)
    uint16_t conv_pad_w,          // Convolution Param (pad_width)
    uint16_t conv_pad_h,          // Convolution Param (pad_heigth)
    uint16_t conv_stride,         // Convolution Param (stride)
    uint16_t conv_dilation,       // Convolution Param (dilation)
    uint16_t conv_has_bias,       // Convolution Param (has_bias)
    uint16_t act_type,            // Activation type
    uint16_t prelu_slope,         // prelu slope param
    int16_t conv_q_df,            // Quantize Param (conv_df)
    int16_t conv_q_wf,            // Quantize Param (conv_wf)
    int16_t conv_q_bf,            // Quantize Param (conv_bf)
    int16_t conv_q_ls,            // Quantize Param (conv_ls)
    int16_t conv_q_loadmode,      // Quantized Input load type (conv_load_mode)
    uint16_t mapper_cnfg_num_tle, // Mapper's config number of tile
    uint16_t mapper_cnfg_num_tlt, // Mapper's config number of tilelets per tile
    uint16_t mapper_cnfg_thread_np_y,  // Mapper's config thread partition along
                                       // the ofm_y direction
    uint16_t mapper_cnfg_thread_np_ch, // Mapper's config thread partition along
                                       // the ofm_ch direction per tle
    uint16_t mapper_cnfg_thread_conv_wgts_mb, // Mapper's config thread the
                                              // weight numbers per memblob
    uint16_t mapper_cnfg_slice_ofm_w,  // Mapper's config sliding window width
    uint16_t mapper_cnfg_slice_ofm_h,  // Mapper's config sliding window height
    uint16_t mapper_cnfg_slice_ifm_ch, // Mapper's config sliding window channel
    uint16_t mapper_cnfg_mb_size)      // Mapper's config memblob size of tlt
{
  return nmp_conv_layer_istn_wgt(
      dfxp_mode, ifm_ddr_base, ofm_ddr_base, wgts_ddr_base, bias_ddr_base,
      ifm_w, ifm_h, ifm_ch, ifm_size, ofm_w, ofm_h, ofm_size, conv_kx, conv_ky,
      conv_num, conv_weight_size_2d, conv_weight_size_3d, conv_pad_w,
      conv_pad_h, conv_stride, conv_dilation, conv_has_bias, act_type,
      prelu_slope, conv_q_df, conv_q_wf, conv_q_bf, conv_q_ls, conv_q_loadmode,
      mapper_cnfg_num_tle, mapper_cnfg_num_tlt, mapper_cnfg_thread_np_y,
      mapper_cnfg_thread_np_ch, mapper_cnfg_thread_conv_wgts_mb,
      mapper_cnfg_slice_ofm_w, mapper_cnfg_slice_ofm_h,
      mapper_cnfg_slice_ifm_ch, mapper_cnfg_mb_size);
}

/*************************************************************
 * Convolution layer - ostn ifm (ofm stationary/ ifm priority)
 *************************************************************/

// 1. Convolution layer Supports.
//   - any size for kernel width and height
//   - any size for kernel stride
//   - 2^n size for kernel dilation
//
// 2. Partition scheme
//   : Partition scheme is applied to output featuremap partition
//   : Every partition exception(remainder) can be taken by the functions.
//     you can use representative values which are frequently used in the
//     partition.
//   2.1. TLE partitioning (Using mapper_cnfg_thread_np_y,
//   mapper_cnfg_thread_np_ch)
//     : mapper_cnfg_thread_np_y * mapper_cnfg_thread_np_ch <=
//     mapper_cnfg_num_tle) a. Height partition (mapper_cnfg_thread_np_y)
//        : Each TLE takes a different group of job threads which is seperated
//        by mapper_cnfg_thread_np_y.
//     b. Channel partition (mapper_cnfg_thread_np_ch)
//        : Each TLE takes a different job thread which is seperated by
//        mapper_cnfg_thread_np_ch.
//   2.2. TLT partitioning (Atomatically using mapper_cnfg_num_tlt)
//     a. Channel partition (mapper_cnfg_num_tlt)
//        : Each TLT takes a different part of job thread which is seperated
//        along the output featuremap channels.
//          It will automatically assigned by mapper_cnfg_num_tlt.
//
//   Example)
//     * Convolution config
//       .ofm_h    = 128
//       .conv_num = 64
//     * Partition config
//       .mapper_cnfg_num_tle = 4  (TLE0 ~ TLE3)
//       .mapper_cnfg_num_tlt = 16 (TLT0 ~ TLT15)
//       .mapper_cnfg_thread_np_y  = 2
//       .mapper_cnfg_thread_np_ch = 2
//
//     - OFM partition
//       =========================
//       |                       |
//       |      TLE0 / TLE1      |
//       |                       |
//       ========================= OFM(1 ~ N/2)
//       |                       |
//       |      TLE2 / TLE3      |
//       |                       |
//       ========================= OFM((N/2+1) ~ N)
//
//     - OFM_H partition
//       =========================
//       |         TLE0          |
//       -.-.-.-.-.-.-.-.-.-.-.-.- ofm_h/2
//       |         TLE1          |
//       ------------------------- OFM1
//       |         TLE0          |
//       -.-.-.-.-.-.-.-.-.-.-.-.-
//       |         TLE1          |
//       ------------------------- OFM2
//                  ...
//       ------------------------- OFM (N/2-1)
//       |         TLE0          |
//       -.-.-.-.-.-.-.-.-.-.-.-.-
//       |         TLE1          |
//       ========================= OFM (N/2)
//

int _nmp_conv_layer_ostn_ifm(
    uint16_t dfxp_mode,     // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,  // Input base address
    uint32_t ofm_ddr_base,  // Output base address
    uint32_t wgts_ddr_base, // Param(weight) base address
    uint32_t bias_ddr_base, // Param(bias) base address
    uint16_t ifm_w,         // Input width
    uint16_t ifm_h,         // Input height
    uint16_t ifm_ch,        // Input channel
    uint16_t ofm_w,         // Output width
    uint16_t ofm_h,         // Output height
    uint16_t conv_kernel_w, // Convolution Param (kernel_width = kernel_ext_w |
                            // kernel_w)
    uint16_t conv_kernel_h, // Convolution Param (kernel_height = kernel_ext_h |
                            // kernel_h)
    uint16_t conv_num,      // Convolution Param (kernel_num)
    uint16_t
        conv_pad_w, // Convolution Param (pad_width = pad_right << 8 | pad_left)
    uint16_t conv_pad_h,     // Convolution Param (pad_heigth = pad_lower << 8 |
                             // pad_upper)
    uint16_t conv_stride,    // Convolution Param (stride)
    uint16_t conv_dilation,  // Convolution Param (dilation)
    uint16_t act_type,       // Activation type
    uint16_t prelu_slope,    // prelu slope param
    int16_t conv_q_df,       // Quantize Param (conv_df)
    int16_t conv_q_wf,       // Quantize Param (conv_wf)
    int16_t conv_q_bf,       // Quantize Param (conv_bf)
    int16_t conv_q_ls,       // Quantize Param (conv_ls)
    int16_t conv_q_loadmode, // Quantized Input load type (conv_load_mode)
    uint16_t mapper_cnfg_num_tlt, // Mapper's config number of tilelets per tile
    uint16_t mapper_cnfg_thread_np_y,  // Mapper's config thread partition along
                                       // the ofm_y direction
    uint16_t mapper_cnfg_thread_np_ch, // Mapper's config thread partition along
                                       // the ofm_ch direction per tle
    uint16_t mapper_cnfg_thread_conv_wgts_mb, // Mapper's config thread the
                                              // weight numbers per memblob
    uint16_t mapper_cnfg_slice_ofm_w,  // Mapper's config sliding window width
    uint16_t mapper_cnfg_slice_ofm_h,  // Mapper's config sliding window height
    uint16_t mapper_cnfg_slice_ifm_ch, // Mapper's config sliding window channel
    uint16_t mapper_cnfg_mb_size)      // Mapper's config memblob size of tlt
{
  return nmp_conv_layer_ostn_ifm(
      dfxp_mode, ifm_ddr_base, ofm_ddr_base, wgts_ddr_base, bias_ddr_base,
      ifm_w, ifm_h, ifm_ch, ofm_w, ofm_h, conv_kernel_w, conv_kernel_h,
      conv_num, conv_pad_w, conv_pad_h, conv_stride, conv_dilation, act_type,
      prelu_slope, conv_q_df, conv_q_wf, conv_q_bf, conv_q_ls, conv_q_loadmode,
      mapper_cnfg_num_tlt, mapper_cnfg_thread_np_y, mapper_cnfg_thread_np_ch,
      mapper_cnfg_thread_conv_wgts_mb, mapper_cnfg_slice_ofm_w,
      mapper_cnfg_slice_ofm_h, mapper_cnfg_slice_ifm_ch, mapper_cnfg_mb_size);
}

/***************************************************************************
 * Convolution layer wstn ofm (weight stationary/ ofm priority + Activation)
 ***************************************************************************/

// 1. Convolution layer Supports.
//   - any size for kernel width and height
//   - any size for kernel stride
//   - 2^n size for kernel dilation
//
// 2. This function support all type of Activation EXCEPT Leaky ReLU
//
// 3. Partition scheme
//   : Partition scheme is applied to output featuremap partition
//   : Every partition exception(remainder) can be taken by the functions.
//     you can use representative values which are frequently used in the
//     partition.
//   3.1. TLE partitioning (Using mapper_cnfg_thread_np_y,
//   mapper_cnfg_thread_np_ch)
//     : mapper_cnfg_thread_np_y * mapper_cnfg_thread_np_ch <=
//     mapper_cnfg_num_tle) a. Height partition (mapper_cnfg_thread_np_y)
//        : Each TLE takes a different group of job threads which is seperated
//        by mapper_cnfg_thread_np_y.
//     b. Channel partition (mapper_cnfg_thread_np_ch)
//        : Each TLE takes a different job thread which is seperated by
//        mapper_cnfg_thread_np_ch.
//   3.2. TLT partitioning (Atomatically using mapper_cnfg_num_tlt)
//     a. Channel partition (mapper_cnfg_num_tlt)
//        : Each TLT takes a different part of job thread which is seperated
//        along the output featuremap channels.
//          It will automatically assigned by mapper_cnfg_num_tlt.
//
//   Example)
//     * Convolution config
//       .ofm_h    = 128
//       .conv_num = 64
//     * Partition config
//       .mapper_cnfg_num_tle = 4  (TLE0 ~ TLE3)
//       .mapper_cnfg_num_tlt = 16 (TLT0 ~ TLT15)
//       .mapper_cnfg_thread_np_y  = 2
//       .mapper_cnfg_thread_np_ch = 2
//
//     - OFM partition
//       =========================
//       |                       |
//       |      TLE0 / TLE1      |
//       |                       |
//       ========================= OFM(1 ~ N/2)
//       |                       |
//       |      TLE2 / TLE3      |
//       |                       |
//       ========================= OFM((N/2+1) ~ N)
//
//     - OFM_H partition
//       =========================
//       |         TLE0          |
//       -.-.-.-.-.-.-.-.-.-.-.-.- ofm_h/2
//       |         TLE1          |
//       ------------------------- OFM1
//       |         TLE0          |
//       -.-.-.-.-.-.-.-.-.-.-.-.-
//       |         TLE1          |
//       ------------------------- OFM2
//                  ...
//       ------------------------- OFM (N/2-1)
//       |         TLE0          |
//       -.-.-.-.-.-.-.-.-.-.-.-.-
//       |         TLE1          |
//       ========================= OFM (N/2)
//

int _nmp_conv_layer_wstn_ofm(
    uint16_t dfxp_mode,     // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,  // Input base address
    uint32_t ofm_ddr_base,  // Output base address
    uint32_t wgts_ddr_base, // Param(weight) base address
    uint32_t bias_ddr_base, // Param(bias) base address
    uint16_t ifm_w,         // Input width
    uint16_t ifm_h,         // Input height
    uint16_t ifm_ch,        // Input channel
    uint16_t ofm_w,         // Output width
    uint16_t ofm_h,         // Output height
    uint16_t conv_kernel_w, // Convolution Param (kernel_width = kernel_ext_w | kernel_w)
    uint16_t conv_kernel_h, // Convolution Param (kernel_height = kernel_ext_h | kernel_h)
    uint16_t conv_num,      // Convolution Param (kernel_num)
    uint16_t conv_pad_w,    // Convolution Param (pad_width = pad_right << 8 | pad_left)
    uint16_t conv_pad_h,     // Convolution Param (pad_heigth = pad_lower << 8 | pad_upper)
    uint16_t conv_stride,    // Convolution Param (stride)
    uint16_t conv_dilation,  // Convolution Param (dilation)
    uint16_t act_type,       // Activation type (ReLU/pReLU/ReLU6)
    uint16_t prelu_slope,    // prelu slope param
    int16_t conv_q_df,       // Quantize Param (conv_df)
    int16_t conv_q_wf,       // Quantize Param (conv_wf)
    int16_t conv_q_bf,       // Quantize Param (conv_bf)
    int16_t conv_q_ls,       // Quantize Param (conv_ls)
    int16_t conv_q_loadmode, // Quantized Input load type (conv_load_mode)
    uint16_t mapper_cnfg_num_tlt, // Mapper's config number of tilelets per tile
    uint16_t mapper_cnfg_thread_np_y,  // Mapper's config thread partition along the ofm_y direction
    uint16_t mapper_cnfg_thread_np_ch, // Mapper's config thread partition along the ofm_ch direction per tle
    uint16_t mapper_cnfg_thread_conv_wgts_mb, // Mapper's config thread the weight numbers per memblob
    uint16_t mapper_cnfg_slice_ofm_w,  // Mapper's config sliding window width
    uint16_t mapper_cnfg_slice_ofm_h,  // Mapper's config sliding window height
    uint16_t mapper_cnfg_slice_ifm_ch, // Mapper's config sliding window channel
    uint16_t mapper_cnfg_mb_size)      // Mapper's config memblob size of tlt
{
  return nmp_conv_layer_wstn_ofm(
      dfxp_mode, ifm_ddr_base, ofm_ddr_base, wgts_ddr_base, bias_ddr_base,
      ifm_w, ifm_h, ifm_ch, ofm_w, ofm_h, conv_kernel_w, conv_kernel_h,
      conv_num, conv_pad_w, conv_pad_h, conv_stride, conv_dilation, act_type,
      prelu_slope, conv_q_df, conv_q_wf, conv_q_bf, conv_q_ls, conv_q_loadmode,
      mapper_cnfg_num_tlt, mapper_cnfg_thread_np_y, mapper_cnfg_thread_np_ch,
      mapper_cnfg_thread_conv_wgts_mb, mapper_cnfg_slice_ofm_w,
      mapper_cnfg_slice_ofm_h, mapper_cnfg_slice_ifm_ch, mapper_cnfg_mb_size);
}

/**************************************
 * Convolution layer wstn ofm deepwise
 **************************************/

// Convolution layer - weight stationary/ ofm priority for depthwise convolution)
// Difference between depthwise convolution and normal convolution:
// - There is no stack convolution, slice_ifm_ch = 1
// - No IFM reuse between weight: 2D convolution between 1 IFM plane and 1 weight plane

int _nmp_conv_layer_wstn_ofm_dw(
    uint16_t dfxp_mode,     // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,  // Input base address
    uint32_t ofm_ddr_base,  // Output base address
    uint32_t wgts_ddr_base, // Param(weight) base address
    uint32_t bias_ddr_base, // Param(bias) base address
    uint16_t ifm_w,         // Input width
    uint16_t ifm_h,         // Input height
    uint16_t ofm_w,  // Output width
    uint16_t ofm_h,  // Output height
    uint16_t conv_kernel_w, // Convolution Param (kernel_width = kernel_ext_w |
                            // kernel_w)
    uint16_t conv_kernel_h, // Convolution Param (kernel_height = kernel_ext_h |
                            // kernel_h)
    uint16_t conv_num,      // Convolution Param (kernel_num)
    uint16_t conv_pad_w,    // Convolution Param (pad_width  = pad_right << 8 |
                            // pad_left)
    uint16_t conv_pad_h,    // Convolution Param (pad_heigth = pad_lower << 8 |
                            // pad_upper)
    uint16_t conv_stride,   // Convolution Param (stride)
    uint16_t conv_dilation, // Convolution Param (dilation)
    uint16_t act_type,      // Activation type
    uint16_t prelu_slope,   // prelu slope param
    int16_t conv_q_df,      // Quantize Param (conv_df)
    int16_t conv_q_wf,      // Quantize Param (conv_wf)
    int16_t conv_q_bf,      // Quantize Param (conv_bf)
    int16_t conv_q_ls,      // Quantize Param (conv_ls)
    int16_t conv_q_loadmode,      // Quantized Input load type (conv_load_mode)
    uint16_t mapper_cnfg_num_tle, // number of tile
    uint16_t mapper_cnfg_num_tlt, // number of tilelets per tile
    uint16_t
        mapper_cnfg_thread_np_y, // thread partition along the ofm_y direction
    uint16_t mapper_cnfg_slice_ofm_w,    // sliding window width
    uint16_t mapper_cnfg_slice_ofm_h,    // sliding window height
    uint16_t mapper_cnfg_slice_ofm_ch,   // slice ofm channel to load to MB
    uint16_t mapper_cnfg_ofm_ch_per_tlt, // slice ofm channel per tlt
    uint16_t mapper_cnfg_mb_size)        // memblob size of tlt
{
  return nmp_conv_layer_wstn_ofm_dw(
      dfxp_mode, ifm_ddr_base, ofm_ddr_base, wgts_ddr_base, bias_ddr_base,
      ifm_w, ifm_h, ofm_w, ofm_h, conv_kernel_w, conv_kernel_h, conv_num,
      conv_pad_w, conv_pad_h, conv_stride, conv_dilation, act_type, prelu_slope, conv_q_df,
      conv_q_wf, conv_q_bf, conv_q_ls, conv_q_loadmode, mapper_cnfg_num_tle,
      mapper_cnfg_num_tlt, mapper_cnfg_thread_np_y, mapper_cnfg_slice_ofm_w,
      mapper_cnfg_slice_ofm_h, mapper_cnfg_slice_ofm_ch,
      mapper_cnfg_ofm_ch_per_tlt, mapper_cnfg_mb_size);
}

/************
 * Pool Layer
 ************/

// Support Idx pooling only for the Max pooling on 2x2 stride 2
// This can be scheduled by data partition/slicing and channel Pipelining
// - Channel Pipelining - TLE and TLT slice by channel
//   ex)
//
//     * Partitioned by unique TLT id
//                  ...
//       ------------------------- TLE(n-1)
//       |         TLE0          |
//       -------------------------
//       |         TLE1          |
//       -------------------------
//                  ...
//       -------------------------
//       |         TLE(N-1)      |
//       ------------------------- TLE(n)
//                  ...

int _nmp_pool_layer(
    uint16_t dfxp_mode,        // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,     // Layer Input base address
    uint32_t ofm_ddr_base,     // Layer Output(XXX) base address
    uint32_t ofm_ddr_base_idx, // Layer Output(IDX) base address
    uint16_t ifm_w,            // Layer Input width
    uint16_t ifm_h,            // Layer Input height
    uint16_t ifm_ch,           // Layer Input channel
    uint16_t ofm_w,            // Layer Output width
    uint16_t ofm_h,            // Layer Output height
    uint16_t pool_type,        // Pool Param (pool_type)
    uint16_t pool_pad_type,    // Pool Param (padding type)
    uint16_t pool_kx,          // Pool Param (pool_kx)
    uint16_t pool_ky,          // Pool Param (pool_ky)
    uint16_t pool_pad_w, // Pool Param (pad_width = pad_right << 8 | pad_left)
    uint16_t pool_pad_h, // Pool Param (pad_heigth= pad_lower << 8 | pad_upper)
    uint16_t pool_stride_w,       // Pool Param (pool_stride)
    uint16_t pool_stride_h,       // Pool Param (pool_stride)
    int16_t pool_q_df,            // Quantize Param (pool_df)
    int16_t pool_q_wf,            // Quantize Param (pool_wf)
    int16_t pool_q_ls,            // Quantize Param (pool_ls)
    int16_t scale_q,              // Quantize Param (scale_q)
    int16_t pool_q_loadmode,      // Quantized Input load type (pool_load_mode)
    uint16_t mapper_cnfg_num_tlt, // Mapper's config number of tilelets per tile
    uint16_t mapper_cnfg_slice_ifm_c, // Mapper's config ifm_ch per tilelet
    uint16_t mapper_cnfg_slice_ofm_w, // Mapper's config sliding window width
    uint16_t mapper_cnfg_slice_ofm_h) // Mapper's config sliding window height
{
  return nmp_pool_layer(
      dfxp_mode, ifm_ddr_base, ofm_ddr_base, ofm_ddr_base_idx, ifm_w, ifm_h,
      ifm_ch, ofm_w, ofm_h, pool_type, pool_pad_type, pool_kx, pool_ky,
      pool_pad_w, pool_pad_h, pool_stride_w, pool_stride_h, pool_q_df,
      pool_q_wf, pool_q_ls, scale_q, pool_q_loadmode, mapper_cnfg_num_tlt,
      mapper_cnfg_slice_ifm_c, mapper_cnfg_slice_ofm_w, mapper_cnfg_slice_ofm_h);
}

/****************
 * MatMul Layer
 ***************/

int _nmp_matmul_layer(uint16_t dfxp_mode,
                      uint32_t lhs_addr,
                      uint32_t rhs_addr,
                      uint32_t bias_addr,
                      uint32_t out_addr,
                      uint16_t m,
                      uint16_t n,
                      uint16_t k,
                      uint16_t act_type,
                      uint16_t dot_q_df,
                      uint16_t dot_q_wf,
                      uint16_t dot_q_bf,
                      uint16_t dot_q_ls,
                      uint16_t matmul_q_loadmode,
                      uint32_t num_iters,
                      uint32_t num_rows,
                      uint32_t num_rows_remaining,
                      uint16_t num_tle,
                      uint16_t num_tlt)
{
  return nmp_matmul_layer(dfxp_mode, lhs_addr, rhs_addr, bias_addr, out_addr,
                          m, n, k, act_type, dot_q_df, dot_q_wf, dot_q_bf,
                          dot_q_ls, matmul_q_loadmode, num_iters,
                          num_rows, num_rows_remaining, num_tle, num_tlt);
}

/****************
 * SoftMax Layer
 ***************/

// Softmax instruction is not supported by NPU, so the job is computed by
// VSCALE. The exponential and some other instructions on Softmax(Logistic
// regression) are expensive for VSCALE, so it recommands to deploy softmax
// layer on host.
//
// This can be scheduled by data partitioning with # of entire TLTs on NMP.
// And there are two kinds of usecase of softmax layer on CNN world.
// The one is a softmax layer with only encoding layer (Usually Detection or
// Classification network) and the other is softmax layer with encoding and
// decoding layer (Usaually Scene labeling)
//
// - Data Pipelining (DP) - TLE thread and TLT slice
//   ex)
//     * Partitioned by TLE
//       -------------------------
//       |         TLE0          |
//       -------------------------
//       |         TLE1          |
//       -------------------------
//                   ...
//       -------------------------
//       |         TLE(N-1)      |
//       ------------------------- IFM(x,y,ch=fixed by ifm_ch)
//
//     * Partitioned by TLT
//       --------------------------------
//       |   TLT0   |  ...   | TLT(N-1) |
//       -------------------------------- TLEn

int _nmp_softmax_layer(
    uint16_t dfxp_mode,         // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,      // Layer Input base address
    uint32_t ofm_ddr_base,      // Layer Output base address
    uint16_t ifm_dim,           // Layer Input dim
    uint16_t ifm_w,             // Layer Input width
    uint16_t ifm_h,             // Layer Input height
    uint16_t ifm_ch,            // Layer Input channel
    uint16_t ifm_num,           // Layer Input num
    int16_t softmax_q_df,       // Quantize Param (softmax_df)
    int16_t softmax_q_wf,       // Quantize Param (softmax_wf)
    int16_t softmax_q_ls,       // Quantize Param (softmax_ls)
    int16_t softmax_q_loadmode, // Quantized Input load type (softmax_load_mode)
    uint16_t mapper_cnfg_num_tle, // Mapper's config number of tile
    uint16_t mapper_cnfg_num_tlt, // Mapper's config number of tilelet per tile
    uint16_t mapper_cnfg_mb_size) // Mapper's config memblob size of tlt
{
  return nmp_softmax_layer(
      dfxp_mode, ifm_ddr_base, ofm_ddr_base, ifm_dim, ifm_w, ifm_h, ifm_ch,
      ifm_num, softmax_q_df, softmax_q_wf, softmax_q_ls, softmax_q_loadmode,
      mapper_cnfg_num_tle, mapper_cnfg_num_tlt, mapper_cnfg_mb_size);
}

/*************
 * Scale Layer
 *************/
 
// This can be scheduled by data partition/slicing and channel Pipelining
//
// - Data Partitioning (thread) - TLE thread
//   NUM_TLE = mapper_cnfg_dp_nx * mapper_cnfg_dp_ny
//
//   ex1)
//     * mapper_cnfg_dp_nx = 2
//       mapper_cnfg_dp_ny = 1
//       -------------------------
//       |           |           |
//       |           |           |
//       |   TLE0    |    TLE1   |
//       |           |           |
//       |           |           |
//       ------------------------- OFM(1 ~ N)
//   ex2)
//     * mapper_cnfg_dp_nx = 2
//       mapper_cnfg_dp_ny = 2
//       -------------------------
//       |   TLE0    |  TLE1     |
//       |           |           |
//       -------------------------
//       |   TLE2    |  TLE3     |
//       |           |           |
//       ------------------------- OFM(1 ~ N)
//
// - Data Slicing (slice) - sliding window
//   Sliding window size is limitted by TLT's MEMBLOB size
//
//   ex1)
//       mapper_cnfg_slice_ifm_w = A
//       mapper_cnfg_slice_ifm_h = B
//       -------------------------w
//       |   | A                 |
//       |----     TLT0          |
//       | B                     |
//      h-------------------------OFM(1~4)
//       |   | A                 |
//       |----     TLT1          |
//       | B                     |
//       -------------------------OFM(5~8)
//       |   | A                 |
//       |---      TLT2          |
//       | B                     |
//       -------------------------OFM(9~12)
//       |                       |
//                 ....
//       |                       |
//       -------------------------
//       |   | A                 |
//       |----     TLT16         |
//       | B                     |
//       -------------------------OFM(61~64)
//
// - Channel Pipelining (CP) - TLT partition
//    different output featuremaps are assigned to each TLT
//     * mapper_cnfg_num_tlt = 16

int _nmp_scale_layer(
    uint16_t dfxp_mode,            // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,         // Layer Input base address
    uint32_t ofm_ddr_base,         // Layer Output base address
    uint32_t scale_alpha_ddr_base, // Layer Param(alpha) base address
    uint32_t scale_beta_ddr_base,  // Layer Param(beta) base address
    uint16_t ifm_w,                // Layer Input width
    uint16_t ifm_h,                // Layer Input height
    uint16_t ifm_ch,               // Layer Input channel
    int16_t scale_num_axis,        // Scale param (num_axis)
    int16_t scale_axis,            // Scale param (axis)
    int16_t scale_q_df,            // Scale Param (scale_df)
    int16_t scale_q_wf,            // Scale Param (scale_wf)
    int16_t scale_q_bf,            // Quantize Param (scale_bf)
    int16_t scale_q_ls,            // Scale Param (scale_ls)
    int16_t scale_q_loadmode,   // Quantized Input load type (scale_load_mode)
    uint16_t mapper_cnfg_dp_nx, // Mapper's config data partition axis-x
    uint16_t mapper_cnfg_dp_ny, // Mapper's config data partition axis-y
    uint16_t mapper_cnfg_slice_ifm_w, // Mapper's config sliding window width
    uint16_t mapper_cnfg_slice_ifm_h, // Mapper's config sliding window height
    uint16_t mapper_cnfg_num_tlt // Mapper's config number of tilelets per tile
) {
  return nmp_scale_layer(
      dfxp_mode, ifm_ddr_base, ofm_ddr_base, scale_alpha_ddr_base,
      scale_beta_ddr_base, ifm_w, ifm_h, ifm_ch, scale_num_axis, scale_axis,
      scale_q_df, scale_q_wf, scale_q_bf, scale_q_ls, scale_q_loadmode,
      mapper_cnfg_dp_nx, mapper_cnfg_dp_ny, mapper_cnfg_slice_ifm_w,
      mapper_cnfg_slice_ifm_h, mapper_cnfg_num_tlt);
}


/*************
 * Core Layers
 *************/

void _nmp_sync_gsema(uint32_t mst_tle_id, // The Master Tile ID
                    uint32_t tle_id,     // Tile ID
                    uint32_t gsema_loc,  // To generate flip-flop pattern
                    uint32_t num_tle)    // Number of Tiles
{
  nmp_sync_gsema(mst_tle_id, tle_id, gsema_loc, num_tle);
}

void _nmp_sync_tsema(uint32_t mst_tlt_id, // The Master Tilelet ID
                    uint32_t tlt_id,     // Tilelet ID
                    uint32_t num_tlt)    // Number of Tilelets per Tile
{
  nmp_sync_tsema(mst_tlt_id, tlt_id, num_tlt);
}

} // extern "C"
