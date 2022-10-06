#ifndef __NMP_RTLIB_H__
#define __NMP_RTLIB_H__

/*
 * Copyright (c) 2017-2019, LG Electronics Inc.
 *
 * This program or software including the accompanying associated documentation
 * ("Software") is the proprietary software of LG Electronics Inc. and or its
 * licensors, and may only be used, duplicated, modified or distributed pursuant
 * to the terms and conditions of a separate written license agreement between
 * you and LG Electronics Inc. ("Authorized License"). Except as set forth in an
 * Authorized License, LG Electronics Inc. grants no license (express or
 * implied), rights to use, or waiver of any kind with respect to the Software,
 * and LG Electronics Inc. expressly reserves all rights in and to the Software
 * and all intellectual property therein. If you have no Authorized License,
 * then you have no rights to use the Software in any ways, and should
 * immediately notify LG Electronics Inc. and discontinue all use of the
 * Software.
 *
 * Author:
 *  Jonghwan Lee <jonghwan1.lee@lge.com>
 *  SeungRyeol Lee <zizztux.lee@lge.com>
 */

/* vi: set ts=4 sw=2 sts=2 expandtab: */

#include <cstdint>

extern "C" {

void nmp_halt_layer();

void nmp_init_layer();

void nmp_sync_layer(int32_t num_tls, int32_t num_tlts, int32_t mask);

uint16_t
nmp_load_subblob(uint16_t self,       // my tilelet mask
                 uint16_t tilelets,   // destination tilelets
                 uint16_t inner_mem,  // destination inner memory
                 uint16_t inner_addr, // destination inner memory address
                 uint32_t outer_addr, // source outer memory address
                 uint16_t blob_w,     // blob width (excluding pad)
                 uint16_t blob_h,     // blob height (excluding pad)
                 uint16_t subblob_w,  // sub-blob width to load (including pad)
                 uint16_t subblob_h,  // sub-blob height to load (including pad)
                 uint16_t subblob_d,  // sub-blob depth to load
                 uint8_t pad_north,   // number of padding on the north
                 uint8_t pad_south,   // number of padding on the south
                 uint8_t pad_west,    // number of padding on the west
                 uint8_t pad_east,    // number of padding on the east
                 uint16_t dtype_inner, // data type in inner memory
                 uint16_t dtype_outer, // data type in outer memory
                 uint16_t dtypecast,   // data type conversion
                 uint16_t signal_lsb); // LSB mask for signaling

uint16_t nmp_load_subblob_ow(
    uint16_t self,        // my tilelet mask
    uint16_t tilelets,    // destination tilelets
    uint16_t inner_mem,   // destination inner memory
    uint16_t inner_addr,  // destination inner memory address
    uint32_t outer_addr,  // source outer memory address
    uint16_t blob_w,      // blob width
    uint16_t blob_h,      // blob height
    uint16_t subblob_w,   // sub-blob width to load (including pad)
    uint16_t subblob_h,   // sub-blob height to load (including pad)
    uint16_t subblob_d,   // sub-blob depth to load
    uint8_t pad_north,    // number of padding on the north
    uint8_t pad_south,    // number of padding on the south
    uint8_t pad_west,     // number of padding on the west
    uint8_t pad_east,     // number of padding on the east
    uint16_t dtype_inner, // data type in inner memory
    uint16_t dtype_outer, // data type in outer memory
    uint16_t dtypecast,   // data type conversion
    uint16_t signal_lsb); // LSB mask for signaling

uint16_t nmp_load_subblob_pad(
    uint16_t self,        // my tilelet mask
    uint16_t tilelets,    // destination tilelets
    uint16_t inner_mem,   // destination inner memory
    uint16_t inner_addr,  // destination inner memory address
    uint32_t outer_addr,  // source outer memory address
    uint16_t blob_w,      // blob width
    uint16_t blob_h,      // blob height
    uint16_t subblob_w,   // sub-blob width to load (including pad)
    uint16_t subblob_h,   // sub-blob height to load (including pad)
    uint16_t subblob_d,   // sub-blob depth to load
    uint8_t pad_north,    // number of padding on the north
    uint8_t pad_south,    // number of padding on the south
    uint8_t pad_west,     // number of padding on the west
    uint8_t pad_east,     // number of padding on the east
    uint16_t dtype_inner, // data type in inner memory
    uint16_t dtype_outer, // data type in outer memory
    uint16_t dtypecast,   // data type conversion
    uint16_t signal_lsb); // LSB mask for signaling

uint16_t nmp_load_subblob_avg_pad(
    uint16_t self,       // my tilelet mask
    uint16_t tilelets,   // destination tilelets
    uint16_t inner_mem,  // destination inner memory
    uint16_t inner_addr, // destination inner memory address
    uint32_t outer_addr, // source outer memory address
    uint16_t blob_w,     // blob width
    uint16_t blob_h,     // blob height
    uint16_t subblob_w,  // sub-blob width to load (including pad)
    uint16_t subblob_h,  // sub-blob height to load (including pad)
    uint16_t subblob_d,  // sub-blob depth to load
    uint8_t pad_north,   // number of padding on the north
    uint8_t pad_south,   // number of padding on the south
    uint8_t pad_west,    // number of padding on the west
    uint8_t pad_east,    // number of padding on the east
    uint16_t ksize,
    uint16_t dtype_inner, // data type in inner memory
    uint16_t dtype_outer, // data type in outer memory
    uint16_t dtypecast,   // data type conversion
    uint16_t signal_lsb); // LSB mask for signaling

uint16_t nmp_load_featuremap(
    uint32_t src_ddr_base,  // Source DDR address
    uint16_t dst_mb_base,   // Target MBLOB address
    uint16_t dst_mb,        // Target MBLOB
    uint16_t num_ch,        // Layer Input Slice_ch
    uint16_t fm_w,          // Layer Input width
    uint16_t fm_h,          // Layer Input height
    uint16_t slice_w,       // Layer Input Slice width
    uint16_t slice_h,       // Layer Input Slice height
    uint16_t pad_dx,        // Padding dx ([15:8] pad_bottom, [7:0] pad_top)
    uint16_t pad_dy,        // Padding dy ([15:8] pad_right, [7:0] pad_left)
    uint16_t sema_lsb,      // LSB of 5-bits semaphore
    uint16_t dfxp_mode_src, // Data Precision (NMP_CFG_DFXP_SIZE_DXX) in DDR
    uint16_t dfxp_mode_dst, // Data Precision (NMP_CFG_DFXP_SIZE_DXX) in MEMBLOB
    uint16_t dfxp_load_type, // Data Load Type (0 : 3)
    uint16_t multicast // Multicast mode (NMP_MULTICAST or ONE_HOT(TLT_ID))
);

uint16_t nmp_load_featuremap_pad(
    uint32_t src_ddr_base,  // Source DDR address
    uint16_t dst_mb_base,   // Target MBLOB address
    uint16_t dst_mb,        // Target MBLOB
    uint16_t num_ch,        // Layer Input Slice_ch
    uint16_t fm_w,          // Layer Input width
    uint16_t fm_h,          // Layer Input height
    uint16_t slice_w,       // Layer Input Slice width
    uint16_t slice_h,       // Layer Input Slice height
    uint8_t pad_dx_upper,   // Padding upper
    uint8_t pad_dx_lower,   // Padding lower
    uint8_t pad_dy_left,    // Padding left
    uint8_t pad_dy_right,   // Padding right
    uint16_t sema_lsb,      // LSB of 5-bits semaphore
    uint16_t dfxp_mode_src, // Data Precision (NMP_CFG_DFXP_SIZE_DXX) in DDR
    uint16_t dfxp_mode_dst, // Data Precision (NMP_CFG_DFXP_SIZE_DXX) in MEMBLOB
    uint16_t dfxp_load_type, // Data Load Type (0 : 3)
    uint16_t multicast // Multicast mode (NMP_MULTICAST or ONE_HOT(TLT_ID))
);

uint16_t nmp_load_featuremap_non_zero_pad(
    uint32_t src_ddr_base,  // Source DDR address
    uint16_t dst_mb_base,   // Target MBLOB address
    uint16_t dst_mb,        // Target MBLOB -> NMP_TLT_MBLOB0
    uint16_t num_ch,        // Layer Input Slice_ch
    uint16_t fm_w,          // Layer Input width
    uint16_t fm_h,          // Layer Input height
    uint16_t slice_w,       // Layer Input Slice width
    uint16_t slice_h,       // Layer Input Slice height
    uint8_t pad_dx_upper,   // Padding dx ([15:8] pad_bottom, [7:0] pad_top)
    uint8_t pad_dx_lower,   // Padding dx ([15:8] pad_bottom, [7:0] pad_top)
    uint8_t pad_dy_left,    // Padding dy ([15:8] pad_right, [7:0] pad_left)
    uint8_t pad_dy_right,   // Padding dy ([15:8] pad_right, [7:0] pad_left)
    uint16_t sema_lsb,      // LSB of 1-bits semaphore
    uint16_t dfxp_mode_src, // Data Precision (NMP_CFG_DFXP_SIZE_DXX) in DDR
    uint16_t dfxp_mode_dst, // Data Precision (NMP_CFG_DFXP_SIZE_DXX) in MEMBLOB
    uint16_t dfxp_load_type, // Data Load Type (0 : 3)
    uint16_t multicast // Multicast mode (NMP_MULTICAST or ONE_HOT(TLT_ID))
);

uint16_t nmp_load_param_bn(
    uint32_t bn_alpha_ddr_base, // Param(alpha) DDR address
    uint32_t bn_beta_ddr_base,  // Param(beta) DDR address
    uint16_t bn_param_mb_base,  // Param(weight) MBLOB address
    uint16_t bn_param_mb,       // Param(weight) MBLOB
    uint16_t num_ch,            // Layer Param Slice_ch
    uint16_t sema_lsb,          // LSB of 2-bits semaphore
    uint16_t dfxp_mode,         // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint16_t multicast // Multicast mode (NMP_MULTICAST or ONE_HOT(TLT_ID))
);

uint16_t nmp_load_param_conv(
    uint32_t conv_weight_ddr_base, // Param(weight) DDR address
    uint32_t conv_bias_ddr_base,   // Param(bias) DDR address
    uint16_t conv_weight_mb_base,  // Param(weight) MBLOB address
    uint16_t conv_weight_mb,       // Param(weight) MBLOB
    uint16_t conv_bias_mb_base,    // Param(bias) MBLOB address
    uint16_t conv_bias_mb,         // Param(bias) MBLOB
    uint16_t conv_num,             // Layer Param Slice_ch
    uint16_t conv_kx,              // Layer Param(weight width)
    uint16_t conv_ky,              // Layer Param(weight height)
    uint16_t conv_ch,              // Lyaer Param(weight channel)
    uint16_t sema_lsb,             // LSB of 2-bits semaphore
    uint16_t dfxp_mode,            // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint16_t multicast, // Multicast mode (NMP_MULTICAST or ONE_HOT(TLT_ID))
    uint16_t has_bias   // Whether adding bias or not
);

int nmp_absval_layer(
    uint16_t dfxp_mode,       // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,    // Layer Input base address
    uint32_t ofm_ddr_base,    // Layer Output base address
    uint32_t ifm_size,        // Layer ifm_c * ifm_h * ifm_w
    uint32_t cnfg_ifm_thread, // Mapper's config ifm size processed per tlt
    uint32_t cnfg_ifm_slice,  // Mapper's config sliding window
    uint16_t cnfg_num_tlt     // Mapper's config number of tilelets per tle
);

int nmp_activation_layer(
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
    uint16_t alpha,          // alpha value for activation type NMP_ACT_PRELU
    uint16_t ntlts_per_tle); // the nubmer of tilelets in a tile

int nmp_activation_layer_leaky(
    uint16_t dfxp_mode,       // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,    // Layer Input base address
    uint32_t ofm_ddr_base,    // Layer Output base address
    uint32_t ifm_size,        // Layer ifm_c * ifm_h * ifm_w
    int16_t act_type,         // Activation Mode (NMP_ACT_XXX)
    int16_t act_q_df,         // Quantize Param (act_df)
    int16_t act_q_wf,         // Quantize Param (act_wf)
    int16_t act_q_ls,         // Quantize Param (act_ls)
    int16_t act_q_loadmode,   // Quantized Input load type (act_load_mode)
    uint32_t cnfg_ifm_thread, // Mapper's config ifm size processed per tlt
    uint32_t cnfg_ifm_slice,  // Mapper's config sliding window
    uint16_t neg_slope,       // negative slope
    uint16_t cnfg_num_tlt     // Mapper's config number of tilelets per tle
);

int nmp_activation_layer_relu6(
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
    uint16_t cnfg_num_tlt);   // Mapper's config number of tilelets per tle

int nmp_activation_layer_prelu(
    uint16_t dfxp_mode,         // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,      // Layer Input base address
    uint32_t ofm_ddr_base,      // Layer Output base address
    uint32_t prelu_weight_base, // Prelu weight base address
    uint16_t ifm_ch,            // Layer ifm_c
    uint32_t ifm_area,          // Layer ifm_h * ifm_w
    uint16_t is_log_slope,      // Layer slope type
    int16_t act_q_df,           // Quantize Param (act_df)
    int16_t act_q_wf,           // Quantize Param (act_wf)
    int16_t act_q_ls,           // Quantize Param (act_ls)
    int16_t act_q_loadmode,     // Quantized Input load type (act_load_mode)
    uint16_t cnfg_ifm_nx,       // TLE partition: number of ifm area division
    uint16_t cnfg_ifm_nch,      // TLE partition: number of ifm channel division
    uint32_t cnfg_ifm_thread,   // Mapper's config ifm size processed per tlt
    uint32_t cnfg_ifm_slice,    // Mapper's config sliding window
    uint16_t cnfg_num_tlt);     // Mapper's config number of tilelets per tle

int nmp_argmax_layer(
    uint16_t dfxp_mode,    // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base, // Layer Input base address
    uint32_t ofm_ddr_base, // Layer Output base address
    uint16_t ifm_dim,      // Layer Input dimension (to decide threading scheme)
    uint16_t ifm_w,        // Layer Input width
    uint16_t ifm_h,        // Layer Input height
    uint16_t ifm_ch,       // Layer Input channel
    uint16_t ifm_num,      // Layer Input num
    int16_t argmax_q_df,   // Quantize Param (argmax_df)
    int16_t argmax_q_wf,   // Quantize Param (argmax_wf)
    int16_t argmax_q_ls,   // Quantize Param (argmax_ls)
    int16_t argmax_q_loadmode, // Quantized Input load type (argmax_load_mode)
    uint16_t mapper_cnfg_num_tle, // Mapper's config a number of tile
    uint16_t
        mapper_cnfg_num_tlt,     // Mapper's config a number of tilelet per tile
    uint16_t mapper_cnfg_mb_size // Mapper's config memblob size of tlt
);

int nmp_bn_layer(
    uint16_t dfxp_mode,         // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,      // Layer Input base address
    uint32_t ofm_ddr_base,      // Layer Output base address
    uint32_t bn_alpha_ddr_base, // Layer Param(alpha) base address
    uint32_t bn_beta_ddr_base,  // Layer Param(beta) base address
    uint16_t ifm_area,          // Layer Input size = ifm_h * ifm_w
    uint16_t ifm_ch,            // Layer Input channel
    int16_t bn_q_df,            // Quantize Param (bn_df)
    int16_t bn_q_wf,            // Quantize Param (bn_wf)
    int16_t bn_q_bf,            // Quantize Param (bn_bf)
    int16_t bn_q_ls,            // Quantize Param (bn_ls)
    int16_t bn_q_loadmode,      // Quantize Param (bn_q_loadmode)
    uint16_t cnfg_ifm_nx,       // TLE partition: number of ifm area division
    uint16_t cnfg_ifm_nch,      // TLE partition: number of ifm channel division
    uint16_t cnfg_ifm_thread,   // IFM data size process by a tile
    uint16_t cnfg_ifm_slice,    // IFM data size process per load
    uint16_t num_tlts);         // number of tilelets per tile

int nmp_deconv_layer(
    const uint16_t dfxp_mode,            // Precision (NMP_CFG_DFXP_SIZE_DXX)
    const uint32_t ifm_ddr_base,         // Layer Input base address
    const uint32_t ofm_ddr_base,         // Layer Output base address
    const uint32_t conv_weight_ddr_base, // Layer Param(weight) base address
    const uint32_t conv_bias_ddr_base,   // Layer Param(bias) base address
    const uint16_t ifm_w,                // Layer Input width
    const uint16_t ifm_h,                // Layer Input height
    const uint16_t ifm_ch,               // Layer Input channel
    const uint16_t ofm_w,                // Layer Output width
    const uint16_t ofm_h,                // Layer Output height
    const uint16_t conv_ksize,           // Convolution Param (kernel_size)
    const uint16_t conv_num,             // Convolution Param (kernel_num)
    const uint16_t conv_ch,              // Convolution Param (kernel_ch)
    const uint16_t conv_group,           // Convolution Param (group for ch)
    const uint16_t conv_pad,             // Convolution Param (pad_size)
    const uint16_t conv_stride,          // Convolution Param (stride)
    const uint16_t conv_dilation,        // Convolution Param (dilation)
    const uint16_t conv_has_bias,        // Convolution Param (has_bias)
    const int16_t conv_q_df,             // Quantize Param (conv_df)
    const int16_t conv_q_wf,             // Quantize Param (conv_wf)
    const int16_t conv_q_ls,             // Quantize Param (conv_ls)
    const int16_t conv_q_loadmode, // Quantized Input load type (conv_load_mode)
    const uint16_t mapper_cnfg_num_tle, // Mapper's config number of tile
    const uint16_t
        mapper_cnfg_num_tlt, // Mapper's config number of tilelets per tile
    const uint16_t mapper_cnfg_mb_size // Mapper's config memblob size of tlt
);

int nmp_conv_layer_istn_ofm(
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
    uint16_t prelu_slope,    // pReLU slope param
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
    uint16_t mapper_cnfg_mb_size);     // Mapper's config memblob size of tlt

int nmp_conv_layer_istn_wgt(
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
    uint16_t mapper_cnfg_mb_size);     // Mapper's config memblob size of tlt

int nmp_conv_layer_ostn_ifm(
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
    uint16_t prelu_slope,    // pReLU slope param
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
    uint16_t mapper_cnfg_mb_size);     // Mapper's config memblob size of tlt

int nmp_conv_layer_wstn_ofm(
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
    uint16_t act_type,       // Activation type (ReLU/pReLU/ReLU6)
    uint16_t prelu_slope,    // pReLU slope param
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
    uint16_t mapper_cnfg_mb_size);     // Mapper's config memblob size of tlt

int nmp_conv_layer_wstn_ofm_dw(
    uint16_t dfxp_mode,     // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,  // Input base address
    uint32_t ofm_ddr_base,  // Output base address
    uint32_t wgts_ddr_base, // Param(weight) base address
    uint32_t bias_ddr_base, // Param(bias) base address
    uint16_t ifm_w,         // Input width
    uint16_t ifm_h,         // Input height
    uint16_t ofm_w,         // Output width
    uint16_t ofm_h,         // Output height
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
    uint16_t prelu_slope,   // pReLU slope param
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
    uint16_t mapper_cnfg_mb_size);       // memblob size of tlt

int nmp_eltwise_layer(
    uint16_t dfxp_mode,              // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm1_ddr_base,          // Layer Input1 base address
    uint32_t ifm2_ddr_base,          // Layer Input2 base address
    uint32_t ofm_ddr_base,           // Layer Output base address
    uint32_t ifm_size,               // Layer Input size = IFM_C * IFM_H * IFM_W
    int16_t eltwise_type,            // NMP_ELTWISE_SUM:0x2, NMP_ELTWISE_MUL:0x3
    int16_t eltwise_q_df,            // Quantize Param (eltwise_df)
    int16_t eltwise_q_wf,            // Quantize Param (eltwise_wf)
    int16_t eltwise_q_ls,            // Quantize Param (eltwise_ls)
    int16_t eltwise_q_ifm1_loadmode, // Quantized Input load type
                                     // (eltwise_load_mode)
    int16_t eltwise_q_ifm2_loadmode, // Quantized Input load type
                                     // (eltwise_load_mode)
    uint32_t cnfg_ifm_thread, // Mapper's config ifm size processed per tlt
    uint16_t cnfg_ifm_slice,  // Mapper's config sliding window
    uint16_t
        mapper_cnfg_num_tlt); // Mapper's config number of tilelets per tile

int nmp_innerp_layer(
    uint16_t dfxp_mode,             // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,          // Layer Input base address
    uint32_t ofm_ddr_base,          // Layer Output base address
    uint32_t inner_weight_ddr_base, // Layer Param(weight) base address
    uint32_t inner_bias_ddr_base,   // Layer Param(bias) base address
    uint32_t ifm_size,              // Layer Input width * height * channel
    uint16_t ofm_ch,                // Layer Output channel
    uint16_t ofm_num,               // Layer Output num
    uint16_t activation_type,       // Activation type
    uint16_t neg_slope,             // Negative slope of LReLU & PReLU
    int16_t inner_ifm_q,            // IFM Q for inner-product
    int16_t inner_wgt_q,            // Weight Q for inner-product
    int16_t inner_ls,               // Inter-layer shift for inner-product
    int16_t inner_q_loadmode, // Quantized Input load type (inner_load_mode)
    int16_t inner_bias_q,     // Bias Q
    int16_t act_slope_q,      // Quantize Param lrelu param dataQ
    int16_t act_ls,           // Quantize Param (lrelu layer level shift)
    uint16_t tle_ofm_ch,   // The number of weight vectors processed in a tile
    uint16_t slice_ofm_ch, // The number of weight vectors processed by a single
                           // percept operation
    uint16_t slice_inner,  // The number of scala values in a weight vector
                           // processed by a single percept operation
    uint16_t cnfg_num_tlt);

int nmp_input_transform_layer(
    uint16_t dfxp_mode,      // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,   // Layer Input base address
    uint32_t ofm_ddr_base,   // Layer Output base address
    uint32_t mean_ddr_base,  // Layer Mean value base address
    uint32_t ifm_area,       // Layer Input width x height
    uint16_t ifm_ch,         // Layer Input channel
    uint16_t operation_type, // SCALE / MIN_CH / MIN_PX
    int32_t scale_value,     // if dfxp = 16 bit: MSB[31:16] is alpha value LSB
                         // [15:0] is beta value if dfxp = 8 bit: MSB[15:8] is
                         // alpha value LSB [7:0] is beta value
    int16_t input_q_df,       // Quantize Param (scale_df)
    int16_t input_q_wf,       // Quantize Param (scale_wf)
    int16_t input_q_ls,       // Quantize Param (scale_ls)
    int16_t input_q_loadmode, // Quantized Input load type (scale_load_mode)
    uint32_t cnfg_ifm_thread, // 2D IFM data partition per tlt
    uint16_t cnfg_ifm_slice,  // 2D IFM data slicing per load
    uint16_t cnfg_num_tlt     // Number of tilelets per tile
);

int nmp_percept_layer(
    uint16_t dfxp_mode,             // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,          // Layer Input base address
    uint32_t ofm_ddr_base,          // Layer Output base address
    uint32_t inner_weight_ddr_base, // Layer Param(weight) base address
    uint32_t inner_bias_ddr_base,   // Layer Param(bias)  base address
    uint32_t ifm_size,              // Layer Input width * height * channel
    uint16_t ofm_ch,                // Layer Output channel
    uint16_t ofm_num,               // Layer Output num
    uint16_t inner_has_bias,        // InnerProduct Param (inner_has_bias)
    uint16_t activation_type,       // Activation type
    uint16_t neg_slope,             // Negative slope for LReLU & PReLU
    int16_t inner_q_df,             // Quantize Param input dataQ
    int16_t inner_q_wf,             // Quantize Param inner weight dataQ
    int16_t inner_q_ls,       // Quantize Param (innerproduct layer level shift)
    int16_t inner_q_loadmode, // Quantized Input load type (inner_load_mode)
    int16_t lrelu_q_wf,       // Quantize Param lrelu param dataQ
    int16_t lrelu_q_ls,       // Quantize Param (lrelu layer level shift)
    uint16_t tle_ofm_ch,      // Number of ofm channel processed by 1 TLE
    uint16_t slice_ofm_ch, // Number of ofm channel processed by 1 tlt at a time
    uint16_t slice_inner,  // The kernel size per ofm to load at a time
    uint16_t cnfg_num_tlt);

int nmp_pool_layer(
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
    uint16_t mapper_cnfg_slice_ofm_h  // Mapper's config sliding window height
);

int nmp_power_layer(
    uint16_t dfxp_mode,          // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,       // Layer Input base address
    uint32_t ofm_ddr_base,       // Layer Output base address
    uint32_t pow_scale_ddr_base, // Layer Param(scale) base address
    uint32_t pow_shift_ddr_base, // Layer Param(shift) base address
    uint16_t ifm_w,              // Layer Input width
    uint16_t ifm_h,              // Layer Input height
    uint16_t ifm_ch,             // Layer Input channel
    int16_t pow_q_df,            // Quantize Param (bn_df)
    int16_t pow_q_wf,            // Quantize Param (bn_wf)
    int16_t pow_q_ls,            // Quantize Param (bn_ls)
    int16_t pow_q_loadmode,      // Quantized Input load type (pow_load_mode)
    int16_t param_power,         // Param_power
    uint16_t mapper_cnfg_dp_nx,  // Mapper's config data partition axis-x
    uint16_t mapper_cnfg_dp_ny,  // Mapper's config data partition axis-y
    uint16_t mapper_cnfg_slice_ifm_w, // Mapper's config sliding window width
    uint16_t mapper_cnfg_slice_ifm_h, // Mapper's config sliding window height
    uint16_t mapper_cnfg_num_tlt // Mapper's config number of tilelets per tile
);

int nmp_proposal_layer(
    const uint16_t dfxp_mode, //! @param Precision ( NMP_CFG_DFXP_SIZE_DXX )
    const uint32_t
        ifm_rpn_score_ddr_base,           //! @param Layer Input(1) base address
    const uint32_t ifm_rpn_bbox_ddr_base, //! @param Layer Input(2) base address
    const uint32_t mapper_prop_coord_ddr_base, //! @param proposals'
                                               //! coordinates(x1,y1,x2,y2);
                                               //! pre-computed by Mapper
    const uint32_t
        mapper_prop_info_ddr_base, //! @param proposals' info(w,h,ctr_x,ctry);
                                   //! pre-computed by Mapper;
    const uint32_t mapper_splitters_ddr_base, //! @param pre-selected splitter
                                              //! values by mapper
    const uint32_t adj_prop_coord_ddr_base,   //! @param adjusted postions after
                                              //! enum_proposals()
    const uint32_t adj_prop_score_ddr_base, //! @param adjusted scores according
                                            //! to the transformed box

    const uint32_t bucket_idx_ddr_base, //! @param assigned bucket index for
                                        //! each scores before sort_box()
    const uint32_t sorted_n_ddr_base,
    const uint32_t sorted_score_ddr_base, //! @param for the storing the
                                          //! sorted_score after sort_box()
    const uint32_t sorted_index_ddr_base, //! @param for the storing the
                                          //! sorted_index after sort_box();
    const uint32_t sorted_coord_ddr_base, //! @param for the retrieved
                                          //! porposals' coordinates;
    const uint32_t ofm_rois_ddr_base, //! @param Layer Output(1) base address
    const uint32_t num_rois_ddr_base, //! @param Layer Output(2) base address

    const int num_anchors,   //! @param num_anchors = anchor_sizes*anchor_ratios
    const int num_proposals, //! @param num_proposals =
                             //! rpn_ifm_width*rpn_ifm_height*num_anchors
    const int bottom_H,      //! @param IFM's height; IFM(1) = IFM(2)
    const int bottom_W,      //! @param IFM's width; IFM(1) = IFM(2)
    const int bottom_area,   //! @param bottom_area = bottom_H * bottom_W

    const uint16_t img_H,     //! @param original image's info; height
    const uint16_t img_W,     //! @param original image's info; width
    const uint16_t min_box_H, //! @param minimum size for a proposal; height
    const uint16_t min_box_W, //! @param minimum size for a proposal; height
    const uint16_t nms_thresh,

    const int pre_nms_topn,  //! @param number for keeping proposals before NMS
    const int post_nms_topn, //! @param number for keeping proposals after NMS

    const int16_t q_rpn_score,  //! @param Quantize Param
    const int16_t q_rpn_bbox,   //! @param Quantize Param
    const int16_t q_prop_coord, //! @param Quantize Param
    const int16_t q_prop_info,  //! @param Quantize Param
    const int16_t q_prop_param, //! @param Quantize Param
    const int16_t q_rois,       //! @param Quantize Param
    const uint16_t
        mapper_cnfg_num_tle, //! @param Mapper's config number of tile
    const uint16_t mapper_cnfg_num_tlt, //! @param Mapper's config number of
                                        //! tilets per tile
    const uint16_t
        mapper_cnfg_mb_size //! @param Mapper's config memblob size of tlt
);

int nmp_bounded_relu_layer(
    uint16_t dfxp_mode,         // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,      // Layer Input base address
    uint32_t ofm_ddr_base,      // Layer Output base address
    uint16_t ifm_w,             // Layer Input width
    uint16_t ifm_h,             // Layer Input height
    uint16_t ifm_ch,            // Layer Input channel
    int16_t act_bound,          // Activation Bound threshold Param
    int16_t act_q_df,           // Quantize Param (act_df)
    int16_t act_q_wf,           // Quantize Param (act_wf)
    int16_t act_q_ls,           // Quantize Param (act_ls)
    int16_t act_q_loadmode,     // Quantized Input load type (act_load_mode)
    uint16_t mapper_cnfg_dp_nx, // Mapper's config data partition axis-x
    uint16_t mapper_cnfg_dp_ny, // Mapper's config data partition axis-y
    uint16_t mapper_cnfg_slice_ifm_w, // Mapper's config sliding window width
    uint16_t mapper_cnfg_slice_ifm_h, // Mapper's config sliding window height
    uint16_t mapper_cnfg_num_tlt // Mapper's config number of tilelets per tile
);

int nmp_scale_layer(
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
);

int nmp_slice_layer(
    uint16_t dfxp_mode,      // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,   // Layer Input base address
    uint32_t ofm_ddr_base[], // Layer Output base address
    uint32_t ofm_num,        // Layer Output base address
    uint16_t ifm_shape[],    // Layer Input shape (num, channel, height, width)
    uint16_t ifm_size,       // Layer Input size (width * height)
    uint16_t slice_axis,     // Slice Param(axis)
    uint16_t slice_points,   // Slice Param(point)
    uint16_t slice_size,     // Slice Param(point)
    uint16_t mapper_cnfg_num_tle, // Mapper's config number of tile
    uint16_t mapper_cnfg_num_tlt, // Mapper's config number of tilelets per tile
    uint16_t mapper_cnfg_mb_size  // Mapper's config memblob size
);

int nmp_softmax_layer(
    uint16_t dfxp_mode,    // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base, // Layer Input base address
    uint32_t ofm_ddr_base, // Layer Output base address
    uint16_t ifm_dim,      // Layer Input dimension (to decide threading scheme)
    uint16_t ifm_w,        // Layer Input width
    uint16_t ifm_h,        // Layer Input height
    uint16_t ifm_ch,       // Layer Input channel
    uint16_t ifm_num,      // Layer Input num
    int16_t softmax_q_df,  // Quantize Param (softmax_df)
    int16_t softmax_q_wf,  // Quantize Param (softmax_wf)
    int16_t softmax_q_ls,  // Quantize Param (softmax_ls)
    int16_t softmax_q_loadmode, // Quantized Input load type (softmax_load_mode)
    uint16_t mapper_cnfg_num_tle, // Mapper's config number of tile
    uint16_t mapper_cnfg_num_tlt, // Mapper's config number of tilelet per tile
    uint16_t mapper_cnfg_mb_size // Mapper's config memblob size of tlt
);

int nmp_upsample_layer(
    uint16_t dfxp_mode,        // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm1_ddr_base,    // Layer Input1 base address
    uint32_t ifm2_ddr_base,    // Layer Input2 base address
    uint32_t ofm_ddr_base,     // Layer Output base address
    uint16_t ifm_w,            // Layer Input width
    uint16_t ifm_h,            // Layer Input height
    uint16_t ifm_ch,           // Layer Input channel
    uint16_t ofm_w,            // Layer Output width
    uint16_t ofm_h,            // Layer Output height
    uint16_t upsample_scale,   // Upsample Param (upsample_scale)
    uint16_t upsample_has_idx, // Upsample Param (upsample_has_idx)
    int16_t upsample_q_df,     // Quantize Param (upsample_df)
    int16_t upsample_q_wf,     // Quantize Param (upsample_wf)
    int16_t upsample_q_ls,     // Quantize Param (upsample_ls)
    int16_t
        upsample_q_loadmode, // Quantized Input load type (upsample_load_mode)
    int16_t upsample_idx_q_loadmode,  // Quantized Input load type
                                      // (upsample_load_mode)
    uint16_t mapper_cnfg_dp_nx,       // Mapper's config data partition axis-x
    uint16_t mapper_cnfg_dp_ny,       // Mapper's config data partition axis-y
    uint16_t mapper_cnfg_slice_ofm_w, // Mapper's config sliding window width
    uint16_t mapper_cnfg_slice_ofm_h, // Mapper's config sliding window height
    uint16_t mapper_cnfg_num_tlt // Mapper's config number of tilelets per tile
);

int nmp_deconv_ifm_ext_layer(
    uint16_t dfxp_mode,      // Precision (NMP_CFG_DFXP_SIZE_DXX)
    uint32_t ifm_ddr_base,   // Input base address
    uint32_t ofm_ddr_base,   // Output base address
    uint16_t ifm_w,          // Input width
    uint16_t ifm_h,          // Input height
    uint16_t ifm_ch,         // Input channel
    uint16_t stride,         // Deconvolution Param (stride)
    uint16_t has_neg_data,   // optimization option: is the ifm data contain
                             // negative data
    int16_t ifm_dq,          // Quantize Param
    uint16_t num_tlts,       // TLE partition: number of tilelets per tile
    uint16_t mb_size,        // Double mblob size divided by dfxp_mode
    uint16_t tle_part_ifm_h, // TLE partition: thread partition along the ofm_y
                             // direction
    uint16_t tle_part_knl_n, // TLE partition: thread partition along the ofm_ch
                             // direction per tle
    uint16_t tlt_part_slice_ifm_h,   // TLT partition: sliding ifm height
    uint16_t tlt_part_slice_ifm_ch); // TLT partition: sliding ifm channel

void nmp_sync_gsema(uint32_t mst_tle_id, // The Master Tile ID
                    uint32_t tle_id,     // Tile ID
                    uint32_t gsema_loc,  // To generate flip-flop pattern
                    uint32_t num_tle     // Number of Tiles
);

void nmp_sync_tsema(uint32_t mst_tlt_id, // The Master Tilelet ID
                    uint32_t tlt_id,     // Tilelet ID
                    uint32_t num_tlt     // Number of Tilelets per Tile
);

int nmp_matmul_layer(uint16_t dfxp_mode,
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
                     uint16_t num_tlt);

} // extern "C"

#endif
