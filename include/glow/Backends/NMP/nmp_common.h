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
 * @file: nmp_common.h
 * @brief description: Contains Defines from lib/Backends/NMP/libjit/nmp.h &
 *        lib/Backends/NMP/libjit/lib_nmp.h to simplify the build system.
 * @date: 11 17, 2021
 */

#ifndef __NMP_COMMON_H__
#define __NMP_COMMON_H__

#include <cstdint>

#define NMP_CFG_DFXP_SIZE_D08 (0x1 << 4)
#define NMP_CFG_DFXP_SIZE_D16 (0x0 << 4)
#define NMP_BASE_ADDR         0x15000000
#define NMP_BASE_DATA_ADDR    (NMP_BASE_ADDR + 0x01000000)

#define NMP_ELTWISE_SUM 0x2
#define NMP_ELTWISE_MUL 0x3
#define NMP_ELTWISE_MAX 0xf

#define NMP_POOL_AVG 0
#define NMP_POOL_MAX 1
#define NMP_POOL_PAD_ZERO 0
#define NMP_POOL_PAD_NEG 1
#define NMP_POOL_PAD_AVG 2

#define NMP_ELMSIZE_SAME 0x0
#define NMP_ELMSIZE_TRUNCATE_LSB 0x1
#define NMP_ELMSIZE_SIGNEXT_MSB 0x2
#define NMP_ELMSIZE_ZEROPAD_LSB 0x3

#define NMP_ACT_NONE 0x0
#define NMP_ACT_RELU 0x1
/* #define NMP_ACT_RELU6 2 */
/* #define NMP_ACT_PRELU 3 */
#define NMP_ACT_SIGM 0x10
#define NMP_ACT_TANH 0x11

#ifndef NMP_NUM_TLE
#define NMP_NUM_TLE 4
#endif

#ifndef NMP_NUM_TLT
#define NMP_NUM_TLT 8
#endif

#define NMP_NUM_TLTS (NMP_NUM_TLE * NMP_NUM_TLT)

#ifndef NMP_MBLOB_SIZE
#define NMP_MBLOB_SIZE 8192
#endif

#endif
