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
 * @file: nmp_dim_t.h
 * @brief description: This file is directly copied from glow/Base/DimType.h 
 *        to simplify the build system
 * @date: 11 17, 2021
 */

#ifndef GLOW_LLVMIRCODEGEN_LIBJIT_NMP_DIM_T_H
#define GLOW_LLVMIRCODEGEN_LIBJIT_NMP_DIM_T_H

#include <cinttypes>
#include <cstddef>
#include <cstdint>

#ifdef DIM_T_32
// The dimensions of Tensors are stored with this type. Note: The same
// fixed width type is used both in the host and the possible co-processors
// handling tensor data. The bit width should be chosen carefully for maximum
// data level parallel execution.
using dim_t = uint32_t;
using sdim_t = int32_t;

#define PRIdDIM PRId32
#define PRIuDIM PRIu32

#else // DIM_T_32
using dim_t = uint64_t;
using sdim_t = int64_t;

#define PRIdDIM PRId64
#define PRIuDIM PRIu64

#endif // DIM_T_32

constexpr unsigned DIM_T_BITWIDTH = sizeof(dim_t) * 8;
constexpr unsigned SDIM_T_BITWIDTH = sizeof(sdim_t) * 8;

#endif // GLOW_LLVMIRCODEGEN_LIBJIT_NMP_DIM_T_H
