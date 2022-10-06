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
 * @file: NMPTensorAnalysis.cpp
 * @brief description: Implement Tensor Slice Optmizations for NMP Codegen.
 * @date: 11 18, 2021
 */

#include "NMPTensorAnalysis.h"
#include "glow/Backends/NMP/nmp_common.h"

#include <iostream>
#include <string>

void TLEPartition::Execute() {
  double min_time = std::numeric_limits<double>::max();

  for (int num_thread = 1; num_thread <= arch_info_.num_tle; num_thread++) {
    if (arch_info_.num_tle % num_thread == 0) {
      // Identify the part of the data the TLE0 will work on
      int thread_ofm_h = std::ceil((double)(conv_info_.ofm_h) / num_thread);
      int thread_ofm_c = std::ceil(
          (double)(conv_info_.num_filters * num_thread) / (arch_info_.num_tle));

      ConvSliceInfo slice = TLTPartitioning(
          thread_ofm_h, thread_ofm_c, conv_info_.is_dw, conv_info_.has_bias);
      double exec_time = slice.total_time;

      if (exec_time < min_time) {
        min_time = exec_time;
        // TLE
        data_out_.thread_ofm_h = thread_ofm_h;
        data_out_.thread_ofm_c = thread_ofm_c;
        data_out_.thread_np_y = num_thread;
        data_out_.thread_np_c = arch_info_.num_tle / num_thread;

        // ConvSliceInfo
        data_out_.slice = slice;

        // Update fn_call
        UpdateFnCall();
      }
    }
  }
}

ConvSliceInfo TLEPartition::TLTPartitioning(uint16_t thread_ofm_h, uint16_t thread_ofm_c,
                                            bool is_dw, bool has_bias) {
  // If it is DW, then only one TLT Partitioning Strategy is executed
  if (is_dw) {
    auto conv_dw = std::unique_ptr<PartitionStrategy>(new WeightStationaryOFMDW(
        thread_ofm_h, thread_ofm_c, arch_info_, conv_info_, has_bias));
    conv_dw->Execute();
    return conv_dw->data_out();
  }

  constexpr size_t num_array = 3;
  std::array<std::unique_ptr<PartitionStrategy>, num_array> tlt_part = {
      std::unique_ptr<PartitionStrategy>(new WeightStationaryOFM(
          thread_ofm_h, thread_ofm_c, arch_info_, conv_info_, has_bias)),
      std::unique_ptr<PartitionStrategy>(new OFMStationaryIFM(
          thread_ofm_h, thread_ofm_c, arch_info_, conv_info_, has_bias)),
      std::unique_ptr<PartitionStrategy>(new IFMStationaryOFM(
          thread_ofm_h, thread_ofm_c, arch_info_, conv_info_, has_bias))};

  std::array<ConvSliceInfo, num_array> slices;
  std::array<double, num_array> times;

  for (size_t i = 0; i < num_array; i++) {
    times[i] = tlt_part[i]->Execute();
    slices[i] = tlt_part[i]->data_out();
  }

  // TODO: Improve the way I select the TLTStrategy
  if ((times[0] < times[1]) && (times[0] < times[2])) {
    return slices[0];
  } else if (times[2] <= times[1]) {
    return slices[2];
  } else {
    return slices[1];
  }
}

bool PartitionStrategy::HasTLTStrategyConstraint(ConvSliceInfo slice) {
  Conv2dInfo &conv_info = this->conv_info();
  ArchitectureInfo &info_arch = this->info_arch();

  if (HasTLTStrategySpecificConstraint(slice)) {
    return true;
  }

  // Check W slice
  int filter_size = 0;
  if (strategy() == TLTStrategy::wstn_ofm) {
    filter_size = conv_info.kernel_h * conv_info.kernel_w * conv_info.ifm_c *
                  info_arch.f_size;
  } else {
    filter_size = conv_info.kernel_h * conv_info.kernel_w * slice._ifm_c *
                  info_arch.f_size;
  }

  int additional_bias = 0;
  if (hasBias()) {
    additional_bias = slice._filters * info_arch.f_size;
  }

  int slice_w_size = (filter_size * slice._filters) + additional_bias;

  // Check IN slice
  int slice_in_size =
      slice._ifm_w * slice._ifm_h * slice._ifm_c * info_arch.f_size;

  // Check OUT slice -- there is another check later (StackConv)
  int slice_out_size =
      slice._ofm_w * slice._ofm_h * slice._filters * info_arch.f_size;

  if (slice_in_size > info_arch.mblob_size ||
      slice_w_size > info_arch.mblob_size ||
      slice_out_size > info_arch.mblob_size) {
    return true;
  }

  return false;
}

// Check if three is two slices that covers the padding
// Only 1 slice can have overlap with the padding
// There are 4 checks to evaluate this constraints:
// 1) padding-top    (Check any TLT of the first TLE)
// 2) padding-bottom (Check any TLT of the last TLE)
// 3) padding-left   (Can check any TLT)
// 4) padding-right  (Can check any TLT)
bool PartitionStrategy::HasPaddingConstraint(ConvSliceInfo slice) {
  Conv2dInfo &conv_info = conv_info_;
  if (conv_info.padding_top == 0 && conv_info.padding_bottom == 0 &&
      conv_info.padding_left == 0 && conv_info.padding_right == 0) {
    return false;
  }

  int min_slice_ofm_h = ((conv_info.kernel_h - 1) / 2) - (conv_info.stride - 1);
  int min_slice_ofm_w = ((conv_info.kernel_w - 1) / 2) - (conv_info.stride - 1);

  // Check 1) padding-top
  if (conv_info.padding_top > 0) {
    if (slice._ofm_h < min_slice_ofm_h) {
      return true;
    }
  }

  // Check 2) padding-bottom
  if (conv_info.padding_bottom > 0) {
    int thread_ofm_h_last_tle;
    if (conv_info.ofm_h % slice.thread_ofm_h == 0) {
      thread_ofm_h_last_tle = slice.thread_ofm_h;
    } else {
      thread_ofm_h_last_tle = slice.thread_ofm_h - 1;
    }

    int slice_ofm_h_remain_last_tle = thread_ofm_h_last_tle % slice._ofm_h;
    if (slice_ofm_h_remain_last_tle > 0 &&
        slice_ofm_h_remain_last_tle < min_slice_ofm_h) {
      return true;
    }
  }

  // Check 3) padding-left
  if (conv_info.padding_left > 0) {
    if (slice._ofm_w < min_slice_ofm_w) {
      return true;
    }
  }

  // Check 4) padding-righ
  if (conv_info.padding_right > 0) {
    if (slice._ofm_w_remain < min_slice_ofm_w) {
      return true;
    }
  }

  return false;
}

bool PartitionStrategy::HasStackedConvConstraint(ConvSliceInfo slice) {
  Conv2dInfo &conv_info = conv_info_;
  ArchitectureInfo &info_arch = this->info_arch();

  if (slice._ifm_c == conv_info.ifm_c) {
    return false;
  }

  int out_intermediate_size = slice._ofm_h * slice._ofm_w * slice._filters *
                              info_arch.f_size * 4 /* quad-precision */;

  if (out_intermediate_size > info_arch.mblob_size) {
    return true;
  }

  // Check kernel's volume (It has to be greater than 12)
  int kernel_volume = conv_info.kernel_h * conv_info.kernel_w * slice._ifm_c;
  if (kernel_volume < 12) {
    return true;
  }

  // Check the last iteration over the kernels
  int kernel_volume_remain =
      conv_info.kernel_h * conv_info.kernel_w * slice._ifm_c_remain;
  if (kernel_volume_remain < 12) {
    return true;
  }

  return false;
}

double PartitionStrategy::GenSlices() {
  ArchitectureInfo &info_arch = this->info_arch();

  double min_value = std::numeric_limits<double>::max();

  // For the case CONV-DW, identify the number of filters
  // the TLT will work on; For the other TLT strategies,
  // the depth is the number of channels in the IN
  int max_depth = conv_info_.ifm_c;
  int first_width = 1;
  if (isConvDW()) {
    max_depth = (int)ceil((double)thread_ofm_c_ / info_arch.num_tlt_per_tle);
    first_width = conv_info_.ofm_w;
  }

  for (int w = first_width; w <= conv_info_.ofm_w; w++) {
    for (int h = 1; h <= thread_ofm_h_; h++) {
      for (int c = 1; c <= max_depth; c++) {

        ConvSliceInfo test_slice = PrepareSlices(w, h, c);

        if (!isConvDW()) {
          if (HasStackedConvConstraint(test_slice)) {
            continue;
          }
        }

        if (HasTLTStrategyConstraint(test_slice) ||
            HasPaddingConstraint(test_slice)) {
          continue;
        }

        double tmp = CalculateConvolutionTime(test_slice);

        if (tmp <= min_value) {
          test_slice.strategy = strategy();
          data_out_ = test_slice;
          min_value = tmp;
        }
      }
    }
  }

  return min_value;
}

/* In this version it considers only the execution of TL0 and TLT0;
 * Should I consider the execution of all the TLTs of a single TLE?
 * TODO(Rafael): Check this */
ConvSliceInfo PartitionStrategy::PrepareSlices(int slice_ofm_w, int slice_ofm_h,
                                               int slice_ifm_c) {
  Conv2dInfo &conv_info = this->conv_info();
  ArchitectureInfo &info_arch = this->info_arch();
  ConvSliceInfo slice;

  // TLE
  slice.thread_ofm_h = thread_ofm_h_;
  slice.thread_ofm_w = conv_info.ofm_w;
  slice.thread_ofm_c = thread_ofm_c_;

  // TLT - Identify the #filters TLE0-TLT0 has to processs for this slice
  slice._ofm_c = slice.thread_ofm_c / info_arch.num_tlt_per_tle;
  slice._ofm_c_remain = slice.thread_ofm_c % info_arch.num_tlt_per_tle;
  if (slice._ofm_c_remain > 0) {
    slice._ofm_c++;
    slice.tlt_balanced = false;
  } else {
    slice.tlt_balanced = true;
  }

  // [SLICE_IFM_c]
  slice._ifm_c = slice_ifm_c;
  slice._ifm_nc = conv_info.ifm_c / slice._ifm_c;
  slice._ifm_c_remain = conv_info.ifm_c % slice._ifm_c;
  if (slice._ifm_c_remain != 0) {
    slice._ifm_nc++;
  } else {
    slice._ifm_c_remain = slice._ifm_c;
  }

  // [SLICE_W]
  slice._filters =
      GetSliceNumFilters(slice_ofm_w, slice_ofm_h, slice._ifm_c, slice._ofm_c);
  slice._ofm_nc = slice._ofm_c / slice._filters;
  slice._ofm_c_remain = slice._ofm_c % slice._filters;
  if (slice._ofm_c_remain != 0) {
    slice._ofm_nc++;
  } else {
    slice._ofm_c_remain = slice._filters;
  }

  // [SLICE_OUT]
  slice._ofm_w = slice_ofm_w;
  slice._ofm_h = slice_ofm_h;
  slice._ofm_nx = slice.thread_ofm_w / slice._ofm_w;
  slice._ofm_w_remain = slice.thread_ofm_w % slice._ofm_w;
  slice._ofm_ny = slice.thread_ofm_h / slice._ofm_h;
  slice._ofm_h_remain = slice.thread_ofm_h % slice._ofm_h;

  // [SLICE_IN]
  slice._ifm_w = (slice._ofm_w - 1) * conv_info.stride + conv_info.kernel_w;
  slice._ifm_w_remain =
      (slice._ofm_w_remain - 1) * conv_info.stride + conv_info.kernel_w;
  slice._ifm_h = (slice._ofm_h - 1) * conv_info.stride + conv_info.kernel_h;
  slice._ifm_h_remain =
      (slice._ofm_h_remain - 1) * conv_info.stride + conv_info.kernel_h;

  if (slice._ofm_w_remain != 0) {
    slice._ofm_nx += 1;
  } else {
    slice._ifm_w_remain = slice._ifm_w;
    slice._ofm_w_remain = slice._ofm_w;
  }

  if (slice._ofm_h_remain != 0) {
    slice._ofm_ny += 1;
  } else {
    slice._ifm_h_remain = slice._ifm_h;
    slice._ofm_h_remain = slice._ofm_h;
  }

  return slice;
}

double PartitionStrategy::ifm_ddr_timeV2(int slice_i, int slice_j, int slice_k,
                                         int slice_ifm_w, int slice_ifm_h,
                                         int slice_ifm_c) {
  int burst_cnt = 0;
  int last_burst = -1;
  uint32_t slice_addr = (slice_k * (conv_info_.ifm_h * conv_info_.ifm_w) +
                         (slice_i * conv_info_.ifm_w) + slice_j) *
                        info_arch_.f_size;
  int slice_row_size = slice_ifm_w * info_arch_.f_size;
  int channel_size = conv_info_.ifm_h * conv_info_.ifm_w * info_arch_.f_size;

  // compute the burst count of the slice
  for (int k = 0; k < slice_ifm_c; k++) {
    // traversing one channel
    for (int i = 0; i < slice_ifm_h; i++) {
      // compute row bounds addresses in memory
      uint32_t row_begin =
          slice_addr + (((i * conv_info_.ifm_w) + slice_j) * info_arch_.f_size);
      uint32_t row_end = row_begin + slice_row_size - info_arch_.f_size;

      // compute burst begin and end addresses of frist burst
      int burst_begin =
          (int)floor((double)row_begin / info_arch_.max_burst_size) *
          info_arch_.max_burst_size;
      int burst_end =
          burst_begin + info_arch_.max_burst_size - info_arch_.f_size;

      // slice will have at least one burst if not seen before
      if (burst_begin != last_burst)
        burst_cnt += 1;

      // add the remaining bursts on the row
      if (burst_end < row_end) {
        int tmp_burst_cnt =
            (int)ceil((row_end - burst_end) / info_arch_.max_burst_size);
        burst_cnt += tmp_burst_cnt;
        last_burst = burst_begin + (tmp_burst_cnt * info_arch_.max_burst_size);
      } else {
        last_burst = burst_begin;
      }
    }
    slice_addr += channel_size;
  }

  double mb_total = ((double)(burst_cnt * 128.0) / 1024) / 1024;
  double transfer_time = (mb_total / info_arch_.max_ddr_bw) * 1000 /* ms */;
  double cas_latency = burst_cnt * info_arch_.cas_latency;
  double total_ms = transfer_time + cas_latency;

  // return slice burst count
  return total_ms;
}

double PartitionStrategy::ifm_ddr_time(int slice_ifm_w, int slice_ifm_h,
                                       int slice_ifm_c) {
  int burst_cnt = 0;
  int data_len;
  int factor;
  double mb_total;

  if (slice_ifm_w < conv_info_.ifm_w) {
    data_len = slice_ifm_w;
    factor = slice_ifm_c * slice_ifm_h;
  } else {
    if (slice_ifm_h < conv_info_.ifm_h) {
      data_len = slice_ifm_w * slice_ifm_h;
      factor = slice_ifm_c;
    } else {
      data_len = slice_ifm_w * slice_ifm_h * slice_ifm_c;
      factor = 1;
    }
  }

  // Count Bursts
  burst_cnt = (int)ceil((double)data_len * info_arch_.f_size /
                        info_arch_.max_burst_size);
  burst_cnt *= factor;

  mb_total = ((double)(burst_cnt * 128.0) / 1024) / 1024;

  double transfer_time = (mb_total / info_arch_.max_ddr_bw) * 1000 /* ms */;
  double cas_latency = burst_cnt * info_arch_.cas_latency;
  double total_ms = transfer_time + cas_latency;

  return total_ms;
}

double PartitionStrategy::weight_ddr_time(int slice_kernel_c, int num_kernels) {
  int burst_cnt = 0;
  int datalen;
  int factor;
  double mb_total;

  if (conv_info_.ifm_c == slice_kernel_c || isConvDW()) {
    datalen = conv_info_.kernel_h * conv_info_.kernel_w * slice_kernel_c *
              num_kernels;
    factor = 1;
  } else {
    datalen = conv_info_.kernel_h * conv_info_.kernel_w * slice_kernel_c;
    factor = num_kernels;
  }

  // Count Bursts
  burst_cnt = (int)ceil((double)datalen * info_arch_.f_size /
                        info_arch_.max_burst_size);
  burst_cnt *= factor;

  mb_total = ((double)(burst_cnt * 128.0) / 1024) / 1024;

  double transfer_time = (mb_total / info_arch_.max_ddr_bw) * 1000 /* ms */;
  double cas_latency = burst_cnt * info_arch_.cas_latency;
  double total_ms = transfer_time + cas_latency;

  return total_ms;
}

double PartitionStrategy::ofm_ddr_time(int slice_ofm_w, int slice_ofm_h,
                                       int slice_ofm_c) {
  int burst_cnt = 0;
  int data_len;
  int factor;
  double mb_total;

  if (slice_ofm_w < conv_info_.ofm_w) {
    data_len = slice_ofm_w;
    factor = slice_ofm_h * slice_ofm_c;
  } else {
    if (slice_ofm_h < conv_info_.ofm_h) {
      data_len = slice_ofm_w * slice_ofm_h;
      factor = slice_ofm_c;
    } else {
      data_len = slice_ofm_w * slice_ofm_h * slice_ofm_c;
      factor = 1;
    }
  }

  // Count Bursts
  burst_cnt = (int)ceil((double)data_len * info_arch_.f_size /
                        info_arch_.max_burst_size);
  burst_cnt *= factor;

  mb_total = ((double)(burst_cnt * 128.0) / 1024) / 1024;

  double transfer_time = (mb_total / info_arch_.max_ddr_bw) * 1000 /* ms */;
  double cas_latency = burst_cnt * info_arch_.cas_latency;
  double total_ms = transfer_time + cas_latency;

  return total_ms;
}

long PartitionStrategy::useful_mac(int slice_ofm_w, int slice_ofm_h,
                                   int slice_ifm_c, int slice_filters) {
  long total_useful_mac = slice_ofm_w * slice_ofm_h * slice_ifm_c *
                          slice_filters * conv_info_.kernel_h *
                          conv_info_.kernel_w;

  return total_useful_mac;
}

long PartitionStrategy::idle_mac(int slice_ifm_w, int slice_ifm_h,
                                 int slice_ifm_c, int slice_filters) {
  int n = info_arch_.mac_per_tlt / info_arch_.f_size;

  long full_slice_w = n + conv_info_.kernel_w - 1;
  while (full_slice_w < slice_ifm_w) {
    full_slice_w += n;
  }

  long remain_macs = full_slice_w - slice_ifm_w;
  long total_idle_macs = remain_macs * slice_ifm_h * slice_ifm_c *
                         slice_filters * conv_info_.kernel_h *
                         conv_info_.kernel_w;

  return total_idle_macs;
}

double PartitionStrategy::calculation_time(int slice_ofm_w, int slice_ofm_h,
                                           int slice_ifm_w, int slice_ifm_h,
                                           int slice_ifm_c, int slice_filters) {
  int n = info_arch_.mac_per_tlt / info_arch_.f_size;
  long num_useful_macs =
      useful_mac(slice_ofm_w, slice_ofm_h, slice_ifm_c, slice_filters);
  long num_idle_macs =
      idle_mac(slice_ifm_w, slice_ifm_h, slice_ifm_c, slice_filters);
  long total_macs = num_useful_macs + num_idle_macs;
  long total_cycles = (long)ceil((double)total_macs / n);
  double num_cycles_per_second = info_arch_.frequency * pow(10, 6);

  return ((double)total_cycles / num_cycles_per_second) * 1000;
}

double WeightStationaryOFM::SoftwareTime(int slice_ofm_nc, int slice_ofm_ny,
                                         int slice_ofm_nx, int slice_ifm_nc) {
  ArchitectureInfo &info_arch = this->info_arch();
  // 100 is an approximation
  int sofware_cycles =
      slice_ofm_nc * slice_ofm_ny * slice_ofm_nx * slice_ifm_nc * 120;
  double num_cycles_per_second = info_arch.frequency * pow(10, 6);

  return (sofware_cycles / num_cycles_per_second) * 1000;
}

double WeightStationaryOFM::CalculateConvolutionTime(ConvSliceInfo &slice) {
  Conv2dInfo &conv_info = this->conv_info();
  double conv_time = 0.0f;
  double load_in_time = 0.0f;
  double load_w_time = 0.0f;
  double store_out_time = 0.0f;

  for (int f = 0; f < slice._ofm_nc; f++) {
    int _tmp_slice_filters = slice._filters;
    if (f == (slice._ofm_nc - 1)) {
      _tmp_slice_filters = slice._ofm_c_remain;
    }

    // Load W
    load_w_time += weight_ddr_time(conv_info.ifm_c, _tmp_slice_filters);

    int ifm_h_idx = 0;
    for (int y = 0; y < slice._ofm_ny; y++) {
      int _tmp_slice_ifm_y = slice._ifm_h;
      int _tmp_slice_ofm_y = slice._ofm_h;
      if (y == (slice._ofm_ny - 1)) {
        _tmp_slice_ifm_y = slice._ifm_h_remain;
        _tmp_slice_ofm_y = slice._ofm_h_remain;
      }

      int ifm_w_idx = 0;
      for (int x = 0; x < slice._ofm_nx; x++) {
        int _tmp_slice_ifm_x = slice._ifm_w;
        int _tmp_slice_ofm_x = slice._ofm_w;
        if (x == (slice._ofm_nx - 1)) {
          _tmp_slice_ifm_x = slice._ifm_w_remain;
          _tmp_slice_ofm_x = slice._ofm_w_remain;
        }

        int ifm_c_idx = 0;
        for (int ich = 0; ich < slice._ifm_nc; ich++) {
          int _tmp_slice_ifm_c = slice._ifm_c;
          if (ich == (slice._ifm_nc - 1)) {
            _tmp_slice_ifm_c = slice._ifm_c_remain;
          }

          // Load IN
          load_in_time += ifm_ddr_time(_tmp_slice_ifm_x, _tmp_slice_ifm_y,
                                       _tmp_slice_ifm_c);
          // Convolution
          conv_time += calculation_time(_tmp_slice_ofm_x, _tmp_slice_ofm_y,
                                        _tmp_slice_ifm_x, _tmp_slice_ifm_y,
                                        _tmp_slice_ifm_c, 1) *
                       _tmp_slice_filters;

          if (ich == (slice._ifm_nc - 1)) {
            store_out_time += ofm_ddr_time(_tmp_slice_ofm_x, _tmp_slice_ofm_y,
                                           _tmp_slice_filters);
          }
          ifm_c_idx += _tmp_slice_ifm_c;
        }
        ifm_w_idx += _tmp_slice_ifm_x;
      }
      ifm_h_idx += _tmp_slice_ifm_y;
    }
  }

  // Consolidate W_TIME
  // load_w_time *= info_arch.num_tlt_per_tle;

  double sw_time =
      SoftwareTime(slice._ofm_nc, slice._ofm_ny, slice._ofm_nx, slice._ifm_nc);

  double total_computation =
      conv_time + load_in_time + load_w_time + store_out_time + sw_time;

  slice.load_in_time = load_in_time;
  slice.load_w_time = load_w_time;
  slice.store_time = store_out_time;
  slice.conv_time = conv_time;
  slice.sw_time = sw_time;
  slice.total_time = total_computation;

  return total_computation;
}

int WeightStationaryOFM::GetSliceNumFilters(int slice_ofm_w, int slice_ofm_h,
                                            int slice_ifm_c,
                                            int total_filters_tlt) {
  Conv2dInfo &conv_info = this->conv_info();
  ArchitectureInfo &info_arch = this->info_arch();

  int factor = 1;
  if (slice_ifm_c != conv_info.ifm_c) {
    factor = 4;
  }

  int slice_ofm_size = slice_ofm_w * slice_ofm_h * info_arch.f_size * factor;
  int filter_size = conv_info.kernel_h * conv_info.kernel_w * conv_info.ifm_c *
                    info_arch.f_size;

  int additional_bias = 0;
  int slice_filters = 1;
  for (int f = 1; f <= total_filters_tlt; f++) {
    if (hasBias()) {
      additional_bias = f * info_arch.f_size;
    }
    int slice_ofm_size_x_f = slice_ofm_size * f;
    int filter_size_x_f = (filter_size * f) + additional_bias;

    if (slice_ofm_size_x_f <= info_arch.mblob_size &&
        filter_size_x_f <= info_arch.mblob_size) {
      slice_filters = f;
    } else {
      break;
    }
  }

  return slice_filters;
}

// A entire filter has to fit in the MBLOB
bool WeightStationaryOFM::HasTLTStrategySpecificConstraint(
    ConvSliceInfo slice) {
  return false;
}

double IFMStationaryOFM::SoftwareTime(int slice_ofm_nc, int slice_ofm_ny,
                                      int slice_ofm_nx, int slice_ifm_nc) {
  ArchitectureInfo &info_arch = this->info_arch();
  // 100 is an approximation
  int sofware_cycles = slice_ofm_ny * slice_ifm_nc * 100;
  double num_cycles_per_second = info_arch.frequency * pow(10, 6);

  return (sofware_cycles / num_cycles_per_second) * 1000;
}

double IFMStationaryOFM::CalculateConvolutionTime(ConvSliceInfo &slice) {
  double conv_time = 0.0f;
  double load_in_time = 0.0f;
  double load_w_time = 0.0f;
  double store_out_time = 0.0f;

  int ifm_h_idx = 0;
  for (int y = 0; y < slice._ofm_ny; y++) {
    int _tmp_slice_ifm_y = slice._ifm_h;
    int _tmp_slice_ofm_y = slice._ofm_h;
    if (y == (slice._ofm_ny - 1)) {
      _tmp_slice_ifm_y = slice._ifm_h_remain;
      _tmp_slice_ofm_y = slice._ofm_h_remain;
    }

    int ifm_c_idx = 0;
    for (int ich = 0; ich < slice._ifm_nc; ich++) {
      int _tmp_slice_ifm_c = slice._ifm_c;
      if (ich == (slice._ifm_nc - 1)) {
        _tmp_slice_ifm_c = slice._ifm_c_remain;
      }

      // Load W
      load_w_time += weight_ddr_time(_tmp_slice_ifm_c, slice._filters);

      // Load IN
      load_in_time +=
          ifm_ddr_time(slice._ifm_w, _tmp_slice_ifm_y, _tmp_slice_ifm_c);

      // Convolution
      conv_time +=
          calculation_time(slice._ofm_w, _tmp_slice_ofm_y, slice._ifm_w,
                           _tmp_slice_ifm_y, _tmp_slice_ifm_c, slice._filters);

      if (ich == (slice._ifm_nc - 1)) {
        store_out_time +=
            ofm_ddr_time(slice._ofm_w, _tmp_slice_ofm_y, slice._filters);
      }
      ifm_c_idx += _tmp_slice_ifm_c;
    }
    ifm_h_idx += _tmp_slice_ifm_y;
  }

  // Consolidate W_TIME
  // load_w_time *= info_arch.num_tlt_per_tle;

  double sw_time =
      SoftwareTime(slice._ofm_nc, slice._ofm_ny, slice._ofm_nx, slice._ifm_nc);

  double total_computation =
      conv_time + load_in_time + load_w_time + store_out_time + sw_time;

  slice.load_in_time = load_in_time;
  slice.load_w_time = load_w_time;
  slice.store_time = store_out_time;
  slice.conv_time = conv_time;
  slice.sw_time = sw_time;
  slice.total_time = total_computation;

  return total_computation;
}

int IFMStationaryOFM::GetSliceNumFilters(int slice_ofm_w, int slice_ofm_h,
                                         int slice_ifm_c,
                                         int total_filters_tlt) {
  return total_filters_tlt;
}

// A entire filter has to fit in the MBLOB
bool IFMStationaryOFM::HasTLTStrategySpecificConstraint(ConvSliceInfo slice) {
  Conv2dInfo &conv_info = this->conv_info();

  // Check if W Slice contains all the filters given to the TLT
  if (slice._ofm_c != slice._filters) {
    return true;
  }

  // slice_ofm_w has to be equal to ofm_w
  if (slice._ofm_w != conv_info.ofm_w) {
    return true;
  }

  return false;
}

double OFMStationaryIFM::SoftwareTime(int slice_ofm_nc, int slice_ofm_ny,
                                      int slice_ofm_nx, int slice_ifm_nc) {
  ArchitectureInfo &info_arch = this->info_arch();
  // 100 is an approximation
  int sofware_cycles =
      slice_ofm_nc * slice_ofm_ny * slice_ofm_nx * slice_ifm_nc * 120;
  double num_cycles_per_second = info_arch.frequency * pow(10, 6);

  return (sofware_cycles / num_cycles_per_second) * 1000;
}

double OFMStationaryIFM::CalculateConvolutionTime(ConvSliceInfo &slice) {
  double conv_time = 0.0f;
  double load_in_time = 0.0f;
  double load_w_time = 0.0f;
  double store_out_time = 0.0f;

  int ifm_h_idx = 0;
  for (int y = 0; y < slice._ofm_ny; y++) {
    int _tmp_slice_ifm_y = slice._ifm_h;
    int _tmp_slice_ofm_y = slice._ofm_h;
    if (y == (slice._ofm_ny - 1)) {
      _tmp_slice_ifm_y = slice._ifm_h_remain;
      _tmp_slice_ofm_y = slice._ofm_h_remain;
    }

    int ifm_w_idx = 0;
    for (int x = 0; x < slice._ofm_nx; x++) {
      int _tmp_slice_ifm_x = slice._ifm_w;
      int _tmp_slice_ofm_x = slice._ofm_w;
      if (x == (slice._ofm_nx - 1)) {
        _tmp_slice_ifm_x = slice._ifm_w_remain;
        _tmp_slice_ofm_x = slice._ofm_w_remain;
      }

      int ifm_c_idx = 0;
      for (int f = 0; f < slice._ofm_nc; f++) {
        int _tmp_slice_filters = slice._filters;
        if (f == (slice._ofm_nc - 1)) {
          _tmp_slice_filters = slice._ofm_c_remain;
        }

        for (int ich = 0; ich < slice._ifm_nc; ich++) {
          int _tmp_slice_ifm_c = slice._ifm_c;
          if (ich == (slice._ifm_nc - 1)) {
            _tmp_slice_ifm_c = slice._ifm_c_remain;
          }

          // Load W
          load_w_time += weight_ddr_time(_tmp_slice_ifm_c, _tmp_slice_filters);

          // Load IN
          load_in_time += ifm_ddr_time(_tmp_slice_ifm_x, _tmp_slice_ifm_y,
                                       _tmp_slice_ifm_c);

          // Convolution
          conv_time += calculation_time(_tmp_slice_ofm_x, _tmp_slice_ofm_y,
                                        _tmp_slice_ifm_x, _tmp_slice_ifm_y,
                                        _tmp_slice_ifm_c, 1) *
                       _tmp_slice_filters;

          if (ich == (slice._ifm_nc - 1)) {
            store_out_time += ofm_ddr_time(_tmp_slice_ofm_x, _tmp_slice_ofm_y,
                                           _tmp_slice_filters);
          }
          ifm_c_idx += _tmp_slice_ifm_c;
        }
        ifm_w_idx += _tmp_slice_ifm_x;
      }
      ifm_h_idx += _tmp_slice_ifm_y;
    }
  }

  // Consolidate W_TIME
  // load_w_time *= info_arch.num_tlt_per_tle;

  double sw_time =
      SoftwareTime(slice._ofm_nc, slice._ofm_ny, slice._ofm_nx, slice._ifm_nc);

  double total_computation =
      conv_time + load_in_time + load_w_time + store_out_time + sw_time;

  slice.load_in_time = load_in_time;
  slice.load_w_time = load_w_time;
  slice.store_time = store_out_time;
  slice.conv_time = conv_time;
  slice.sw_time = sw_time;
  slice.total_time = total_computation;

  return total_computation;
}

int OFMStationaryIFM::GetSliceNumFilters(int slice_ofm_w, int slice_ofm_h,
                                         int slice_ifm_c,
                                         int total_filters_tlt) {
  Conv2dInfo &conv_info = this->conv_info();
  ArchitectureInfo &info_arch = this->info_arch();

  int factor = 1;
  if (slice_ifm_c != conv_info.ifm_c) {
    factor = 4;
  }

  int slice_ofm_size = slice_ofm_w * slice_ofm_h * info_arch.f_size * factor;
  int filter_size =
      conv_info.kernel_h * conv_info.kernel_w * slice_ifm_c * info_arch.f_size;

  int slice_filters = 1;
  int additional_bias = 0;
  for (int f = 1; f <= total_filters_tlt; f++) {
    if (hasBias()) {
      additional_bias = f * info_arch.f_size;
    }
    int slice_ofm_size_x_f = slice_ofm_size * f;
    int filter_size_x_f = (filter_size * f) + additional_bias;

    if (slice_ofm_size_x_f <= info_arch.mblob_size &&
        filter_size_x_f <= info_arch.mblob_size) {
      slice_filters = f;
    } else {
      break;
    }
  }

  return slice_filters;
}

bool OFMStationaryIFM::HasTLTStrategySpecificConstraint(ConvSliceInfo slice) {
  return false;
}

double WeightStationaryOFMDW::SoftwareTime(int slice_ofm_nc, int slice_ofm_ny,
                                           int slice_ofm_nx, int slice_ifm_nc) {
  ArchitectureInfo &info_arch = this->info_arch();
  // 100 is an approximation
  int sofware_cycles = slice_ofm_nc * slice_ofm_ny * slice_ofm_nx * 120;
  double num_cycles_per_second = info_arch.frequency * pow(10, 6);

  return (sofware_cycles / num_cycles_per_second) * 1000;
}

double WeightStationaryOFMDW::CalculateConvolutionTime(ConvSliceInfo &slice) {
  double conv_time = 0.0f;
  double load_in_time = 0.0f;
  double load_w_time = 0.0f;
  double store_out_time = 0.0f;

  for (int f = 0; f < slice._ofm_nc; f++) {
    int _tmp_slice_filters = slice._filters;
    if (f == (slice._ofm_nc - 1)) {
      _tmp_slice_filters = slice._ofm_c_remain;
    }

    // Load W
    load_w_time += weight_ddr_time(1, _tmp_slice_filters);

    for (int y = 0; y < slice._ofm_ny; y++) {
      int _tmp_slice_ifm_y = slice._ifm_h;
      int _tmp_slice_ofm_y = slice._ofm_h;
      if (y == (slice._ofm_ny - 1)) {
        _tmp_slice_ifm_y = slice._ifm_h_remain;
        _tmp_slice_ofm_y = slice._ofm_h_remain;
      }

      for (int x = 0; x < slice._ofm_nx; x++) {
        int _tmp_slice_ifm_x = slice._ifm_w;
        int _tmp_slice_ofm_x = slice._ofm_w;
        if (x == (slice._ofm_nx - 1)) {
          _tmp_slice_ifm_x = slice._ifm_w_remain;
          _tmp_slice_ofm_x = slice._ofm_w_remain;
        }

        // Load IN
        load_in_time += ifm_ddr_time(_tmp_slice_ifm_x, _tmp_slice_ifm_y,
                                     _tmp_slice_filters);
        // Convolution
        conv_time +=
            calculation_time(_tmp_slice_ofm_x, _tmp_slice_ofm_y,
                             _tmp_slice_ifm_x, _tmp_slice_ifm_y, 1, 1) *
            _tmp_slice_filters;
        // Store
        store_out_time += ofm_ddr_time(_tmp_slice_ofm_x, _tmp_slice_ofm_y,
                                       _tmp_slice_filters);
      }
    }
  }

  // Consolidate W_TIME
  // load_w_time *= info_arch.num_tlt_per_tle;

  double sw_time =
      SoftwareTime(slice._ofm_nc, slice._ofm_ny, slice._ofm_nx, slice._ifm_nc);

  double total_computation =
      conv_time + load_in_time + load_w_time + store_out_time + sw_time;

  slice.load_in_time = load_in_time;
  slice.load_w_time = load_w_time;
  slice.store_time = store_out_time;
  slice.conv_time = conv_time;
  slice.sw_time = sw_time;
  slice.total_time = total_computation;

  return total_computation;
}

int WeightStationaryOFMDW::GetSliceNumFilters(int slice_ofm_w, int slice_ofm_h,
                                              int filters_per_iter,
                                              int total_filters_tlt) {
  return filters_per_iter;
}

// A entire filter has to fit in the MBLOB
bool WeightStationaryOFMDW::HasTLTStrategySpecificConstraint(
    ConvSliceInfo slice) {
  return false;
}
