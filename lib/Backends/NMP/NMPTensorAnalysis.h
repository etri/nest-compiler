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
 * @file: NMPTensorAnalysis.h
 * @brief description: Implement Tensor Slice Optmizations for NMP Codegen.
 * @date: 11 18, 2021
 */

#ifndef __NMP_TENSOR_ANALYSIS_H__
#define __NMP_TENSOR_ANALYSIS_H__

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

struct ArchitectureInfo {
  uint16_t f_size;
  uint16_t mac_per_tlt;
  uint16_t frequency;
  uint16_t mblob_size;
  uint16_t num_tle;
  uint16_t num_tlt_per_tle;
  double cas_latency;
  uint16_t max_burst_size;
  double max_ddr_bw;
};

enum TLTStrategy {
  istn_ofm,
  /* istn_wgt, */ // check
  ostn_ifm,
  wstn_ofm,
  wstn_ofm_dw,
};

struct Conv2dInfo {
  uint16_t ifm_c;
  uint16_t ifm_h;
  uint16_t ifm_w;
  uint16_t ofm_h;
  uint16_t ofm_w;
  uint16_t kernel_h;
  uint16_t kernel_w;
  uint16_t num_filters;
  uint16_t padding_top;
  uint16_t padding_bottom;
  uint16_t padding_left;
  uint16_t padding_right;
  uint16_t stride;
  uint16_t dilation;
  bool is_dw;
  bool has_bias;
};

struct ConvSliceInfo {
  TLTStrategy strategy;
  uint16_t thread_ofm_h;
  uint16_t thread_ofm_w;
  uint16_t thread_ofm_c;
  uint16_t _filters;
  uint16_t _ofm_c;
  uint16_t _ofm_c_remain;
  uint16_t _ofm_nc;
  uint16_t _ifm_c;
  uint16_t _ifm_c_remain;
  uint16_t _ifm_nc;
  uint16_t _ofm_w;
  uint16_t _ofm_h;
  uint16_t _ofm_nx;
  uint16_t _ofm_w_remain;
  uint16_t _ofm_ny;
  uint16_t _ofm_h_remain;
  uint16_t _ifm_w;
  uint16_t _ifm_w_remain;
  uint16_t _ifm_h;
  uint16_t _ifm_h_remain;
  double load_in_time;
  double load_w_time;
  double store_time;
  double conv_time;
  double sw_time;
  double total_time;
  bool tlt_balanced;
};

struct MapperInfo {
  /* TLE */
  uint16_t thread_ofm_h;
  uint16_t thread_ofm_c;
  uint16_t thread_np_y;
  uint16_t thread_np_c;
  /* TLT */
  ConvSliceInfo slice;
  std::string fn_name;
};

class TLEPartition {
public:
  // ArchitectureInfo carries architecture parameters like #TLE & #TLT
  // Conv2dInfo stores the convolution information
  TLEPartition(const ArchitectureInfo &arch_info, const Conv2dInfo &conv_info)
      : arch_info_{arch_info}, conv_info_{conv_info} {}

  ConvSliceInfo TLTPartitioning(uint16_t thread_ofm_h, uint16_t thread_ofm_c,
                                bool is_dw, bool has_bias);

  // Executes the partitioning resulting from TLEPartition
  void Execute();

  // Gets the result of the partition of the convolution
  MapperInfo GetResult() { return data_out_; }

  void UpdateFnCall() {
    switch (data_out_.slice.strategy) {
    case TLTStrategy::istn_ofm:
      data_out_.fn_name = "conv_layer_istn_ofm";
      break;
    /* case TLTStrategy::istn_wgt: */ // check
    /*   data_out_.fn_name = "conv_layer_istn_wgt"; */
    /*   break; */
    case TLTStrategy::ostn_ifm:
      data_out_.fn_name = "conv_layer_ostn_ifm";
      break;
    case TLTStrategy::wstn_ofm:
      data_out_.fn_name = "conv_layer_wstn_ofm";
      break;
    case TLTStrategy::wstn_ofm_dw:
      data_out_.fn_name = "conv_layer_wstn_ofm_dw";
      break;
    }
  }

private:
  const ArchitectureInfo &arch_info_;
  const Conv2dInfo &conv_info_;
  MapperInfo data_out_;
};

class PartitionStrategy {
public:
  virtual ~PartitionStrategy() = default;

  PartitionStrategy(const int thread_ofm_h, const int thread_ofm_c,
                    const ArchitectureInfo &info, const Conv2dInfo &conv_info,
                    bool has_bias)
      : is_dw_(false), has_bias_(has_bias), thread_ofm_h_(thread_ofm_h),
        thread_ofm_c_(thread_ofm_c), info_arch_{info}, conv_info_{conv_info} {}

  double Execute() { return GenSlices(); }

  ConvSliceInfo &data_out() { return data_out_; }

  virtual TLTStrategy strategy() = 0;

  virtual double CalculateConvolutionTime(ConvSliceInfo &slice) = 0;

  virtual int GetSliceNumFilters(int slice_ofm_w, int slice_ofm_h,
                                 int slice_ifm_c, int total_filters_tlt) = 0;

  virtual double SoftwareTime(int slice_ofm_nc, int slice_ofm_ny,
                              int slice_ofm_nx, int slice_ifm_nc) = 0;

  virtual bool HasTLTStrategySpecificConstraint(ConvSliceInfo slice) = 0;

  bool HasTLTStrategyConstraint(ConvSliceInfo slice);

  bool HasPaddingConstraint(ConvSliceInfo slice);

  bool HasStackedConvConstraint(ConvSliceInfo slice);

  double GenSlices();

  ConvSliceInfo PrepareSlices(int slice_ofm_w, int slice_ofm_h,
                              int slice_ifm_c);

protected:
  double ifm_ddr_timeV2(int slice_i, int slice_j, int slice_k, int slice_ifm_w,
                        int slice_ifm_h, int slice_ifm_c);

  double ifm_ddr_time(int slice_ifm_w, int slice_ifm_h, int slice_ifm_c);

  double weight_ddr_time(int slice_kernel_c, int num_kernels);

  double ofm_ddr_time(int slice_ofm_w, int slice_ofm_h, int slice_ofm_c);

  long useful_mac(int slice_ofm_w, int slice_ofm_h, int slice_ifm_c,
                  int slice_num_filters);

  long idle_mac(int slice_ifm_w, int slice_ifm_h, int slice_ifm_c,
                int slice_num_filters);

  double calculation_time(int slice_ofm_w, int slice_ofm_h, int slice_ifm_w,
                          int slice_ifm_h, int slice_ifm_c,
                          int slice_num_filters);

  ArchitectureInfo &info_arch() { return info_arch_; }

  Conv2dInfo &conv_info() { return conv_info_; }

  void SetConvDW() { is_dw_ = true; }

  bool isConvDW() { return (is_dw_) ? true : false; }

  bool hasBias() { return (has_bias_) ? true : false; }

protected:
  bool is_dw_;
  bool has_bias_;

private:
  int thread_ofm_h_;
  int thread_ofm_c_;
  ArchitectureInfo info_arch_;
  ConvSliceInfo data_out_;
  Conv2dInfo conv_info_;
};

class WeightStationaryOFM : public PartitionStrategy {
public:
  ~WeightStationaryOFM() override = default;

  WeightStationaryOFM(int thread_ofm_h, int thread_ofm_c,
                      const ArchitectureInfo &info, const Conv2dInfo &conv_info,
                      bool has_bias)
      : PartitionStrategy(thread_ofm_h, thread_ofm_c, info, conv_info,
                          has_bias) {}

  TLTStrategy strategy() override { return TLTStrategy::wstn_ofm; }

  double SoftwareTime(int slice_ofm_nc, int slice_ofm_ny, int slice_ofm_nx,
                      int slice_ifm_nc) override;

  double CalculateConvolutionTime(ConvSliceInfo &slice) override;

  int GetSliceNumFilters(int slice_ofm_w, int slice_ofm_h, int slice_ifm_c,
                         int total_filters_tlt) override;

  bool HasTLTStrategySpecificConstraint(ConvSliceInfo slice) override;
};

class IFMStationaryOFM : public PartitionStrategy {
public:
  IFMStationaryOFM(int thread_ofm_h, int thread_ofm_c,
                   const ArchitectureInfo &info, const Conv2dInfo &conv_info,
                   bool has_bias)
      : PartitionStrategy(thread_ofm_h, thread_ofm_c, info, conv_info,
                          has_bias) {}

  TLTStrategy strategy() override { return TLTStrategy::istn_ofm; }

  double SoftwareTime(int slice_ofm_nc, int slice_ofm_ny, int slice_ofm_nx,
                      int slice_ifm_nc) override;

  double CalculateConvolutionTime(ConvSliceInfo &slice) override;

  int GetSliceNumFilters(int slice_ofm_w, int slice_ofm_h, int slice_ifm_c,
                         int total_filters_tlt) override;

  bool HasTLTStrategySpecificConstraint(ConvSliceInfo slice) override;
};

class OFMStationaryIFM : public PartitionStrategy {
public:
  OFMStationaryIFM(int thread_ofm_h, int thread_ofm_c,
                   const ArchitectureInfo &info, const Conv2dInfo &conv_info,
                   bool has_bias)
      : PartitionStrategy(thread_ofm_h, thread_ofm_c, info, conv_info,
                          has_bias) {}

  TLTStrategy strategy() override { return TLTStrategy::ostn_ifm; }

  double SoftwareTime(int slice_ofm_nc, int slice_ofm_ny, int slice_ofm_nx,
                      int slice_ifm_nc) override;

  double CalculateConvolutionTime(ConvSliceInfo &slice) override;

  int GetSliceNumFilters(int slice_ofm_w, int slice_ofm_h, int slice_ifm_c,
                         int total_filters_tlt) override;

  bool HasTLTStrategySpecificConstraint(ConvSliceInfo slice) override;
};

class WeightStationaryOFMDW : public PartitionStrategy {
public:
  WeightStationaryOFMDW(int thread_ofm_h, int thread_ofm_c,
                        const ArchitectureInfo &info,
                        const Conv2dInfo &conv_info, bool has_bias)
      : PartitionStrategy(thread_ofm_h, thread_ofm_c, info, conv_info,
                          has_bias) {
    this->SetConvDW();
  }

  TLTStrategy strategy() override { return TLTStrategy::wstn_ofm_dw; }

  double SoftwareTime(int slice_ofm_nc, int slice_ofm_ny, int slice_ofm_nx,
                      int slice_ifm_nc) override;

  double CalculateConvolutionTime(ConvSliceInfo &slice) override;

  int GetSliceNumFilters(int slice_ofm_w, int slice_ofm_h, int filters_per_iter,
                         int total_filters_tlt) override;

  bool HasTLTStrategySpecificConstraint(ConvSliceInfo slice) override;
};

#endif
