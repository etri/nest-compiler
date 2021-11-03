struct ConvolutionTuneElement{
  unsigned_t inputN_;
  unsigned_t inputH_;
  unsigned_t inputW_;
  unsigned_t inputC_;
  unsigned_t filterN_;
  unsigned_t filterH_;
  unsigned_t filterW_;
  unsigned_t filterC_;
  unsigned_t stride_;
  unsigned_t pad_;
  unsigned_t nVirtualThread_;
  unsigned_t tileHSize_;
  unsigned_t tileWSize_;

  bool isMatch(unsigned_t inputN, unsigned_t inputH, unsigned_t inputW, unsigned_t inputC,
               unsigned_t filterN, unsigned_t filterH, unsigned_t filterW, unsigned_t filterC,
               unsigned_t stride, unsigned_t pad){
    if(inputN_==inputN && inputH_==inputH && inputW_==inputW && inputC_==inputC &&
        filterN_==filterN && filterH_==filterH && filterW_==filterW && filterC_==filterC &&
        stride_==stride && pad_==pad)
      return true;
    return false;
  }
};


struct ConvolutionTune{
  std::vector<ConvolutionTuneElement> ConvolutionTune_;
};

struct ConvolutionTune convTune;
convTune.ConvolutionTune_.clear();

//inputN, inputH, inputW, inputC, filterN, filterH, filterW, filterC, stride, pad, nVirtualThread, tileHSize, tileWSize
convTune.ConvolutionTune_.push_back({1, 56, 56, 64, 64, 3, 3, 64, 1, 1, 2, 8, 56});
convTune.ConvolutionTune_.push_back({1, 56, 56, 64, 128, 1, 1, 64, 2, 0, 2, 7, 28});
convTune.ConvolutionTune_.push_back({1, 56, 56, 64, 128, 3, 3, 64, 2, 1, 2, 8, 28});
convTune.ConvolutionTune_.push_back({1, 28, 28, 128, 128, 3, 3, 128, 1, 1, 2, 14, 28});
convTune.ConvolutionTune_.push_back({1, 28, 28, 128, 256, 1, 1, 128, 2, 0, 2, 7, 14});
convTune.ConvolutionTune_.push_back({1, 28, 28, 128, 256, 3, 3, 128, 2, 1, 2, 14, 14});
convTune.ConvolutionTune_.push_back({1, 14, 14, 256, 256, 3, 3, 256, 1, 1, 2, 14, 14});
convTune.ConvolutionTune_.push_back({1, 14, 14, 256, 512, 3, 3, 256, 2, 0, 2, 7, 7});
convTune.ConvolutionTune_.push_back({1, 14, 14, 256, 512, 1, 1, 256, 2, 0, 2, 7, 7});
convTune.ConvolutionTune_.push_back({1, 14, 14, 256, 512, 3, 3, 256, 2, 1, 2, 7, 7});
convTune.ConvolutionTune_.push_back({1, 7, 7, 512, 512, 3, 3, 512, 1, 1, 2, 7, 7});