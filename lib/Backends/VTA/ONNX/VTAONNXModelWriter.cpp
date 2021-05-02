
Error ONNXModelWriter::writeVTAConvolution(
    glow::VTAConvolutionNode const *, GraphType &graph) {
  return MAKE_ERR("Unsupported Op for ONNX");
}