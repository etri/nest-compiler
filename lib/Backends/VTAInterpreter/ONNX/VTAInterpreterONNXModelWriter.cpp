
Error ONNXModelWriter::writeVTAInterpreterConvolution(
    glow::VTAInterpreterConvolutionNode const *, GraphType &graph) {
  return MAKE_ERR("Unsupported Op for ONNX");
}