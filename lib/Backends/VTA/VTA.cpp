#include "VTA.h"
#include "VTAFunction.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include "VTACodeGen/VTASaver.h"
#include <vector>
namespace glow {
    namespace runtime {
        extern unsigned GlowVTAMemory;
    }
} // namespace glow
using namespace glow;

Expected<std::unique_ptr<CompiledFunction>>
VTA::compile(Function *F, const BackendOptions &opts) const {
    TraceInfo traceInfo = buildManualTraceInfo(F);
    auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

    if (!opts.backendSpecificOpts.empty()) {
        parseBackendSpecificOptions(opts);
    }

    if (opts.autoInstrument) {
        autoInstrument(traceInfo, IR.get());
    }

    std::unique_ptr<CompiledFunction> compiledFunc;
    if (opts.collectConstants) {
        compiledFunc = compileIR(std::move(IR));
    } else {
        compiledFunc = compileIRWithoutConstants(std::move(IR));
    }

    compiledFunc->setTraceInfo(std::move(traceInfo));
    return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}

std::unique_ptr<CompiledFunction>
VTA::compileIR(std::unique_ptr<IRFunction> IR) const {
    auto *mod = IR->getGraph()->getParent();
    auto function = compileIRWithoutConstants(std::move(IR));
    auto IFunction = static_cast<VTAFunction *>(function.get());
    IFunction->collectConstants(mod);
    return function;
}

std::unique_ptr<CompiledFunction>
VTA::compileIRWithoutConstants(std::unique_ptr<IRFunction> IR) const {
    MemoryAllocator constantWeightsAllocator("ConstantWeights", 0);
    MemoryAllocator placeholderWeightsAllocator("PlaceholderWeights", 0);
    MemoryAllocator activationsAllocator("Activations", 0);
    runtime::RuntimeBundle bundle = runtime::RuntimeBundle::create(
            *IR, constantWeightsAllocator, placeholderWeightsAllocator,
            activationsAllocator);
    auto compiledFunction =
            glow::make_unique<VTAFunction>(std::move(IR), std::move(bundle));
    compiledFunction->setIRInstructionProcessingHandler(
            getIRInstructionProcessingHandler());
    return compiledFunction;
}

bool VTA::isOpSupported(const NodeInfo &NI) const {
    switch (NI.getKind()) {
      case Kinded::Kind::VTAConvolutionNodeKind:
        return true;
        case Kinded::Kind::BatchedReduceMinNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int32ITy,
                     ElemKind::Int64ITy});

        case Kinded::Kind::AddNodeKind:
        case Kinded::Kind::SubNodeKind:
        case Kinded::Kind::MulNodeKind:
        case Kinded::Kind::MaxNodeKind:
        case Kinded::Kind::MinNodeKind:
        case Kinded::Kind::ReluNodeKind:
        case Kinded::Kind::ClipNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                     ElemKind::Int32ITy, ElemKind::Int64ITy});

        case Kinded::Kind::ResizeNearestNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                     ElemKind::Int16QTy, ElemKind::Int32QTy, ElemKind::Int32ITy,
                     ElemKind::Int64ITy});
        case Kinded::Kind::ResizeBilinearNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                     ElemKind::Int16QTy, ElemKind::Int32QTy, ElemKind::Int32ITy,
                     ElemKind::Int64ITy});

        case Kinded::Kind::AvgPoolNodeKind:
        case Kinded::Kind::AdaptiveAvgPoolNodeKind:
        case Kinded::Kind::BatchedReduceAddNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});

        case Kinded::Kind::MatMulNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                     ElemKind::Int16QTy});

        case Kinded::Kind::FullyConnectedNodeKind:
            if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
                return NI.allInputsAndOutputsHaveSameElemKind(
                        {ElemKind::FloatTy, ElemKind::Float16Ty});
            }
            return (NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::Int8QTy}, {FullyConnectedNode::BiasIdx}) &&
                    (NI.getInElemTy(FullyConnectedNode::BiasIdx) == ElemKind::Int8QTy ||
                     NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                     ElemKind::Int32QTy ||
                     NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                     ElemKind::FloatTy)) ||
                   (NI.allInputsAndOutputsHaveSameElemKind(
                           {ElemKind::Int16QTy}, {FullyConnectedNode::BiasIdx}) &&
                    (NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                     ElemKind::Int16QTy ||
                     NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                     ElemKind::Int32QTy));

        case Kinded::Kind::MaxPoolNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
                    {MaxPoolNode::ArgmaxIdx}) &&
                   (NI.getOutElemTy(MaxPoolNode::ArgmaxIdx) == ElemKind::Int64ITy);

        case Kinded::Kind::ArgMaxNodeKind:
        case Kinded::Kind::ArgMinNodeKind:
          return NI.allInputsAndOutputsHaveSameElemKind(
              {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
              {ArgMaxNode::ResultIdx}) &&
              (NI.getOutElemTy(ArgMaxNode::ResultIdx) == ElemKind::Int64ITy ||
                  NI.getOutElemTy(ArgMaxNode::ResultIdx) == ElemKind::Int32ITy);

        case Kinded::Kind::PowNodeKind:
        case Kinded::Kind::LocalResponseNormalizationNodeKind:
        case Kinded::Kind::LogNodeKind:
        case Kinded::Kind::TanhNodeKind:
        case Kinded::Kind::ExpNodeKind:
        case Kinded::Kind::SigmoidNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty});
        case Kinded::Kind::SliceNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                     ElemKind::Int32QTy, ElemKind::Int64ITy, ElemKind::Int32ITy});
        case Kinded::Kind::DivNodeKind:
        case Kinded::Kind::SpaceToDepthNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                     ElemKind::Int64ITy});

        case Kinded::Kind::SplatNodeKind:
        case Kinded::Kind::InsertTensorNodeKind:
        case Kinded::Kind::ConcatNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                     ElemKind::Int32ITy, ElemKind::Int64ITy, ElemKind::BoolTy});

        case Kinded::Kind::SelectNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
                    {SelectNode::CondIdx}) &&
                   (NI.getInElemTy(SelectNode::CondIdx) == ElemKind::BoolTy);

        case Kinded::Kind::CmpLTENodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
                    {CmpLTENode::ResultIdx}) &&
                   (NI.getOutElemTy(CmpLTENode::ResultIdx) == ElemKind::BoolTy);

        case Kinded::Kind::CmpLTNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                     ElemKind::Int32ITy, ElemKind::Int64ITy},
                    {}, {CmpLTNode::ResultIdx}) &&
                   (NI.getOutElemTy(CmpLTNode::ResultIdx) == ElemKind::BoolTy);

        case Kinded::Kind::IsNaNNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty}, {},
                    {CmpLTENode::ResultIdx}) &&
                   (NI.getOutElemTy(CmpLTENode::ResultIdx) == ElemKind::BoolTy);

        case Kinded::Kind::CmpEQNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int32ITy,
                     ElemKind::Int64ITy},
                    {}, {CmpEQNode::ResultIdx}) &&
                   (NI.getOutElemTy(CmpEQNode::ResultIdx) == ElemKind::BoolTy);

        case Kinded::Kind::ModuloNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::Int32ITy, ElemKind::Int64ITy});

        case Kinded::Kind::ConvolutionNodeKind:
            if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
                return NI.allInputsAndOutputsHaveSameElemKind(
                        {ElemKind::FloatTy, ElemKind::Float16Ty});
            }
            return (NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::Int8QTy}, {ConvolutionNode::BiasIdx}) &&
                    (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int8QTy ||
                     NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy)) ||
                   (NI.allInputsAndOutputsHaveSameElemKind(
                           {ElemKind::Int16QTy}, {ConvolutionNode::BiasIdx}) &&
                    (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int16QTy ||
                     NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy));
/*
        case Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind:
            return (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::InputIdx) ==
                    ElemKind::Int8QTy) &&
                   (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::FilterIdx) ==
                    ElemKind::Int8QTy) &&
                   (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::BiasIdx) ==
                    ElemKind::Int32QTy) &&
                   (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::ScalesIdx) ==
                    ElemKind::FloatTy) &&
                   (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::OffsetsIdx) ==
                    ElemKind::Int32ITy) &&
                   (NI.getOutElemTy(ChannelwiseQuantizedConvolutionNode::ResultIdx) ==
                    ElemKind::Int8QTy);
*/
        case Kinded::Kind::Convolution3DNodeKind:
            if (!NI.getInTy(Convolution3DNode::InputIdx)->isQuantizedType()) {
                return NI.allInputsAndOutputsHaveSameElemKind(
                        {ElemKind::FloatTy, ElemKind::Float16Ty});
            }
            return (NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::Int8QTy}, {Convolution3DNode::BiasIdx}) &&
                    (NI.getInElemTy(Convolution3DNode::BiasIdx) == ElemKind::Int8QTy ||
                     NI.getInElemTy(Convolution3DNode::BiasIdx) ==
                     ElemKind::Int32QTy)) ||
                   (NI.allInputsAndOutputsHaveSameElemKind(
                           {ElemKind::Int16QTy}, {Convolution3DNode::BiasIdx}) &&
                    (NI.getInElemTy(Convolution3DNode::BiasIdx) == ElemKind::Int16QTy ||
                     NI.getInElemTy(Convolution3DNode::BiasIdx) == ElemKind::Int32QTy));

        case Kinded::Kind::ConvTransposeNodeKind:
            // TODO - support other types.
            return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

        case Kinded::Kind::BatchedAddNodeKind:
            if (!NI.getInTy(BatchedAddNode::BatchIdx)->isQuantizedType()) {
                return NI.allInputsAndOutputsHaveSameElemKind(
                        {ElemKind::FloatTy, ElemKind::Float16Ty});
            }
            return (NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::Int8QTy}, {BatchedAddNode::SliceIdx}) &&
                    (NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int8QTy ||
                     NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int32QTy)) ||
                   (NI.allInputsAndOutputsHaveSameElemKind(
                           {ElemKind::Int16QTy}, {BatchedAddNode::SliceIdx}) &&
                    (NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int16QTy ||
                     NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int32QTy));

        case Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind:
            return (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::InputIdx) ==
                    ElemKind::Int8QTy) &&
                   (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::WeightsIdx) ==
                    ElemKind::Int8QTy) &&
                   (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::ScalesIdx) ==
                    ElemKind::FloatTy) &&
                   (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::OffsetsIdx) ==
                    ElemKind::Int32ITy) &&
                   (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::BiasIdx) ==
                    ElemKind::Int8QTy ||
                    NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::BiasIdx) ==
                    ElemKind::Int32QTy) &&
                   (NI.getOutElemTy(RowwiseQuantizedFullyConnectedNode::ResultIdx) ==
                    ElemKind::Int8QTy);

        case Kinded::Kind::SparseLengthsSumNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
                    {SparseLengthsSumNode::IndicesIdx,
                     SparseLengthsSumNode::LengthsIdx}) &&
                   (NI.getInElemTy(SparseLengthsSumNode::IndicesIdx) ==
                    ElemKind::Int64ITy ||
                    NI.getInElemTy(SparseLengthsSumNode::IndicesIdx) ==
                    ElemKind::Int32ITy) &&
                   (NI.getInElemTy(SparseLengthsSumNode::LengthsIdx) ==
                    ElemKind::Int32ITy);

        case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
                    {SparseLengthsWeightedSumNode::IndicesIdx,
                     SparseLengthsWeightedSumNode::LengthsIdx}) &&
                   (NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
                    ElemKind::Int64ITy ||
                    NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
                    ElemKind::Int32ITy) &&
                   (NI.getInElemTy(SparseLengthsWeightedSumNode::LengthsIdx) ==
                    ElemKind::Int32ITy);

        case Kinded::Kind::EmbeddingBagNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty},
                    {EmbeddingBagNode::IndicesIdx, EmbeddingBagNode::OffsetsIdx}) &&
                   (NI.getInElemTy(EmbeddingBagNode::IndicesIdx) ==
                    ElemKind::Int64ITy) &&
                   (NI.getInElemTy(EmbeddingBagNode::OffsetsIdx) == ElemKind::Int64ITy);

        case Kinded::Kind::SparseLengthsWeightedSumGradNodeKind:
            // GradOfInputNamedIndicesIdx and GradOfInputNamedLengthsIdx do not need to
            // be checked because they are not used.
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy},
                    {SparseLengthsWeightedSumGradNode::IndicesIdx,
                     SparseLengthsWeightedSumGradNode::LengthsIdx},
                    {SparseLengthsWeightedSumGradNode::GradOfInputNamedIndicesIdx,
                     SparseLengthsWeightedSumGradNode::
                     GradOfInputNamedLengthsIdx}) &&
                   (NI.getInElemTy(SparseLengthsWeightedSumGradNode::IndicesIdx) ==
                    ElemKind::Int64ITy) &&
                   (NI.getInElemTy(SparseLengthsWeightedSumGradNode::LengthsIdx) ==
                    ElemKind::Int32ITy);

        case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty},
                    {RowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx,
                     RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx,
                     RowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx}) &&
                   (NI.getInElemTy(
                           RowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx) ==
                    ElemKind::UInt8QTy) &&
                   (NI.getInElemTy(
                           RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) ==
                    ElemKind::Int64ITy ||
                    NI.getInElemTy(
                            RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) ==
                    ElemKind::Int32ITy) &&
                   (NI.getInElemTy(
                           RowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx) ==
                    ElemKind::Int32ITy);

        case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind: {
            if (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::IndicesIdx) !=
                ElemKind::Int64ITy ||
                NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::OffsetsIdx) !=
                ElemKind::Int64ITy) {
                return false;
            }

            switch (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::DataIdx)) {
                case ElemKind::UInt4FusedFP16QTy:
                case ElemKind::UInt8FusedFP16QTy:
                    return (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::WeightsIdx) ==
                            ElemKind::Float16Ty) &&
                           (NI.getOutElemTy(EmbeddingBagByteRowwiseOffsetsNode::ResultIdx) ==
                            ElemKind::Float16Ty);
                case ElemKind::UInt8FusedQTy:
                    return (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::WeightsIdx) ==
                            ElemKind::FloatTy) &&
                           (NI.getOutElemTy(EmbeddingBagByteRowwiseOffsetsNode::ResultIdx) ==
                            ElemKind::FloatTy);
                default:
                    return false;
            }
        }

        case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind: {
            if ((NI.getInElemTy(
                    FusedRowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) !=
                 ElemKind::Int64ITy &&
                 NI.getInElemTy(
                         FusedRowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) !=
                 ElemKind::Int32ITy) ||
                NI.getInElemTy(
                        FusedRowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx) !=
                ElemKind::Int32ITy) {
                return false;
            }

            switch (NI.getInElemTy(
                    FusedRowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx)) {
                case ElemKind::UInt4FusedFP16QTy:
                case ElemKind::UInt8FusedFP16QTy:
                    return (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                           WeightsIdx) == ElemKind::Float16Ty ||
                            NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                           WeightsIdx) == ElemKind::FloatTy) &&
                           (NI.getOutElemTy(
                                   FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                   ResultIdx) == ElemKind::Float16Ty);
                case ElemKind::UInt8FusedQTy:
                    if ((NI.getInElemTy(
                            FusedRowwiseQuantizedSparseLengthsWeightedSumNode::WeightsIdx) ==
                         ElemKind::FloatTy) &&
                        (NI.getOutElemTy(
                                FusedRowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx) ==
                         ElemKind::FloatTy)) {
                        return true;
                    }
                    return (
                            (NI.getInElemTy(
                                    FusedRowwiseQuantizedSparseLengthsWeightedSumNode::WeightsIdx) ==
                             ElemKind::Float16Ty) &&
                            (NI.getOutElemTy(
                                    FusedRowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx) ==
                             ElemKind::Float16Ty));
                default:
                    return false;
            }
        }

        case Kinded::Kind::LengthsRangeFillNodeKind:
        case Kinded::Kind::LengthsToRangesNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});

        case Kinded::Kind::GatherNodeKind:
            // Note: Data and Result can be any data type, but must match.
            return (NI.getInElemTy(GatherNode::DataIdx) ==
                    NI.getOutElemTy(GatherNode::ResultIdx)) &&
                   ((NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int32ITy) ||
                    (NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int64ITy));

        case Kinded::Kind::BatchOneHotNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                     ElemKind::Int32ITy, ElemKind::Int64ITy},
                    {BatchOneHotNode::LengthsIdx}) &&
                   (NI.getInElemTy(BatchOneHotNode::LengthsIdx) == ElemKind::Int32ITy);

        case Kinded::Kind::QuantizationProfileNodeKind:
        case Kinded::Kind::AvgPoolGradNodeKind:
        case Kinded::Kind::AdaptiveAvgPoolGradNodeKind:
        case Kinded::Kind::LocalResponseNormalizationGradNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

        case Kinded::Kind::MaxPoolGradNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy},
                    {MaxPoolGradNode::OriginalOutputForArgmaxIdx,
                     MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx}) &&
                   (NI.getInElemTy(MaxPoolGradNode::OriginalOutputForArgmaxIdx) ==
                    ElemKind::Int64ITy) &&
                   (NI.getInElemTy(
                           MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx) ==
                    ElemKind::Int64ITy);

        case Kinded::Kind::QuantizeNodeKind:
            return ((NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::FloatTy) ||
                    (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::Float16Ty)) &&
                   ((NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int8QTy) ||
                    (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::UInt8QTy) ||
                    (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int16QTy) ||
                    (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int32QTy));

        case Kinded::Kind::DequantizeNodeKind:
            return ((NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int8QTy) ||
                    (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::UInt8QTy) ||
                    (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int16QTy) ||
                    (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int32QTy)) &&
                   ((NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::FloatTy) ||
                    (NI.getOutElemTy(DequantizeNode::ResultIdx) ==
                     ElemKind::Float16Ty));

        case Kinded::Kind::RescaleQuantizedNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::Int8QTy, ElemKind::Int16QTy, ElemKind::Int32QTy});

        case Kinded::Kind::IntLookupTableNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy});

        case Kinded::Kind::ConvertToNodeKind: {
            auto isConversionSupportedFor = [](ElemKind kind) {
                switch (kind) {
                    case ElemKind::Float16Ty:
                    case ElemKind::FloatTy:
                    case ElemKind::Int32ITy:
                    case ElemKind::Int64ITy:
                    case ElemKind::BoolTy:
                        return true;
                    default:
                        return false;
                }
            };
            return isConversionSupportedFor(NI.getInElemTy(ConvertToNode::InputIdx)) &&
                   isConversionSupportedFor(NI.getOutElemTy(ConvertToNode::ResultIdx));
        }

        case Kinded::Kind::TopKNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
                    {TopKNode::IndicesIdx}) &&
                   ((NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int64ITy) ||
                    (NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int32ITy));

        case Kinded::Kind::ScatterDataNodeKind:
            return (NI.getInElemTy(ScatterDataNode::IndicesIdx) ==
                    ElemKind::Int64ITy) &&
                   (NI.getOutElemTy(ScatterDataNode::ResultIdx) ==
                    NI.getInElemTy(ScatterDataNode::DataIdx)) &&
                   (NI.getOutElemTy(ScatterDataNode::ResultIdx) ==
                    NI.getInElemTy(ScatterDataNode::SlicesIdx));

            // We just clip 64 to 32 SelectedIdx silently with the SoftMax
            // SelectedIdx in case dim_t is 32b.
        case Kinded::Kind::SoftMaxNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty},
                    {SoftMaxNode::SelectedIdx}) &&
                   (NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int32ITy ||
                    NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int64ITy);

        case Kinded::Kind::GatherRangesNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::Int32ITy, ElemKind::Int64ITy},
                    {GatherRangesNode::DataIdx}, {GatherRangesNode::OutputIdx}) &&
                   (NI.getOutElemTy(GatherRangesNode::OutputIdx) ==
                    NI.getInElemTy(GatherRangesNode::DataIdx));

        case Kinded::Kind::CrossEntropyLossNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty},
                    {CrossEntropyLossNode::LabelsIdx}) &&
                   (NI.getInElemTy(CrossEntropyLossNode::LabelsIdx) ==
                    ElemKind::Int64ITy);

        case Kinded::Kind::CumSumNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int32ITy,
                     ElemKind::Int64ITy});

        case Kinded::Kind::LengthsSumNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty},
                    {LengthsSumNode::LengthsIdx}) &&
                   (NI.getInElemTy(LengthsSumNode::LengthsIdx) == ElemKind::Int32ITy);

        case Kinded::Kind::SparseToDenseNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int64ITy},
                    {SparseToDenseNode::IndicesIdx}) &&
                   (NI.getInElemTy(SparseToDenseNode::IndicesIdx) ==
                    ElemKind::Int64ITy);

        case Kinded::Kind::SparseToDenseMaskNodeKind:
            return (NI.getInElemTy(SparseToDenseMaskNode::IndicesIdx) ==
                    ElemKind::Int64ITy) &&
                   (NI.getInElemTy(SparseToDenseMaskNode::LengthsIdx) ==
                    ElemKind::Int32ITy) &&
                   (NI.getInElemTy(SparseToDenseMaskNode::ValuesIdx) ==
                    NI.getInElemTy(SparseToDenseMaskNode::DefaultValueIdx)) &&
                   (NI.getInElemTy(SparseToDenseMaskNode::ValuesIdx) ==
                    NI.getOutElemTy(SparseToDenseMaskNode::ResultIdx));

        case Kinded::Kind::TraceEventNodeKind:
            return NI.getInElemTy(TraceEventNode::DataIdx) == ElemKind::Int64ITy;

        case Kinded::Kind::TransposeNodeKind:
        case Kinded::Kind::ReshapeNodeKind:
        case Kinded::Kind::SaveNodeKind:
        case Kinded::Kind::FlipNodeKind:
            // These work regardless of the underlying type.
            return true;

        case Kinded::Kind::NonMaxSuppressionNodeKind:
            return NI.getInElemTy(NonMaxSuppressionNode::BoxesIdx) ==
                   ElemKind::FloatTy &&
                   NI.getInElemTy(NonMaxSuppressionNode::ScoresIdx) ==
                   ElemKind::FloatTy &&
                   (NI.getOutElemTy(NonMaxSuppressionNode::IndicesIdx) ==
                    ElemKind::Int32ITy ||
                    NI.getOutElemTy(NonMaxSuppressionNode::IndicesIdx) ==
                    ElemKind::Int64ITy) &&
                   (NI.getOutElemTy(
                           NonMaxSuppressionNode::NumberOfSelectedIndicesIdx) ==
                    ElemKind::Int32ITy ||
                    NI.getOutElemTy(
                            NonMaxSuppressionNode::NumberOfSelectedIndicesIdx) ==
                    ElemKind::Int64ITy) &&
                   (NI.getOutElemTy(
                           NonMaxSuppressionNode::NumberOfSelectedIndicesIdx) ==
                    NI.getOutElemTy(NonMaxSuppressionNode::IndicesIdx));

        case Kinded::Kind::AudioSpectrogramNodeKind:
            return NI.getInElemTy(AudioSpectrogramNode::InputIdx) ==
                   ElemKind::FloatTy &&
                   NI.getOutElemTy(AudioSpectrogramNode::SpectrogramIdx) ==
                   ElemKind::FloatTy;

        case Kinded::Kind::MFCCNodeKind:
            return NI.getInElemTy(MFCCNode::SpectrogramIdx) == ElemKind::FloatTy &&
                   NI.getOutElemTy(MFCCNode::CoefficientsIdx) == ElemKind::FloatTy;

        case Kinded::Kind::SoftMaxGradNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy}, {SoftMaxGradNode::SelectedIdx},
                    {SoftMaxGradNode::GradOfInputNamedSelectedIdx}) &&
                   (NI.getInElemTy(SoftMaxGradNode::SelectedIdx) == ElemKind::Int64ITy);

        case Kinded::Kind::ConvolutionGradNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy}, {},
                    {ConvolutionGradNode::GradOfInputNamedInputIdx});

        case Kinded::Kind::CrossEntropyLossGradNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind(
                    {ElemKind::FloatTy}, {CrossEntropyLossGradNode::LabelsIdx},
                    {CrossEntropyLossGradNode::GradOfInputNamedLabelsIdx}) &&
                   (NI.getInElemTy(CrossEntropyLossGradNode::LabelsIdx) ==
                    ElemKind::Int64ITy) &&
                   (NI.getOutElemTy(
                           CrossEntropyLossGradNode::GradOfInputNamedLabelsIdx) ==
                    ElemKind::Int64ITy);

        case Kinded::Kind::BatchedPairwiseDotProductNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

        case Kinded::Kind::BatchedPairwiseDotProductGradNodeKind:
            return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

        default:
            return false;
    }
}

/// Use template meta-programming to check if typename ClassName contains
/// has_getLayout() method. Below generates a struct named has_getLayout that
/// looks for said method.
CLASS_CONTAINS_METHOD(getLayout)

template <typename T, std::enable_if_t<
        !has_getLayout<T, ConvolutionLayout>::value, int> = 0>
static bool checkLayout(const T &I) {
    (void)I;
    return true;
}

template <typename T,
        std::enable_if_t<has_getLayout<T, ConvolutionLayout>::value, int> = 0>
static bool checkLayout(const T &I) {
    if (I.getLayout() != NHWC) {
        report("Glow VTA supports only NHWC");
        return false;
    }
    return true;
}

static bool checkLayoutForNode(const Node &N) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case Kinded::Kind::CLASS##Kind: {                                            \
    const CLASS *CI = llvm::cast<CLASS>(&N);                                   \
    return checkLayout(*CI);                                                   \
    break;                                                                     \
  }
    switch (N.getKind()) {
#include "glow/AutoGenNodes.def"
        default:
            llvm_unreachable("Invalid instruction.");
    }
    return true;
}

bool VTA::verify(const Function &F, bool verbose) const {
    if (!F.verify(this)) {
        return false;
    }
    if (!checkAllNodesSupported(F, verbose)) {
        return false;
    }
    for (const Node &N : F.getNodes()) {
        if (!checkLayoutForNode(N)) {
            return false;
        }
        if (!(N.getKind() == Kinded::Kind::ConvolutionNodeKind &&
              llvm::cast<ConvolutionNode>(&N)->getFusedActivation() ==
              FusedActivation::RELU) &&
              !(N.getKind() == Kinded::Kind::VTAConvolutionNodeKind &&
              llvm::cast<VTAConvolutionNode>(&N)->getFusedActivation() ==
              FusedActivation::RELU) &&
            !checkNoFusionForNode(N)) {
          return false;
        }
    }
    return true;
}

static bool checkLayoutForInstr(const Instruction &I) {
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    const CLASS *CI = llvm::cast<CLASS>(&I);                                   \
    return checkLayout(*CI);                                                   \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
    switch (I.getKind()) {
#include "glow/AutoGenInstr.def"
        default:
            llvm_unreachable("Invalid instruction.");
    }
    return true;
}

bool VTA::verify(const IRFunction &IR) const {
    for (const auto &I : IR.getInstrs()) {
        // Only support convolution+relu fusions for now.
        if (!(I.getKind() == Kinded::Kind::ConvolutionInstKind &&
              llvm::cast<ConvolutionInst>(&I)->getFusedActivation() ==
              FusedActivation::RELU) &&
            !(I.getKind() == Kinded::Kind::VTAConvolutionInstKind &&
              llvm::cast<VTAConvolutionInst>(&I)->getFusedActivation() ==
              FusedActivation::RELU) &&
            !checkNoFusionForInstr(I)) {
          return false;
        }

        if (I.getKind() == Kinded::Kind::VTAConvolutionInstKind){
          continue;
        }

        if (!checkLayoutForInstr(I)) {
            return false;
        }
    }
    return true;
}

bool VTA::shouldLower(const Node *N) const {
    switch (N->getKind()) {
        case Kinded::Kind::ConvolutionNodeKind:
        case Kinded::Kind::Convolution3DNodeKind:
        case Kinded::Kind::SparseLengthsSumNodeKind:
        case Kinded::Kind::FullyConnectedNodeKind:
        case Kinded::Kind::ReluNodeKind:
          return false;
        default:
            return true;
    }
}

/// Quantize the given float \p bias as int32 using \p inputScale,
/// weight \p scales and \p offset=0. \returns false if the bias was already
/// quantized and thus no change was made and true otherwise.
static bool quantizeFloatBias(Function *F, FullyConnectedNode &fullyConnected) {
    if (fullyConnected.getBias().getType()->isQuantizedType() ||
        (!fullyConnected.getWeights().getType()->isQuantizedType())) {
        return false;
    }
    assert(fullyConnected.getBias().getElementType() == ElemKind::FloatTy &&
           "Bias type must be a float in order to quantize it.");
    Constant *biasC =
            llvm::dyn_cast<Constant>(fullyConnected.getBias().getNode());
    assert(biasC && "bias input to ChannelwiseQuantizedConvolutionNode "
                    "must be a Constant in order to quantize the bias");
    const auto &biasUnquantizedH = biasC->getPayload().getHandle<float>();
    // biasQuantizedT is Int32QTy
    const float inputScale = fullyConnected.getInput().getType()->getScale();
    const float weigthScale = fullyConnected.getWeights().getType()->getScale();
    const float scale = inputScale * weigthScale;
    auto biasQuantizedT = Tensor(ElemKind::Int32QTy, biasUnquantizedH.dims(),
            /* scale */ scale, /* offset */ 0);
    auto biasQuantizedH = biasQuantizedT.getHandle<int32_t>();
    TensorQuantizationParams tqp;
    tqp.scale = scale;
    tqp.offset = 0;
    for (dim_t i = 0; i < biasQuantizedH.size(); ++i) {
        biasQuantizedH.raw(i) =
                quantization::quantize<int32_t>(biasUnquantizedH.raw(i), tqp);
    }
    auto biasQuantizedC = F->getParent()->createConstant(
            biasC->getName(), std::move(biasQuantizedT));
    auto newFullyConnectedNode = F->createFullyConnected(
            fullyConnected.getName(), fullyConnected.getInput(),
            fullyConnected.getWeights(), biasQuantizedC,
            fullyConnected.getResult().getType(), /* axis doens't matter */ 1);
    fullyConnected.getResult().replaceAllUsesOfWith(newFullyConnectedNode);
    return true;
}

/// Channelwise quantize the given float \p bias as int32 using \p inputScale,
/// per-channel \p scales and \p offsets. \returns false if the bias was already
/// quantized and thus no change was made and true otherwise.
static bool channelwiseQuantizeFloatBias(
        Function *F, ChannelwiseQuantizedConvolutionNode &channelwiseConv) {
    return false;
}

Expected<bool> VTA::transformPostLowering(
        Function *F, CompilationContext &cctx,
        const glow::runtime::DeviceInfo *devInfo) const {
    LOG_SCOPE(F->getLogContext(), "VTA::transformPostLowering")

    bool changed = false;
    for (auto &node : F->getNodes()) {
        if (auto *channelwiseConv =
                llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(&node)) {
            changed |= channelwiseQuantizeFloatBias(F, *channelwiseConv);
        } else if (auto *fullyConnected =
                llvm::dyn_cast<FullyConnectedNode>(&node)) {
            changed |= quantizeFloatBias(F, *fullyConnected);
        }
    }
    return changed;
}

void VTA::parseBackendSpecificOptions(
        const BackendOptions &opts) const {
    auto vtaMaxMemOpt =
            opts.backendSpecificOpts.find("vta-memory");
    if (vtaMaxMemOpt != opts.backendSpecificOpts.end()) {
        glow::runtime::GlowVTAMemory =
                std::stoi(vtaMaxMemOpt->second);
        llvm::outs() << "VTA memory set to "
                     << glow::runtime::GlowVTAMemory << "\n";
    }
}
