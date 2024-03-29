/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Modifications copyright (C) 2022 <ETRI/Yongin Kwon>
 */

BB.newNode("VTAInterpreterConvolution")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads", /* addSetter */ true)
    .addMember(MemberType::Unsigned, "Group", /* addSetter */ true)
    .addMember(MemberType::VectorUnsigned, "Dilation")
    .addMember(MEMBER_TYPE_INFO(glow::ConvolutionLayout), "Layout")
    .addFusedActivation()
    .addResultFromCtorArg()
    // .addGradient()
    .setDocstring("This is a VTA-specific convolution node that is "
                  "identical to the normal ConvolutionNode. That node "
                  "and convolution + relu are replaced with this one "
                  "for backend-specific code generation");

BB.includeBackendSpecificVerification(
    "glow/VTAInterpreterSpecificNodesVerification.h");
