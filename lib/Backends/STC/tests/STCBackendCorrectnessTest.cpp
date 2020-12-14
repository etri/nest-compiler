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
 */
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {
        // Interpreter does not support kernel stacking yet.
        "dataParallelStackingTest/0",
        "extract3Dtest/0",
        "convGradTest/0",
        "localResponseNormalizationTest/0",
        "localResponseNormalizationGradTest/0",
        "dataParallelStackingTest/0",
        "AvgPoolGradTest/0",
        "MaxPoolGradTest/0",
        "intLookupTable/0",
        "smallConv/0",
        "groupConvTest/0",
        "nonSquarePaddingConvTest/0",
        "nonSquareKernelConvTest/0",
        "nonSquareStrideConvTest/0",
        "convDKKC8Test/0",
        "softmaxGradTest/0",
        "convOps/0",
        "basicFCNet/0",
        "basicFCNetQuantized/0",
        "complexNet1/0",
        "tinyResnet/0",
        "maxSplatTest/0"
};
