#include "main_template.h"

#include "p0.h"
#include "p1.h"
#include "p2.h"

#include <time.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

int main(int argc, char **argv) {
    parseCommandLineOptions(argc, argv);

    //======== initialize variables =======//

    uint8_t *constantWeightVarsAddr0 = NULL;
    uint8_t *mutableWeightVarsAddr0 = initMutableWeightVars(p0_config, "data");
    uint8_t *activationsAddr0 = initActivations(p0_config);

    uint8_t *constantWeightVarsAddr1 =
            initConstantWeights("p1.weights.bin", p1_config);
    uint8_t *activationsAddr1 = initActivations(p1_config);

    uint8_t *constantWeightVarsAddr2 =
            initConstantWeights("p2.weights.bin", p2_config);
    uint8_t *activationsAddr2 = initActivations(p2_config);

    //======== initialize mutableWeightVarsAddr =======//

    uint8_t *mutableWeightVarsAddr1 =  allocateMutableWeightVars(p1_config);
    uint8_t *mutableWeightVarsAddr2 =  allocateMutableWeightVars(p2_config);


    //======== multiple partitions =======//

    clock_t start_total = 0;
    start_total = clock();
    p0(constantWeightVarsAddr0, mutableWeightVarsAddr0, activationsAddr0);

    float* resultVar0 = getInferenceResultVar(p0_config, mutableWeightVarsAddr0, "resnetv10_pool0_fwd__1");
    copyMutableWeightVarsWithoutAlloc(p1_config, mutableWeightVarsAddr1, "resnetv10_pool0_fwd__1", resultVar0);

    p1(constantWeightVarsAddr1, mutableWeightVarsAddr1, activationsAddr1);

    float* resultVar1 = getInferenceResultVar(p1_config, mutableWeightVarsAddr1, "resnetv10_stage1_conv0_fwd__2");
    copyMutableWeightVarsWithoutAlloc(p2_config, mutableWeightVarsAddr2, "resnetv10_pool0_fwd__1", resultVar0);
    copyMutableWeightVarsWithoutAlloc(p2_config, mutableWeightVarsAddr2, "resnetv10_stage1_conv0_fwd__2", resultVar1);

    p2(constantWeightVarsAddr2, mutableWeightVarsAddr2, activationsAddr2);

    clock_t now_total;
    now_total = clock();
    unsigned long inftime_total = (unsigned long)(now_total - start_total);
    printf("\n====> [Total inference time] %lu microsec.\n", inftime_total);

    // Report the results.
    printf("====== final result of this neural net ========\n");
    dumpInferenceResults(p2_config, mutableWeightVarsAddr2, "resnetv10_dense0_fwd");


    // Free all resources.
    free(activationsAddr0);
    free(constantWeightVarsAddr0);
    free(mutableWeightVarsAddr0);

    free(activationsAddr1);
    free(constantWeightVarsAddr1);
    free(mutableWeightVarsAddr1);

    free(activationsAddr2);
    free(constantWeightVarsAddr2);
    free(mutableWeightVarsAddr2);

    return 0;
}
