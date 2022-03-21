/*****************************************************************************
	*
	* Copyright Next-Generation System Software Research Group, All rights reserved.
	* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
	* Electronics and Telecommunications Research Institute (ETRI)
	*
	* THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
	* WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
	* TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE"
	* REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
	"* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
	* IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
	* COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
	*
	* LICENSE file : README_LICENSE_ETRI located in the top directory
	*
*****************************************************************************/

#define VTAMAIN

#include "main_template.h"

#ifdef VTAMAIN
#include "VTARuntime.h"
#endif

#include "p0.h"
#include "p1.h"
#include "p2.h"
#include "p3.h"
#include "p4.h"
#include "p5.h"
#include "p6.h"
#include "p7.h"
#include "p8.h"
#include "p9.h"
#include "p10.h"
#include "p11.h"
#include "p12.h"
#include "p13.h"
#include "p14.h"
#include "p15.h"
#include "p16.h"
#include "p17.h"

#include <time.h>

int main(int argc, char **argv) {
	parseCommandLineOptions(argc, argv);

#ifdef VTAMAIN
	initVTARuntime();
#endif
	//======== initialize variables =======//

	uint8_t *constantWeightVarsAddr0 =
	    NULL;
//		initConstantWeights("p0.weights.bin", p0_config);
	uint8_t *mutableWeightVarsAddr0 = initMutableWeightVars(p0_config, "data");
	uint8_t *activationsAddr0 = initActivations(p0_config);

	uint8_t *constantWeightVarsAddr1 =
		initConstantWeights("p1.weights.bin", p1_config);
	uint8_t *activationsAddr1 = initActivations(p1_config);

	uint8_t *constantWeightVarsAddr2 =
	    //NULL;
		initConstantWeights("p2.weights.bin", p2_config);
	uint8_t *activationsAddr2 = initActivations(p2_config);

	uint8_t *constantWeightVarsAddr3 =
		initConstantWeights("p3.weights.bin", p3_config);
	uint8_t *activationsAddr3 = initActivations(p3_config);

	uint8_t *constantWeightVarsAddr4 =
	    //NULL;
		initConstantWeights("p4.weights.bin", p4_config);
	uint8_t *activationsAddr4 = initActivations(p4_config);

	uint8_t *constantWeightVarsAddr5 =
	    //NULL;
		initConstantWeights("p5.weights.bin", p5_config);
	uint8_t *activationsAddr5 = initActivations(p5_config);

	uint8_t *constantWeightVarsAddr6 =
		initConstantWeights("p6.weights.bin", p6_config);
	uint8_t *activationsAddr6 = initActivations(p6_config);

	uint8_t *constantWeightVarsAddr7 =
	    //NULL;
		initConstantWeights("p7.weights.bin", p7_config);
	uint8_t *activationsAddr7 = initActivations(p7_config);

	uint8_t *constantWeightVarsAddr8 =
		initConstantWeights("p8.weights.bin", p8_config);
	uint8_t *activationsAddr8 = initActivations(p8_config);

	uint8_t *constantWeightVarsAddr9 =
	    //NULL;
		initConstantWeights("p9.weights.bin", p9_config);
	uint8_t *activationsAddr9 = initActivations(p9_config);

	uint8_t *constantWeightVarsAddr10 =
		initConstantWeights("p10.weights.bin", p10_config);
	uint8_t *activationsAddr10 = initActivations(p10_config);

	uint8_t *constantWeightVarsAddr11 =
	    //NULL;
		initConstantWeights("p11.weights.bin", p11_config);
	uint8_t *activationsAddr11 = initActivations(p11_config);

	uint8_t *constantWeightVarsAddr12 =
		initConstantWeights("p12.weights.bin", p12_config);
	uint8_t *activationsAddr12 = initActivations(p12_config);

	uint8_t *constantWeightVarsAddr13 =
	    //NULL;
		initConstantWeights("p13.weights.bin", p13_config);
	uint8_t *activationsAddr13 = initActivations(p13_config);

	uint8_t *constantWeightVarsAddr14 =
		initConstantWeights("p14.weights.bin", p14_config);
	uint8_t *activationsAddr14 = initActivations(p14_config);

	uint8_t *constantWeightVarsAddr15 =
	    //NULL;
		initConstantWeights("p15.weights.bin", p15_config);
	uint8_t *activationsAddr15 = initActivations(p15_config);

	uint8_t *constantWeightVarsAddr16 =
		initConstantWeights("p16.weights.bin", p16_config);
	uint8_t *activationsAddr16 = initActivations(p16_config);

	uint8_t *constantWeightVarsAddr17 =
	    //NULL;
		initConstantWeights("p17.weights.bin", p17_config);
	uint8_t *activationsAddr17 = initActivations(p17_config);

	//======== initialize mutableWeightVarsAddr =======//

	uint8_t *mutableWeightVarsAddr1 =  allocateMutableWeightVars(p1_config);
	uint8_t *mutableWeightVarsAddr2 =  allocateMutableWeightVars(p2_config);
	uint8_t *mutableWeightVarsAddr3 =  allocateMutableWeightVars(p3_config);
	uint8_t *mutableWeightVarsAddr4 =  allocateMutableWeightVars(p4_config);
	uint8_t *mutableWeightVarsAddr5 =  allocateMutableWeightVars(p5_config);
	uint8_t *mutableWeightVarsAddr6 =  allocateMutableWeightVars(p6_config);
	uint8_t *mutableWeightVarsAddr7 =  allocateMutableWeightVars(p7_config);
	uint8_t *mutableWeightVarsAddr8 =  allocateMutableWeightVars(p8_config);
	uint8_t *mutableWeightVarsAddr9 =  allocateMutableWeightVars(p9_config);
	uint8_t *mutableWeightVarsAddr10 =  allocateMutableWeightVars(p10_config);
	uint8_t *mutableWeightVarsAddr11 =  allocateMutableWeightVars(p11_config);
	uint8_t *mutableWeightVarsAddr12 =  allocateMutableWeightVars(p12_config);
	uint8_t *mutableWeightVarsAddr13 =  allocateMutableWeightVars(p13_config);
	uint8_t *mutableWeightVarsAddr14 =  allocateMutableWeightVars(p14_config);
	uint8_t *mutableWeightVarsAddr15 =  allocateMutableWeightVars(p15_config);
	uint8_t *mutableWeightVarsAddr16 =  allocateMutableWeightVars(p16_config);
	uint8_t *mutableWeightVarsAddr17 =  allocateMutableWeightVars(p17_config);

    p0_load_module(0);
//    p2_load_module(0);
//    p4_load_module(0);
//    p5_load_module(0);
//    p7_load_module(0);
//    p9_load_module(0);
//    p11_load_module(0);
//    p13_load_module(0);
//    p15_load_module(0);
//    p17_load_module(0);
	p1_load_module(constantWeightVarsAddr1);
	p3_load_module(constantWeightVarsAddr3);
	p6_load_module(constantWeightVarsAddr6);
	p8_load_module(constantWeightVarsAddr8);
	p10_load_module(constantWeightVarsAddr10);
	p12_load_module(constantWeightVarsAddr12);
	p14_load_module(constantWeightVarsAddr14);
	p16_load_module(constantWeightVarsAddr16);

	//======== multiple partitions =======//

	clock_t start_total = 0;
	start_total = clock();
	p0(constantWeightVarsAddr0, mutableWeightVarsAddr0, activationsAddr0);

	float* resultVar0 = getInferenceResultVar(p0_config, mutableWeightVarsAddr0, "resnetv10_pool0_fwd__1");
	copyMutableWeightVarsWithoutAlloc(p1_config, mutableWeightVarsAddr1, "resnetv10_pool0_fwd__1", resultVar0);

	p1(constantWeightVarsAddr1, mutableWeightVarsAddr1, activationsAddr1);

	float* resultVar1 = getInferenceResultVar(p1_config, mutableWeightVarsAddr1, "resnetv10_stage1_conv1_fwd__2");
	copyMutableWeightVarsWithoutAlloc(p2_config, mutableWeightVarsAddr2, "resnetv10_pool0_fwd__1", resultVar0);
	copyMutableWeightVarsWithoutAlloc(p2_config, mutableWeightVarsAddr2, "resnetv10_stage1_conv1_fwd__2", resultVar1);

	p2(constantWeightVarsAddr2, mutableWeightVarsAddr2, activationsAddr2);

	float* resultVar2 = getInferenceResultVar(p2_config, mutableWeightVarsAddr2, "resnetv10_stage1_activation0__1");
	copyMutableWeightVarsWithoutAlloc(p3_config, mutableWeightVarsAddr3, "resnetv10_stage1_activation0__1", resultVar2);

	p3(constantWeightVarsAddr3, mutableWeightVarsAddr3, activationsAddr3);

	float* resultVar3 = getInferenceResultVar(p3_config, mutableWeightVarsAddr3, "resnetv10_stage1_conv3_fwd__2");
	copyMutableWeightVarsWithoutAlloc(p4_config, mutableWeightVarsAddr4, "resnetv10_stage1_activation0__1", resultVar2);
	copyMutableWeightVarsWithoutAlloc(p4_config, mutableWeightVarsAddr4, "resnetv10_stage1_conv3_fwd__2", resultVar3);

	p4(constantWeightVarsAddr4, mutableWeightVarsAddr4, activationsAddr4);

	float* resultVar4 = getInferenceResultVar(p4_config, mutableWeightVarsAddr4, "resnetv10_stage1_activation1__1");
	copyMutableWeightVarsWithoutAlloc(p5_config, mutableWeightVarsAddr5, "resnetv10_stage1_activation1__1", resultVar4);

	p5(constantWeightVarsAddr5, mutableWeightVarsAddr5, activationsAddr5);

	float* resultVar5 = getInferenceResultVar(p5_config, mutableWeightVarsAddr5, "resnetv10_stage2_conv2_fwd__2");
	copyMutableWeightVarsWithoutAlloc(p6_config, mutableWeightVarsAddr6, "resnetv10_stage1_activation1__1", resultVar4);

	p6(constantWeightVarsAddr6, mutableWeightVarsAddr6, activationsAddr6);

	float* resultVar6 = getInferenceResultVar(p6_config, mutableWeightVarsAddr6, "resnetv10_stage2_conv1_fwd__2");
	copyMutableWeightVarsWithoutAlloc(p7_config, mutableWeightVarsAddr7, "resnetv10_stage2_conv2_fwd__2", resultVar5);
	copyMutableWeightVarsWithoutAlloc(p7_config, mutableWeightVarsAddr7, "resnetv10_stage2_conv1_fwd__2", resultVar6);

	p7(constantWeightVarsAddr7, mutableWeightVarsAddr7, activationsAddr7);

	float* resultVar7 = getInferenceResultVar(p7_config, mutableWeightVarsAddr7, "resnetv10_stage2_activation0__1");
	copyMutableWeightVarsWithoutAlloc(p8_config, mutableWeightVarsAddr8, "resnetv10_stage2_activation0__1", resultVar7);

	p8(constantWeightVarsAddr8, mutableWeightVarsAddr8, activationsAddr8);

	float* resultVar8 = getInferenceResultVar(p8_config, mutableWeightVarsAddr8, "resnetv10_stage2_conv4_fwd__2");
	copyMutableWeightVarsWithoutAlloc(p9_config, mutableWeightVarsAddr9, "resnetv10_stage2_activation0__1", resultVar7);
	copyMutableWeightVarsWithoutAlloc(p9_config, mutableWeightVarsAddr9, "resnetv10_stage2_conv4_fwd__2", resultVar8);

	p9(constantWeightVarsAddr9, mutableWeightVarsAddr9, activationsAddr9);

	float* resultVar9 = getInferenceResultVar(p9_config, mutableWeightVarsAddr9, "resnetv10_stage2_activation1__1");
	copyMutableWeightVarsWithoutAlloc(p10_config, mutableWeightVarsAddr10, "resnetv10_stage2_activation1__1", resultVar9);

	p10(constantWeightVarsAddr10, mutableWeightVarsAddr10, activationsAddr10);

	float* resultVar10_1 = getInferenceResultVar(p10_config, mutableWeightVarsAddr10, "resnetv10_stage3_conv2_fwd__2");
	float* resultVar10_2 = getInferenceResultVar(p10_config, mutableWeightVarsAddr10, "resnetv10_stage3_conv1_fwd__2");
	copyMutableWeightVarsWithoutAlloc(p11_config, mutableWeightVarsAddr11, "resnetv10_stage3_conv2_fwd__2", resultVar10_1);
	copyMutableWeightVarsWithoutAlloc(p11_config, mutableWeightVarsAddr11, "resnetv10_stage3_conv1_fwd__2", resultVar10_2);

	p11(constantWeightVarsAddr11, mutableWeightVarsAddr11, activationsAddr11);

	float* resultVar11 = getInferenceResultVar(p11_config, mutableWeightVarsAddr11, "resnetv10_stage3_activation0__1");
	copyMutableWeightVarsWithoutAlloc(p12_config, mutableWeightVarsAddr12, "resnetv10_stage3_activation0__1", resultVar11);

	p12(constantWeightVarsAddr12, mutableWeightVarsAddr12, activationsAddr12);

	float* resultVar12 = getInferenceResultVar(p12_config, mutableWeightVarsAddr12, "resnetv10_stage3_conv4_fwd__2");
	copyMutableWeightVarsWithoutAlloc(p13_config, mutableWeightVarsAddr13, "resnetv10_stage3_activation0__1", resultVar11);
	copyMutableWeightVarsWithoutAlloc(p13_config, mutableWeightVarsAddr13, "resnetv10_stage3_conv4_fwd__2", resultVar12);

	p13(constantWeightVarsAddr13, mutableWeightVarsAddr13, activationsAddr13);

	float* resultVar13 = getInferenceResultVar(p13_config, mutableWeightVarsAddr13, "resnetv10_stage3_activation1__1");
	copyMutableWeightVarsWithoutAlloc(p14_config, mutableWeightVarsAddr14, "resnetv10_stage3_activation1__1", resultVar13);

	p14(constantWeightVarsAddr14, mutableWeightVarsAddr14, activationsAddr14);

	float* resultVar14_1 = getInferenceResultVar(p14_config, mutableWeightVarsAddr14, "resnetv10_stage4_conv2_fwd__2");
	float* resultVar14_2 = getInferenceResultVar(p14_config, mutableWeightVarsAddr14, "resnetv10_stage4_conv1_fwd__2");
	copyMutableWeightVarsWithoutAlloc(p15_config, mutableWeightVarsAddr15, "resnetv10_stage4_conv2_fwd__2", resultVar14_1);
	copyMutableWeightVarsWithoutAlloc(p15_config, mutableWeightVarsAddr15, "resnetv10_stage4_conv1_fwd__2", resultVar14_2);

	p15(constantWeightVarsAddr15, mutableWeightVarsAddr15, activationsAddr15);

	float* resultVar15 = getInferenceResultVar(p15_config, mutableWeightVarsAddr15, "resnetv10_stage4_activation0__1");
	copyMutableWeightVarsWithoutAlloc(p16_config, mutableWeightVarsAddr16, "resnetv10_stage4_activation0__1", resultVar15);

	p16(constantWeightVarsAddr16, mutableWeightVarsAddr16, activationsAddr16);

	float* resultVar16 = getInferenceResultVar(p16_config, mutableWeightVarsAddr16, "resnetv10_stage4_conv4_fwd__2");
	copyMutableWeightVarsWithoutAlloc(p17_config, mutableWeightVarsAddr17, "resnetv10_stage4_activation0__1", resultVar15);
	copyMutableWeightVarsWithoutAlloc(p17_config, mutableWeightVarsAddr17, "resnetv10_stage4_conv4_fwd__2", resultVar16);

	p17(constantWeightVarsAddr17, mutableWeightVarsAddr17, activationsAddr17);

	clock_t now_total;
	now_total = clock();
	unsigned long inftime_total = (unsigned long)(now_total - start_total);
	printf("\n====> [Total inference time] %lu microsec.\n", inftime_total);

	// Report the results.
	printf("====== final result of this neural net ========\n");
	int maxIdx = dumpInferenceResults(p17_config, mutableWeightVarsAddr17, "resnetv10_dense0_fwd");

	p1_destroy_module();
	p3_destroy_module();
	p6_destroy_module();
	p8_destroy_module();
	p10_destroy_module();
	p12_destroy_module();
	p14_destroy_module();
	p16_destroy_module();

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

	free(activationsAddr3);
	free(constantWeightVarsAddr3);
	free(mutableWeightVarsAddr3);

	free(activationsAddr4);
	free(constantWeightVarsAddr4);
	free(mutableWeightVarsAddr4);

	free(activationsAddr5);
	free(constantWeightVarsAddr5);
	free(mutableWeightVarsAddr5);

	free(activationsAddr6);
	free(constantWeightVarsAddr6);
	free(mutableWeightVarsAddr6);

	free(activationsAddr7);
	free(constantWeightVarsAddr7);
	free(mutableWeightVarsAddr7);

	free(activationsAddr8);
	free(constantWeightVarsAddr8);
	free(mutableWeightVarsAddr8);

	free(activationsAddr9);
	free(constantWeightVarsAddr9);
	free(mutableWeightVarsAddr9);

	free(activationsAddr10);
	free(constantWeightVarsAddr10);
	free(mutableWeightVarsAddr10);

	free(activationsAddr11);
	free(constantWeightVarsAddr11);
	free(mutableWeightVarsAddr11);

	free(activationsAddr12);
	free(constantWeightVarsAddr12);
	free(mutableWeightVarsAddr12);

	free(activationsAddr13);
	free(constantWeightVarsAddr13);
	free(mutableWeightVarsAddr13);

	free(activationsAddr14);
	free(constantWeightVarsAddr14);
	free(mutableWeightVarsAddr14);

	free(activationsAddr15);
	free(constantWeightVarsAddr15);
	free(mutableWeightVarsAddr15);

	free(activationsAddr16);
	free(constantWeightVarsAddr16);
	free(mutableWeightVarsAddr16);

	free(activationsAddr17);
	free(constantWeightVarsAddr17);
	free(mutableWeightVarsAddr17);

#ifdef VTAMAIN
	destroyVTARuntime();
#endif
	return !(maxIdx==281);

}

