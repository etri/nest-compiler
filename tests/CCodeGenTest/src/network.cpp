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

#include "network.h"

float (*varresnetv22_batchnorm0_gamma)[3];
float (*varresnetv22_batchnorm0_beta)[3];
float (*varresnetv22_batchnorm0_running_mean)[3];
float (*varresnetv22_batchnorm0_running_var)[3];
float (*varresnetv22_batchnorm0_fwd)[1][3][224][224];
float (*vardata)[1][3][224][224];
//BatchNormalization
unsigned int channelVar0;
float epsilonVar0;
float pad0;
float (*varresnetv22_conv0_weight)[64][3][7][7];
float (*varresnetv22_conv0_fwd)[64][7][7][3];
//Transpose
vector<size_t> shuffle225081392;
float (*varresnetv22_conv0_fwd__1)[1][224][224][3];
//Transpose
vector<size_t> shuffle225082800;
float (*varconv_bias)[64];
float (*varresnetv22_conv0_fwd__2)[1][112][112][64];
//Convolution
vector<size_t> filter225083184;
vector<size_t> kernel225083184;
vector<size_t> stride225083184;
vector<size_t> pad225083184;
size_t group225083184;
float (*varresnetv22_conv0_fwd__3)[1][64][112][112];
//Transpose
vector<size_t> shuffle225083600;
float (*varresnetv22_batchnorm1_gamma)[64];
float (*varresnetv22_batchnorm1_beta)[64];
float (*varresnetv22_batchnorm1_running_mean)[64];
float (*varresnetv22_batchnorm1_running_var)[64];
float (*varresnetv22_batchnorm1_fwd)[1][64][112][112];
//BatchNormalization
unsigned int channelVar1;
float epsilonVar1;
float pad1;
float (*varresnetv22_relu0_fwd)[1][64][112][112];
float (*varresnetv22_pool0_fwd)[1][112][112][64];
//Transpose
vector<size_t> shuffle225086048;
float (*varresnetv22_pool0_fwd__1)[1][56][56][64];
//MaxPool
vector<size_t> kernel225086496;
vector<size_t> stride225086496;
vector<size_t> pad225086496;
float (*varresnetv22_pool0_fwd__2)[1][64][56][56];
//Transpose
vector<size_t> shuffle225086896;
float (*varresnetv22_stage1_batchnorm0_gamma)[64];
float (*varresnetv22_stage1_batchnorm0_beta)[64];
float (*varresnetv22_stage1_batchnorm0_running_mean)[64];
float (*varresnetv22_stage1_batchnorm0_running_var)[64];
float (*varresnetv22_stage1_batchnorm0_fwd)[1][64][56][56];
//BatchNormalization
unsigned int channelVar2;
float epsilonVar2;
float pad2;
float (*varresnetv22_stage1_activation0)[1][64][56][56];
float (*varresnetv22_stage1_conv0_weight)[64][64][3][3];
float (*varresnetv22_stage1_conv0_fwd)[64][3][3][64];
//Transpose
vector<size_t> shuffle225089136;
float (*varresnetv22_stage1_conv0_fwd__1)[1][56][56][64];
//Transpose
vector<size_t> shuffle225080240;
float (*varconv_bias__1)[64];
float (*varresnetv22_stage1_conv0_fwd__2)[1][56][56][64];
//Convolution
vector<size_t> filter225091856;
vector<size_t> kernel225091856;
vector<size_t> stride225091856;
vector<size_t> pad225091856;
size_t group225091856;
float (*varresnetv22_stage1_conv0_fwd__3)[1][64][56][56];
//Transpose
vector<size_t> shuffle225092304;
float (*varresnetv22_stage1_batchnorm1_gamma)[64];
float (*varresnetv22_stage1_batchnorm1_beta)[64];
float (*varresnetv22_stage1_batchnorm1_running_mean)[64];
float (*varresnetv22_stage1_batchnorm1_running_var)[64];
float (*varresnetv22_stage1_batchnorm1_fwd)[1][64][56][56];
//BatchNormalization
unsigned int channelVar3;
float epsilonVar3;
float pad3;
float (*varresnetv22_stage1_activation1)[1][64][56][56];
float (*varresnetv22_stage1_conv1_weight)[64][64][3][3];
float (*varresnetv22_stage1_conv1_fwd)[64][3][3][64];
//Transpose
vector<size_t> shuffle225094320;
float (*varresnetv22_stage1_conv1_fwd__1)[1][56][56][64];
//Transpose
vector<size_t> shuffle225096096;
float (*varconv_bias__2)[64];
float (*varresnetv22_stage1_conv1_fwd__2)[1][56][56][64];
//Convolution
vector<size_t> filter225096368;
vector<size_t> kernel225096368;
vector<size_t> stride225096368;
vector<size_t> pad225096368;
size_t group225096368;
float (*varresnetv22_stage1_conv1_fwd__3)[1][64][56][56];
//Transpose
vector<size_t> shuffle225096784;
float (*varresnetv22_stage1__plus0)[1][64][56][56];
float (*varresnetv22_stage1_batchnorm2_gamma)[64];
float (*varresnetv22_stage1_batchnorm2_beta)[64];
float (*varresnetv22_stage1_batchnorm2_running_mean)[64];
float (*varresnetv22_stage1_batchnorm2_running_var)[64];
float (*varresnetv22_stage1_batchnorm2_fwd)[1][64][56][56];
//BatchNormalization
unsigned int channelVar4;
float epsilonVar4;
float pad4;
float (*varresnetv22_stage1_activation2)[1][64][56][56];
float (*varresnetv22_stage1_conv2_weight)[64][64][3][3];
float (*varresnetv22_stage1_conv2_fwd)[64][3][3][64];
//Transpose
vector<size_t> shuffle225090144;
float (*varresnetv22_stage1_conv2_fwd__1)[1][56][56][64];
//Transpose
vector<size_t> shuffle225101952;
float (*varconv_bias__3)[64];
float (*varresnetv22_stage1_conv2_fwd__2)[1][56][56][64];
//Convolution
vector<size_t> filter225102288;
vector<size_t> kernel225102288;
vector<size_t> stride225102288;
vector<size_t> pad225102288;
size_t group225102288;
float (*varresnetv22_stage1_conv2_fwd__3)[1][64][56][56];
//Transpose
vector<size_t> shuffle225102704;
float (*varresnetv22_stage1_batchnorm3_gamma)[64];
float (*varresnetv22_stage1_batchnorm3_beta)[64];
float (*varresnetv22_stage1_batchnorm3_running_mean)[64];
float (*varresnetv22_stage1_batchnorm3_running_var)[64];
float (*varresnetv22_stage1_batchnorm3_fwd)[1][64][56][56];
//BatchNormalization
unsigned int channelVar5;
float epsilonVar5;
float pad5;
float (*varresnetv22_stage1_activation3)[1][64][56][56];
float (*varresnetv22_stage1_conv3_weight)[64][64][3][3];
float (*varresnetv22_stage1_conv3_fwd)[64][3][3][64];
//Transpose
vector<size_t> shuffle225104784;
float (*varresnetv22_stage1_conv3_fwd__1)[1][56][56][64];
//Transpose
vector<size_t> shuffle225106608;
float (*varconv_bias__4)[64];
float (*varresnetv22_stage1_conv3_fwd__2)[1][56][56][64];
//Convolution
vector<size_t> filter225106944;
vector<size_t> kernel225106944;
vector<size_t> stride225106944;
vector<size_t> pad225106944;
size_t group225106944;
float (*varresnetv22_stage1_conv3_fwd__3)[1][64][56][56];
//Transpose
vector<size_t> shuffle225107360;
float (*varresnetv22_stage1__plus1)[1][64][56][56];
float (*varresnetv22_stage2_batchnorm0_gamma)[64];
float (*varresnetv22_stage2_batchnorm0_beta)[64];
float (*varresnetv22_stage2_batchnorm0_running_mean)[64];
float (*varresnetv22_stage2_batchnorm0_running_var)[64];
float (*varresnetv22_stage2_batchnorm0_fwd)[1][64][56][56];
//BatchNormalization
unsigned int channelVar6;
float epsilonVar6;
float pad6;
float (*varresnetv22_stage2_activation0)[1][64][56][56];
float (*varresnetv22_stage2_conv0_weight)[128][64][3][3];
float (*varresnetv22_stage2_conv0_fwd)[128][3][3][64];
//Transpose
vector<size_t> shuffle225109968;
float (*varresnetv22_stage2_conv0_fwd__1)[1][56][56][64];
//Transpose
vector<size_t> shuffle225112192;
float (*varconv_bias__5)[128];
float (*varresnetv22_stage2_conv0_fwd__2)[1][28][28][128];
//Convolution
vector<size_t> filter225112672;
vector<size_t> kernel225112672;
vector<size_t> stride225112672;
vector<size_t> pad225112672;
size_t group225112672;
float (*varresnetv22_stage2_conv0_fwd__3)[1][128][28][28];
//Transpose
vector<size_t> shuffle225113088;
float (*varresnetv22_stage2_batchnorm1_gamma)[128];
float (*varresnetv22_stage2_batchnorm1_beta)[128];
float (*varresnetv22_stage2_batchnorm1_running_mean)[128];
float (*varresnetv22_stage2_batchnorm1_running_var)[128];
float (*varresnetv22_stage2_batchnorm1_fwd)[1][128][28][28];
//BatchNormalization
unsigned int channelVar7;
float epsilonVar7;
float pad7;
float (*varresnetv22_stage2_activation1)[1][128][28][28];
float (*varresnetv22_stage2_conv1_weight)[128][128][3][3];
float (*varresnetv22_stage2_conv1_fwd)[128][3][3][128];
//Transpose
vector<size_t> shuffle225115312;
float (*varresnetv22_stage2_conv1_fwd__1)[1][28][28][128];
//Transpose
vector<size_t> shuffle225117504;
float (*varconv_bias__6)[128];
float (*varresnetv22_stage2_conv1_fwd__2)[1][28][28][128];
//Convolution
vector<size_t> filter225117840;
vector<size_t> kernel225117840;
vector<size_t> stride225117840;
vector<size_t> pad225117840;
size_t group225117840;
float (*varresnetv22_stage2_conv1_fwd__3)[1][128][28][28];
//Transpose
vector<size_t> shuffle225118256;
float (*varresnetv22_stage2_conv2_weight)[128][64][1][1];
float (*varresnetv22_stage2_conv2_fwd)[128][1][1][64];
//Transpose
vector<size_t> shuffle225115376;
float (*varresnetv22_stage2_conv2_fwd__1)[1][56][56][64];
//Transpose
vector<size_t> shuffle225100464;
float (*varconv_bias__7)[128];
float (*varresnetv22_stage2_conv2_fwd__2)[1][28][28][128];
//Convolution
vector<size_t> filter225122880;
vector<size_t> kernel225122880;
vector<size_t> stride225122880;
vector<size_t> pad225122880;
size_t group225122880;
float (*varresnetv22_stage2_conv2_fwd__3)[1][128][28][28];
//Transpose
vector<size_t> shuffle225123472;
float (*varresnetv22_stage2__plus0)[1][128][28][28];
float (*varresnetv22_stage2_batchnorm2_gamma)[128];
float (*varresnetv22_stage2_batchnorm2_beta)[128];
float (*varresnetv22_stage2_batchnorm2_running_mean)[128];
float (*varresnetv22_stage2_batchnorm2_running_var)[128];
float (*varresnetv22_stage2_batchnorm2_fwd)[1][128][28][28];
//BatchNormalization
unsigned int channelVar8;
float epsilonVar8;
float pad8;
float (*varresnetv22_stage2_activation2)[1][128][28][28];
float (*varresnetv22_stage2_conv3_weight)[128][128][3][3];
float (*varresnetv22_stage2_conv3_fwd)[128][3][3][128];
//Transpose
vector<size_t> shuffle225125664;
float (*varresnetv22_stage2_conv3_fwd__1)[1][28][28][128];
//Transpose
vector<size_t> shuffle225127744;
float (*varconv_bias__8)[128];
float (*varresnetv22_stage2_conv3_fwd__2)[1][28][28][128];
//Convolution
vector<size_t> filter225128080;
vector<size_t> kernel225128080;
vector<size_t> stride225128080;
vector<size_t> pad225128080;
size_t group225128080;
float (*varresnetv22_stage2_conv3_fwd__3)[1][128][28][28];
//Transpose
vector<size_t> shuffle225128496;
float (*varresnetv22_stage2_batchnorm3_gamma)[128];
float (*varresnetv22_stage2_batchnorm3_beta)[128];
float (*varresnetv22_stage2_batchnorm3_running_mean)[128];
float (*varresnetv22_stage2_batchnorm3_running_var)[128];
float (*varresnetv22_stage2_batchnorm3_fwd)[1][128][28][28];
//BatchNormalization
unsigned int channelVar9;
float epsilonVar9;
float pad9;
float (*varresnetv22_stage2_activation3)[1][128][28][28];
float (*varresnetv22_stage2_conv4_weight)[128][128][3][3];
float (*varresnetv22_stage2_conv4_fwd)[128][3][3][128];
//Transpose
vector<size_t> shuffle225130576;
float (*varresnetv22_stage2_conv4_fwd__1)[1][28][28][128];
//Transpose
vector<size_t> shuffle225132656;
float (*varconv_bias__9)[128];
float (*varresnetv22_stage2_conv4_fwd__2)[1][28][28][128];
//Convolution
vector<size_t> filter225132992;
vector<size_t> kernel225132992;
vector<size_t> stride225132992;
vector<size_t> pad225132992;
size_t group225132992;
float (*varresnetv22_stage2_conv4_fwd__3)[1][128][28][28];
//Transpose
vector<size_t> shuffle225133408;
float (*varresnetv22_stage2__plus1)[1][128][28][28];
float (*varresnetv22_stage3_batchnorm0_gamma)[128];
float (*varresnetv22_stage3_batchnorm0_beta)[128];
float (*varresnetv22_stage3_batchnorm0_running_mean)[128];
float (*varresnetv22_stage3_batchnorm0_running_var)[128];
float (*varresnetv22_stage3_batchnorm0_fwd)[1][128][28][28];
//BatchNormalization
unsigned int channelVar10;
float epsilonVar10;
float pad10;
float (*varresnetv22_stage3_activation0)[1][128][28][28];
float (*varresnetv22_stage3_conv0_weight)[256][128][3][3];
float (*varresnetv22_stage3_conv0_fwd)[256][3][3][128];
//Transpose
vector<size_t> shuffle225136016;
float (*varresnetv22_stage3_conv0_fwd__1)[1][28][28][128];
//Transpose
vector<size_t> shuffle225105424;
float (*varconv_bias__10)[256];
float (*varresnetv22_stage3_conv0_fwd__2)[1][14][14][256];
//Convolution
vector<size_t> filter225138848;
vector<size_t> kernel225138848;
vector<size_t> stride225138848;
vector<size_t> pad225138848;
size_t group225138848;
float (*varresnetv22_stage3_conv0_fwd__3)[1][256][14][14];
//Transpose
vector<size_t> shuffle225139264;
float (*varresnetv22_stage3_batchnorm1_gamma)[256];
float (*varresnetv22_stage3_batchnorm1_beta)[256];
float (*varresnetv22_stage3_batchnorm1_running_mean)[256];
float (*varresnetv22_stage3_batchnorm1_running_var)[256];
float (*varresnetv22_stage3_batchnorm1_fwd)[1][256][14][14];
//BatchNormalization
unsigned int channelVar11;
float epsilonVar11;
float pad11;
float (*varresnetv22_stage3_activation1)[1][256][14][14];
float (*varresnetv22_stage3_conv1_weight)[256][256][3][3];
float (*varresnetv22_stage3_conv1_fwd)[256][3][3][256];
//Transpose
vector<size_t> shuffle225141616;
float (*varresnetv22_stage3_conv1_fwd__1)[1][14][14][256];
//Transpose
vector<size_t> shuffle225137808;
float (*varconv_bias__11)[256];
float (*varresnetv22_stage3_conv1_fwd__2)[1][14][14][256];
//Convolution
vector<size_t> filter225144464;
vector<size_t> kernel225144464;
vector<size_t> stride225144464;
vector<size_t> pad225144464;
size_t group225144464;
float (*varresnetv22_stage3_conv1_fwd__3)[1][256][14][14];
//Transpose
vector<size_t> shuffle225144880;
float (*varresnetv22_stage3_conv2_weight)[256][128][1][1];
float (*varresnetv22_stage3_conv2_fwd)[256][1][1][128];
//Transpose
vector<size_t> shuffle225141680;
float (*varresnetv22_stage3_conv2_fwd__1)[1][28][28][128];
//Transpose
vector<size_t> shuffle225143472;
float (*varconv_bias__12)[256];
float (*varresnetv22_stage3_conv2_fwd__2)[1][14][14][256];
//Convolution
vector<size_t> filter225148640;
vector<size_t> kernel225148640;
vector<size_t> stride225148640;
vector<size_t> pad225148640;
size_t group225148640;
float (*varresnetv22_stage3_conv2_fwd__3)[1][256][14][14];
//Transpose
vector<size_t> shuffle225149056;
float (*varresnetv22_stage3__plus0)[1][256][14][14];
float (*varresnetv22_stage3_batchnorm2_gamma)[256];
float (*varresnetv22_stage3_batchnorm2_beta)[256];
float (*varresnetv22_stage3_batchnorm2_running_mean)[256];
float (*varresnetv22_stage3_batchnorm2_running_var)[256];
float (*varresnetv22_stage3_batchnorm2_fwd)[1][256][14][14];
//BatchNormalization
unsigned int channelVar12;
float epsilonVar12;
float pad12;
float (*varresnetv22_stage3_activation2)[1][256][14][14];
float (*varresnetv22_stage3_conv3_weight)[256][256][3][3];
float (*varresnetv22_stage3_conv3_fwd)[256][3][3][256];
//Transpose
vector<size_t> shuffle225151664;
float (*varresnetv22_stage3_conv3_fwd__1)[1][14][14][256];
//Transpose
vector<size_t> shuffle225154176;
float (*varconv_bias__13)[256];
float (*varresnetv22_stage3_conv3_fwd__2)[1][14][14][256];
//Convolution
vector<size_t> filter225154400;
vector<size_t> kernel225154400;
vector<size_t> stride225154400;
vector<size_t> pad225154400;
size_t group225154400;
float (*varresnetv22_stage3_conv3_fwd__3)[1][256][14][14];
//Transpose
vector<size_t> shuffle225154816;
float (*varresnetv22_stage3_batchnorm3_gamma)[256];
float (*varresnetv22_stage3_batchnorm3_beta)[256];
float (*varresnetv22_stage3_batchnorm3_running_mean)[256];
float (*varresnetv22_stage3_batchnorm3_running_var)[256];
float (*varresnetv22_stage3_batchnorm3_fwd)[1][256][14][14];
//BatchNormalization
unsigned int channelVar13;
float epsilonVar13;
float pad13;
float (*varresnetv22_stage3_activation3)[1][256][14][14];
float (*varresnetv22_stage3_conv4_weight)[256][256][3][3];
float (*varresnetv22_stage3_conv4_fwd)[256][3][3][256];
//Transpose
vector<size_t> shuffle225156976;
float (*varresnetv22_stage3_conv4_fwd__1)[1][14][14][256];
//Transpose
vector<size_t> shuffle225159488;
float (*varconv_bias__14)[256];
float (*varresnetv22_stage3_conv4_fwd__2)[1][14][14][256];
//Convolution
vector<size_t> filter225159712;
vector<size_t> kernel225159712;
vector<size_t> stride225159712;
vector<size_t> pad225159712;
size_t group225159712;
float (*varresnetv22_stage3_conv4_fwd__3)[1][256][14][14];
//Transpose
vector<size_t> shuffle225160128;
float (*varresnetv22_stage3__plus1)[1][256][14][14];
float (*varresnetv22_stage4_batchnorm0_gamma)[256];
float (*varresnetv22_stage4_batchnorm0_beta)[256];
float (*varresnetv22_stage4_batchnorm0_running_mean)[256];
float (*varresnetv22_stage4_batchnorm0_running_var)[256];
float (*varresnetv22_stage4_batchnorm0_fwd)[1][256][14][14];
//BatchNormalization
unsigned int channelVar14;
float epsilonVar14;
float pad14;
float (*varresnetv22_stage4_activation0)[1][256][14][14];
float (*varresnetv22_stage4_conv0_weight)[512][256][3][3];
float (*varresnetv22_stage4_conv0_fwd)[512][3][3][256];
//Transpose
vector<size_t> shuffle225120448;
float (*varresnetv22_stage4_conv0_fwd__1)[1][14][14][256];
//Transpose
vector<size_t> shuffle225122032;
float (*varconv_bias__15)[512];
float (*varresnetv22_stage4_conv0_fwd__2)[1][7][7][512];
//Convolution
vector<size_t> filter225170432;
vector<size_t> kernel225170432;
vector<size_t> stride225170432;
vector<size_t> pad225170432;
size_t group225170432;
float (*varresnetv22_stage4_conv0_fwd__3)[1][512][7][7];
//Transpose
vector<size_t> shuffle225122544;
float (*varresnetv22_stage4_batchnorm1_gamma)[512];
float (*varresnetv22_stage4_batchnorm1_beta)[512];
float (*varresnetv22_stage4_batchnorm1_running_mean)[512];
float (*varresnetv22_stage4_batchnorm1_running_var)[512];
float (*varresnetv22_stage4_batchnorm1_fwd)[1][512][7][7];
//BatchNormalization
unsigned int channelVar15;
float epsilonVar15;
float pad15;
float (*varresnetv22_stage4_activation1)[1][512][7][7];
float (*varresnetv22_stage4_conv1_weight)[512][512][3][3];
float (*varresnetv22_stage4_conv1_fwd)[512][3][3][512];
//Transpose
vector<size_t> shuffle225172784;
float (*varresnetv22_stage4_conv1_fwd__1)[1][7][7][512];
//Transpose
vector<size_t> shuffle225170320;
float (*varconv_bias__16)[512];
float (*varresnetv22_stage4_conv1_fwd__2)[1][7][7][512];
//Convolution
vector<size_t> filter225176656;
vector<size_t> kernel225176656;
vector<size_t> stride225176656;
vector<size_t> pad225176656;
size_t group225176656;
float (*varresnetv22_stage4_conv1_fwd__3)[1][512][7][7];
//Transpose
vector<size_t> shuffle225177072;
float (*varresnetv22_stage4_conv2_weight)[512][256][1][1];
float (*varresnetv22_stage4_conv2_fwd)[512][1][1][256];
//Transpose
vector<size_t> shuffle225172848;
float (*varresnetv22_stage4_conv2_fwd__1)[1][14][14][256];
//Transpose
vector<size_t> shuffle225175664;
float (*varconv_bias__17)[512];
float (*varresnetv22_stage4_conv2_fwd__2)[1][7][7][512];
//Convolution
vector<size_t> filter225181856;
vector<size_t> kernel225181856;
vector<size_t> stride225181856;
vector<size_t> pad225181856;
size_t group225181856;
float (*varresnetv22_stage4_conv2_fwd__3)[1][512][7][7];
//Transpose
vector<size_t> shuffle225182272;
float (*varresnetv22_stage4__plus0)[1][512][7][7];
float (*varresnetv22_stage4_batchnorm2_gamma)[512];
float (*varresnetv22_stage4_batchnorm2_beta)[512];
float (*varresnetv22_stage4_batchnorm2_running_mean)[512];
float (*varresnetv22_stage4_batchnorm2_running_var)[512];
float (*varresnetv22_stage4_batchnorm2_fwd)[1][512][7][7];
//BatchNormalization
unsigned int channelVar16;
float epsilonVar16;
float pad16;
float (*varresnetv22_stage4_activation2)[1][512][7][7];
float (*varresnetv22_stage4_conv3_weight)[512][512][3][3];
float (*varresnetv22_stage4_conv3_fwd)[512][3][3][512];
//Transpose
vector<size_t> shuffle225184880;
float (*varresnetv22_stage4_conv3_fwd__1)[1][7][7][512];
//Transpose
vector<size_t> shuffle225188416;
float (*varconv_bias__18)[512];
float (*varresnetv22_stage4_conv3_fwd__2)[1][7][7][512];
//Convolution
vector<size_t> filter225188640;
vector<size_t> kernel225188640;
vector<size_t> stride225188640;
vector<size_t> pad225188640;
size_t group225188640;
float (*varresnetv22_stage4_conv3_fwd__3)[1][512][7][7];
//Transpose
vector<size_t> shuffle225189056;
float (*varresnetv22_stage4_batchnorm3_gamma)[512];
float (*varresnetv22_stage4_batchnorm3_beta)[512];
float (*varresnetv22_stage4_batchnorm3_running_mean)[512];
float (*varresnetv22_stage4_batchnorm3_running_var)[512];
float (*varresnetv22_stage4_batchnorm3_fwd)[1][512][7][7];
//BatchNormalization
unsigned int channelVar17;
float epsilonVar17;
float pad17;
float (*varresnetv22_stage4_activation3)[1][512][7][7];
float (*varresnetv22_stage4_conv4_weight)[512][512][3][3];
float (*varresnetv22_stage4_conv4_fwd)[512][3][3][512];
//Transpose
vector<size_t> shuffle225191216;
float (*varresnetv22_stage4_conv4_fwd__1)[1][7][7][512];
//Transpose
vector<size_t> shuffle225194752;
float (*varconv_bias__19)[512];
float (*varresnetv22_stage4_conv4_fwd__2)[1][7][7][512];
//Convolution
vector<size_t> filter225194976;
vector<size_t> kernel225194976;
vector<size_t> stride225194976;
vector<size_t> pad225194976;
size_t group225194976;
float (*varresnetv22_stage4_conv4_fwd__3)[1][512][7][7];
//Transpose
vector<size_t> shuffle225195392;
float (*varresnetv22_stage4__plus1)[1][512][7][7];
float (*varresnetv22_batchnorm2_gamma)[512];
float (*varresnetv22_batchnorm2_beta)[512];
float (*varresnetv22_batchnorm2_running_mean)[512];
float (*varresnetv22_batchnorm2_running_var)[512];
float (*varresnetv22_batchnorm2_fwd)[1][512][7][7];
//BatchNormalization
unsigned int channelVar18;
float epsilonVar18;
float pad18;
float (*varresnetv22_relu1_fwd)[1][512][7][7];
float (*varresnetv22_pool1_fwd)[1][7][7][512];
//Transpose
vector<size_t> shuffle225198176;
float (*varresnetv22_pool1_fwd__1)[1][1][1][512];
//AvgPool
vector<size_t> kernel225198864;
vector<size_t> stride225198896;
vector<size_t> pad225198928;
float (*varresnetv22_pool1_fwd__2)[1][512][1][1];
//Transpose
vector<size_t> shuffle225198800;
float (*varresnetv22_flatten0_reshape0)[1][512];
float (*varresnetv22_dense0_weight)[1000][512];
float (*varresnetv22_dense0_fwd)[512][1000];
//Transpose
vector<size_t> shuffle225201120;
float (*varresnetv22_dense0_fwd__1)[1][1000];
//MatMul
float (*varresnetv22_dense0_bias)[1000];
float (*varresnetv22_dense0_fwd_reshape)[1][1000];
float (*varresnetv22_dense0_fwd__2)[1][1000];

void initializeVariables(std::string dataFilePath)
 {
	string binfile = dataFilePath;
	std::ifstream readFileBin;
	readFileBin.open(binfile.data(), ios::binary);

	//Constant input
	varresnetv22_batchnorm0_gamma = (float(*)[3])malloc(sizeof(float[3]));
	readFileBin.read((char *)varresnetv22_batchnorm0_gamma,4*3);

	//Constant input
	varresnetv22_batchnorm0_beta = (float(*)[3])malloc(sizeof(float[3]));
	readFileBin.read((char *)varresnetv22_batchnorm0_beta,4*3);

	//Constant input
	varresnetv22_batchnorm0_running_mean = (float(*)[3])malloc(sizeof(float[3]));
	readFileBin.read((char *)varresnetv22_batchnorm0_running_mean,4*3);

	//Constant input
	varresnetv22_batchnorm0_running_var = (float(*)[3])malloc(sizeof(float[3]));
	readFileBin.read((char *)varresnetv22_batchnorm0_running_var,4*3);

	//BatchNormalization
	readFileBin.read((char *)&channelVar0, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar0, sizeof(float));
	readFileBin.read((char *)&pad0, sizeof(float));
	varresnetv22_batchnorm0_fwd = (float(*)[1][3][224][224])malloc(sizeof(float[1][3][224][224]));
	if(varresnetv22_batchnorm0_fwd != NULL)
	{
		memset(varresnetv22_batchnorm0_fwd, 0x00, sizeof(float[1][3][224][224]));
	}
	if(varresnetv22_batchnorm0_fwd != NULL)
	{
		varresnetv22_batchnorm0_fwd = (float(*)[1][3][224][224])malloc(sizeof(float[1][3][224][224]));
	}

	//Constant input
	varresnetv22_conv0_weight = (float(*)[64][3][7][7])malloc(sizeof(float[64][3][7][7]));
	readFileBin.read((char *)varresnetv22_conv0_weight,4*64*3*7*7);
//
	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225081392, &readFileBin);
	varresnetv22_conv0_fwd = (float(*)[64][7][7][3])malloc(sizeof(float[64][7][7][3]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225082800, &readFileBin);
	varresnetv22_conv0_fwd__1 = (float(*)[1][224][224][3])malloc(sizeof(float[1][224][224][3]));

	//Constant input
	varconv_bias = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varconv_bias,4*64);

	//Convolution
	//64 7 7 3 
	read_vector(&filter225083184, &readFileBin);
	//7 7 
	read_vector(&kernel225083184, &readFileBin);
	//2 2 
	read_vector(&stride225083184, &readFileBin);
	//3 3 3 3 
	read_vector(&pad225083184, &readFileBin);
	readFileBin.read((char *)&group225083184, sizeof(size_t));
	varresnetv22_conv0_fwd__2 = (float(*)[1][112][112][64])malloc(sizeof(float[1][112][112][64]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225083600, &readFileBin);
	varresnetv22_conv0_fwd__3 = (float(*)[1][64][112][112])malloc(sizeof(float[1][64][112][112]));

	//Constant input
	varresnetv22_batchnorm1_gamma = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_batchnorm1_gamma,4*64);

	//Constant input
	varresnetv22_batchnorm1_beta = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_batchnorm1_beta,4*64);

	//Constant input
	varresnetv22_batchnorm1_running_mean = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_batchnorm1_running_mean,4*64);

	//Constant input
	varresnetv22_batchnorm1_running_var = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_batchnorm1_running_var,4*64);

	//BatchNormalization
	readFileBin.read((char *)&channelVar1, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar1, sizeof(float));
	readFileBin.read((char *)&pad1, sizeof(float));
	varresnetv22_batchnorm1_fwd = (float(*)[1][64][112][112])malloc(sizeof(float[1][64][112][112]));
	if(varresnetv22_batchnorm1_fwd != NULL)
	{
		memset(varresnetv22_batchnorm1_fwd, 0x00, sizeof(float[1][64][112][112]));
	}
	if(varresnetv22_batchnorm1_fwd != NULL)
	{
		varresnetv22_batchnorm1_fwd = (float(*)[1][64][112][112])malloc(sizeof(float[1][64][112][112]));
	}
	if(varresnetv22_relu0_fwd != NULL)
	{
		varresnetv22_relu0_fwd = (float(*)[1][64][112][112])malloc(sizeof(float[1][64][112][112]));
	}

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225086048, &readFileBin);
	varresnetv22_pool0_fwd = (float(*)[1][112][112][64])malloc(sizeof(float[1][112][112][64]));

	//MaxPool
	//3 3 
	read_vector(&kernel225086496, &readFileBin);
	//2 2 
	read_vector(&stride225086496, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225086496, &readFileBin);
	varresnetv22_pool0_fwd__1 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225086896, &readFileBin);
	varresnetv22_pool0_fwd__2 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));

	//Constant input
	varresnetv22_stage1_batchnorm0_gamma = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm0_gamma,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm0_beta = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm0_beta,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm0_running_mean = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm0_running_mean,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm0_running_var = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm0_running_var,4*64);

	//BatchNormalization
	readFileBin.read((char *)&channelVar2, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar2, sizeof(float));
	readFileBin.read((char *)&pad2, sizeof(float));
	varresnetv22_stage1_batchnorm0_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	if(varresnetv22_stage1_batchnorm0_fwd != NULL)
	{
		memset(varresnetv22_stage1_batchnorm0_fwd, 0x00, sizeof(float[1][64][56][56]));
	}
	if(varresnetv22_stage1_batchnorm0_fwd != NULL)
	{
		varresnetv22_stage1_batchnorm0_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}
	if(varresnetv22_stage1_activation0 != NULL)
	{
		varresnetv22_stage1_activation0 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}

	//Constant input
	varresnetv22_stage1_conv0_weight = (float(*)[64][64][3][3])malloc(sizeof(float[64][64][3][3]));
	readFileBin.read((char *)varresnetv22_stage1_conv0_weight,4*64*64*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225089136, &readFileBin);
	varresnetv22_stage1_conv0_fwd = (float(*)[64][3][3][64])malloc(sizeof(float[64][3][3][64]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225080240, &readFileBin);
	varresnetv22_stage1_conv0_fwd__1 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Constant input
	varconv_bias__1 = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varconv_bias__1,4*64);

	//Convolution
	//64 3 3 64 
	read_vector(&filter225091856, &readFileBin);
	//3 3 
	read_vector(&kernel225091856, &readFileBin);
	//1 1 
	read_vector(&stride225091856, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225091856, &readFileBin);
	readFileBin.read((char *)&group225091856, sizeof(size_t));
	varresnetv22_stage1_conv0_fwd__2 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225092304, &readFileBin);
	varresnetv22_stage1_conv0_fwd__3 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));

	//Constant input
	varresnetv22_stage1_batchnorm1_gamma = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm1_gamma,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm1_beta = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm1_beta,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm1_running_mean = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm1_running_mean,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm1_running_var = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm1_running_var,4*64);

	//BatchNormalization
	readFileBin.read((char *)&channelVar3, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar3, sizeof(float));
	readFileBin.read((char *)&pad3, sizeof(float));
	varresnetv22_stage1_batchnorm1_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	if(varresnetv22_stage1_batchnorm1_fwd != NULL)
	{
		memset(varresnetv22_stage1_batchnorm1_fwd, 0x00, sizeof(float[1][64][56][56]));
	}
	if(varresnetv22_stage1_batchnorm1_fwd != NULL)
	{
		varresnetv22_stage1_batchnorm1_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}
	if(varresnetv22_stage1_activation1 != NULL)
	{
		varresnetv22_stage1_activation1 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}

	//Constant input
	varresnetv22_stage1_conv1_weight = (float(*)[64][64][3][3])malloc(sizeof(float[64][64][3][3]));
	readFileBin.read((char *)varresnetv22_stage1_conv1_weight,4*64*64*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225094320, &readFileBin);
	varresnetv22_stage1_conv1_fwd = (float(*)[64][3][3][64])malloc(sizeof(float[64][3][3][64]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225096096, &readFileBin);
	varresnetv22_stage1_conv1_fwd__1 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Constant input
	varconv_bias__2 = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varconv_bias__2,4*64);

	//Convolution
	//64 3 3 64 
	read_vector(&filter225096368, &readFileBin);
	//3 3 
	read_vector(&kernel225096368, &readFileBin);
	//1 1 
	read_vector(&stride225096368, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225096368, &readFileBin);
	readFileBin.read((char *)&group225096368, sizeof(size_t));
	varresnetv22_stage1_conv1_fwd__2 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225096784, &readFileBin);
	varresnetv22_stage1_conv1_fwd__3 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	if(varresnetv22_stage1__plus0 != NULL)
	{
		varresnetv22_stage1__plus0 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}

	//Constant input
	varresnetv22_stage1_batchnorm2_gamma = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm2_gamma,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm2_beta = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm2_beta,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm2_running_mean = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm2_running_mean,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm2_running_var = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm2_running_var,4*64);

	//BatchNormalization
	readFileBin.read((char *)&channelVar4, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar4, sizeof(float));
	readFileBin.read((char *)&pad4, sizeof(float));
	varresnetv22_stage1_batchnorm2_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	if(varresnetv22_stage1_batchnorm2_fwd != NULL)
	{
		memset(varresnetv22_stage1_batchnorm2_fwd, 0x00, sizeof(float[1][64][56][56]));
	}
	if(varresnetv22_stage1_batchnorm2_fwd != NULL)
	{
		varresnetv22_stage1_batchnorm2_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}
	if(varresnetv22_stage1_activation2 != NULL)
	{
		varresnetv22_stage1_activation2 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}
	
	//Constant input
	varresnetv22_stage1_conv2_weight = (float(*)[64][64][3][3])malloc(sizeof(float[64][64][3][3]));
	readFileBin.read((char *)varresnetv22_stage1_conv2_weight,4*64*64*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225090144, &readFileBin);
	varresnetv22_stage1_conv2_fwd = (float(*)[64][3][3][64])malloc(sizeof(float[64][3][3][64]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225101952, &readFileBin);
	varresnetv22_stage1_conv2_fwd__1 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Constant input
	varconv_bias__3 = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varconv_bias__3,4*64);

	//Convolution
	//64 3 3 64 
	read_vector(&filter225102288, &readFileBin);
	//3 3 
	read_vector(&kernel225102288, &readFileBin);
	//1 1 
	read_vector(&stride225102288, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225102288, &readFileBin);
	readFileBin.read((char *)&group225102288, sizeof(size_t));
	varresnetv22_stage1_conv2_fwd__2 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225102704, &readFileBin);
	varresnetv22_stage1_conv2_fwd__3 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));

	//Constant input
	varresnetv22_stage1_batchnorm3_gamma = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm3_gamma,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm3_beta = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm3_beta,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm3_running_mean = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm3_running_mean,4*64);

	//Constant input
	varresnetv22_stage1_batchnorm3_running_var = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage1_batchnorm3_running_var,4*64);

	//BatchNormalization
	readFileBin.read((char *)&channelVar5, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar5, sizeof(float));
	readFileBin.read((char *)&pad5, sizeof(float));
	varresnetv22_stage1_batchnorm3_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	if(varresnetv22_stage1_batchnorm3_fwd != NULL)
	{
		memset(varresnetv22_stage1_batchnorm3_fwd, 0x00, sizeof(float[1][64][56][56]));
	}
	if(varresnetv22_stage1_batchnorm3_fwd != NULL)
	{
		varresnetv22_stage1_batchnorm3_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}
	
	if(varresnetv22_stage1_activation3 != NULL)
	{
		varresnetv22_stage1_activation3 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}

	//Constant input
	varresnetv22_stage1_conv3_weight = (float(*)[64][64][3][3])malloc(sizeof(float[64][64][3][3]));
	readFileBin.read((char *)varresnetv22_stage1_conv3_weight,4*64*64*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225104784, &readFileBin);
	varresnetv22_stage1_conv3_fwd = (float(*)[64][3][3][64])malloc(sizeof(float[64][3][3][64]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225106608, &readFileBin);
	varresnetv22_stage1_conv3_fwd__1 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Constant input
	varconv_bias__4 = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varconv_bias__4,4*64);

	//Convolution
	//64 3 3 64 
	read_vector(&filter225106944, &readFileBin);
	//3 3 
	read_vector(&kernel225106944, &readFileBin);
	//1 1 
	read_vector(&stride225106944, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225106944, &readFileBin);
	readFileBin.read((char *)&group225106944, sizeof(size_t));
	varresnetv22_stage1_conv3_fwd__2 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225107360, &readFileBin);
	varresnetv22_stage1_conv3_fwd__3 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	if(varresnetv22_stage1__plus1 != NULL)
	{
		varresnetv22_stage1__plus1 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}

	//Constant input
	varresnetv22_stage2_batchnorm0_gamma = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm0_gamma,4*64);

	//Constant input
	varresnetv22_stage2_batchnorm0_beta = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm0_beta,4*64);

	//Constant input
	varresnetv22_stage2_batchnorm0_running_mean = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm0_running_mean,4*64);

	//Constant input
	varresnetv22_stage2_batchnorm0_running_var = (float(*)[64])malloc(sizeof(float[64]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm0_running_var,4*64);

	//BatchNormalization
	readFileBin.read((char *)&channelVar6, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar6, sizeof(float));
	readFileBin.read((char *)&pad6, sizeof(float));
	varresnetv22_stage2_batchnorm0_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	if(varresnetv22_stage2_batchnorm0_fwd != NULL)
	{
		memset(varresnetv22_stage2_batchnorm0_fwd, 0x00, sizeof(float[1][64][56][56]));
	}
	if(varresnetv22_stage2_batchnorm0_fwd != NULL)
	{
		varresnetv22_stage2_batchnorm0_fwd = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}
	if(varresnetv22_stage2_activation0 != NULL)
	{
		varresnetv22_stage2_activation0 = (float(*)[1][64][56][56])malloc(sizeof(float[1][64][56][56]));
	}

	//Constant input
	varresnetv22_stage2_conv0_weight = (float(*)[128][64][3][3])malloc(sizeof(float[128][64][3][3]));
	readFileBin.read((char *)varresnetv22_stage2_conv0_weight,4*128*64*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225109968, &readFileBin);
	varresnetv22_stage2_conv0_fwd = (float(*)[128][3][3][64])malloc(sizeof(float[128][3][3][64]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225112192, &readFileBin);
	varresnetv22_stage2_conv0_fwd__1 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Constant input
	varconv_bias__5 = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varconv_bias__5,4*128);

	//Convolution
	//128 3 3 64 
	read_vector(&filter225112672, &readFileBin);
	//3 3 
	read_vector(&kernel225112672, &readFileBin);
	//2 2 
	read_vector(&stride225112672, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225112672, &readFileBin);
	readFileBin.read((char *)&group225112672, sizeof(size_t));
	varresnetv22_stage2_conv0_fwd__2 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225113088, &readFileBin);
	varresnetv22_stage2_conv0_fwd__3 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));

	//Constant input
	varresnetv22_stage2_batchnorm1_gamma = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm1_gamma,4*128);

	//Constant input
	varresnetv22_stage2_batchnorm1_beta = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm1_beta,4*128);

	//Constant input
	varresnetv22_stage2_batchnorm1_running_mean = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm1_running_mean,4*128);

	//Constant input
	varresnetv22_stage2_batchnorm1_running_var = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm1_running_var,4*128);

	//BatchNormalization
	readFileBin.read((char *)&channelVar7, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar7, sizeof(float));
	readFileBin.read((char *)&pad7, sizeof(float));
	varresnetv22_stage2_batchnorm1_fwd = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	if(varresnetv22_stage2_batchnorm1_fwd != NULL)
	{
		memset(varresnetv22_stage2_batchnorm1_fwd, 0x00, sizeof(float[1][128][28][28]));
	}
	if(varresnetv22_stage2_batchnorm1_fwd != NULL)
	{
		varresnetv22_stage2_batchnorm1_fwd = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}
	
	if(varresnetv22_stage2_activation1 != NULL)
	{
		varresnetv22_stage2_activation1 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}

	//Constant input
	varresnetv22_stage2_conv1_weight = (float(*)[128][128][3][3])malloc(sizeof(float[128][128][3][3]));
	readFileBin.read((char *)varresnetv22_stage2_conv1_weight,4*128*128*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225115312, &readFileBin);
	varresnetv22_stage2_conv1_fwd = (float(*)[128][3][3][128])malloc(sizeof(float[128][3][3][128]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225117504, &readFileBin);
	varresnetv22_stage2_conv1_fwd__1 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Constant input
	varconv_bias__6 = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varconv_bias__6,4*128);

	//Convolution
	//128 3 3 128 
	read_vector(&filter225117840, &readFileBin);
	//3 3 
	read_vector(&kernel225117840, &readFileBin);
	//1 1 
	read_vector(&stride225117840, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225117840, &readFileBin);
	readFileBin.read((char *)&group225117840, sizeof(size_t));
	varresnetv22_stage2_conv1_fwd__2 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225118256, &readFileBin);
	varresnetv22_stage2_conv1_fwd__3 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));

	//Constant input
	varresnetv22_stage2_conv2_weight = (float(*)[128][64][1][1])malloc(sizeof(float[128][64][1][1]));
	readFileBin.read((char *)varresnetv22_stage2_conv2_weight,4*128*64*1*1);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225115376, &readFileBin);
	varresnetv22_stage2_conv2_fwd = (float(*)[128][1][1][64])malloc(sizeof(float[128][1][1][64]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225100464, &readFileBin);
	varresnetv22_stage2_conv2_fwd__1 = (float(*)[1][56][56][64])malloc(sizeof(float[1][56][56][64]));

	//Constant input
	varconv_bias__7 = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varconv_bias__7,4*128);

	//Convolution
	//128 1 1 64 
	read_vector(&filter225122880, &readFileBin);
	//1 1 
	read_vector(&kernel225122880, &readFileBin);
	//2 2 
	read_vector(&stride225122880, &readFileBin);
	//0 0 0 0 
	read_vector(&pad225122880, &readFileBin);
	readFileBin.read((char *)&group225122880, sizeof(size_t));
	varresnetv22_stage2_conv2_fwd__2 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225123472, &readFileBin);
	varresnetv22_stage2_conv2_fwd__3 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	if(varresnetv22_stage2__plus0 != NULL)
	{
		varresnetv22_stage2__plus0 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}

	//Constant input
	varresnetv22_stage2_batchnorm2_gamma = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm2_gamma,4*128);

	//Constant input
	varresnetv22_stage2_batchnorm2_beta = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm2_beta,4*128);

	//Constant input
	varresnetv22_stage2_batchnorm2_running_mean = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm2_running_mean,4*128);

	//Constant input
	varresnetv22_stage2_batchnorm2_running_var = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm2_running_var,4*128);

	//BatchNormalization
	readFileBin.read((char *)&channelVar8, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar8, sizeof(float));
	readFileBin.read((char *)&pad8, sizeof(float));
	varresnetv22_stage2_batchnorm2_fwd = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	if(varresnetv22_stage2_batchnorm2_fwd != 0)
	{
		memset(varresnetv22_stage2_batchnorm2_fwd, 0x00, sizeof(float[1][128][28][28]));
	}
	if(varresnetv22_stage2_batchnorm2_fwd != NULL)
	{
		varresnetv22_stage2_batchnorm2_fwd = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}
	if(varresnetv22_stage2_activation2 != NULL)
	{
		varresnetv22_stage2_activation2 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}

	//Constant input
	varresnetv22_stage2_conv3_weight = (float(*)[128][128][3][3])malloc(sizeof(float[128][128][3][3]));
	readFileBin.read((char *)varresnetv22_stage2_conv3_weight,4*128*128*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225125664, &readFileBin);
	varresnetv22_stage2_conv3_fwd = (float(*)[128][3][3][128])malloc(sizeof(float[128][3][3][128]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225127744, &readFileBin);
	varresnetv22_stage2_conv3_fwd__1 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Constant input
	varconv_bias__8 = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varconv_bias__8,4*128);

	//Convolution
	//128 3 3 128 
	read_vector(&filter225128080, &readFileBin);
	//3 3 
	read_vector(&kernel225128080, &readFileBin);
	//1 1 
	read_vector(&stride225128080, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225128080, &readFileBin);
	readFileBin.read((char *)&group225128080, sizeof(size_t));
	varresnetv22_stage2_conv3_fwd__2 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225128496, &readFileBin);
	varresnetv22_stage2_conv3_fwd__3 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));

	//Constant input
	varresnetv22_stage2_batchnorm3_gamma = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm3_gamma,4*128);

	//Constant input
	varresnetv22_stage2_batchnorm3_beta = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm3_beta,4*128);

	//Constant input
	varresnetv22_stage2_batchnorm3_running_mean = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm3_running_mean,4*128);

	//Constant input
	varresnetv22_stage2_batchnorm3_running_var = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage2_batchnorm3_running_var,4*128);

	//BatchNormalization
	readFileBin.read((char *)&channelVar9, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar9, sizeof(float));
	readFileBin.read((char *)&pad9, sizeof(float));
	varresnetv22_stage2_batchnorm3_fwd = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	if(varresnetv22_stage2_batchnorm3_fwd != NULL)
	{
		memset(varresnetv22_stage2_batchnorm3_fwd, 0x00, sizeof(float[1][128][28][28]));
	}
	if(varresnetv22_stage2_batchnorm3_fwd != NULL)
	{
		varresnetv22_stage2_batchnorm3_fwd = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}
	if(varresnetv22_stage2_activation3 != NULL)
	{
		varresnetv22_stage2_activation3 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}

	//Constant input
	varresnetv22_stage2_conv4_weight = (float(*)[128][128][3][3])malloc(sizeof(float[128][128][3][3]));
	readFileBin.read((char *)varresnetv22_stage2_conv4_weight,4*128*128*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225130576, &readFileBin);
	varresnetv22_stage2_conv4_fwd = (float(*)[128][3][3][128])malloc(sizeof(float[128][3][3][128]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225132656, &readFileBin);
	varresnetv22_stage2_conv4_fwd__1 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Constant input
	varconv_bias__9 = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varconv_bias__9,4*128);

	//Convolution
	//128 3 3 128 
	read_vector(&filter225132992, &readFileBin);
	//3 3 
	read_vector(&kernel225132992, &readFileBin);
	//1 1 
	read_vector(&stride225132992, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225132992, &readFileBin);
	readFileBin.read((char *)&group225132992, sizeof(size_t));
	varresnetv22_stage2_conv4_fwd__2 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225133408, &readFileBin);
	varresnetv22_stage2_conv4_fwd__3 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	if(varresnetv22_stage2__plus1 != NULL)
	{
		varresnetv22_stage2__plus1 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}

	//Constant input
	varresnetv22_stage3_batchnorm0_gamma = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm0_gamma,4*128);

	//Constant input
	varresnetv22_stage3_batchnorm0_beta = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm0_beta,4*128);

	//Constant input
	varresnetv22_stage3_batchnorm0_running_mean = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm0_running_mean,4*128);

	//Constant input
	varresnetv22_stage3_batchnorm0_running_var = (float(*)[128])malloc(sizeof(float[128]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm0_running_var,4*128);

	//BatchNormalization
	readFileBin.read((char *)&channelVar10, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar10, sizeof(float));
	readFileBin.read((char *)&pad10, sizeof(float));
	varresnetv22_stage3_batchnorm0_fwd = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	if(varresnetv22_stage3_batchnorm0_fwd != NULL)
	{
		memset(varresnetv22_stage3_batchnorm0_fwd, 0x00, sizeof(float[1][128][28][28]));
	}
	if(varresnetv22_stage3_batchnorm0_fwd != NULL)
	{
		varresnetv22_stage3_batchnorm0_fwd = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}
	if(varresnetv22_stage3_activation0 != NULL)
	{
		varresnetv22_stage3_activation0 = (float(*)[1][128][28][28])malloc(sizeof(float[1][128][28][28]));
	}

	//Constant input
	varresnetv22_stage3_conv0_weight = (float(*)[256][128][3][3])malloc(sizeof(float[256][128][3][3]));
	readFileBin.read((char *)varresnetv22_stage3_conv0_weight,4*256*128*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225136016, &readFileBin);
	varresnetv22_stage3_conv0_fwd = (float(*)[256][3][3][128])malloc(sizeof(float[256][3][3][128]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225105424, &readFileBin);
	varresnetv22_stage3_conv0_fwd__1 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Constant input
	varconv_bias__10 = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varconv_bias__10,4*256);

	//Convolution
	//256 3 3 128 
	read_vector(&filter225138848, &readFileBin);
	//3 3 
	read_vector(&kernel225138848, &readFileBin);
	//2 2 
	read_vector(&stride225138848, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225138848, &readFileBin);
	readFileBin.read((char *)&group225138848, sizeof(size_t));
	varresnetv22_stage3_conv0_fwd__2 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225139264, &readFileBin);
	varresnetv22_stage3_conv0_fwd__3 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));

	//Constant input
	varresnetv22_stage3_batchnorm1_gamma = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm1_gamma,4*256);

	//Constant input
	varresnetv22_stage3_batchnorm1_beta = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm1_beta,4*256);

	//Constant input
	varresnetv22_stage3_batchnorm1_running_mean = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm1_running_mean,4*256);

	//Constant input
	varresnetv22_stage3_batchnorm1_running_var = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm1_running_var,4*256);

	//BatchNormalization
	readFileBin.read((char *)&channelVar11, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar11, sizeof(float));
	readFileBin.read((char *)&pad11, sizeof(float));
	varresnetv22_stage3_batchnorm1_fwd = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	if(varresnetv22_stage3_batchnorm1_fwd != NULL)
	{
		memset(varresnetv22_stage3_batchnorm1_fwd, 0x00, sizeof(float[1][256][14][14]));
	}
	if(varresnetv22_stage3_batchnorm1_fwd != NULL)
	{
		varresnetv22_stage3_batchnorm1_fwd = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}
	if(varresnetv22_stage3_activation1 != NULL)
	{
		varresnetv22_stage3_activation1 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}

	//Constant input
	varresnetv22_stage3_conv1_weight = (float(*)[256][256][3][3])malloc(sizeof(float[256][256][3][3]));
	readFileBin.read((char *)varresnetv22_stage3_conv1_weight,4*256*256*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225141616, &readFileBin);
	varresnetv22_stage3_conv1_fwd = (float(*)[256][3][3][256])malloc(sizeof(float[256][3][3][256]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225137808, &readFileBin);
	varresnetv22_stage3_conv1_fwd__1 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Constant input
	varconv_bias__11 = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varconv_bias__11,4*256);

	//Convolution
	//256 3 3 256 
	read_vector(&filter225144464, &readFileBin);
	//3 3 
	read_vector(&kernel225144464, &readFileBin);
	//1 1 
	read_vector(&stride225144464, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225144464, &readFileBin);
	readFileBin.read((char *)&group225144464, sizeof(size_t));
	varresnetv22_stage3_conv1_fwd__2 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225144880, &readFileBin);
	varresnetv22_stage3_conv1_fwd__3 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));

	//Constant input
	varresnetv22_stage3_conv2_weight = (float(*)[256][128][1][1])malloc(sizeof(float[256][128][1][1]));
	readFileBin.read((char *)varresnetv22_stage3_conv2_weight,4*256*128*1*1);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225141680, &readFileBin);
	varresnetv22_stage3_conv2_fwd = (float(*)[256][1][1][128])malloc(sizeof(float[256][1][1][128]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225143472, &readFileBin);
	varresnetv22_stage3_conv2_fwd__1 = (float(*)[1][28][28][128])malloc(sizeof(float[1][28][28][128]));

	//Constant input
	varconv_bias__12 = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varconv_bias__12,4*256);

	//Convolution
	//256 1 1 128 
	read_vector(&filter225148640, &readFileBin);
	//1 1 
	read_vector(&kernel225148640, &readFileBin);
	//2 2 
	read_vector(&stride225148640, &readFileBin);
	//0 0 0 0 
	read_vector(&pad225148640, &readFileBin);
	readFileBin.read((char *)&group225148640, sizeof(size_t));
	varresnetv22_stage3_conv2_fwd__2 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225149056, &readFileBin);
	varresnetv22_stage3_conv2_fwd__3 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	if(varresnetv22_stage3__plus0 != NULL)
	{
		varresnetv22_stage3__plus0 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}
	
	//Constant input
	varresnetv22_stage3_batchnorm2_gamma = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm2_gamma,4*256);

	//Constant input
	varresnetv22_stage3_batchnorm2_beta = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm2_beta,4*256);

	//Constant input
	varresnetv22_stage3_batchnorm2_running_mean = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm2_running_mean,4*256);

	//Constant input
	varresnetv22_stage3_batchnorm2_running_var = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm2_running_var,4*256);

	//BatchNormalization
	readFileBin.read((char *)&channelVar12, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar12, sizeof(float));
	readFileBin.read((char *)&pad12, sizeof(float));
	varresnetv22_stage3_batchnorm2_fwd = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	if(varresnetv22_stage3_batchnorm2_fwd != NULL)
	{
		memset(varresnetv22_stage3_batchnorm2_fwd, 0x00, sizeof(float[1][256][14][14]));
	}
	if(varresnetv22_stage3_batchnorm2_fwd != NULL)
	{
		varresnetv22_stage3_batchnorm2_fwd = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}
	if(varresnetv22_stage3_activation2 != NULL)
	{
		varresnetv22_stage3_activation2 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}

	//Constant input
	varresnetv22_stage3_conv3_weight = (float(*)[256][256][3][3])malloc(sizeof(float[256][256][3][3]));
	readFileBin.read((char *)varresnetv22_stage3_conv3_weight,4*256*256*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225151664, &readFileBin);
	varresnetv22_stage3_conv3_fwd = (float(*)[256][3][3][256])malloc(sizeof(float[256][3][3][256]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225154176, &readFileBin);
	varresnetv22_stage3_conv3_fwd__1 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Constant input
	varconv_bias__13 = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varconv_bias__13,4*256);

	//Convolution
	//256 3 3 256 
	read_vector(&filter225154400, &readFileBin);
	//3 3 
	read_vector(&kernel225154400, &readFileBin);
	//1 1 
	read_vector(&stride225154400, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225154400, &readFileBin);
	readFileBin.read((char *)&group225154400, sizeof(size_t));
	varresnetv22_stage3_conv3_fwd__2 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225154816, &readFileBin);
	varresnetv22_stage3_conv3_fwd__3 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));

	//Constant input
	varresnetv22_stage3_batchnorm3_gamma = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm3_gamma,4*256);

	//Constant input
	varresnetv22_stage3_batchnorm3_beta = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm3_beta,4*256);

	//Constant input
	varresnetv22_stage3_batchnorm3_running_mean = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm3_running_mean,4*256);

	//Constant input
	varresnetv22_stage3_batchnorm3_running_var = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage3_batchnorm3_running_var,4*256);

	//BatchNormalization
	readFileBin.read((char *)&channelVar13, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar13, sizeof(float));
	readFileBin.read((char *)&pad13, sizeof(float));
	varresnetv22_stage3_batchnorm3_fwd = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	if(varresnetv22_stage3_batchnorm3_fwd != NULL)
	{
		memset(varresnetv22_stage3_batchnorm3_fwd, 0x00, sizeof(float[1][256][14][14]));
	}
	if(varresnetv22_stage3_batchnorm3_fwd != NULL)
	{
		varresnetv22_stage3_batchnorm3_fwd = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}
	if(varresnetv22_stage3_activation3 != NULL)
	{
		varresnetv22_stage3_activation3 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}
	
	//Constant input
	varresnetv22_stage3_conv4_weight = (float(*)[256][256][3][3])malloc(sizeof(float[256][256][3][3]));
	readFileBin.read((char *)varresnetv22_stage3_conv4_weight,4*256*256*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225156976, &readFileBin);
	varresnetv22_stage3_conv4_fwd = (float(*)[256][3][3][256])malloc(sizeof(float[256][3][3][256]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225159488, &readFileBin);
	varresnetv22_stage3_conv4_fwd__1 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Constant input
	varconv_bias__14 = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varconv_bias__14,4*256);

	//Convolution
	//256 3 3 256 
	read_vector(&filter225159712, &readFileBin);
	//3 3 
	read_vector(&kernel225159712, &readFileBin);
	//1 1 
	read_vector(&stride225159712, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225159712, &readFileBin);
	readFileBin.read((char *)&group225159712, sizeof(size_t));
	varresnetv22_stage3_conv4_fwd__2 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225160128, &readFileBin);
	varresnetv22_stage3_conv4_fwd__3 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	if(varresnetv22_stage3__plus1 != NULL)
	{
		varresnetv22_stage3__plus1 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}

	//Constant input
	varresnetv22_stage4_batchnorm0_gamma = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm0_gamma,4*256);

	//Constant input
	varresnetv22_stage4_batchnorm0_beta = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm0_beta,4*256);

	//Constant input
	varresnetv22_stage4_batchnorm0_running_mean = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm0_running_mean,4*256);

	//Constant input
	varresnetv22_stage4_batchnorm0_running_var = (float(*)[256])malloc(sizeof(float[256]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm0_running_var,4*256);

	//BatchNormalization
	readFileBin.read((char *)&channelVar14, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar14, sizeof(float));
	readFileBin.read((char *)&pad14, sizeof(float));
	varresnetv22_stage4_batchnorm0_fwd = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	if(varresnetv22_stage4_batchnorm0_fwd != NULL)
	{
		memset(varresnetv22_stage4_batchnorm0_fwd, 0x00, sizeof(float[1][256][14][14]));
	}
	if(varresnetv22_stage4_batchnorm0_fwd != NULL)
	{
		varresnetv22_stage4_batchnorm0_fwd = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}
	if(varresnetv22_stage4_activation0 != NULL)
	{
		varresnetv22_stage4_activation0 = (float(*)[1][256][14][14])malloc(sizeof(float[1][256][14][14]));
	}

	//Constant input
	varresnetv22_stage4_conv0_weight = (float(*)[512][256][3][3])malloc(sizeof(float[512][256][3][3]));
	readFileBin.read((char *)varresnetv22_stage4_conv0_weight,4*512*256*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225120448, &readFileBin);
	varresnetv22_stage4_conv0_fwd = (float(*)[512][3][3][256])malloc(sizeof(float[512][3][3][256]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225122032, &readFileBin);
	varresnetv22_stage4_conv0_fwd__1 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Constant input
	varconv_bias__15 = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varconv_bias__15,4*512);

	//Convolution
	//512 3 3 256 
	read_vector(&filter225170432, &readFileBin);
	//3 3 
	read_vector(&kernel225170432, &readFileBin);
	//2 2 
	read_vector(&stride225170432, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225170432, &readFileBin);
	readFileBin.read((char *)&group225170432, sizeof(size_t));
	varresnetv22_stage4_conv0_fwd__2 = (float(*)[1][7][7][512])malloc(sizeof(float[1][7][7][512]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225122544, &readFileBin);
	varresnetv22_stage4_conv0_fwd__3 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));

	//Constant input
	varresnetv22_stage4_batchnorm1_gamma = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm1_gamma,4*512);

	//Constant input
	varresnetv22_stage4_batchnorm1_beta = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm1_beta,4*512);

	//Constant input
	varresnetv22_stage4_batchnorm1_running_mean = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm1_running_mean,4*512);

	//Constant input
	varresnetv22_stage4_batchnorm1_running_var = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm1_running_var,4*512);

	//BatchNormalization
	readFileBin.read((char *)&channelVar15, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar15, sizeof(float));
	readFileBin.read((char *)&pad15, sizeof(float));
	varresnetv22_stage4_batchnorm1_fwd = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	if(varresnetv22_stage4_batchnorm1_fwd != NULL)
	{
		memset(varresnetv22_stage4_batchnorm1_fwd, 0x00, sizeof(float[1][512][7][7]));
	}
	if(varresnetv22_stage4_batchnorm1_fwd != NULL)
	{
		varresnetv22_stage4_batchnorm1_fwd = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}
	if(varresnetv22_stage4_activation1 != NULL)
	{
		varresnetv22_stage4_activation1 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}

	//Constant input
	varresnetv22_stage4_conv1_weight = (float(*)[512][512][3][3])malloc(sizeof(float[512][512][3][3]));
	readFileBin.read((char *)varresnetv22_stage4_conv1_weight,4*512*512*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225172784, &readFileBin);
	varresnetv22_stage4_conv1_fwd = (float(*)[512][3][3][512])malloc(sizeof(float[512][3][3][512]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225170320, &readFileBin);
	varresnetv22_stage4_conv1_fwd__1 = (float(*)[1][7][7][512])malloc(sizeof(float[1][7][7][512]));

	//Constant input
	varconv_bias__16 = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varconv_bias__16,4*512);

	//Convolution
	//512 3 3 512 
	read_vector(&filter225176656, &readFileBin);
	//3 3 
	read_vector(&kernel225176656, &readFileBin);
	//1 1 
	read_vector(&stride225176656, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225176656, &readFileBin);
	readFileBin.read((char *)&group225176656, sizeof(size_t));
	varresnetv22_stage4_conv1_fwd__2 = (float(*)[1][7][7][512])malloc(sizeof(float[1][7][7][512]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225177072, &readFileBin);
	varresnetv22_stage4_conv1_fwd__3 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));

	//Constant input
	varresnetv22_stage4_conv2_weight = (float(*)[512][256][1][1])malloc(sizeof(float[512][256][1][1]));
	readFileBin.read((char *)varresnetv22_stage4_conv2_weight,4*512*256*1*1);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225172848, &readFileBin);
	varresnetv22_stage4_conv2_fwd = (float(*)[512][1][1][256])malloc(sizeof(float[512][1][1][256]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225175664, &readFileBin);
	varresnetv22_stage4_conv2_fwd__1 = (float(*)[1][14][14][256])malloc(sizeof(float[1][14][14][256]));

	//Constant input
	varconv_bias__17 = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varconv_bias__17,4*512);

	//Convolution
	//512 1 1 256 
	read_vector(&filter225181856, &readFileBin);
	//1 1 
	read_vector(&kernel225181856, &readFileBin);
	//2 2 
	read_vector(&stride225181856, &readFileBin);
	//0 0 0 0 
	read_vector(&pad225181856, &readFileBin);
	readFileBin.read((char *)&group225181856, sizeof(size_t));
	varresnetv22_stage4_conv2_fwd__2 = (float(*)[1][7][7][512])malloc(sizeof(float[1][7][7][512]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225182272, &readFileBin);
	varresnetv22_stage4_conv2_fwd__3 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	if(varresnetv22_stage4__plus0 != NULL)
	{
		varresnetv22_stage4__plus0 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}

	//Constant input
	varresnetv22_stage4_batchnorm2_gamma = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm2_gamma,4*512);

	//Constant input
	varresnetv22_stage4_batchnorm2_beta = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm2_beta,4*512);

	//Constant input
	varresnetv22_stage4_batchnorm2_running_mean = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm2_running_mean,4*512);

	//Constant input
	varresnetv22_stage4_batchnorm2_running_var = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm2_running_var,4*512);

	//BatchNormalization
	readFileBin.read((char *)&channelVar16, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar16, sizeof(float));
	readFileBin.read((char *)&pad16, sizeof(float));
	varresnetv22_stage4_batchnorm2_fwd = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	if(varresnetv22_stage4_batchnorm2_fwd != NULL)
	{
		memset(varresnetv22_stage4_batchnorm2_fwd, 0x00, sizeof(float[1][512][7][7]));
	}
	if(varresnetv22_stage4_batchnorm2_fwd != NULL)
	{
		varresnetv22_stage4_batchnorm2_fwd = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}
	if(varresnetv22_stage4_activation2 != NULL)
	{
		varresnetv22_stage4_activation2 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}
	
	//Constant input
	varresnetv22_stage4_conv3_weight = (float(*)[512][512][3][3])malloc(sizeof(float[512][512][3][3]));
	readFileBin.read((char *)varresnetv22_stage4_conv3_weight,4*512*512*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225184880, &readFileBin);
	varresnetv22_stage4_conv3_fwd = (float(*)[512][3][3][512])malloc(sizeof(float[512][3][3][512]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225188416, &readFileBin);
	varresnetv22_stage4_conv3_fwd__1 = (float(*)[1][7][7][512])malloc(sizeof(float[1][7][7][512]));

	//Constant input
	varconv_bias__18 = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varconv_bias__18,4*512);

	//Convolution
	//512 3 3 512 
	read_vector(&filter225188640, &readFileBin);
	//3 3 
	read_vector(&kernel225188640, &readFileBin);
	//1 1 
	read_vector(&stride225188640, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225188640, &readFileBin);
	readFileBin.read((char *)&group225188640, sizeof(size_t));
	varresnetv22_stage4_conv3_fwd__2 = (float(*)[1][7][7][512])malloc(sizeof(float[1][7][7][512]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225189056, &readFileBin);
	varresnetv22_stage4_conv3_fwd__3 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));

	//Constant input
	varresnetv22_stage4_batchnorm3_gamma = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm3_gamma,4*512);

	//Constant input
	varresnetv22_stage4_batchnorm3_beta = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm3_beta,4*512);

	//Constant input
	varresnetv22_stage4_batchnorm3_running_mean = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm3_running_mean,4*512);

	//Constant input
	varresnetv22_stage4_batchnorm3_running_var = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_stage4_batchnorm3_running_var,4*512);

	//BatchNormalization
	readFileBin.read((char *)&channelVar17, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar17, sizeof(float));
	readFileBin.read((char *)&pad17, sizeof(float));
	varresnetv22_stage4_batchnorm3_fwd = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	if(varresnetv22_stage4_batchnorm3_fwd != NULL)
	{
		memset(varresnetv22_stage4_batchnorm3_fwd, 0x00, sizeof(float[1][512][7][7]));
	}
	if(varresnetv22_stage4_batchnorm3_fwd != NULL)
	{
		varresnetv22_stage4_batchnorm3_fwd = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}
	if(varresnetv22_stage4_activation3 != NULL)
	{
		varresnetv22_stage4_activation3 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}

	//Constant input
	varresnetv22_stage4_conv4_weight = (float(*)[512][512][3][3])malloc(sizeof(float[512][512][3][3]));
	readFileBin.read((char *)varresnetv22_stage4_conv4_weight,4*512*512*3*3);

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225191216, &readFileBin);
	varresnetv22_stage4_conv4_fwd = (float(*)[512][3][3][512])malloc(sizeof(float[512][3][3][512]));

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225194752, &readFileBin);
	varresnetv22_stage4_conv4_fwd__1 = (float(*)[1][7][7][512])malloc(sizeof(float[1][7][7][512]));

	//Constant input
	varconv_bias__19 = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varconv_bias__19,4*512);

	//Convolution
	//512 3 3 512 
	read_vector(&filter225194976, &readFileBin);
	//3 3 
	read_vector(&kernel225194976, &readFileBin);
	//1 1 
	read_vector(&stride225194976, &readFileBin);
	//1 1 1 1 
	read_vector(&pad225194976, &readFileBin);
	readFileBin.read((char *)&group225194976, sizeof(size_t));
	varresnetv22_stage4_conv4_fwd__2 = (float(*)[1][7][7][512])malloc(sizeof(float[1][7][7][512]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225195392, &readFileBin);
	varresnetv22_stage4_conv4_fwd__3 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	if(varresnetv22_stage4__plus1 != NULL)
	{
		varresnetv22_stage4__plus1 = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}

	//Constant input
	varresnetv22_batchnorm2_gamma = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_batchnorm2_gamma,4*512);

	//Constant input
	varresnetv22_batchnorm2_beta = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_batchnorm2_beta,4*512);

	//Constant input
	varresnetv22_batchnorm2_running_mean = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_batchnorm2_running_mean,4*512);

	//Constant input
	varresnetv22_batchnorm2_running_var = (float(*)[512])malloc(sizeof(float[512]));
	readFileBin.read((char *)varresnetv22_batchnorm2_running_var,4*512);

	//BatchNormalization
	readFileBin.read((char *)&channelVar18, sizeof(unsigned int));
	readFileBin.read((char *)&epsilonVar18, sizeof(float));
	readFileBin.read((char *)&pad18, sizeof(float));
	varresnetv22_batchnorm2_fwd = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	if(varresnetv22_batchnorm2_fwd != NULL)
	{
		memset(varresnetv22_batchnorm2_fwd, 0x00, sizeof(float[1][512][7][7]));
	}
	if(varresnetv22_batchnorm2_fwd != NULL)
	{
		varresnetv22_batchnorm2_fwd = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}
	if(varresnetv22_relu1_fwd != NULL)
	{
		varresnetv22_relu1_fwd = (float(*)[1][512][7][7])malloc(sizeof(float[1][512][7][7]));
	}
	

	//Transpose
	//0 2 3 1 
	read_vector(&shuffle225198176, &readFileBin);
	varresnetv22_pool1_fwd = (float(*)[1][7][7][512])malloc(sizeof(float[1][7][7][512]));

	//MaxPool
	//7 7 
	read_vector(&kernel225198864, &readFileBin);
	//1 1 
	read_vector(&stride225198896, &readFileBin);
	//0 0 0 0 
	read_vector(&pad225198928, &readFileBin);
	varresnetv22_pool1_fwd__1 = (float(*)[1][1][1][512])malloc(sizeof(float[1][1][1][512]));

	//Transpose
	//0 3 1 2 
	read_vector(&shuffle225198800, &readFileBin);
	varresnetv22_pool1_fwd__2 = (float(*)[1][512][1][1])malloc(sizeof(float[1][512][1][1]));

	//Reshape
	if(varresnetv22_flatten0_reshape0 != NULL)
	{
		varresnetv22_flatten0_reshape0 = (float(*)[1][512])malloc(sizeof(float[1][512]));
		varresnetv22_flatten0_reshape0 = (float(*)[1][512])malloc(sizeof(float[1][512]));
	}
	
	
	
	//Constant input
	varresnetv22_dense0_weight = (float(*)[1000][512])malloc(sizeof(float[1000][512]));
	readFileBin.read((char *)varresnetv22_dense0_weight,4*1000*512);

	//Transpose
	//1 0 
	read_vector(&shuffle225201120, &readFileBin);
	varresnetv22_dense0_fwd = (float(*)[512][1000])malloc(sizeof(float[512][1000]));

	varresnetv22_dense0_fwd__1 = (float(*)[1][1000])malloc(sizeof(float[1][1000]));

	//Constant input
	varresnetv22_dense0_bias = (float(*)[1000])malloc(sizeof(float[1000]));
	readFileBin.read((char *)varresnetv22_dense0_bias,4*1000);

	//Reshape
	if(varresnetv22_dense0_fwd_reshape != NULL)
	{
		varresnetv22_dense0_fwd_reshape = (float(*)[1][1000])malloc(sizeof(float[1][1000]));
		varresnetv22_dense0_fwd_reshape = (float(*)[1][1000])malloc(sizeof(float[1][1000]));
	}
	
	if(varresnetv22_dense0_fwd__2 != NULL)
	{
		varresnetv22_dense0_fwd__2 = (float(*)[1][1000])malloc(sizeof(float[1][1000]));
	}

}

void inference(float(*vardata)[1][3][224][224], float result[1][1000])
 {
	//BatchNormalization
	batchNorm(*vardata, *varresnetv22_batchnorm0_gamma, *varresnetv22_batchnorm0_beta, *varresnetv22_batchnorm0_running_mean, *varresnetv22_batchnorm0_running_var, channelVar0, epsilonVar0, pad0, *varresnetv22_batchnorm0_fwd);

	//Transpose
	transpose(*varresnetv22_conv0_weight, shuffle225081392, *varresnetv22_conv0_fwd);

	//Transpose
	transpose(*varresnetv22_batchnorm0_fwd, shuffle225082800, *varresnetv22_conv0_fwd__1);

	//Convolution
	conv(*varresnetv22_conv0_fwd__1, *varresnetv22_conv0_fwd, filter225083184, *varconv_bias, stride225083184, pad225083184, group225083184, *varresnetv22_conv0_fwd__2);

	//Transpose
	transpose(*varresnetv22_conv0_fwd__2, shuffle225083600, *varresnetv22_conv0_fwd__3);

	//BatchNormalization
	batchNorm(*varresnetv22_conv0_fwd__3, *varresnetv22_batchnorm1_gamma, *varresnetv22_batchnorm1_beta, *varresnetv22_batchnorm1_running_mean, *varresnetv22_batchnorm1_running_var, channelVar1, epsilonVar1, pad1, *varresnetv22_batchnorm1_fwd);

	//Relu
	relu(*varresnetv22_batchnorm1_fwd);
	varresnetv22_relu0_fwd = varresnetv22_batchnorm1_fwd;
	//Transpose
	transpose(*varresnetv22_relu0_fwd, shuffle225086048, *varresnetv22_pool0_fwd);

	//MaxPool
	maxpool(*varresnetv22_pool0_fwd, kernel225086496, stride225086496, pad225086496, *varresnetv22_pool0_fwd__1);

	//Transpose
	transpose(*varresnetv22_pool0_fwd__1, shuffle225086896, *varresnetv22_pool0_fwd__2);

	//BatchNormalization
	batchNorm(*varresnetv22_pool0_fwd__2, *varresnetv22_stage1_batchnorm0_gamma, *varresnetv22_stage1_batchnorm0_beta, *varresnetv22_stage1_batchnorm0_running_mean, *varresnetv22_stage1_batchnorm0_running_var, channelVar2, epsilonVar2, pad2, *varresnetv22_stage1_batchnorm0_fwd);

	//Relu
	relu(*varresnetv22_stage1_batchnorm0_fwd);
	varresnetv22_stage1_activation0 = varresnetv22_stage1_batchnorm0_fwd;
	//Transpose
	transpose(*varresnetv22_stage1_conv0_weight, shuffle225089136, *varresnetv22_stage1_conv0_fwd);

	//Transpose
	transpose(*varresnetv22_stage1_activation0, shuffle225080240, *varresnetv22_stage1_conv0_fwd__1);

	//Convolution
	conv(*varresnetv22_stage1_conv0_fwd__1, *varresnetv22_stage1_conv0_fwd, filter225091856, *varconv_bias__1, stride225091856, pad225091856, group225091856, *varresnetv22_stage1_conv0_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage1_conv0_fwd__2, shuffle225092304, *varresnetv22_stage1_conv0_fwd__3);

	//BatchNormalization
	batchNorm(*varresnetv22_stage1_conv0_fwd__3, *varresnetv22_stage1_batchnorm1_gamma, *varresnetv22_stage1_batchnorm1_beta, *varresnetv22_stage1_batchnorm1_running_mean, *varresnetv22_stage1_batchnorm1_running_var, channelVar3, epsilonVar3, pad3, *varresnetv22_stage1_batchnorm1_fwd);

	//Relu
	relu(*varresnetv22_stage1_batchnorm1_fwd);
	varresnetv22_stage1_activation1 = varresnetv22_stage1_batchnorm1_fwd;
	//Transpose
	transpose(*varresnetv22_stage1_conv1_weight, shuffle225094320, *varresnetv22_stage1_conv1_fwd);

	//Transpose
	transpose(*varresnetv22_stage1_activation1, shuffle225096096, *varresnetv22_stage1_conv1_fwd__1);

	//Convolution
	conv(*varresnetv22_stage1_conv1_fwd__1, *varresnetv22_stage1_conv1_fwd, filter225096368, *varconv_bias__2, stride225096368, pad225096368, group225096368, *varresnetv22_stage1_conv1_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage1_conv1_fwd__2, shuffle225096784, *varresnetv22_stage1_conv1_fwd__3);

	//Add
	add(*varresnetv22_stage1_conv1_fwd__3, *varresnetv22_pool0_fwd__2);
	varresnetv22_stage1__plus0 = varresnetv22_stage1_conv1_fwd__3;
	//BatchNormalization
	batchNorm(*varresnetv22_stage1__plus0, *varresnetv22_stage1_batchnorm2_gamma, *varresnetv22_stage1_batchnorm2_beta, *varresnetv22_stage1_batchnorm2_running_mean, *varresnetv22_stage1_batchnorm2_running_var, channelVar4, epsilonVar4, pad4, *varresnetv22_stage1_batchnorm2_fwd);

	//Relu
	relu(*varresnetv22_stage1_batchnorm2_fwd);
	varresnetv22_stage1_activation2 = varresnetv22_stage1_batchnorm2_fwd;
	//Transpose
	transpose(*varresnetv22_stage1_conv2_weight, shuffle225090144, *varresnetv22_stage1_conv2_fwd);

	//Transpose
	transpose(*varresnetv22_stage1_activation2, shuffle225101952, *varresnetv22_stage1_conv2_fwd__1);

	//Convolution
	conv(*varresnetv22_stage1_conv2_fwd__1, *varresnetv22_stage1_conv2_fwd, filter225102288, *varconv_bias__3, stride225102288, pad225102288, group225102288, *varresnetv22_stage1_conv2_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage1_conv2_fwd__2, shuffle225102704, *varresnetv22_stage1_conv2_fwd__3);

	//BatchNormalization
	batchNorm(*varresnetv22_stage1_conv2_fwd__3, *varresnetv22_stage1_batchnorm3_gamma, *varresnetv22_stage1_batchnorm3_beta, *varresnetv22_stage1_batchnorm3_running_mean, *varresnetv22_stage1_batchnorm3_running_var, channelVar5, epsilonVar5, pad5, *varresnetv22_stage1_batchnorm3_fwd);

	//Relu
	relu(*varresnetv22_stage1_batchnorm3_fwd);
	varresnetv22_stage1_activation3 = varresnetv22_stage1_batchnorm3_fwd;
	//Transpose
	transpose(*varresnetv22_stage1_conv3_weight, shuffle225104784, *varresnetv22_stage1_conv3_fwd);

	//Transpose
	transpose(*varresnetv22_stage1_activation3, shuffle225106608, *varresnetv22_stage1_conv3_fwd__1);

	//Convolution
	conv(*varresnetv22_stage1_conv3_fwd__1, *varresnetv22_stage1_conv3_fwd, filter225106944, *varconv_bias__4, stride225106944, pad225106944, group225106944, *varresnetv22_stage1_conv3_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage1_conv3_fwd__2, shuffle225107360, *varresnetv22_stage1_conv3_fwd__3);

	//Add
	add(*varresnetv22_stage1_conv3_fwd__3, *varresnetv22_stage1__plus0);
	varresnetv22_stage1__plus1 = varresnetv22_stage1_conv3_fwd__3;
	//BatchNormalization
	batchNorm(*varresnetv22_stage1__plus1, *varresnetv22_stage2_batchnorm0_gamma, *varresnetv22_stage2_batchnorm0_beta, *varresnetv22_stage2_batchnorm0_running_mean, *varresnetv22_stage2_batchnorm0_running_var, channelVar6, epsilonVar6, pad6, *varresnetv22_stage2_batchnorm0_fwd);

	//Relu
	relu(*varresnetv22_stage2_batchnorm0_fwd);
	varresnetv22_stage2_activation0 = varresnetv22_stage2_batchnorm0_fwd;
	//Transpose
	transpose(*varresnetv22_stage2_conv0_weight, shuffle225109968, *varresnetv22_stage2_conv0_fwd);

	//Transpose
	transpose(*varresnetv22_stage2_activation0, shuffle225112192, *varresnetv22_stage2_conv0_fwd__1);

	//Convolution
	conv(*varresnetv22_stage2_conv0_fwd__1, *varresnetv22_stage2_conv0_fwd, filter225112672, *varconv_bias__5, stride225112672, pad225112672, group225112672, *varresnetv22_stage2_conv0_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage2_conv0_fwd__2, shuffle225113088, *varresnetv22_stage2_conv0_fwd__3);

	//BatchNormalization
	batchNorm(*varresnetv22_stage2_conv0_fwd__3, *varresnetv22_stage2_batchnorm1_gamma, *varresnetv22_stage2_batchnorm1_beta, *varresnetv22_stage2_batchnorm1_running_mean, *varresnetv22_stage2_batchnorm1_running_var, channelVar7, epsilonVar7, pad7, *varresnetv22_stage2_batchnorm1_fwd);

	//Relu
	relu(*varresnetv22_stage2_batchnorm1_fwd);
	varresnetv22_stage2_activation1 = varresnetv22_stage2_batchnorm1_fwd;
	//Transpose
	transpose(*varresnetv22_stage2_conv1_weight, shuffle225115312, *varresnetv22_stage2_conv1_fwd);

	//Transpose
	transpose(*varresnetv22_stage2_activation1, shuffle225117504, *varresnetv22_stage2_conv1_fwd__1);

	//Convolution
	conv(*varresnetv22_stage2_conv1_fwd__1, *varresnetv22_stage2_conv1_fwd, filter225117840, *varconv_bias__6, stride225117840, pad225117840, group225117840, *varresnetv22_stage2_conv1_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage2_conv1_fwd__2, shuffle225118256, *varresnetv22_stage2_conv1_fwd__3);

	//Transpose
	transpose(*varresnetv22_stage2_conv2_weight, shuffle225115376, *varresnetv22_stage2_conv2_fwd);

	//Transpose
	transpose(*varresnetv22_stage2_activation0, shuffle225100464, *varresnetv22_stage2_conv2_fwd__1);

	//Convolution
	conv(*varresnetv22_stage2_conv2_fwd__1, *varresnetv22_stage2_conv2_fwd, filter225122880, *varconv_bias__7, stride225122880, pad225122880, group225122880, *varresnetv22_stage2_conv2_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage2_conv2_fwd__2, shuffle225123472, *varresnetv22_stage2_conv2_fwd__3);

	//Add
	add(*varresnetv22_stage2_conv1_fwd__3, *varresnetv22_stage2_conv2_fwd__3);
	varresnetv22_stage2__plus0 = varresnetv22_stage2_conv1_fwd__3;
	//BatchNormalization
	batchNorm(*varresnetv22_stage2__plus0, *varresnetv22_stage2_batchnorm2_gamma, *varresnetv22_stage2_batchnorm2_beta, *varresnetv22_stage2_batchnorm2_running_mean, *varresnetv22_stage2_batchnorm2_running_var, channelVar8, epsilonVar8, pad8, *varresnetv22_stage2_batchnorm2_fwd);

	//Relu
	relu(*varresnetv22_stage2_batchnorm2_fwd);
	varresnetv22_stage2_activation2 = varresnetv22_stage2_batchnorm2_fwd;
	//Transpose
	transpose(*varresnetv22_stage2_conv3_weight, shuffle225125664, *varresnetv22_stage2_conv3_fwd);

	//Transpose
	transpose(*varresnetv22_stage2_activation2, shuffle225127744, *varresnetv22_stage2_conv3_fwd__1);

	//Convolution
	conv(*varresnetv22_stage2_conv3_fwd__1, *varresnetv22_stage2_conv3_fwd, filter225128080, *varconv_bias__8, stride225128080, pad225128080, group225128080, *varresnetv22_stage2_conv3_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage2_conv3_fwd__2, shuffle225128496, *varresnetv22_stage2_conv3_fwd__3);

	//BatchNormalization
	batchNorm(*varresnetv22_stage2_conv3_fwd__3, *varresnetv22_stage2_batchnorm3_gamma, *varresnetv22_stage2_batchnorm3_beta, *varresnetv22_stage2_batchnorm3_running_mean, *varresnetv22_stage2_batchnorm3_running_var, channelVar9, epsilonVar9, pad9, *varresnetv22_stage2_batchnorm3_fwd);

	//Relu
	relu(*varresnetv22_stage2_batchnorm3_fwd);
	varresnetv22_stage2_activation3 = varresnetv22_stage2_batchnorm3_fwd;
	//Transpose
	transpose(*varresnetv22_stage2_conv4_weight, shuffle225130576, *varresnetv22_stage2_conv4_fwd);

	//Transpose
	transpose(*varresnetv22_stage2_activation3, shuffle225132656, *varresnetv22_stage2_conv4_fwd__1);

	//Convolution
	conv(*varresnetv22_stage2_conv4_fwd__1, *varresnetv22_stage2_conv4_fwd, filter225132992, *varconv_bias__9, stride225132992, pad225132992, group225132992, *varresnetv22_stage2_conv4_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage2_conv4_fwd__2, shuffle225133408, *varresnetv22_stage2_conv4_fwd__3);

	//Add
	add(*varresnetv22_stage2_conv4_fwd__3, *varresnetv22_stage2__plus0);
	varresnetv22_stage2__plus1 = varresnetv22_stage2_conv4_fwd__3;
	//BatchNormalization
	batchNorm(*varresnetv22_stage2__plus1, *varresnetv22_stage3_batchnorm0_gamma, *varresnetv22_stage3_batchnorm0_beta, *varresnetv22_stage3_batchnorm0_running_mean, *varresnetv22_stage3_batchnorm0_running_var, channelVar10, epsilonVar10, pad10, *varresnetv22_stage3_batchnorm0_fwd);

	//Relu
	relu(*varresnetv22_stage3_batchnorm0_fwd);
	varresnetv22_stage3_activation0 = varresnetv22_stage3_batchnorm0_fwd;
	//Transpose
	transpose(*varresnetv22_stage3_conv0_weight, shuffle225136016, *varresnetv22_stage3_conv0_fwd);

	//Transpose
	transpose(*varresnetv22_stage3_activation0, shuffle225105424, *varresnetv22_stage3_conv0_fwd__1);

	//Convolution
	conv(*varresnetv22_stage3_conv0_fwd__1, *varresnetv22_stage3_conv0_fwd, filter225138848, *varconv_bias__10, stride225138848, pad225138848, group225138848, *varresnetv22_stage3_conv0_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage3_conv0_fwd__2, shuffle225139264, *varresnetv22_stage3_conv0_fwd__3);

	//BatchNormalization
	batchNorm(*varresnetv22_stage3_conv0_fwd__3, *varresnetv22_stage3_batchnorm1_gamma, *varresnetv22_stage3_batchnorm1_beta, *varresnetv22_stage3_batchnorm1_running_mean, *varresnetv22_stage3_batchnorm1_running_var, channelVar11, epsilonVar11, pad11, *varresnetv22_stage3_batchnorm1_fwd);

	//Relu
	relu(*varresnetv22_stage3_batchnorm1_fwd);
	varresnetv22_stage3_activation1 = varresnetv22_stage3_batchnorm1_fwd;
	//Transpose
	transpose(*varresnetv22_stage3_conv1_weight, shuffle225141616, *varresnetv22_stage3_conv1_fwd);

	//Transpose
	transpose(*varresnetv22_stage3_activation1, shuffle225137808, *varresnetv22_stage3_conv1_fwd__1);

	//Convolution
	conv(*varresnetv22_stage3_conv1_fwd__1, *varresnetv22_stage3_conv1_fwd, filter225144464, *varconv_bias__11, stride225144464, pad225144464, group225144464, *varresnetv22_stage3_conv1_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage3_conv1_fwd__2, shuffle225144880, *varresnetv22_stage3_conv1_fwd__3);

	//Transpose
	transpose(*varresnetv22_stage3_conv2_weight, shuffle225141680, *varresnetv22_stage3_conv2_fwd);

	//Transpose
	transpose(*varresnetv22_stage3_activation0, shuffle225143472, *varresnetv22_stage3_conv2_fwd__1);

	//Convolution
	conv(*varresnetv22_stage3_conv2_fwd__1, *varresnetv22_stage3_conv2_fwd, filter225148640, *varconv_bias__12, stride225148640, pad225148640, group225148640, *varresnetv22_stage3_conv2_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage3_conv2_fwd__2, shuffle225149056, *varresnetv22_stage3_conv2_fwd__3);

	//Add
	add(*varresnetv22_stage3_conv1_fwd__3, *varresnetv22_stage3_conv2_fwd__3);
	varresnetv22_stage3__plus0 = varresnetv22_stage3_conv1_fwd__3;
	//BatchNormalization
	batchNorm(*varresnetv22_stage3__plus0, *varresnetv22_stage3_batchnorm2_gamma, *varresnetv22_stage3_batchnorm2_beta, *varresnetv22_stage3_batchnorm2_running_mean, *varresnetv22_stage3_batchnorm2_running_var, channelVar12, epsilonVar12, pad12, *varresnetv22_stage3_batchnorm2_fwd);

	//Relu
	relu(*varresnetv22_stage3_batchnorm2_fwd);
	varresnetv22_stage3_activation2 = varresnetv22_stage3_batchnorm2_fwd;
	//Transpose
	transpose(*varresnetv22_stage3_conv3_weight, shuffle225151664, *varresnetv22_stage3_conv3_fwd);

	//Transpose
	transpose(*varresnetv22_stage3_activation2, shuffle225154176, *varresnetv22_stage3_conv3_fwd__1);

	//Convolution
	conv(*varresnetv22_stage3_conv3_fwd__1, *varresnetv22_stage3_conv3_fwd, filter225154400, *varconv_bias__13, stride225154400, pad225154400, group225154400, *varresnetv22_stage3_conv3_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage3_conv3_fwd__2, shuffle225154816, *varresnetv22_stage3_conv3_fwd__3);

	//BatchNormalization
	batchNorm(*varresnetv22_stage3_conv3_fwd__3, *varresnetv22_stage3_batchnorm3_gamma, *varresnetv22_stage3_batchnorm3_beta, *varresnetv22_stage3_batchnorm3_running_mean, *varresnetv22_stage3_batchnorm3_running_var, channelVar13, epsilonVar13, pad13, *varresnetv22_stage3_batchnorm3_fwd);

	//Relu
	relu(*varresnetv22_stage3_batchnorm3_fwd);
	varresnetv22_stage3_activation3 = varresnetv22_stage3_batchnorm3_fwd;
	//Transpose
	transpose(*varresnetv22_stage3_conv4_weight, shuffle225156976, *varresnetv22_stage3_conv4_fwd);

	//Transpose
	transpose(*varresnetv22_stage3_activation3, shuffle225159488, *varresnetv22_stage3_conv4_fwd__1);

	//Convolution
	conv(*varresnetv22_stage3_conv4_fwd__1, *varresnetv22_stage3_conv4_fwd, filter225159712, *varconv_bias__14, stride225159712, pad225159712, group225159712, *varresnetv22_stage3_conv4_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage3_conv4_fwd__2, shuffle225160128, *varresnetv22_stage3_conv4_fwd__3);

	//Add
	add(*varresnetv22_stage3_conv4_fwd__3, *varresnetv22_stage3__plus0);
	varresnetv22_stage3__plus1 = varresnetv22_stage3_conv4_fwd__3;
	//BatchNormalization
	batchNorm(*varresnetv22_stage3__plus1, *varresnetv22_stage4_batchnorm0_gamma, *varresnetv22_stage4_batchnorm0_beta, *varresnetv22_stage4_batchnorm0_running_mean, *varresnetv22_stage4_batchnorm0_running_var, channelVar14, epsilonVar14, pad14, *varresnetv22_stage4_batchnorm0_fwd);

	//Relu
	relu(*varresnetv22_stage4_batchnorm0_fwd);
	varresnetv22_stage4_activation0 = varresnetv22_stage4_batchnorm0_fwd;
	//Transpose
	transpose(*varresnetv22_stage4_conv0_weight, shuffle225120448, *varresnetv22_stage4_conv0_fwd);

	//Transpose
	transpose(*varresnetv22_stage4_activation0, shuffle225122032, *varresnetv22_stage4_conv0_fwd__1);

	//Convolution
	conv(*varresnetv22_stage4_conv0_fwd__1, *varresnetv22_stage4_conv0_fwd, filter225170432, *varconv_bias__15, stride225170432, pad225170432, group225170432, *varresnetv22_stage4_conv0_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage4_conv0_fwd__2, shuffle225122544, *varresnetv22_stage4_conv0_fwd__3);

	//BatchNormalization
	batchNorm(*varresnetv22_stage4_conv0_fwd__3, *varresnetv22_stage4_batchnorm1_gamma, *varresnetv22_stage4_batchnorm1_beta, *varresnetv22_stage4_batchnorm1_running_mean, *varresnetv22_stage4_batchnorm1_running_var, channelVar15, epsilonVar15, pad15, *varresnetv22_stage4_batchnorm1_fwd);

	//Relu
	relu(*varresnetv22_stage4_batchnorm1_fwd);
	varresnetv22_stage4_activation1 = varresnetv22_stage4_batchnorm1_fwd;
	//Transpose
	transpose(*varresnetv22_stage4_conv1_weight, shuffle225172784, *varresnetv22_stage4_conv1_fwd);

	//Transpose
	transpose(*varresnetv22_stage4_activation1, shuffle225170320, *varresnetv22_stage4_conv1_fwd__1);

	//Convolution
	conv(*varresnetv22_stage4_conv1_fwd__1, *varresnetv22_stage4_conv1_fwd, filter225176656, *varconv_bias__16, stride225176656, pad225176656, group225176656, *varresnetv22_stage4_conv1_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage4_conv1_fwd__2, shuffle225177072, *varresnetv22_stage4_conv1_fwd__3);

	//Transpose
	transpose(*varresnetv22_stage4_conv2_weight, shuffle225172848, *varresnetv22_stage4_conv2_fwd);

	//Transpose
	transpose(*varresnetv22_stage4_activation0, shuffle225175664, *varresnetv22_stage4_conv2_fwd__1);

	//Convolution
	conv(*varresnetv22_stage4_conv2_fwd__1, *varresnetv22_stage4_conv2_fwd, filter225181856, *varconv_bias__17, stride225181856, pad225181856, group225181856, *varresnetv22_stage4_conv2_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage4_conv2_fwd__2, shuffle225182272, *varresnetv22_stage4_conv2_fwd__3);

	//Add
	add(*varresnetv22_stage4_conv1_fwd__3, *varresnetv22_stage4_conv2_fwd__3);
	varresnetv22_stage4__plus0 = varresnetv22_stage4_conv1_fwd__3;
	//BatchNormalization
	batchNorm(*varresnetv22_stage4__plus0, *varresnetv22_stage4_batchnorm2_gamma, *varresnetv22_stage4_batchnorm2_beta, *varresnetv22_stage4_batchnorm2_running_mean, *varresnetv22_stage4_batchnorm2_running_var, channelVar16, epsilonVar16, pad16, *varresnetv22_stage4_batchnorm2_fwd);

	//Relu
	relu(*varresnetv22_stage4_batchnorm2_fwd);
	varresnetv22_stage4_activation2 = varresnetv22_stage4_batchnorm2_fwd;
	//Transpose
	transpose(*varresnetv22_stage4_conv3_weight, shuffle225184880, *varresnetv22_stage4_conv3_fwd);

	//Transpose
	transpose(*varresnetv22_stage4_activation2, shuffle225188416, *varresnetv22_stage4_conv3_fwd__1);

	//Convolution
	conv(*varresnetv22_stage4_conv3_fwd__1, *varresnetv22_stage4_conv3_fwd, filter225188640, *varconv_bias__18, stride225188640, pad225188640, group225188640, *varresnetv22_stage4_conv3_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage4_conv3_fwd__2, shuffle225189056, *varresnetv22_stage4_conv3_fwd__3);

	//BatchNormalization
	batchNorm(*varresnetv22_stage4_conv3_fwd__3, *varresnetv22_stage4_batchnorm3_gamma, *varresnetv22_stage4_batchnorm3_beta, *varresnetv22_stage4_batchnorm3_running_mean, *varresnetv22_stage4_batchnorm3_running_var, channelVar17, epsilonVar17, pad17, *varresnetv22_stage4_batchnorm3_fwd);

	//Relu
	relu(*varresnetv22_stage4_batchnorm3_fwd);
	varresnetv22_stage4_activation3 = varresnetv22_stage4_batchnorm3_fwd;
	//Transpose
	transpose(*varresnetv22_stage4_conv4_weight, shuffle225191216, *varresnetv22_stage4_conv4_fwd);

	//Transpose
	transpose(*varresnetv22_stage4_activation3, shuffle225194752, *varresnetv22_stage4_conv4_fwd__1);

	//Convolution
	conv(*varresnetv22_stage4_conv4_fwd__1, *varresnetv22_stage4_conv4_fwd, filter225194976, *varconv_bias__19, stride225194976, pad225194976, group225194976, *varresnetv22_stage4_conv4_fwd__2);

	//Transpose
	transpose(*varresnetv22_stage4_conv4_fwd__2, shuffle225195392, *varresnetv22_stage4_conv4_fwd__3);

	//Add
	add(*varresnetv22_stage4_conv4_fwd__3, *varresnetv22_stage4__plus0);
	varresnetv22_stage4__plus1 = varresnetv22_stage4_conv4_fwd__3;
	//BatchNormalization
	batchNorm(*varresnetv22_stage4__plus1, *varresnetv22_batchnorm2_gamma, *varresnetv22_batchnorm2_beta, *varresnetv22_batchnorm2_running_mean, *varresnetv22_batchnorm2_running_var, channelVar18, epsilonVar18, pad18, *varresnetv22_batchnorm2_fwd);

	//Relu
	relu(*varresnetv22_batchnorm2_fwd);
	varresnetv22_relu1_fwd = varresnetv22_batchnorm2_fwd;
	//Transpose
	transpose(*varresnetv22_relu1_fwd, shuffle225198176, *varresnetv22_pool1_fwd);

	//AvgPool
	avgpool(*varresnetv22_pool1_fwd, kernel225198864, stride225198896, pad225198928, *varresnetv22_pool1_fwd__1);

	//Transpose
	transpose(*varresnetv22_pool1_fwd__1, shuffle225198800, *varresnetv22_pool1_fwd__2);

	//Reshape
	reshape(*varresnetv22_pool1_fwd__2, *varresnetv22_flatten0_reshape0);

	//Transpose
	transpose(*varresnetv22_dense0_weight, shuffle225201120, *varresnetv22_dense0_fwd);

	//Matmul
	matmul(*varresnetv22_flatten0_reshape0, *varresnetv22_dense0_fwd, *varresnetv22_dense0_fwd__1);

	//Reshape
	memcpy((*varresnetv22_dense0_fwd_reshape)[0],varresnetv22_dense0_bias, sizeof(float[1][1000]));
	//Add
	add(*varresnetv22_dense0_fwd__1, *varresnetv22_dense0_fwd_reshape);
	varresnetv22_dense0_fwd__2 = varresnetv22_dense0_fwd__1;
	memcpy(result,varresnetv22_dense0_fwd__2, 4*1*1000);

}

void memFree(float(*vardata)[1][3][224][224], float result[1][1000])
{
	free(vardata);
	free(varresnetv22_conv0_weight);
	free(varresnetv22_batchnorm0_fwd);
	free(varresnetv22_conv0_fwd__1);
	//Convolution
	free(varresnetv22_conv0_fwd);
	free(varconv_bias);
	free(varresnetv22_conv0_fwd__2);
	free(varresnetv22_conv0_fwd__3);
	free(varresnetv22_batchnorm1_fwd);
	free(varresnetv22_pool0_fwd);
	free(varresnetv22_pool0_fwd__1);
	free(varresnetv22_pool0_fwd__2);
	free(varresnetv22_stage1_batchnorm0_fwd);
	free(varresnetv22_stage1_conv0_weight);
	free(varresnetv22_stage1_conv0_fwd__1);
	//Convolution
	free(varresnetv22_stage1_conv0_fwd);
	free(varconv_bias__1);
	free(varresnetv22_stage1_conv0_fwd__2);
	free(varresnetv22_stage1_conv0_fwd__3);
	free(varresnetv22_stage1_batchnorm1_fwd);
	free(varresnetv22_stage1_conv1_weight);
	free(varresnetv22_stage1_conv1_fwd__1);
	//Convolution
	free(varresnetv22_stage1_conv1_fwd);
	free(varconv_bias__2);
	free(varresnetv22_stage1_conv1_fwd__2);
	free(varresnetv22_stage1_conv1_fwd__3);
	//Add
	free(varresnetv22_stage1_batchnorm2_fwd);
	free(varresnetv22_stage1_conv2_weight);
	free(varresnetv22_stage1_conv2_fwd__1);
	//Convolution
	free(varresnetv22_stage1_conv2_fwd);
	free(varconv_bias__3);
	free(varresnetv22_stage1_conv2_fwd__2);
	free(varresnetv22_stage1_conv2_fwd__3);
	free(varresnetv22_stage1_batchnorm3_fwd);
	free(varresnetv22_stage1_conv3_weight);
	free(varresnetv22_stage1_conv3_fwd__1);
	//Convolution
	free(varresnetv22_stage1_conv3_fwd);
	free(varconv_bias__4);
	free(varresnetv22_stage1_conv3_fwd__2);
	free(varresnetv22_stage1_conv3_fwd__3);
	//Add
	free(varresnetv22_stage2_batchnorm0_fwd);
	free(varresnetv22_stage2_conv0_weight);
	free(varresnetv22_stage2_conv0_fwd__1);
	//Convolution
	free(varresnetv22_stage2_conv0_fwd);
	free(varconv_bias__5);
	free(varresnetv22_stage2_conv0_fwd__2);
	free(varresnetv22_stage2_conv0_fwd__3);
	free(varresnetv22_stage2_batchnorm1_fwd);
	free(varresnetv22_stage2_conv1_weight);
	free(varresnetv22_stage2_conv1_fwd__1);
	//Convolution
	free(varresnetv22_stage2_conv1_fwd);
	free(varconv_bias__6);
	free(varresnetv22_stage2_conv1_fwd__2);
	free(varresnetv22_stage2_conv2_weight);
	free(varresnetv22_stage2_conv2_fwd__1);
	//Convolution
	free(varresnetv22_stage2_conv2_fwd);
	free(varconv_bias__7);
	free(varresnetv22_stage2_conv2_fwd__2);
	free(varresnetv22_stage2_conv1_fwd__3);
	//Add
	free(varresnetv22_stage2_conv2_fwd__3);
	free(varresnetv22_stage2_batchnorm2_fwd);
	free(varresnetv22_stage2_conv3_weight);
	free(varresnetv22_stage2_conv3_fwd__1);
	//Convolution
	free(varresnetv22_stage2_conv3_fwd);
	free(varconv_bias__8);
	free(varresnetv22_stage2_conv3_fwd__2);
	free(varresnetv22_stage2_conv3_fwd__3);
	free(varresnetv22_stage2_batchnorm3_fwd);
	free(varresnetv22_stage2_conv4_weight);
	free(varresnetv22_stage2_conv4_fwd__1);
	//Convolution
	free(varresnetv22_stage2_conv4_fwd);
	free(varconv_bias__9);
	free(varresnetv22_stage2_conv4_fwd__2);
	free(varresnetv22_stage2_conv4_fwd__3);
	//Add
	free(varresnetv22_stage3_batchnorm0_fwd);
	free(varresnetv22_stage3_conv0_weight);
	free(varresnetv22_stage3_conv0_fwd__1);
	//Convolution
	free(varresnetv22_stage3_conv0_fwd);
	free(varconv_bias__10);
	free(varresnetv22_stage3_conv0_fwd__2);
	free(varresnetv22_stage3_conv0_fwd__3);
	free(varresnetv22_stage3_batchnorm1_fwd);
	free(varresnetv22_stage3_conv1_weight);
	free(varresnetv22_stage3_conv1_fwd__1);
	//Convolution
	free(varresnetv22_stage3_conv1_fwd);
	free(varconv_bias__11);
	free(varresnetv22_stage3_conv1_fwd__2);
	free(varresnetv22_stage3_conv2_weight);
	free(varresnetv22_stage3_conv2_fwd__1);
	//Convolution
	free(varresnetv22_stage3_conv2_fwd);
	free(varconv_bias__12);
	free(varresnetv22_stage3_conv2_fwd__2);
	free(varresnetv22_stage3_conv1_fwd__3);
	//Add
	free(varresnetv22_stage3_conv2_fwd__3);
	free(varresnetv22_stage3_batchnorm2_fwd);
	free(varresnetv22_stage3_conv3_weight);
	free(varresnetv22_stage3_conv3_fwd__1);
	//Convolution
	free(varresnetv22_stage3_conv3_fwd);
	free(varconv_bias__13);
	free(varresnetv22_stage3_conv3_fwd__2);
	free(varresnetv22_stage3_conv3_fwd__3);
	free(varresnetv22_stage3_batchnorm3_fwd);
	free(varresnetv22_stage3_conv4_weight);
	free(varresnetv22_stage3_conv4_fwd__1);
	//Convolution
	free(varresnetv22_stage3_conv4_fwd);
	free(varconv_bias__14);
	free(varresnetv22_stage3_conv4_fwd__2);
	free(varresnetv22_stage3_conv4_fwd__3);
	//Add
	free(varresnetv22_stage4_batchnorm0_fwd);
	free(varresnetv22_stage4_conv0_weight);
	free(varresnetv22_stage4_conv0_fwd__1);
	//Convolution
	free(varresnetv22_stage4_conv0_fwd);
	free(varconv_bias__15);
	free(varresnetv22_stage4_conv0_fwd__2);
	free(varresnetv22_stage4_conv0_fwd__3);
	free(varresnetv22_stage4_batchnorm1_fwd);
	free(varresnetv22_stage4_conv1_weight);
	free(varresnetv22_stage4_conv1_fwd__1);
	//Convolution
	free(varresnetv22_stage4_conv1_fwd);
	free(varconv_bias__16);
	free(varresnetv22_stage4_conv1_fwd__2);
	free(varresnetv22_stage4_conv2_weight);
	free(varresnetv22_stage4_conv2_fwd__1);
	//Convolution
	free(varresnetv22_stage4_conv2_fwd);
	free(varconv_bias__17);
	free(varresnetv22_stage4_conv2_fwd__2);
	free(varresnetv22_stage4_conv1_fwd__3);
	//Add
	free(varresnetv22_stage4_conv2_fwd__3);
	free(varresnetv22_stage4_batchnorm2_fwd);
	free(varresnetv22_stage4_conv3_weight);
	free(varresnetv22_stage4_conv3_fwd__1);
	//Convolution
	free(varresnetv22_stage4_conv3_fwd);
	free(varconv_bias__18);
	free(varresnetv22_stage4_conv3_fwd__2);
	free(varresnetv22_stage4_conv3_fwd__3);
	free(varresnetv22_stage4_batchnorm3_fwd);
	free(varresnetv22_stage4_conv4_weight);
	free(varresnetv22_stage4_conv4_fwd__1);
	//Convolution
	free(varresnetv22_stage4_conv4_fwd);
	free(varconv_bias__19);
	free(varresnetv22_stage4_conv4_fwd__2);
	free(varresnetv22_stage4_conv4_fwd__3);
	//Add
	free(varresnetv22_batchnorm2_fwd);
	free(varresnetv22_pool1_fwd);
	free(varresnetv22_pool1_fwd__1);
	free(varresnetv22_pool1_fwd__2);
	free(varresnetv22_dense0_weight);
	free(varresnetv22_flatten0_reshape0);
	//MatMul
	free(varresnetv22_dense0_fwd);
	free(varresnetv22_dense0_bias);
	free(varresnetv22_dense0_fwd__1);
	//Add
	free(varresnetv22_dense0_fwd_reshape);

}
