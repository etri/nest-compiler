//============================================================================
// Name        : NNCompGen.cpp
// Author      :
// Version     :
// Copyright   :
// Description :
//============================================================================
//void memFree(input, *result);
//bool readPngImage(*input ,imageFilePath);

#include <iostream>
#include <string.h>
#include <time.h>
#include <algorithm>

#include "network.h"
#include "Image.h"
#include "library.h"


using namespace std;



int main(int argc, char *argv[]) {

	clock_t start, end;
	double inferenceTime;

	cout << "= Library Integration & code execution = " << endl;

	char* imageFilePath = 0, *dataFilePath = 0;
	if(argc < 3) {
		cout << "~ Random value is used for input, and the data.bin file in the current directory is used." << endl;
	} else {
		imageFilePath = argv[1];
        dataFilePath = argv[2];
		cout << "Image file: " << imageFilePath << endl;
        cout << "data file: " << dataFilePath << endl;
	}

//
//	{//model_num: 0, add.onnx
//
//	float (* input)[1][224][224][3];
//	input = (float (*)[1][224][224][3])malloc(sizeof(float[1][224][224][3]));
//
//	for(int i = 0; i < 1; i++)
//		for(int j = 0; j < 224; j++)
//			for(int k = 0; k < 224; k++)
//				for(int l = 0; l < 3; l++)
//					(*input)[i][j][k][l] = 0.5;
//
//	float (* result)[1][224][224][3];
//	result = (float (*)[1][224][224][3])malloc(sizeof(float[1][224][224][3]));
//	memset(result, 0x00, sizeof(float[1][224][224][3]));
//
//	time_t istart,iend;
//	time (&istart);
//
//	//cout << "start network_func" << endl;
//	//network_func(input, result);
//
//	initializeVariables();
//	inference(input, *result);
//	memFree(input, *result);
//
//	time (&iend);
//	double dif = difftime (iend,istart);
//	printf ("Inference time is %.2lf seconds.", dif );
//
//
//	cout <<"-----------------------------" << endl;
//
//	for(int i = 0; i < 1; i++) {
//		for(int j = 0; j < 224; j++) {
//			for(int k = 0; k < 224; k++) {
//				for(int l = 0; l < 3; l++) {
//					cout <<(*result)[i][j][k][l] << "\t";
//				}
//				cout<<endl;
//			}
//			cout <<endl;
//		}
//		cout <<endl;
//	}
//
//
//	free(result);
//	}

//	{//model_num: 1 or 3, avgpool_4d_pad.onnx  or maxpool_4d_with_padding.onnx
//
//	float (* input)[1][3][28][28];
//	input = (float (*)[1][3][28][28])malloc(sizeof(float[1][3][28][28]));
//
//	for(int i = 0; i < 1; i++)
//		for(int j = 0; j < 3; j++)
//			for(int k = 0; k < 28; k++)
//				for(int l = 0; l < 28; l++)
//					(*input)[i][j][k][l] = 0.5*k - l;
//
//	float (* result)[1][3][30][30];
//	result = (float (*)[1][3][30][30])malloc(sizeof(float[1][3][30][30]));
//	memset(result, 0x00, sizeof(float[1][3][30][30]));
//
//	time_t istart,iend;
//	time (&istart);
//
//	cout << "start network_func" << endl;
//	network_func(input, result);
//
//	time (&iend);
//	double dif = difftime (iend,istart);
//	printf ("Inference time is %.2lf seconds.", dif );
//
//
//	cout <<"-----------------------------" << endl;
//
//	for(int i = 0; i < 1; i++) {
//		for(int j = 0; j < 3; j++) {
//			for(int k = 0; k < 30; k++) {
//				for(int l = 0; l < 30; l++) {
//					cout <<(*result)[i][j][k][l] << "\t";
//				}
//				cout<<endl;
//			}
//			cout <<endl;
//		}
//		cout <<endl;
//	}
//
//
//	free(result);
//
//	}

//	{//model_num: 2, avgpool_4d_stride.onnx
//
//	float (* input)[1][1][5][5];
//	input = (float (*)[1][1][5][5])malloc(sizeof(float[1][1][5][5]));
//
//	for(int i = 0; i < 1; i++)
//		for(int j = 0; j < 1; j++)
//			for(int k = 0; k < 5; k++)
//				for(int l = 0; l < 5; l++)
//					(*input)[i][j][k][l] = 0.5;
//
//	float (* result)[1][1][2][2];
//	result = (float (*)[1][1][2][2])malloc(sizeof(float[1][1][2][2]));
//	memset(result, 0x00, sizeof(float[1][1][2][2]));
//
//	time_t istart,iend;
//	time (&istart);
//
//	cout << "start network_func" << endl;
//	network_func(input, result);
//
//	time (&iend);
//	double dif = difftime (iend,istart);
//	printf ("Inference time is %.2lf seconds.", dif );
//
//
//	cout <<"-----------------------------" << endl;
//
//	for(int i = 0; i < 1; i++) {
//		for(int j = 0; j < 1; j++) {
//			for(int k = 0; k < 2; k++) {
//				for(int l = 0; l < 2; l++) {
//					cout <<(*result)[i][j][k][l] << "\t";
//				}
//				cout<<endl;
//			}
//			cout <<endl;
//		}
//		cout <<endl;
//	}
//
//
//	free(result);
//
//	}


//	{//model_num: 4, relu_4d.onnx
//
//	float (* input)[2][3][4][5];
//	input = (float (*)[2][3][4][5])malloc(sizeof(float[2][3][4][5]));
//
//	for(int i = 0; i < 2; i++)
//		for(int j = 0; j < 3; j++)
//			for(int k = 0; k < 4; k++)
//				for(int l = 0; l < 5; l++)
//					(*input)[i][j][k][l] = 3 + (-1)*l;
//
//	float (* result)[2][3][4][5];
//	result = (float (*)[2][3][4][5])malloc(sizeof(float[2][3][4][5]));
//	memset(result, 0x00, sizeof(float[2][3][4][5]));
//
//	time_t istart,iend;
//	time (&istart);
//
//	cout << "start network_func" << endl;
//	network_func(input, result);
//
//	time (&iend);
//	double dif = difftime (iend,istart);
//	printf ("Inference time is %.2lf seconds.", dif );
//
//
//	cout <<"-----------------------------" << endl;
//
//	for(int i = 0; i < 2; i++) {
//		for(int j = 0; j < 3; j++) {
//			for(int k = 0; k < 4; k++) {
//				for(int l = 0; l < 5; l++) {
//					cout <<(*result)[i][j][k][l] << "\t";
//				}
//				cout<<endl;
//			}
//			cout <<endl;
//		}
//		cout <<endl;
//	}
//
//
//	free(result);
//
//	}

//	{//model_num: 5, two_transpose.onnx
//
//	float (* input)[2][3][4];
//	input = (float (*)[2][3][4])malloc(sizeof(float[2][3][4]));
//
//	for(int i = 0; i < 2; i++)
//		for(int j = 0; j < 3; j++)
//			for(int k = 0; k < 4; k++)
//				(*input)[i][j][k] = 0.5*k;
//
//	for(int i = 0; i < 2; i++) {
//		for(int j = 0; j < 3; j++) {
//			for(int k = 0; k < 4; k++) {
//				cout <<(*input)[i][j][k] << "\t";
//			}
//			cout <<endl;
//		}
//		cout <<endl;
//	}
//
//	float (* result)[2][3][4];
//	result = (float (*)[2][3][4])malloc(sizeof(float[2][3][4]));
//	memset(result, 0x00, sizeof(float[3][2][4]));
//
//	time_t istart,iend;
//	time (&istart);
//
//	cout << "start network_func" << endl;
//	network_func(input, result);
//
//	time (&iend);
//	double dif = difftime (iend,istart);
//	printf ("Inference time is %.2lf seconds.", dif );
//
//
//	cout <<"-----------------------------" << endl;
//
//	for(int i = 0; i < 2; i++) {
//		for(int j = 0; j < 3; j++) {
//			for(int k = 0; k < 4; k++) {
//				cout <<(*result)[i][j][k] << "\t";
//			}
//			cout <<endl;
//		}
//		cout <<endl;
//	}
//
//
//	free(result);
//
//	}

//
//	{//model_num: 6, conv.onnx
//
//	float (* input)[20][16][50][40];
//	input = (float (*)[20][16][50][40])malloc(sizeof(float[20][16][50][40]));
//
//	for(int i = 0; i < 20; i++)
//		for(int j = 0; j < 16; j++)
//			for(int k = 0; k < 50; k++)
//				for(int l = 0; l < 40; l++)
//					(*input)[i][j][k][l] = 7.0;
//
//	float (* result)[20][13][48][38];
//	result = (float (*)[20][13][48][38])malloc(sizeof(float[20][13][48][38]));
//	memset(result, 0x00, sizeof(float[20][13][48][38]));
//
////	time_t istart,iend;
////	time (&istart);
////
////	cout << "start network_func" << endl;
////	network_func(input, result);
////
////	time (&iend);
////	double dif = difftime (iend,istart);
////	printf ("Inference time is %.2lf seconds.", dif );
//
//
//
//	cout << "start initializeVariables" << endl;
//	start = clock();
//	initializeVariables();
//	end = clock();
//	inferenceTime = (double)(end - start);
//	cout << "==> Initialization time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//	cout << "start inference" << endl;
//	start = clock();
//	inference(input, *result);
//	end = clock();
//	inferenceTime = (double)(end - start);
//	cout << "==> Inference time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//	//cout << "result[0][0] = " << result[0][0] << endl;//
//	cout << "start memFree" << endl;
//	start = clock();
//	memFree(input, *result);
//	end = clock();
//	inferenceTime = (double)(end - start);
//	cout << "==> Free time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//
//	cout <<"-----------------------------" << endl;
//
//	for(int i = 0; i < 20; i++) {
//		for(int j = 0; j < 13; j++) {
//			for(int k = 0; k < 48; k++) {
//				for(int l = 0; l < 38; l++) {
//					cout <<(*result)[i][j][k][l] << "\t";
//				}
//				cout<<endl;
//			}
//			cout <<endl;
//		}
//		cout <<endl;
//	}
//
//
//	free(result);
//
//	}

//
//	{//model_num: 7, conv_no_group.onnx
//
//	float (* input)[2][3][7][5];
//	input = (float (*)[2][3][7][5])malloc(sizeof(float[2][3][7][5]));
//
//	for(int i = 0; i < 2; i++)
//		for(int j = 0; j < 3; j++)
//			for(int k = 0; k < 7; k++)
//				for(int l = 0; l < 5; l++)
//					(*input)[i][j][k][l] = 1.0;
//
//	float (* result)[2][4][5][4];
//	result = (float (*)[2][4][5][4])malloc(sizeof(float[2][4][5][4]));
//	memset(result, 0x00, sizeof(float[2][4][5][4]));
//
//	time_t istart,iend;
//	time (&istart);
//
//	cout << "start network_func" << endl;
//	network_func(input, result);
//
//	time (&iend);
//	double dif = difftime (iend,istart);
//	printf ("Inference time is %.2lf seconds.", dif );
//
//
//	cout <<"-----------------------------" << endl;
//
//	for(int i = 0; i < 2; i++) {
//		for(int j = 0; j < 4; j++) {
//			for(int k = 0; k < 5; k++) {
//				for(int l = 0; l < 4; l++) {
//					cout <<(*result)[i][j][k][l] << "\t";
//				}
//				cout<<endl;
//			}
//			cout <<endl;
//		}
//		cout <<endl;
//	}
//
//
//	free(result);
//
//	}


//	{//model_num: 11, softmax_axis1.onnx
//
//	float (* input)[3][4][5];
//	input = (float (*)[3][4][5])malloc(sizeof(float[3][4][5]));
//
//	for(int i = 0; i < 3; i++)
//		for(int j = 0; j < 4; j++)
//			for(int k = 0; k < 5; k++)
//				(*input)[i][j][k] = 1.0*k + j;
//
//	float (* result)[3][4][5];
//	result = (float (*)[3][4][5])malloc(sizeof(float[3][4][5]));
//	memset(result, 0x00, sizeof(float[3][4][5]));
//
//	time_t istart,iend;
//	time (&istart);
//
//	cout << "start network_func" << endl;
//	network_func(input, result);
//
//	time (&iend);
//	double dif = difftime (iend,istart);
//	printf ("Inference time is %.2lf seconds.", dif );
//
//
//	cout <<"-----------------------------" << endl;
//
//	for(int i = 0; i < 3; i++) {
//		for(int j = 0; j < 4; j++) {
//			for(int k = 0; k < 5; k++) {
//				cout <<(*result)[i][j][k] << "\t";
//			}
//			cout <<endl;
//		}
//		cout <<endl;
//	}
//
//
//	free(result);
//
//	}
//
//
//
//	{//model_num: 8, super_resolution.onnx
//
//		float (* input)[1][1][224][224];
//		input = (float (*)[1][1][224][224])malloc(sizeof(float[1][1][224][224]));
//
//		for(int i = 0; i < 1; i++)
//			for(int j = 0; j < 1; j++)
//				for(int k = 0; k < 224; k++)
//					for(int l = 0; l < 224; l++)
//						(*input)[i][j][k][l] = 1;
//
//		float (* result)[1][1][672][672];
//		result = (float (*)[1][1][672][672])malloc(sizeof(float[1][1][672][672]));
//		memset(result, 0x00, sizeof(float[1][1][672][672]));
//
////		time_t istart,iend;
////		time (&istart);
////
////		cout << "start network_func" << endl;
////		network_func(input, result);
////
////		time (&iend);
////		double dif = difftime (iend,istart);
////		printf ("Inference time is %.2lf seconds.", dif );
//
//
//
//		cout << "start initializeVariables" << endl;
//		start = clock();
//		initializeVariables();
//		end = clock();
//		inferenceTime = (double)(end - start);
//		cout << "==> Initialization time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//		cout << "start inference" << endl;
//		start = clock();
//		inference(input, *result);
//		end = clock();
//		inferenceTime = (double)(end - start);
//		cout << "==> Inference time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//		cout << "result[0][0] = " << result[0][0] << endl;//
//		cout << "start memFree" << endl;
//		start = clock();
//		memFree(input, *result);
//		end = clock();
//		inferenceTime = (double)(end - start);
//		cout << "==> Free time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//		cout <<"-----------------------------" << endl;
//
//		for(int i = 0; i < 1; i++) {
//			for(int j = 0; j < 1; j++) {
//				for(int k = 0; k < 672/100; k++) {
//					for(int l = 0; l < 672/100; l++) {
//						cout <<(*result)[i][j][k][l] << "\t";
//					}
//					cout<<endl;
//				}
//				cout <<endl;
//			}
//			cout <<endl;
//		}
//
//
//		free(result);
//
//	}


//	{//exam_conv_relu.onnx
//
//
//	float (* input)[1][3][224][224];
//	input = (float (*)[1][3][224][224])malloc(sizeof(float[1][3][224][224]));
//
//	for(int i = 0; i < 1; i++)
//		for(int j = 0; j < 3; j++)
//			for(int k = 0; k < 224; k++)
//				for(int l = 0; l < 224; l++)
//					//(*input)[i][j][k][l] = 0.1* (float)k + (float)i;
//					(*input)[i][j][k][l] = 1.0;
//
//	float (* result)[1][32][224][224];
//	result = (float (*)[1][32][224][224])malloc(sizeof(float[1][32][224][224]));
//	memset(result, 0x00, sizeof(float[1][32][224][224]));
//
////	float (* result)[1][3][224][224];
////	result = (float (*)[1][3][224][224])malloc(sizeof(float[1][3][224][224]));
////	memset(result, 0x00, sizeof(float[1][3][224][224]));
//
//
////	time_t istart,iend;
////	time (&istart);
////
////	cout << "start network_func" << endl;
////	network_func(input, result);
////
////	time (&iend);
////	double dif = difftime (iend,istart);
////	printf ("Inference time is %.2lf seconds.", dif );
//
//	time_t istart,iend;
//	time (&istart);
//
//	cout << "start initializeVariables" << endl;
//	start = clock();
//	initializeVariables();
//	end = clock();
//	inferenceTime = (double)(end - start);
//	cout << "==> Initialization time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//	cout << "start inference" << endl;
//	start = clock();
//	inference(input, *result);
//	end = clock();
//	inferenceTime = (double)(end - start);
//	cout << "==> Inference time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//
//	//cout << "result[0][0] = " << result[0][0] << endl;//
//	cout << "start memFree" << endl;
//	start = clock();
//	memFree(input, *result);
//	end = clock();
//	inferenceTime = (double)(end - start);
//	cout << "==> Free time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//
//	time (&iend);
//	double dif = difftime (iend,istart);
//	printf ("Inference time is %.2lf seconds.", dif);
//
//
//
////	cout <<"-----------------------------" << endl;
////
////	for(int i = 0; i < 1; i++) {
////		for(int j = 0; j < 32; j++) {
////			for(int k = 0; k < 224; k++) {
////				for(int l = 0; l < 224; l++) {
////					printf("%f ", (*result)[i][j][k][l]);
////				}
////				cout << endl;
////			}
////			cout <<endl;
////		}
////		cout <<endl;
////	}
////
////	cout << endl;
//
//
//	free(result);
//
//	}
//


	{//model_num: 8, resnet18v2.onnx, vgg19


	float (* input)[1][3][224][224];
	input = (float (*)[1][3][224][224])malloc(sizeof(float[1][3][224][224]));

	for(int i = 0; i < 1; i++)
		for(int j = 0; j < 3; j++)
			for(int k = 0; k < 224; k++)
				for(int l = 0; l < 224; l++)
					(*input)[i][j][k][l] = 7.0;

	if(imageFilePath != 0) {
		bool isRead = readPngImage(*input, imageFilePath, zeroOne, imagenetNormMean, imagenetNormStd);
		if(isRead) {
			cout << "Error during PNG read!" << endl;
			return 0;
		}
	}




	float (* result)[1][1000];
	result = (float (*)[1][1000])malloc(sizeof(float[1][1000]));
	memset(result, 0x00, sizeof(float[1][1000]));


	cout <<"-----------------------------" << endl<<endl;

//	time_t istart,iend;
//	time (&istart);
//
//	cout << "start network_func" << endl;
//	network_func(input, result);
//
//	time (&iend);
//	double dif = difftime (iend,istart);
//	printf ("Inference time is %.2lf seconds.", dif );


	cout << endl << "Initializing variables (executing initializeVariables())" << endl;
	start = clock();
	initializeVariables(dataFilePath);
	end = clock();
	inferenceTime = (double)(end - start);
    cout << "==> Initialization time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl<<endl;


	cout << "Inferencing (executing inference())" << endl;
	start = clock();
	inference(input, *result);
	end = clock();
	inferenceTime = (double)(end - start);
    cout << "==> Inference time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl<<endl;


	vector<pair<float, int> > ilist;

	int bestIdx = 0;
	float bestProb = (*result)[0][0];
	for (int i = 0; i < 1000; i++) {

	    ilist.push_back(make_pair((*result)[0][i], i));

		if((*result)[0][i] > bestProb) {
			bestProb = (*result)[0][i];
			bestIdx = i;
		}
	}

	sort(ilist.begin(), ilist.end());

	printf("Best index: %d\n", bestIdx);
	cout <<"-----------------------------" << endl <<endl;

	printf("Best 10 indexes and probabilities: \n");

	cout <<"-----------------------------" << endl;
	printf("%6s  %10s: \n", "Index,", "Probability");
	cout <<"-----------------------------" << endl;

	for(int i = 0; i < 10; i++) {
		printf("%6d, %10f\n", ilist[999-i].second, (*result)[0][ilist[999-i].second]);

	}
	cout <<"-----------------------------" << endl;

	free(result);


    //cout << "result[0][0] = " << result[0][0] << endl;//
    cout << "Releasing dynamically allocated memory (executing memFree())" << endl;
	start = clock();
    memFree(input, *result);
	end = clock();
	inferenceTime = (double)(end - start);
    cout << "==> Free time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl << endl;


	cout <<"-----------------------------" << endl;

	}

//
//	{//exam_conv_relu.onnx
//
//	float (* input)[1][3][224][224];
//	input = (float (*)[1][3][224][224])malloc(sizeof(float[1][3][224][224]));
//
//	for(int i = 0; i < 1; i++)
//		for(int j = 0; j < 3; j++)
//			for(int k = 0; k < 224; k++)
//				for(int l = 0; l < 224; l++)
//					(*input)[i][j][k][l] = 1.0;
//
////	if(imageFilePath != 0) {
////		bool isRead = readPngImage(*input, imageFilePath, zeroOne, imagenetNormMean, imagenetNormStd);
////		if(isRead) {
////			cout << "Error during PNG read!" << endl;
////			return 0;
////		}
////	}
//
//
////	float (* result)[1][1000];
////	result = (float (*)[1][1000])malloc(sizeof(float[1][1000]));
////	memset(result, 0x00, sizeof(float[1][1000]));
//
//	float (* result)[1][32][222][222];
//	result = (float (*)[1][32][222][222])malloc(sizeof(float[1][32][222][222]));
//	memset(result, 0x00, sizeof(float[1][32][222][222]));
//
//
//	cout <<"-----------------------------" << endl<<endl;
//
//	cout << endl << "Initializing variables (executing initializeVariables())" << endl;
//	start = clock();
//	initializeVariables();
//	end = clock();
//	inferenceTime = (double)(end - start);
//    cout << "==> Initialization time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl<<endl;
//
//
//	cout << "Inferencing (executing inference())" << endl;
//	start = clock();
//	inference(input, *result);
//	end = clock();
//	inferenceTime = (double)(end - start);
//    cout << "==> Inference time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl<<endl;
//
//    //cout << "result[0][0] = " << result[0][0] << endl;//
//    cout << "Releasing dynamically allocated memory (executing memFree())" << endl;
//	start = clock();
//    memFree(input, *result);
//	end = clock();
//	inferenceTime = (double)(end - start);
//    cout << "==> Free time : " << ((inferenceTime) / CLOCKS_PER_SEC) << " seconds" << endl << endl;
//
//
//	cout <<"-----------------------------" << endl;
//
//	for(int i = 0; i < 1; i++){
//		cout << endl;
//		for(int j = 0; j < 32; j++){
//			cout << endl;
//			for(int k = 0; k < 222; k++){
//				cout << endl;
//				for(int l = 0; l < 222; l++){
//					printf("%f ", (*result)[i][j][k][l]);
//				}
//			}
//		}
//	}
//
//	cout <<"-----------------------------" << endl;
//
//	free(result);
//
//	}

//	cout << endl <<"End" << endl;


	return 0;
}

