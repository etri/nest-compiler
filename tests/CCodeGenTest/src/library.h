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

#ifndef LIBRARY_H_
#define LIBRARY_H_

//#define _DEBUG
//#define THREAD_MATMUL
//#define _USE_OPENBLAS_MATMUL
//#define _USE_OPENBLAS_CONV

//#define _CONV_ORIGINAL
//#define _CONV_GROUP_ORIGINAL
#define _USE_LLVM 1

#ifdef __MACH__
#include <stdlib.h>
#else
#include <malloc.h>
#endif


#include <vector>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>

#include <algorithm>
#include <vector>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <omp.h>
#include <time.h>
#include <assert.h>

#ifdef _USE_OPENBLAS_MATMUL
#include <cblas.h>
#else

#ifdef _USE_OPENBLAS_CONV
#include <cblas.h>
#else

#endif

#endif


#ifdef __MACH__
#include <dispatch/dispatch.h>
#else
#include <semaphore.h>
#endif


using namespace std;

//#define FLOAT    0
//#define DOUBLE   1
//#define INT      2
//#define CHAR     3

typedef int8_t i8;
typedef uint8_t ui8;
typedef int16_t i16;
typedef int32_t i32;

#if defined(__MACH__) || defined(_USE_LLVM)
#define MAX_PAD_SIZE 3 //msyu
#endif

using namespace std;

void read_vector(vector<size_t>* vec, ifstream* readFileBin);

/*
 *
 *
 * Batch Normalization
 *
 *
 */


template <typename T1, size_t N1, size_t N2, size_t N3, size_t N4,
        typename T2>
void splat(T1 (&in1)[N1][N2][N3][N4], int axis, T2 (&result)[N1][N2][N3][N4])
{
#ifdef _DEBUG
    cout << "splat()-4D" << endl;
#endif

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}

template <typename T1, size_t K1, size_t L1, size_t M1, size_t N1,
        typename T2, size_t K2, size_t L2, size_t M2, size_t N2,
        typename T3, size_t K3, size_t L3, size_t M3, size_t N3>
void concat(int axis, T1 (&in1)[N1][M1][K1][L1], T2 (&in2)[N2][M2][K2][L2], T3 (&result)[N3][M3][K3][L3])
{
#ifdef _DEBUG
    cout << "concat()-4D" << endl;
    cout << "axis: " << axis << endl;
#endif
    if(axis == 3) {
        for(size_t i = 0; i < N3; i++) {
            for(size_t j = 0; j < M3; j++) {
                for(size_t k = 0; k < K3; k++) {
                    for(size_t l = 0; l < L3; l++) {
                        if(l < L1) {
                            result[i][j][k][l] = in1[i][j][k][l];
                        } else {
                            result[i][j][k][l] = in2[i][j][k][l-L1];
                        }
                    }
                }
            }
        }
    } else {
        cout << "Not supported. " << axis << endl;
    }
    //	end = clock();
    //    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}

template <typename T1, size_t N, size_t M, size_t K, size_t L>
void batchNorm(T1 (&in)[N][M][K][L], T1 (&scale)[M], T1 (&bias)[M], T1 (&mean)[M], T1 (&var)[M], unsigned int channelIdx, T1 epsilon, T1 momentum, T1 (&result)[N][M][K][L])
{
    //NCHW
//
#ifdef _DEBUG
    cout << "batchNorm() - 1" << endl;
  cout << "channel: " << channelIdx << endl;
  cout << "N: " << N << endl;
  cout << "M: " << M << endl;
  cout << "K: " << K << endl;
  cout << "L: " << L << endl;
#endif

    for(size_t i = 0; i < N; i++) {
        for(size_t j = 0; j < M; j++) {
            for(size_t k = 0; k < K; k++) {
                for(size_t l = 0; l < L; l++) {
                    float norm = (in[i][j][k][l] - mean[j])/sqrtf(epsilon + var[j]);
                    result[i][j][k][l] = scale[j]* norm + bias[j];
                }
            }
        }
    }
}

template <typename T1, size_t N, size_t M, size_t K, size_t L>
void batchNorm(T1 (&in)[N][L][K][M], T1 (&scale)[M], T1 (&bias)[M], T1 (&mean)[M], T1 (&var)[M], unsigned int channelIdx, T1 epsilon, T1 momentum, T1 (&result)[N][L][K][M])
{
    //NCHW
//
#ifdef _DEBUG
    cout << "batchNorm() - 2" << endl;
    cout << "channel: " << channelIdx << endl;
    cout << "N: " << N << endl;
    cout << "L: " << L << endl;
    cout << "K: " << K << endl;
    cout << "M: " << M << endl;

#endif

    for(size_t i = 0; i < N; i++) {
        for(size_t j = 0; j < L; j++) {
            for(size_t k = 0; k < K; k++) {
                for(size_t l = 0; l < M; l++) {
                    float norm = (in[i][j][k][l] - mean[l])/sqrtf(epsilon + var[l]);
                    result[i][j][k][l] = scale[l]* norm + bias[l];
                }
            }
        }
    }
}

/*
 *
 *
 * Convolution
 *
 *
 */

extern int conv_idx;

#ifdef _CONV_ORIGINAL

template <typename T1, size_t BATCH, size_t HEIGHT, size_t WIDTH, size_t CHANNEL,
        typename T2, size_t FILTER_SIZE, size_t FILTER_HEIGHT, size_t FILTER_WIDTH, size_t FILTER_CHANNEL,
        typename T3, size_t N2,
        typename T4, size_t N3, size_t M3, size_t K3, size_t L3>
void conv(T1 (&in1)[BATCH][HEIGHT][WIDTH][CHANNEL], T2 (&filter)[FILTER_SIZE][FILTER_HEIGHT][FILTER_WIDTH][FILTER_CHANNEL], vector<size_t> filter_dim, T3 (&bias)[N2], \
	vector<size_t> stride, vector<size_t> pad, int group, T4 (&result)[N3][M3][K3][L3])
{
#ifdef _DEBUG
    cout << conv_idx++ << ": conv()-original" << endl;

    cout << "BATCH: " << BATCH << endl;
    cout << "HEIGHT: " << HEIGHT << endl;
    cout << "WIDTH: " << WIDTH << endl;
    cout << "CHANNEL: " << CHANNEL << endl;
    cout << "pad: " << pad.at(0) << endl;
    cout << "stride: " << stride.at(0) << endl;
    cout << "FILTER_HEIGHT: " << FILTER_HEIGHT << endl;
    cout << "FILTER_WIDTH: " << FILTER_WIDTH << endl;
    cout << "FILTER_CHANNEL: " << FILTER_CHANNEL << endl;
    //N, H, W, C
    //N == 1
    //group == 1
#endif

    int pad_xleft = pad.at(0);
    int pad_xright = pad.at(1);
    int pad_ytop = pad.at(2);
    int pad_ybottom = pad.at(3);
    int xadd = pad_xleft + pad_xright;
    int yadd = pad_ytop + pad_ybottom;

    int stride_size = stride.at(0);

    int filter_h = filter_dim.at(1);
    int filter_w = filter_dim.at(2);
    int filter_size = filter_dim.at(0);

    int out_h = (HEIGHT + yadd - filter_h)/stride_size + 1;
    int out_w = (WIDTH + xadd - filter_w)/stride_size + 1;

#ifdef _DEBUG
    cout << "FILTER_SIZE: " << FILTER_SIZE << endl;
    cout << "out_h: " << out_h << endl;
    cout << "out_w: " << out_w << endl;

    struct timespec tp, ep;
    double timeCheck;
    clock_gettime(CLOCK_MONOTONIC, &tp);
#endif


#if defined(__MACH__) || defined(_USE_LLVM)
    float (* newin)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
#else
    float (* newin)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
#endif

    for (int b = 0; b < BATCH; b++){
        for (size_t c = 0; c < CHANNEL; c++){
            for (int h = 0; h < HEIGHT; h++){
                for (int w = 0; w < WIDTH; w++){
                    (*newin)[b][pad_ytop+h][pad_xleft+w][c] = in1[b][h][w][c];
                }
            }
        }
    }

    int new_h_idx = 0, new_w_idx = 0;
    for (int b = 0; b < BATCH; b++){
        for(int fc = 0; fc < filter_size; fc++) {
            for(int i = 0; i <out_h; i++) {
                for(int j = 0; j <out_w; j++) {

                    float total = 0.0;
                    for(int fh = 0; fh <filter_h; fh++) {
                        for(int fw = 0; fw <filter_w; fw++) {
                            for(size_t f = 0; f < CHANNEL; f++) {
                                total = total + (*newin)[b][new_h_idx+fh][new_w_idx+fw][f]*filter[fc][fh][fw][f];
                            }
                        }
                    }
                    result[b][i][j][fc] = total + bias[fc];
                    new_w_idx += stride_size;
                }
                new_h_idx += stride_size;
                new_w_idx = 0;
            }
            new_h_idx = 0;
        }
    }

#ifdef _DEBUG
    free(newin);
    clock_gettime(CLOCK_MONOTONIC, &ep);
    timeCheck = ((ep.tv_sec - tp.tv_sec) + (ep.tv_nsec - tp.tv_nsec) / 1000000000.0);
    cout << "==================== " + to_string(conv_idx++) + " convolution time : " << timeCheck <<" seconds  ====================" << endl << endl;
#endif
}
#else

#ifdef __MACH__
extern dispatch_semaphore_t    semaphore;
#else
extern    sem_t  semaphore;
#endif

// 스레드 수 정의
#define MAX_THREAD 2

//스레드를 사용하기 위한 구조체.
template <typename T1, typename T2, typename T3, int tN, int tM, int tM1>
struct tTHREAD{
    T1(*arrr)[tM];
    T2(*brrr1)[tM][tM1];
    T3(*crrr1)[tN][tM1];
    int b;
    int cnt;
};
// 스레드 함수
template <typename T1, typename T2, typename T3, int tN, int tM, int tM1>
void *multi(void *num)
{
    struct tTHREAD<float, float, float, tN, tM, tM1> *numMul = (struct tTHREAD<float, float, float, tN, tM, tM1> *)num;
    // cnt는 현재 사용되려는 스레드 번호.
    int core = numMul->cnt;
    int B = numMul->b;

    int mN = tN / MAX_THREAD * (core + 1);
    if (core == MAX_THREAD - 1) mN = tN;

    for (int i = core * (tN / MAX_THREAD); i < mN; i++) {// 각 스레드는 행렬 행 기준으로 스레드 개수만큼 나눠서 연산
        for (int k = 0; k < tM; k++) {
            for (int j = 0; j < tM1; j++) {
                numMul->crrr1[B][i][j] += numMul->arrr[i][k] * numMul->brrr1[B][k][j];
            }
        }
    }

#ifdef __MACH__
    dispatch_semaphore_signal(semaphore);
#else
    sem_post(&semaphore);
#endif

    return nullptr;
}



#ifdef _CONV_GROUP_ORIGINAL

template <typename T1, size_t BATCH, size_t HEIGHT, size_t WIDTH, size_t CHANNEL,
        typename T2, size_t FILTER_SIZE, size_t FILTER_HEIGHT, size_t FILTER_WIDTH, size_t FILTER_CHANNEL,
        typename T3, size_t N2,
        typename T4, size_t N3, size_t M3, size_t K3, size_t L3>
void conv_group(T1 (&in1)[BATCH][HEIGHT][WIDTH][CHANNEL], T2 (&filter)[FILTER_SIZE][FILTER_HEIGHT][FILTER_WIDTH][FILTER_CHANNEL], vector<size_t> kernel_dim, T3 (&bias)[N2], \
	vector<size_t> stride, vector<size_t> pad, int group, T4 (&result)[N3][M3][K3][L3])
{
#ifdef _DEBUG
    cout << "conv_group()-original" << endl;

    cout << "BATCH: " << BATCH << endl;
    cout << "HEIGHT: " << HEIGHT << endl;
    cout << "WIDTH: " << WIDTH << endl;
    cout << "CHANNEL: " << CHANNEL << endl;
    cout << "pad: " << pad.at(0) << endl;
    cout << "stride: " << stride.at(0) << endl;
    cout << "FILTER_SIZE: " << FILTER_SIZE << endl;
    cout << "FILTER_HEIGHT: " << FILTER_HEIGHT << endl;
    cout << "FILTER_WIDTH: " << FILTER_WIDTH << endl;
    cout << "FILTER_CHANNEL: " << FILTER_CHANNEL << endl;

    cout << "OUT BATCH: " << N3 << endl;
    cout << "OUT HEIGHT: " << M3 << endl;
    cout << "OUT WIDTH: " << K3 << endl;
    cout << "OUT CHANNEL: " << L3 << endl;

    cout << "group: " << group << endl;
    //N, H, W, C
    //N == 1
    //group == 1
#endif

    int pad_xleft = pad.at(0);
    int pad_xright = pad.at(1);
    int pad_ytop = pad.at(2);
    int pad_ybottom = pad.at(3);
    int xadd = pad_xleft + pad_xright;
    int yadd = pad_ytop + pad_ybottom;

    int stride_size = stride.at(0);

    int kernel_h = kernel_dim.at(1);
    int kernel_w = kernel_dim.at(2);
    int kernel_size = kernel_dim.at(0);

    int out_h = (HEIGHT + yadd - kernel_h)/stride_size + 1;
    int out_w = (WIDTH + xadd - kernel_w)/stride_size + 1;

#ifdef _DEBUG
    cout << "FILTER_SIZE: " << FILTER_SIZE << endl;
    cout << "out_h: " << out_h << endl;
    cout << "out_w: " << out_w << endl;
    cout << "kernel_h: " << kernel_h << endl;
    cout << "kernel_w: " << kernel_w << endl;
    cout << "kernel_size: " << kernel_size << endl;

    struct timespec tp, ep;
    double timeCheck;
    clock_gettime(CLOCK_MONOTONIC, &tp);
#endif


#if defined(__MACH__) || defined(_USE_LLVM)
    float (* newin)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
#else
    float (* newin)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
#endif


    for (int b = 0; b < BATCH; b++){
        for (size_t c = 0; c < CHANNEL; c++){
            for (int h = 0; h < HEIGHT; h++){
                for (int w = 0; w < WIDTH; w++){
                    (*newin)[b][pad_ytop+h][pad_xleft+w][c] = in1[b][h][w][c];
                }
            }
        }
    }

    int splitKernelSize = kernel_size/group;
    int new_h_idx = 0, new_w_idx = 0;

    for (int b = 0; b < BATCH; b++) {
        for(int g = 0; g < group; g++) {
            for (int fc = 0; fc < splitKernelSize; fc++) {
                for (int i = 0; i < out_h; i++) {
                    for (int j = 0; j < out_w; j++) {

                        float total = 0.0;
                        for (int fh = 0; fh < kernel_h; fh++) {
                            for (int fw = 0; fw < kernel_w; fw++) {
                                for (size_t f = 0; f < FILTER_CHANNEL; f++) {
                                    total = total + (*newin)[b][new_h_idx + fh][new_w_idx + fw][g*FILTER_CHANNEL + f] *
                                                    filter[g*splitKernelSize + fc][fh][fw][f];
                                }
                            }
                        }
                        result[b][i][j][g*splitKernelSize + fc] = total + bias[g*splitKernelSize + fc];
                        new_w_idx += stride_size;
                    }
                    new_h_idx += stride_size;
                    new_w_idx = 0;
                }
                new_h_idx = 0;
            }
        }
    }



#ifdef _DEBUG
    free(newin);
    clock_gettime(CLOCK_MONOTONIC, &ep);
    timeCheck = ((ep.tv_sec - tp.tv_sec) + (ep.tv_nsec - tp.tv_nsec) / 1000000000.0);
    cout << "==================== " + to_string(conv_idx++) + " convolution time : " << timeCheck <<" seconds  ====================" << endl << endl;
#endif
}
#else
//
//convolution image to columns
template <typename T1, size_t BATCH, size_t HEIGHT, size_t WIDTH, size_t CHANNEL,
        typename T2, size_t FILTER_SIZE, size_t FILTER_HEIGHT, size_t FILTER_WIDTH, size_t FILTER_CHANNEL,
        typename T3, size_t N2,
        typename T4, size_t N3, size_t M3, size_t K3, size_t L3>
void conv_group(T1 (&in1)[BATCH][HEIGHT][WIDTH][CHANNEL], T2 (&filter)[FILTER_SIZE][FILTER_HEIGHT][FILTER_WIDTH][FILTER_CHANNEL], vector<size_t> filter_dim, T3 (&bias)[N2], \
	vector<size_t> stride, vector<size_t> pad, size_t group, T4 (&result)[N3][M3][K3][L3])
{
 //   cout <<  "-- conv() group: " << group << endl;

#ifdef _DEBUG
    cout <<  "-- conv() -- " << endl;

  cout << "BATCH: " << BATCH << endl;
  cout << "HEIGHT: " << HEIGHT << endl;
  cout << "WIDTH: " << WIDTH << endl;
  cout << "CHANNEL: " << CHANNEL << endl;
  cout << "pad: " << pad.at(0) << endl;
  cout << "stride: " << stride.at(0) << endl;
  cout << "FILTER_HEIGHT: " << FILTER_HEIGHT << endl;
  cout << "FILTER_WIDTH: " << FILTER_WIDTH << endl;
  cout << "FILTER_CHANNEL: " << FILTER_CHANNEL << endl;
  cout << "BIAS LEN: " << N2 << endl;
#endif

    int pad_xleft = pad.at(0);
    int pad_xright = pad.at(1);
    int pad_ytop = pad.at(2);
    int pad_ybottom = pad.at(3);
    int xadd = pad_xleft + pad_xright;
    int yadd = pad_ytop + pad_ybottom;

    int stride_size = stride.at(0);

    int out_h = (HEIGHT + yadd - FILTER_HEIGHT)/stride_size + 1;
    int out_w = (WIDTH + xadd - FILTER_WIDTH)/stride_size + 1;

#ifdef _DEBUG
    cout << "FILTER_SIZE: " << FILTER_SIZE << endl;
  cout << "out_h: " << out_h << endl;
  cout << "out_w: " << out_w << endl;

  struct timespec tp, ep;
  double timeCheck;
  clock_gettime(CLOCK_MONOTONIC, &tp);
#endif

#if defined(__MACH__) || defined(_USE_LLVM)
    float (* newin)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
#else
    float (* newin)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL];
  newin = (float (*)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
  memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
#endif

    //step 1. 패딩

    //int jp, kp;
    //printf("패딩 변환 중\n");
    for (int b = 0; b < BATCH; b++){
        for (int h = 0; h < HEIGHT; h++){
            for (int w = 0; w < WIDTH; w++){
                for (size_t c = 0; c < CHANNEL; c++){
                    (*newin)[b][pad_ytop+h][pad_xleft+w][c] = in1[b][h][w][c];
                }
            }
        }
    }

    //step 2. 입력값 차원 변환
    //printf("입력값 차원 변환 중 \n");
    float (*newin2)[BATCH][CHANNEL*FILTER_HEIGHT*FILTER_WIDTH][M3*K3];
    newin2 = (float(*)[BATCH][CHANNEL*FILTER_HEIGHT*FILTER_WIDTH][M3*K3])malloc(sizeof(float[BATCH][CHANNEL*FILTER_HEIGHT*FILTER_WIDTH][M3*K3]));
    //하나의 스트라이트에 각 채널들을 하나의 행으로 구성
    for (int b = 0, p = 0; b < BATCH; b++, p = 0)
        for (int i = 0; i < out_h; i++)
            for (int j = 0, q = 0; j < out_w; j++, q = 0, p++)
                for (int c = 0; c < CHANNEL; c++)
                    for (int ii = i * stride_size; ii < i * stride_size + FILTER_HEIGHT; ii++)
                        for (int jj = j * stride_size; jj < j * stride_size + FILTER_WIDTH; jj++)
                            (*newin2)[b][q++][p] = (*newin)[b][ii][jj][c];

    free(newin);

    // step 3. 필터 평활화
    //printf("필터 평활화 중\n");

    float (*transFilter)[FILTER_SIZE][FILTER_HEIGHT*FILTER_WIDTH*FILTER_CHANNEL]; //패딩 배열
    transFilter = (float(*)[FILTER_SIZE][FILTER_HEIGHT*FILTER_WIDTH*FILTER_CHANNEL])malloc(sizeof(float[FILTER_SIZE][FILTER_WIDTH*FILTER_HEIGHT*FILTER_CHANNEL]));
    memset(transFilter, 0x00, sizeof(float[FILTER_SIZE][FILTER_HEIGHT*FILTER_WIDTH*FILTER_CHANNEL]));

    for (int i1 = 0, q = 0; i1 < FILTER_SIZE; i1++, q = 0)
        for (int i2 = 0; i2 < FILTER_CHANNEL; i2++)
            for (int i3 = 0; i3 < FILTER_HEIGHT; i3++)
                for (int i4 = 0; i4 < FILTER_WIDTH; i4++)
                    (*transFilter)[i1][q++] = filter[i1][i3][i4][i2];



    // step 4. 행렬 곱.
    float (*finalResult)[BATCH][FILTER_SIZE][M3*K3];
    finalResult = (float (*)[BATCH][FILTER_SIZE][M3*K3])malloc(sizeof(float[BATCH][FILTER_SIZE][M3*K3]));
    memset(finalResult, 0x00, sizeof(float[BATCH][FILTER_SIZE][M3*K3]));

    int div = out_h * out_w; // B * out_h * out_w
    //int total2 = BATCH * CHANNEL * FILTER_HEIGHT * FILTER_WIDTH * out_h * out_w;
    int total2 = BATCH * FILTER_CHANNEL * FILTER_HEIGHT * FILTER_WIDTH * out_h * out_w;
    int total3 = total2 / div;
    //printf("행렬 곱 중\n");
#ifdef _USE_OPENBLAS_CONV
  size_t m1len = FILTER_SIZE*total3, m2len = total3*div, m3len = FILTER_SIZE*div;
    float *m1, *m2, *m3;

    m1 = (float *)malloc(sizeof(float)*m1len);
    m2 = (float *)malloc(sizeof(float)*m2len);
    m3 = (float *)malloc(sizeof(float)*m3len);

    for (int b = 0; b < BATCH; b++) {
        for (int i = 0; i < m1len; i++) {
            m1[i] = (*transFilter)[i/total3][i % total3];
        }

        for (int i = 0; i < m2len; i++) {
            m2[i] = (*newin2)[b][i/div][i % div];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, FILTER_SIZE, div, total3, 1.0, m1, total3, m2, div, 0.0, m3, div);

        for (int i = 0; i < m3len; i++) {
            (*finalResult)[b][i/div][i % div] = m3[i];
        }
    }
    free(m1);
    free(m2);
    free(m3);

#else

#ifdef THREAD_MATMUL

    //printf("행렬 곱 중\n");
    struct tTHREAD<float,float,float, FILTER_SIZE, BATCH*FILTER_CHANNEL*FILTER_HEIGHT*FILTER_WIDTH, M3*K3> p;

    // 스레드에 연산할 메모리 복사
    pthread_t threads[MAX_THREAD];
    p.arrr = (*transFilter);
    p.brrr1 =(*newin2);
    p.crrr1 =(*finalResult);

#ifdef __MACH__
    semaphore = dispatch_semaphore_create(1);
#else
    sem_init(&semaphore, 0, 1);
#endif

    for(int b = 0; b < BATCH; b++){
        for (int i = 0; i < MAX_THREAD; i++) {
#ifdef __MACH__
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
#else
            //sem_wait(&semaphore);
            int r;

            do {
                r = sem_wait(&semaphore);
            } while (r == -1 && errno == EINTR);
#endif

            p.cnt = i;
            p.b = b; // 배치 값

            pthread_create(&threads[i], NULL, multi<float, float, float, FILTER_SIZE, BATCH*FILTER_CHANNEL*FILTER_HEIGHT*FILTER_WIDTH, M3*K3>, (void *)&p); // filter_size, total3, div
        }

        for (int t = 0; t < MAX_THREAD; t++)
            pthread_join(threads[t], NULL); // 모든 스레드가 끝날 때까지 결합 및 대기
    }

#ifdef __MACH__
    dispatch_release(semaphore);
#else
    // sem_distroy(&semaphore);
#endif


#else

//total3:  BATCH * FILTER_CHANNEL * FILTER_HEIGHT * FILTER_WIDTH
//div: out_w * out_h
    int splitedKernelSize = FILTER_SIZE/group;
    // k for문과 j for 문을 변경함으로써, 지역성 향상.
    for (int b = 0; b < BATCH; b++) {
        for (int i = 0; i < splitedKernelSize; i++) {
            for(int g = 0; g < group; g++) {
                for (int k = 0; k < total3; k++) {
                    for (int j = 0; j < div; j++) {
                        (*finalResult)[b][g*splitedKernelSize + i][j] += (*transFilter)[g*splitedKernelSize + i][k] * (*newin2)[b][g*FILTER_CHANNEL*FILTER_WIDTH*FILTER_HEIGHT + k][j];
                    }
                }
            }
        }
    }
#endif

#endif
    // step 5. 결과 차원 변환및 편향 합
    //printf("마지막 차원 변환 중\n");
    //int putout_c = (FILTER_SIZE) / (BATCH);


    for (int i1 = 0; i1 < BATCH; i1++) {
        for(int g = 0; g < group; g++) {
            for (int i4 = 0; i4 < splitedKernelSize; i4++) {
                size_t q = 0;
                for (int i2 = 0; i2 < out_h; i2++) {
                    for (int i3 = 0; i3 < out_w; i3++) {
                        result[i1][i2][i3][g*splitedKernelSize + i4] = (*finalResult)[i1][g*splitedKernelSize + i4][q++] + bias[g*splitedKernelSize + i4];
                    }
                }
                //cout << (float)bias[i4] << "\t" << endl;
            }
        }
    }

    free(newin2);
    free(transFilter);
    free(finalResult);

#ifdef _DEBUG
    clock_gettime(CLOCK_MONOTONIC, &ep);
  timeCheck = ((ep.tv_sec - tp.tv_sec) + (ep.tv_nsec - tp.tv_nsec) / 1000000000.0);
  cout << "==================== " + to_string(conv_idx++) + " convolution time : " << timeCheck <<" seconds  ====================" << endl << endl;
#endif
}

#endif

//
//convolution image to columns
template <typename T1, size_t BATCH, size_t HEIGHT, size_t WIDTH, size_t CHANNEL,
        typename T2, size_t FILTER_SIZE, size_t FILTER_HEIGHT, size_t FILTER_WIDTH, size_t FILTER_CHANNEL,
        typename T3, size_t N2,
        typename T4, size_t N3, size_t M3, size_t K3, size_t L3>
void conv(T1 (&in1)[BATCH][HEIGHT][WIDTH][CHANNEL], T2 (&filter)[FILTER_SIZE][FILTER_HEIGHT][FILTER_WIDTH][FILTER_CHANNEL], vector<size_t> filter_dim, T3 (&bias)[N2], \
	vector<size_t> stride, vector<size_t> pad, size_t group, T4 (&result)[N3][M3][K3][L3])
{
    cout <<  "-- conv() group: " << group << endl;

#ifdef _DEBUG
    cout <<  "-- conv() -- " << endl;

  cout << "BATCH: " << BATCH << endl;
  cout << "HEIGHT: " << HEIGHT << endl;
  cout << "WIDTH: " << WIDTH << endl;
  cout << "CHANNEL: " << CHANNEL << endl;
  cout << "pad: " << pad.at(0) << endl;
  cout << "stride: " << stride.at(0) << endl;
  cout << "FILTER_HEIGHT: " << FILTER_HEIGHT << endl;
  cout << "FILTER_WIDTH: " << FILTER_WIDTH << endl;
  cout << "FILTER_CHANNEL: " << FILTER_CHANNEL << endl;
  cout << "BIAS LEN: " << N2 << endl;
#endif

    int pad_xleft = pad.at(0);
    int pad_xright = pad.at(1);
    int pad_ytop = pad.at(2);
    int pad_ybottom = pad.at(3);
    int xadd = pad_xleft + pad_xright;
    int yadd = pad_ytop + pad_ybottom;

    int stride_size = stride.at(0);

    int out_h = (HEIGHT + yadd - FILTER_HEIGHT)/stride_size + 1;
    int out_w = (WIDTH + xadd - FILTER_WIDTH)/stride_size + 1;

#ifdef _DEBUG
    cout << "FILTER_SIZE: " << FILTER_SIZE << endl;
  cout << "out_h: " << out_h << endl;
  cout << "out_w: " << out_w << endl;

  struct timespec tp, ep;
  double timeCheck;
  clock_gettime(CLOCK_MONOTONIC, &tp);
#endif

#if defined(__MACH__) || defined(_USE_LLVM)
    float (* newin)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
#else
    float (* newin)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
#endif

    //step 1. 패딩

    //int jp, kp;
    //printf("패딩 변환 중\n");
    for (int b = 0; b < BATCH; b++){
        for (int h = 0; h < HEIGHT; h++){
            for (int w = 0; w < WIDTH; w++){
                for (size_t c = 0; c < CHANNEL; c++){
                    (*newin)[b][pad_ytop+h][pad_xleft+w][c] = in1[b][h][w][c];
                }
            }
        }
    }

    //step 2. 입력값 차원 변환
    //printf("입력값 차원 변환 중 \n");
    float (*newin2)[BATCH][CHANNEL*FILTER_HEIGHT*FILTER_WIDTH][M3*K3];
    newin2 = (float(*)[BATCH][CHANNEL*FILTER_HEIGHT*FILTER_WIDTH][M3*K3])malloc(sizeof(float[BATCH][CHANNEL*FILTER_HEIGHT*FILTER_WIDTH][M3*K3]));
    //하나의 스트라이트에 각 채널들을 하나의 행으로 구성
    for (int b = 0, p = 0; b < BATCH; b++, p = 0)
        for (int i = 0; i < out_h; i++)
            for (int j = 0, q = 0; j < out_w; j++, q = 0, p++)
                for (int c = 0; c < CHANNEL; c++)
                    for (int ii = i * stride_size; ii < i * stride_size + FILTER_HEIGHT; ii++)
                        for (int jj = j * stride_size; jj < j * stride_size + FILTER_WIDTH; jj++)
                            (*newin2)[b][q++][p] = (*newin)[b][ii][jj][c];

    free(newin);

    // step 3. 필터 평활화
    //printf("필터 평활화 중\n");

    float (*transFilter)[FILTER_SIZE][FILTER_HEIGHT*FILTER_WIDTH*FILTER_CHANNEL]; //패딩 배열
    transFilter = (float(*)[FILTER_SIZE][FILTER_HEIGHT*FILTER_WIDTH*FILTER_CHANNEL])malloc(sizeof(float[FILTER_SIZE][FILTER_WIDTH*FILTER_HEIGHT*FILTER_CHANNEL]));
    memset(transFilter, 0x00, sizeof(float[FILTER_SIZE][FILTER_HEIGHT*FILTER_WIDTH*FILTER_CHANNEL]));

    for (int i1 = 0, q = 0; i1 < FILTER_SIZE; i1++, q = 0)
        for (int i2 = 0; i2 < FILTER_CHANNEL; i2++)
            for (int i3 = 0; i3 < FILTER_HEIGHT; i3++)
                for (int i4 = 0; i4 < FILTER_WIDTH; i4++)
                    (*transFilter)[i1][q++] = filter[i1][i3][i4][i2];



    // step 4. 행렬 곱.
    float (*finalResult)[BATCH][FILTER_SIZE][M3*K3];
    finalResult = (float (*)[BATCH][FILTER_SIZE][M3*K3])malloc(sizeof(float[BATCH][FILTER_SIZE][M3*K3]));
    memset(finalResult, 0x00, sizeof(float[BATCH][FILTER_SIZE][M3*K3]));

    int div = out_h * out_w; // B * out_h * out_w
    int total2 = BATCH * CHANNEL * FILTER_HEIGHT * FILTER_WIDTH * out_h * out_w;
    int total3 = total2 / div;
    //printf("행렬 곱 중\n");
#ifdef _USE_OPENBLAS_CONV
  size_t m1len = FILTER_SIZE*total3, m2len = total3*div, m3len = FILTER_SIZE*div;
    float *m1, *m2, *m3;

    m1 = (float *)malloc(sizeof(float)*m1len);
    m2 = (float *)malloc(sizeof(float)*m2len);
    m3 = (float *)malloc(sizeof(float)*m3len);

    for (int b = 0; b < BATCH; b++) {
        for (int i = 0; i < m1len; i++) {
            m1[i] = (*transFilter)[i/total3][i % total3];
        }

        for (int i = 0; i < m2len; i++) {
            m2[i] = (*newin2)[b][i/div][i % div];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, FILTER_SIZE, div, total3, 1.0, m1, total3, m2, div, 0.0, m3, div);

        for (int i = 0; i < m3len; i++) {
            (*finalResult)[b][i/div][i % div] = m3[i];
        }
    }
    free(m1);
    free(m2);
    free(m3);

#else

#ifdef THREAD_MATMUL

    //printf("행렬 곱 중\n");
    struct tTHREAD<float,float,float, FILTER_SIZE, BATCH*FILTER_CHANNEL*FILTER_HEIGHT*FILTER_WIDTH, M3*K3> p;

    // 스레드에 연산할 메모리 복사
    pthread_t threads[MAX_THREAD];
    p.arrr = (*transFilter);
    p.brrr1 =(*newin2);
    p.crrr1 =(*finalResult);

#ifdef __MACH__
    semaphore = dispatch_semaphore_create(1);
#else
    sem_init(&semaphore, 0, 1);
#endif

    for(int b = 0; b < BATCH; b++){
        for (int i = 0; i < MAX_THREAD; i++) {
#ifdef __MACH__
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
#else
            //sem_wait(&semaphore);
            int r;

            do {
                r = sem_wait(&semaphore);
            } while (r == -1 && errno == EINTR);
#endif

            p.cnt = i;
            p.b = b; // 배치 값

            pthread_create(&threads[i], NULL, multi<float, float, float, FILTER_SIZE, BATCH*FILTER_CHANNEL*FILTER_HEIGHT*FILTER_WIDTH, M3*K3>, (void *)&p); // filter_size, total3, div
        }

        for (int t = 0; t < MAX_THREAD; t++)
            pthread_join(threads[t], NULL); // 모든 스레드가 끝날 때까지 결합 및 대기
    }

#ifdef __MACH__
    dispatch_release(semaphore);
#else
    // sem_distroy(&semaphore);
#endif


#else
    // k for문과 j for 문을 변경함으로써, 지역성 향상.
    for (int b = 0; b < BATCH; b++) {
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int k = 0; k < total3; k++) {
                for (int j = 0; j < div; j++) {
                    (*finalResult)[b][i][j] += (*transFilter)[i][k] * (*newin2)[b][k][j];
                }
            }
        }
    }
#endif

#endif
    // step 5. 결과 차원 변환및 편향 합
    //printf("마지막 차원 변환 중\n");
    //int putout_c = (FILTER_SIZE) / (BATCH);

    for (int i1 = 0; i1 < BATCH; i1++) {
        for (int i4 = 0; i4 < FILTER_SIZE; i4++) {
            size_t q = 0;
            for (int i2 = 0; i2 < out_h; i2++) {
                for (int i3 = 0; i3 < out_w; i3++) {
                    result[i1][i2][i3][i4] = (*finalResult)[i1][i4][q++] + bias[i4];
                }
            }
            //cout << (float)bias[i4] << "\t" << endl;
        }
    }

    free(newin2);
    free(transFilter);
    free(finalResult);

#ifdef _DEBUG
    clock_gettime(CLOCK_MONOTONIC, &ep);
  timeCheck = ((ep.tv_sec - tp.tv_sec) + (ep.tv_nsec - tp.tv_nsec) / 1000000000.0);
  cout << "==================== " + to_string(conv_idx++) + " convolution time : " << timeCheck <<" seconds  ====================" << endl << endl;
#endif
}

#endif

/*
 *
 *
 * Average Pooling
 *
 *
 */

template <typename T1, size_t BATCH, size_t HEIGHT, size_t WIDTH, size_t CHANNEL,
        typename T2, size_t N2, size_t M2, size_t K2, size_t L2>
void avgpool(T1 (&in)[BATCH][HEIGHT][WIDTH][CHANNEL], vector<size_t> windows_dim, vector<size_t> stride, vector<size_t> pad, T2 (&result)[N2][M2][K2][L2])
{
    //	time_t istart,iend;
//	time (&istart);

    //N, H, W, C
    //N == 1
    //group == 1

    int pad_xleft = pad.at(0);
    int pad_xright = pad.at(1);
    int pad_ytop = pad.at(2);
    int pad_ybottom = pad.at(3);
    int xadd = pad_xleft + pad_xright;
    int yadd = pad_ytop + pad_ybottom;

    int stride_size = stride.at(0);

    int window_h = windows_dim.at(0);
    int window_w = windows_dim.at(1);

    int out_h = (HEIGHT + yadd - window_h)/stride_size + 1;
    int out_w = (WIDTH + xadd - window_w)/stride_size + 1;

    const int new_height = HEIGHT + pad.at(0) + pad.at(1);
    const int new_width = WIDTH +  pad.at(2) + pad.at(3);

#ifdef _DEBUG
    cout << "avgpool()" <<endl;

//  cout << "BATCH: " << BATCH << endl;
//  cout << "HEIGHT: " << HEIGHT << endl;
//  cout << "WIDTH: " << WIDTH << endl;
//  cout << "CHANNEL: " << CHANNEL << endl;
//  cout << "pad: " << pad.at(0) << endl;
//  cout << "stride: " << stride.at(0) << endl;
//  cout << "WINDOW_HEIGHT: " << window_h << endl;
//  cout << "WINDOW_WIDTH: " << window_w << endl;
#endif


#if defined(__MACH__) || defined(_USE_LLVM)
    float (* newin)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
#else
    float (*newin)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
#endif


//    float (* newin)[BATCH][HEIGHT+MAX_PAD_SIZE][WIDTH+MAX_PAD_SIZE][CHANNEL];
//    newin = (float (*)[BATCH][HEIGHT+MAX_PAD_SIZE][WIDTH+MAX_PAD_SIZE][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE][WIDTH+MAX_PAD_SIZE][CHANNEL]));

    for (int b = 0; b < BATCH; b++){
        for (size_t h = 0; h < HEIGHT; h++){
            for (size_t w = 0; w < WIDTH; w++){
                for (int c = 0; c < CHANNEL; c++){
                    (*newin)[b][pad_ytop+h][pad_xleft+w][c] = in[b][h][w][c];
                }
            }
        }
    }

    int wnum = window_h*window_w;
    int new_h_idx = 0, new_w_idx = 0;
    for (int b = 0; b < BATCH; b++){
        for(int c = 0; c < CHANNEL; c++) {

            for(int i = 0; i <out_h; i++) {
                for(int j = 0; j <out_w; j++) {

                    int pad_cnt = 0;
                    float total = 0.0;
                    for(int fh = 0; fh <window_h; fh++) {
                        for(int fw = 0; fw <window_w; fw++) {

                            int h_idx = new_h_idx+fh, w_idx = new_w_idx+fw;

                            if(h_idx < pad_ytop || w_idx < pad_xleft || h_idx >= (new_height - pad_ybottom)|| w_idx >= (new_width - pad_xright)) {
                                pad_cnt++;
                            }

                            total = total + (*newin)[b][h_idx][w_idx][c];
                        }
                    }

                    //cout << "pad_cnt = " << pad_cnt << endl;

                    result[b][i][j][c] = total/(wnum-pad_cnt);
                    new_w_idx += stride_size;
                }
                new_h_idx += stride_size;
                new_w_idx = 0;
            }
            new_h_idx = 0;
        }
    }

    free(newin);

//    time (&iend);
//    double dif = difftime (iend,istart);
//    printf ("==> [Avg. pooling time] %.4lf seconds.\n", dif );

}


//N H W C
/*
 *
 *
 * Max Pooling
 *
 *
 */

template <typename T1, size_t BATCH, size_t HEIGHT, size_t WIDTH, size_t CHANNEL,
        typename T3, size_t N2, size_t M2, size_t K2, size_t L2>
void maxpool(T1 (&in1)[BATCH][HEIGHT][WIDTH][CHANNEL], vector<size_t> windows_dim, vector<size_t> stride, vector<size_t> pad, T3 (&result)[N2][M2][K2][L2])
{

#ifdef _DEBUG
    cout << "maxpool()" << endl;
#endif
//	time_t istart,iend;
//	time (&istart);


    //N, H, W, C
    //N == 1
    //group == 1


    int pad_xleft = pad.at(0);
    int pad_xright = pad.at(1);
    int pad_ytop = pad.at(2);
    int pad_ybottom = pad.at(3);
    int xadd = pad_xleft + pad_xright;
    int yadd = pad_ytop + pad_ybottom;

    int stride_size = stride.at(0);

    int window_h = windows_dim.at(0);
    int window_w = windows_dim.at(1);

    int out_h = (HEIGHT + yadd - window_h)/stride_size + 1;
    int out_w = (WIDTH + xadd - window_w)/stride_size + 1;



#if defined(__MACH__) || defined(_USE_LLVM)
    float (* newin)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL];
    newin = (float (*)[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
    memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+MAX_PAD_SIZE*2][WIDTH+MAX_PAD_SIZE*2][CHANNEL]));
#else
    float (* newin)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL];
  newin = (float (*)[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL])malloc(sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
  memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));
#endif

    // memset(newin, 0x00, sizeof(float[BATCH][HEIGHT+yadd][WIDTH+xadd][CHANNEL]));

    for (int b = 0; b < BATCH; b++){
        for (size_t h = 0; h < HEIGHT; h++){
            for (size_t w = 0; w < WIDTH; w++){
                for (int c = 0; c < CHANNEL; c++){
                    (*newin)[b][pad_ytop+h][pad_xleft+w][c] = in1[b][h][w][c];
                }
            }
        }
    }

    int new_h_idx = 0, new_w_idx = 0;
    for (int b = 0; b < BATCH; b++){
        for(int c = 0; c < CHANNEL; c++) {

            for(int i = 0; i <out_h; i++) {
                for(int j = 0; j <out_w; j++) {

                    float max = -INFINITY;;
                    for(int fh = 0; fh <window_h; fh++) {
                        for(int fw = 0; fw <window_w; fw++) {

                            float value = (*newin)[b][new_h_idx+fh][new_w_idx+fw][c];

                            if(max < value)
                                max = value;
                        }
                    }

                    result[b][i][j][c] = max;
                    new_w_idx += stride_size;
                }
                new_h_idx += stride_size;
                new_w_idx = 0;
            }
            new_h_idx = 0;

        }
    }


//    if(pad_size > 0)
    free(newin);

//    time (&iend);
//    double dif = difftime (iend,istart);
//    printf ("==> [Max. pooling time] %.4lf seconds.\n", dif );
} // end maxpool



/*
 *
 *
 * Softmax 2D
 *
 *
 */
template <typename T1, size_t N, size_t M, typename T2>
void softmax(T1 (&in1)[N][M], int axis, T2 (&result)[N][M])
{

#ifdef _DEBUG
    cout << "softmax()-2D" << endl;
#endif
    //axis

    // clock_t start, end;
    // start = clock();

    if(axis == 1) {

        float max[N];
        for(size_t i = 0; i < N; i++) {
            max[i] = -INFINITY;
        }

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                if (in1[i][j] > max[i]) {
                    max[i] = in1[i][j];
                }
            }
        }

        float sum[N];
        memset(sum, 0x00, sizeof(float)*N);
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                sum[i] += expf(in1[i][j]- max[i]);
            }
        }

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                result[i][j] = expf(in1[i][j]-max[i])/sum[i];
            }
        }
    } else {

        float max = -INFINITY;
        float sum = 0.0;


        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                if (in1[i][j] > max) {
                    max = in1[i][j];
                }
            }
        }

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                sum += expf(in1[i][j]- max);
            }
        }

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                result[i][j] = expf(in1[i][j]-max)/sum;
            }
        }
    }

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}



/*
 *
 *
 * Softmax 3D
 *
 *
 *
 */

template <typename T1, size_t N1, size_t N2, size_t N3,
        typename T2>
void softmax(T1 (&in1)[N1][N2][N3], int axis, T2 (&result)[N1][N2][N3])
{
#ifdef _DEBUG
    cout << "softmax()-3D" << endl;
#endif
    //cout << "axis = " << axis << endl;

    //clock_t start, end;
    //start = clock();

    if(axis == 1) {

        float max[N1];
        for(int i = 0; i < N1; i++) {
            max[i] = -INFINITY;
        }

        for (size_t i = 0; i < N1; i++) {
            for (size_t k = 0; k < N3; k++) {
                for (size_t j = 0; j < N2; j++) {
                    if (in1[i][j][k] > max[i]) {
                        max[i] = in1[i][j][k];
                    }
                }
            }
        }

        float sum[N1];
        memset(sum, 0x00, sizeof(float)*N1);
        for (size_t i = 0; i < N1; i++) {
            for (size_t j = 0; j < N2; j++) {
                for (size_t k = 0; k < N3; k++) {
                    sum[i] += expf(in1[i][j][k]- max[i]);
                }
            }
        }

        for (size_t i = 0; i < N1; i++) {
            for (size_t j = 0; j < N2; j++) {
                for (size_t k = 0; k < N3; k++) {
                    result[i][j][k] = expf(in1[i][j][k]-max[i])/sum[i];
                }
            }
        }
    } else {

        float max = -INFINITY;
        float sum = 0.0;


        for (size_t i = 0; i < N1; i++) {
            for (size_t k = 0; k < N3; k++) {
                for (size_t j = 0; j < N2; j++) {
                    if (in1[i][j][k] > max) {
                        max = in1[i][j][k];
                    }
                }
            }
        }

        for (size_t i = 0; i < N1; i++) {
            for (size_t j = 0; j < N2; j++) {
                for (size_t k = 0; k < N3; k++) {
                    sum += expf(in1[i][j][k]- max);
                }
            }
        }

        for (size_t i = 0; i < N1; i++) {
            for (size_t j = 0; j < N2; j++) {
                for (size_t k = 0; k < N3; k++) {
                    result[i][j][k] = expf(in1[i][j][k]-max)/sum;
                }
            }
        }
    }

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}


/*
 *
 *
 * Relu 4D
 *
 *
 *
 */

template <typename T1, size_t N, size_t M, size_t K, size_t L>
void relu(T1 (&in1)[N][M][K][L])
{
#ifdef _DEBUG
    cout << "relu()-4D" << endl;
#endif
    //  clock_t start, end;
//	start = clock();

    for (size_t i = 0; i < N; i++){
        for (size_t j=0; j<M; j++){
            for (size_t k=0; k<K; k++){
                for (size_t l=0; l<L; l++)
                {
                    if (in1[i][j][k][l] < 0.0)
                        in1[i][j][k][l] = 0.0;

                }
            }
        }
    }

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

} //end relu




/*
 *
 *
 * Relu 2D
 *
 *
 */
template <typename T1, size_t N, size_t M>
void relu(T1 (&in1)[N][M])
{
#ifdef _DEBUG
    cout << "relu()-2D" << endl;
#endif
    //    clock_t start, end;
//	start = clock();


    for (size_t i=0; i < N; i++) {
        for (size_t j=0; j<M; j++){
            if (in1[i][j] <= 0.0)
                in1[i][j] = 0.0;
        }
    }

//	end = clock();
//	cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

} //end relu



/*
 *
 *
 * Add 2D + 2D
 *
 *
 */

template <typename T1, size_t N, size_t M>
void add(T1 (&in1)[N][M], T1 (&in2)[N][M], T1 (&result)[N][M])
{

#ifdef _DEBUG
    cout << "add()-2D + 2D" << endl;
#endif
    //    clock_t start, end;
//	start = clock();

    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < M; j++){
            result[i][j] = in1[i][j] + in2[i][j];
        }
    }

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

    //cout << "[LIB] add end" << endl;

}

/*
 *
 *
 * Add 2D + 1D
 *
 *
 */
template <typename T1, size_t N, size_t M>
void add(T1 (&in1)[N][M], T1 (&in2)[M], T1 (&result)[N][M])
{
#ifdef _DEBUG
    cout << "add()-2D + 1D" << endl;
#endif
    //    clock_t start, end;
//	start = clock();

    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < M; j++){
            result[i][j] = in1[i][j] + in2[j];
        }
    }


//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

    //cout << "[LIB] add end" << endl;

}

/*
 *
 *
 * Add 2D + 2D
 *
 *
 */
template <typename T1, size_t N, size_t M>
void add(T1 (&in1)[N][M], T1 (&in2)[N][M])
{
#ifdef _DEBUG
    cout << "add()-2D + 2D" << endl;
#endif
    //    clock_t start, end;
//	start = clock();

    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < M; j++){
            in1[i][j] = in1[i][j] + in2[i][j];
        }
    }
}

/*
 *
 *
 * Add 2D + 1D
 *
 *
 */
template <typename T1, size_t N, size_t M>
void add(T1 (&in1)[N][M], T1 (&in2)[M])
{
#ifdef _DEBUG
    cout << "add()-2D + 1D" << endl;
#endif
    //    clock_t start, end;
//	start = clock();

    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < M; j++){
            in1[i][j] = in1[i][j] + in2[j];
        }
    }


}


/*
 *
 *
 * Add 4D + 1D
 *
 *
 */
template <typename T1, size_t N1, size_t N2, size_t N3, size_t N4,
        typename T2, size_t M>
void add(T1 (&in1)[N1][N2][N3][N4], T2 (&in2)[M], T1 (&result)[N1][N2][N3][N4])
{
#ifdef _DEBUG
    cout << "add()-2D + 1D" << endl;
#endif
    //clock_t start, end;
    //start = clock();

    for (int i1 = 0; i1 < N1; i1++){
        for (int i2 = 0; i2 < N2; i2++){
            for (int i3 = 0; i3 < N3; i3++){
                for (int i4 = 0; i4 < N4; i4++){

                    result[i1][i2][i3][i4] = in1[i1][i2][i3][i4] + in2[i2];
                }
            }
        }
    }

}

/*
 *
 *
 * Add 4D + 1D
 *
 *
 */
template <typename T1, size_t N1, size_t N2, size_t N3, size_t N4,
        typename T2, size_t M>
void add(T1 (&in1)[N1][N2][N3][N4], T2 (&in2)[M])
{
#ifdef _DEBUG
    cout << "add()-2D + 1D" << endl;
#endif
    //clock_t start, end;
    //start = clock();

    for (int i1 = 0; i1 < N1; i1++){
        for (int i2 = 0; i2 < N2; i2++){
            for (int i3 = 0; i3 < N3; i3++){
                for (int i4 = 0; i4 < N4; i4++){

                    in1[i1][i2][i3][i4] = in1[i1][i2][i3][i4] + in2[i2];
                }
            }
        }
    }

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

    //cout << "[LIB] add end" << endl;

}


/*
 *
 *
 * Add 4D + 4D
 *
 *
 */
template <typename T1, size_t N, size_t M, size_t K, size_t L>
void add(T1 (&in1)[N][M][K][L], T1 (&in2)[N][M][K][L], T1 (&result)[N][M][K][L])
{
#ifdef _DEBUG
    cout << "add()" << endl;
#endif
    //    clock_t start, end;
//	start = clock();

    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < K; k++){
                for (int l = 0; l < L; l++){
                    result[i][j][k][l] = in1[i][j][k][l] + in2[i][j][k][l];
                }
            }
        }
    }

//	end = clock();
//	cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;



} //end add


/*
 *
 *
 * Add 4D + 4D
 *
 *
 */
template <typename T1, size_t N, size_t M, size_t K, size_t L>
void add(T1 (&in1)[N][M][K][L], T1 (&in2)[N][M][K][L])
{
#ifdef _DEBUG
    cout << "add()- 4D +4D" << endl;
#endif
    //    clock_t start, end;
//	start = clock();

    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < M; j++){
            for (size_t k = 0; k < K; k++){
                for (size_t l = 0; l < L; l++){
                    in1[i][j][k][l] = in1[i][j][k][l] + in2[i][j][k][l];
                }
            }
        }
    }

//	end = clock();
//	cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;
//
//

}

template <typename T1, size_t N1, size_t M1, typename T2, size_t N2, size_t M2, typename T3, size_t N3, typename T4, size_t N4, size_t M4>
void gemm(T1 (&a)[N1][M1], T2 (&b)[N2][M2], T3 (&c)[N3], float alpha, float beta, int transA, int transB, T4 (&result)[N4][M4]) {

#ifdef _DEBUG
    cout << "gemm()" << endl;
//  cout << "N: " << N << endl;
//  cout << "K: " << K << endl;
//  cout << "M: " << M << endl;

    struct timespec tp, ep;
    double timeCheck;
    clock_gettime(CLOCK_MONOTONIC, &tp);
#endif

//#ifdef _USE_OPENBLAS_MATMUL
//    size_t m1len = N*K, m2len = K*M, m3len = N*M;
//    float *m1, *m2, *m3;
//
//    m1 = (float *)malloc(sizeof(float)*m1len);
//    m2 = (float *)malloc(sizeof(float)*m2len);
//    m3 = (float *)malloc(sizeof(float)*m3len);
//
//    for(int i = 0; i < m1len; i++) {
//        m1[i] = in1[i/K][i%K];
//    }
//
//    for(int i = 0; i < m2len; i++) {
//        m2[i] = weight[i/M][i%M];
//    }
//
//    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1, m1, K, m2, M, 0, m3, M);
//
//    for(int i = 0; i < m3len; i++) {
//        result[i/M][i%M] = m3[i] + bias[i];
//    }
//
//    free(m1);
//    free(m2);
//    free(m3);
//
//#else

//    cout << "transA = " << transA << " transB = " << transB << endl;
    if(transA == 0 && transB == 0) {
        for (size_t i = 0; i < N1; i++){
            for (size_t j = 0;  j <M1; j++){
                for (size_t k = 0; k < M2; k++){
                    result[i][j] += a[i][k]*b[k][j];
                }
            }
        }
    } else if (transA == 0 && transB == 1) {

        float transposedB[M2][N2];
        for (size_t i = 0; i < N2; i++) {
            for (size_t j = 0; j < M2; j++) {
                transposedB[j][i] = b[i][j];
            }
        }

        for (size_t i = 0; i < N1; i++){
            for (size_t j = 0;  j <M2; j++){
                for (size_t k = 0; k < M1; k++){
                    result[i][j] += a[i][k]*transposedB[k][j];
                }
            }
        }
    }  else if (transA == 1 && transB == 0) {
        float transposedA[M1][N1];
        for (size_t i = 0; i < N1; i++) {
            for (size_t j = 0; j < M1; j++) {
                transposedA[j][i] = a[i][j];
            }
        }

        for (size_t i = 0; i < M1; i++){
            for (size_t j = 0;  j <M2; j++){
                for (size_t k = 0; k < N1; k++){
                    result[i][j] += transposedA[i][k]*b[k][j];
                }
            }
        }
    }

    for (size_t i = 0; i < N4; i++){
        for (size_t j = 0; j < M4; j++){
            result[i][j] = result[i][j] + c[j];
        }
    }

//#endif

#ifdef _DEBUG
    clock_gettime(CLOCK_MONOTONIC, &ep);
    timeCheck = ((ep.tv_sec - tp.tv_sec) + (ep.tv_nsec - tp.tv_nsec) / 1000000000.0);
    cout << "====================  gemm time : " << timeCheck <<" seconds  ====================" << endl << endl;
#endif
}

//void cblas_sgemm(const enum CBLAS_ORDER __Order, const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_TRANSPOSE __TransB,
// const int __M, const int __N, const int __K, const float __alpha, const float *__A, const int __lda, const float *__B, const int __ldb, const float __beta, float *__C, const int __ldc);

template <typename T1, size_t N, size_t K, typename T2, size_t M, typename T3, typename T4>
void fullyconnected(T1 (&in1)[N][K], T2 (&weight)[K][M], T3 (&bias)[M], T4 (&result)[N][M]) {

#ifdef _DEBUG
    cout << "fullyconnected()" << endl;
//  cout << "N: " << N << endl;
//  cout << "K: " << K << endl;
//  cout << "M: " << M << endl;

  struct timespec tp, ep;
  double timeCheck;
  clock_gettime(CLOCK_MONOTONIC, &tp);
#endif

#ifdef _USE_OPENBLAS_MATMUL
    size_t m1len = N*K, m2len = K*M, m3len = N*M;
    float *m1, *m2, *m3;

    m1 = (float *)malloc(sizeof(float)*m1len);
    m2 = (float *)malloc(sizeof(float)*m2len);
    m3 = (float *)malloc(sizeof(float)*m3len);

    for(int i = 0; i < m1len; i++) {
        m1[i] = in1[i/K][i%K];
    }

    for(int i = 0; i < m2len; i++) {
        m2[i] = weight[i/M][i%M];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1, m1, K, m2, M, 0, m3, M);

    for(int i = 0; i < m3len; i++) {
        result[i/M][i%M] = m3[i] + bias[i];
    }

    free(m1);
    free(m2);
    free(m3);

#else
    for (size_t i = 0; i < N; i++){
        for (size_t j = 0;  j <M; j++){
            for (size_t k = 0; k < K; k++){
                result[i][j] += in1[i][k]*weight[k][j];
            }
        }
    }

    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < M; j++){
            result[i][j] = result[i][j] + bias[j];
        }
    }

#endif

#ifdef _DEBUG
    clock_gettime(CLOCK_MONOTONIC, &ep);
  timeCheck = ((ep.tv_sec - tp.tv_sec) + (ep.tv_nsec - tp.tv_nsec) / 1000000000.0);
  cout << "====================  fullyconnected time : " << timeCheck <<" seconds  ====================" << endl << endl;
#endif
}

template <typename T1, size_t N, size_t K, typename T2, size_t M, typename T3, typename T4>
void fullyconnected(T1 (&in1)[N][K], T2 (&weight)[K][M], T3 (&bias)[N][M], T4 (&result)[N][M]) {

#ifdef _DEBUG
    cout << "fullyconnected()" << endl;
//  cout << "N: " << N << endl;
//  cout << "K: " << K << endl;
//  cout << "M: " << M << endl;

  struct timespec tp, ep;
  double timeCheck;
  clock_gettime(CLOCK_MONOTONIC, &tp);
#endif

#ifdef _USE_OPENBLAS_MATMUL
    size_t m1len = N*K, m2len = K*M, m3len = N*M;
    float *m1, *m2, *m3;

    m1 = (float *)malloc(sizeof(float)*m1len);
    m2 = (float *)malloc(sizeof(float)*m2len);
    m3 = (float *)malloc(sizeof(float)*m3len);

    for(int i = 0; i < m1len; i++) {
        m1[i] = in1[i/K][i%K];
    }

    for(int i = 0; i < m2len; i++) {
        m2[i] = weight[i/M][i%M];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1, m1, K, m2, M, 0, m3, M);

    for(int i = 0; i < m3len; i++) {
        result[i/M][i%M] = m3[i] + bias[i];
    }

    free(m1);
    free(m2);
    free(m3);

#else
    for (size_t i = 0; i < N; i++){
        for (size_t j = 0;  j <M; j++){
            for (size_t k = 0; k < K; k++){
                result[i][j] += in1[i][k]*weight[k][j];
            }
        }
    }

    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < M; j++){
            result[i][j] = result[i][j] + bias[i][j];
        }
    }

#endif

#ifdef _DEBUG
    clock_gettime(CLOCK_MONOTONIC, &ep);
  timeCheck = ((ep.tv_sec - tp.tv_sec) + (ep.tv_nsec - tp.tv_nsec) / 1000000000.0);
  cout << "====================  fullyconnected time : " << timeCheck <<" seconds  ====================" << endl << endl;
#endif
}

template <typename T1, size_t N, size_t K,
        typename T2, size_t N1, size_t M1,
        typename T3, size_t N2, size_t M2>
void matmul(T1 (&in1)[N][K], T2 (&in2)[N1][M1], T3 (&result)[N2][M2]) {

#ifdef _DEBUG
    cout << "N: " << N << endl;
  cout << "K: " << K << endl;
  cout << "M: " << M1 << endl;

  struct timespec tp, ep;
  double timeCheck;
  clock_gettime(CLOCK_MONOTONIC, &tp);
#endif

#ifdef _USE_OPENBLAS_MATMUL
    size_t m1len = N*K, m2len = N1*M1, m3len = N2*M2;
    float *m1, *m2, *m3;

    m1 = (float *)malloc(sizeof(float)*m1len);
    m2 = (float *)malloc(sizeof(float)*m2len);
    m3 = (float *)malloc(sizeof(float)*m3len);

    for(int i = 0; i < m1len; i++) {
        m1[i] = in1[i/K][i%K];
    }

    for(int i = 0; i < m2len; i++) {
        m2[i] = in2[i/M1][i%M1];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M1, K, 1, m1, K, m2, M1, 0, m3, M1);

    for(int i = 0; i < m3len; i++) {
        result[i/M2][i%M2] = m3[i];
    }

    free(m1);
    free(m2);
    free(m3);

#else
    for (size_t i = 0; i < N; i++){
//        for (size_t k = 0; k < M; k++){
        for (size_t j = 0;  j <M1; j++){
            for (size_t k = 0; k < K; k++){
                result[i][j] += in1[i][k]*in2[k][j];
            }
        }
    }

#endif

#ifdef _DEBUG
    clock_gettime(CLOCK_MONOTONIC, &ep);
  timeCheck = ((ep.tv_sec - tp.tv_sec) + (ep.tv_nsec - tp.tv_nsec) / 1000000000.0);
  cout << "====================  matmul time : " << timeCheck <<" seconds  ====================" << endl << endl;
#endif
}



/*
 *
 *
 * Transpose 2D
 *
 *
 */
template <typename T1, size_t N, size_t M, typename T2, size_t N1, size_t M1>
void transpose(T1 (&in1)[N][M], vector<size_t> shuffle, T2 (&result)[N1][M1])
{
#ifdef _DEBUG
    cout << "transpose()-2D-2D" << endl;
#endif
    //  clock_t start, end;
//	start = clock();

    for (size_t i=0; i<N; i++){
        for (size_t j=0; j<M; j++){
            result[j][i] = in1[i][j];
        }
    }

//
//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}

/*
 *
 *
 * Transpose 3D
 *
 *
 */
template <typename T1, size_t M, size_t K, size_t L,
        typename T2, size_t M1, size_t K1, size_t L1>
void transpose(T1 (&in1)[M][K][L], vector<size_t> shuffle, T2 (&result)[M1][K1][L1])
{
#ifdef _DEBUG
    cout << "transpose()-3D" << endl;
#endif
    //
//    clock_t start, end;
//    start = clock();

    int shuffleType = 0;
 //   int shuffle0 = shuffle.at(0);
    int shuffle1 = shuffle.at(1);
    int shuffle2 = shuffle.at(2);
    int shuffle3 = shuffle.at(3);


    if(shuffle1 == 1 && shuffle2 == 2 && shuffle3  == 0) {
        shuffleType = 120;
    } else if(shuffle1 == 2 && shuffle2 == 0 && shuffle3  == 1) {
        shuffleType = 201;
    } else if(shuffle1 == 0 && shuffle2 == 2 && shuffle3  == 1) {
        shuffleType = 021;
    } else if(shuffle1 == 1 && shuffle2 == 0 && shuffle3  == 2) {
        shuffleType = 102;
    }

    cout << shuffleType <<endl;
    for (int j=0; j<M; j++){
        for (int k=0; k<K; k++){
            for (int l=0; l<L; l++){
                if(shuffleType == 120) {
                    result[k][l][j] = in1[j][k][l]; // 1, 2, 0
                } else if (shuffleType == 201) {
                    result[l][j][k] = in1[j][k][l]; // 2, 0, 1
                } else if (shuffleType == 021) {
                    result[j][l][k] = in1[j][k][l]; // 0, 2, 1
                } else if (shuffleType == 102) {
                    result[k][j][l] = in1[j][k][l]; // 1, 0, 2
                } else {
                    cout << "No!!!!!!!!!!!!";
                }
            }
            // cout <<endl;
        }
        //cout <<endl;
    }
    //cout <<endl;

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;


} // end transpose


/*
 *
 *
 * Transpose 6D
 *
 *
 */
template <typename T1, size_t N1, size_t N2, size_t N3, size_t N4, size_t N5, size_t N6,
        typename T2, size_t M1, size_t M2, size_t M3, size_t M4, size_t M5, size_t M6>
void transpose( T1 (&in1)[N1][N2][N3][N4][N5][N6], vector<size_t> shuffle, T2 (&result)[M1][M2][M3][M4][M5][M6])
{
#ifdef _DEBUG
    cout << "transpose-6D" << endl;
#endif
    //
//    clock_t start, end;
//    start = clock();

    int shuffleType = 0;
    int shuffle0 = shuffle.at(0);
    int shuffle1 = shuffle.at(1);
    int shuffle2 = shuffle.at(2);
    int shuffle3 = shuffle.at(3);
    int shuffle4 = shuffle.at(4);
    int shuffle5 = shuffle.at(5);

    if(shuffle0 == 0 &&shuffle1 == 1 && shuffle2 == 4 && shuffle3  == 2 && shuffle4  == 5 && shuffle5  == 3) {
        shuffleType = 14253;
    }


    //  cout << shuffleType <<endl;
//#pragma omp parallel for
    for (int i1 = 0; i1 < N1; i1++){
        for (int i2 = 0; i2 < N2; i2++){
            for (int i3 = 0; i3 < N3; i3++){
                for (int i4 = 0; i4 < N4; i4++){
                    for (int i5 = 0; i5 < N5; i5++){
                        for (int i6 = 0; i6 < N6; i6++){

                            if(shuffleType == 14253) {
                                result[i1][i2][i5][i3][i6][i4] = in1[i1][i2][i3][i4][i5][i6]; // 0, 1, 4, 2, 5, 3
                            }

                        }

                    }
                }
                // cout <<endl;
            }
            //cout <<endl;
        }
        //cout <<endl;
    }

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}


/*
 *
 *
 * Transpose 4D
 *
 *
 */
template <typename T1, size_t N1, size_t N2, size_t N3, size_t N4,
        typename T2, size_t M1, size_t M2, size_t M3, size_t M4>
void transpose( T1 (&in1)[N1][N2][N3][N4], vector<size_t> shuffle, T2 (&result)[M1][M2][M3][M4])
{
#ifdef _DEBUG
    cout << "transpose()-4D" << endl;
#endif
//    clock_t start, end;
//    start = clock();

    int shuffleType = 0;
//    int shuffle0 = shuffle.at(0);
    int shuffle1 = shuffle.at(1);
    int shuffle2 = shuffle.at(2);
    int shuffle3 = shuffle.at(3);

    if(shuffle1 == 2 && shuffle2 == 3 && shuffle3  == 1) {
        shuffleType = 231;
    } else if(shuffle1 == 3 && shuffle2 == 1 && shuffle3  == 2) {
        shuffleType = 312;
    } else if(shuffle1 == 1 && shuffle2 == 3 && shuffle3  == 2) {
        shuffleType = 132;
    } else if(shuffle1 == 2 && shuffle2 == 1 && shuffle3  == 3) {
        shuffleType = 213;
    } else if(shuffle1 == 3 && shuffle2 == 2 && shuffle3  == 1) {
        shuffleType = 321;
    }else if(shuffle1 == 1 && shuffle2 == 3 && shuffle3  == 2) {
        shuffleType = 132;
    }

//   cout << "shuffle type = " << shuffleType <<endl;
//#pragma omp parallel for
    for (size_t i1 = 0; i1 < N1; i1++){
        for (size_t i2 = 0; i2< N2; i2++){
            for (size_t i3 = 0; i3 < N3; i3++){
                for (size_t i4 = 0; i4 < N4; i4++){
                    if(shuffleType == 231) {
                        result[i1][i3][i4][i2] = in1[i1][i2][i3][i4]; // 0, 2, 3, 1
                    } else if (shuffleType == 312) {
                        result[i1][i4][i2][i3] = in1[i1][i2][i3][i4]; // 0, 3, 1, 2
                    } else if (shuffleType == 132) {
                        result[i1][i2][i4][i3] = in1[i1][i2][i3][i4]; // 0, 1, 3, 2
                    } else if (shuffleType == 213) {
                        result[i1][i3][i2][i4] = in1[i1][i2][i3][i4]; // 0, 2, 1, 3
                    }else if (shuffleType == 321) {
                        result[i1][i4][i3][i2] = in1[i1][i2][i3][i4]; // 0, 2, 1, 3
                    }else if (shuffleType == 132) {
                        result[i1][i2][i4][i3] = in1[i1][i2][i3][i4]; // 0, 2, 1, 3
                    }
                    else {
                        cout << "No!!!!!!!!!!!!";
                    }
                }
                // cout <<endl;
            }
            //cout <<endl;
        }
        //cout <<endl;
    }

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;


}


/*
 *
 *
 * Reshape 4D => 2D
 *
 *
 */
template <typename T1, size_t N, size_t M, size_t L, size_t K,
        typename T2, size_t N1, size_t M1>
void reshape(T1 (&in1)[N][M][L][K], T2 (&result)[N1][M1])
{
#ifdef _DEBUG
    cout << "reshape()-4D -> 2D)" << endl;
#endif
    //
//    clock_t start, end;
//    start = clock();

//    cout << "^^^^^^^^^^" << endl;
//
//	for(int j = 0; j < M; j++){
//		if(j%10 == 0) cout <<endl;
//		cout << in1[0][j][0][0] << "\t";
//	}

    int total = 0;
    for(size_t i1 = 0; i1 < N; i1++) {
        for(size_t i2 = 0; i2 < M; i2++) {
            for(size_t i3 = 0; i3 < L; i3++) {
                for(size_t i4 = 0; i4 < K; i4++) {

                    total = i1*M*L*K + i2*L*K + i3*K + i4;
                    int idx1 = total/M1, idx2 = total%M1;

                    result[idx1][idx2] = in1[i1][i2][i3][i4];

                }
            }
        }
    }
//
//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}



/*
 *
 *
 * Reshape 6D => 4D
 *
 *
 */
template <typename T1, size_t N1, size_t N2, size_t N3, size_t N4, size_t N5, size_t N6,
        typename T2, size_t M1, size_t M2, size_t M3, size_t M4>
void reshape(T1 (&in1)[N1][N2][N3][N4][N5][N6], T2 (&result)[M1][M2][M3][M4])
{
#ifdef _DEBUG
    cout << "reshape()-6D -> 4D" << endl;
#endif
//    clock_t start, end;
//    start = clock();

    int total = 0;
    for(int i1 = 0; i1 < N1; i1++) {
        for(int i2 = 0; i2 < N2; i2++) {
            for(int i3 = 0; i3 <N3; i3++) {
                for(int i4 = 0; i4 < N4; i4++) {
                    for(int i5 = 0; i5 < N5; i5++) {
                        for(int i6 = 0; i6 < N6; i6++) {
                            total = i1*N2*N3*N4*N5 + i2*N3*N4*N5 + i3*N4*N5 + i4*N5 + i6;

                            int m1 = M2*M3*M4;
                            int m2 = M3*M4;

                            int idx1 = total/m1;
                            int idx2 = (total%m1)/m2;
                            int idx3 = (total%m1)%m2/M4;
                            int idx4 = (total%m1)%m2%M4;
                            result[idx1][idx2][idx3][idx4] = in1[i1][i2][i3][i4][i5][i6];
                        }
                    }
                }
            }
        }
    }

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}



/*
 *
 *
 * Reshape 4D => 6D
 *
 *
 */
template <typename T1, size_t N1, size_t N2, size_t N3, size_t N4,
        typename T2, size_t M1, size_t M2, size_t M3, size_t M4, size_t M5, size_t M6>
void reshape(T1 (&in1)[N1][N2][N3][N4], T2 (&result)[M1][M2][M3][M4][M5][M6])
{
#ifdef _DEBUG
    cout << "reshape()4D -> 6D" << endl;
#endif
    //
//    clock_t start, end;
//    start = clock();

    int total = 0;
    for(int i1 = 0; i1 < N1; i1++) {
        for(int i2 = 0; i2 < N2; i2++) {
            for(int i3 = 0; i3 <N3; i3++) {
                for(int i4 = 0; i4 < N4; i4++) {
                    total = i1*N2*N3*N4 + i2*N3*N4 + i3*N4 + i4;

                    int m1 = M2*M3*M4*M5*M6;
                    int m2 = M3*M4*M5*M6;
                    int m3 = M4*M5*M6;
                    int m4 = M5*M6;

                    int idx1 = total/m1;
                    int idx2 = (total%m1)/m2;
                    int idx3 = ((total%m1)%m2)/m3;
                    int idx4 = ((total%m1)%m2%m3)/m4;
                    int idx5 = (((total%m1)%m2%m3)%m4)/M6;
                    int idx6 = (((total%m1)%m2%m3)%m4)%M6;
                    result[idx1][idx2][idx3][idx4][idx5][idx6] = in1[i1][i2][i3][i4];
                }
            }
        }
    }
//
//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}


template<typename T1>
void leakyRelu(T1 in, vector<size_t> idim, T1 result, int type)
{
    cout << "leakyRelu():TBD" << endl;
}


template<typename T1, typename T2>
void LRN(T1 in, vector<size_t> dim, float size, float beta, float bias, float windowSize, T2 result, int type)
{
    cout << "LRN(): TBD" << endl;
}

template<typename T1, size_t N, size_t M, typename T2>
void dropout(T1 (&in)[N][M], vector<size_t> dim, T1 (&result)[N][M], T2 (&mask)[N][M], int type)
{
    cout << "dropout(): TBD" << endl;
}


template<typename T1, typename T2>
void mul(T1 in1, vector<size_t> dim, T2 in2, T1 result, int type)
{
    cout << "mul(): TBD" << endl;
}

template<typename T1, typename T2, size_t N, size_t M>
void splat(T1 value, T2 (&result)[N][M])
{
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            result[i][j] = (T2)value;
        }
    }
}

template <typename T1, size_t N, size_t M, size_t K, size_t L>
void channel_shuffle(T1 (&in1)[N][M][K][L], const int group, int kernel, T1 (&result)[N][M][K][L])
{
#ifdef _DEBUG
    cout << "shuffle()-4D" << endl;
    cout << "group: " << group << endl;
    cout << "kernel: " << kernel << endl;
#endif

    const int gdim = L/group;
    T1 rt[N][M][K][group][gdim];

    //reshape 4d -> 5d
    int total = 0;
    for(int i1 = 0; i1 < N; i1++) {
        for(int i2 = 0; i2 < M; i2++) {
            for(int i3 = 0; i3 <K; i3++) {
                for(int i4 = 0; i4 < L; i4++) {
                    total = i1*M*K*L + i2*K*L + i3*L + i4;

                    int m1 = M*K*group*gdim;
                    int m2 = K*group*gdim;
                    int m3 = group*gdim;
                    int m4 = gdim;

                    int idx1 = total/m1;
                    int idx2 = (total%m1)/m2;
                    int idx3 = ((total%m1)%m2)/m3;
                    int idx4 = ((total%m1)%m2%m3)/m4;
                    int idx5 = ((total%m1)%m2%m3)%m4;
                    rt[idx1][idx2][idx3][idx4][idx5] = in1[i1][i2][i3][i4];
                }
            }
        }
    }

    T1 transrt[N][M][K][gdim][group];
//    //transpose
    for (size_t i1 = 0; i1 < N; i1++){
        for (size_t i2 = 0; i2< M; i2++){
            for (size_t i3 = 0; i3 < K; i3++){
                for (size_t i4 = 0; i4 < group; i4++){
                    for (size_t i5 = 0; i5 < gdim; i5++) {
                        transrt[i1][i2][i3][i5][i4] = rt[i1][i2][i3][i4][i5];
                    }
                }
            }
        }
    }

    //reshape 5d -> 4d
    total = 0;
    for(int i1 = 0; i1 < N; i1++) {
        for(int i2 = 0; i2 < M; i2++) {
            for(int i3 = 0; i3 <K; i3++) {
                for(int i4 = 0; i4 < gdim; i4++) {
                    for(int i5 = 0; i5 < group; i5++) {
                            total = i1*M*K*gdim*group + i2*K*gdim*group + i3*gdim*group + i4*group + i5;

                            int m1 = M*K*L;
                            int m2 = K*L;

                            int idx1 = total/m1;
                            int idx2 = (total%m1)/m2;
                            int idx3 = (total%m1)%m2/L;
                            int idx4 = (total%m1)%m2%L;
                            result[idx1][idx2][idx3][idx4] = transrt[i1][i2][i3][i4][i5];
                        }
                }
            }
        }
    }

//	end = clock();
//    cout << "==> [time] : " << (((double)(end - start)) / CLOCKS_PER_SEC) << " seconds" << endl;

}


#endif /* LIBRARY_H_ */