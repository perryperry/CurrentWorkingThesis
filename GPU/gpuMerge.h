#ifndef GPUMERGE_H
#define GPUMERGE_H
#include <stdio.h>
#include <stdlib.h>

#define SHARED_SIZE_LIMIT 1024

__constant__ float const_histogram[60];

__global__ void staticReverse(float *d, int n);

__global__ void gpuSummationReduce(float * M00_in, float * M00_out, float * M1x_in, float * M1x_out, float * M1y_in, float * M1y_out, int n);

__global__ void gpuBackProjectKernel(float * d_hist, unsigned char * d_hueArray, int hueArrayLength , float * d_M00, float * d_M1x, float * d_M1y, int width, int xOffset, int yOffset);

__global__ void bpTestKernel(unsigned char * d_hueArray, int * d_converted, int hueArrayLength);

__global__ void gpuMeanShiftKernelForSubFrame(unsigned char * g_idata, float *g_odata, int * readyArray, int input_length, int blockCount,int width, int xOffset, int yOffset);

__global__ void gpuMeanShiftKernelForEntireFrame(unsigned char *g_idata, float *g_odata, int subframe_length, int blockCount, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset, int * cxy);

//__device__ void warpReduce(volatile float* sdata, int tid);

__device__ void warpReduce(volatile float* shared_M00, volatile float* shared_M1x, volatile float* shared_M1y, int tid);

__device__ void warpReduceSingleMatrix(volatile float* sdata, int tid) ;

void setConstantMemoryHistogram(float * histogram);


__global__ void  gpuFinalReduce(float * g_odata, int * cxy, int * row_offset, int * col_offset, int sub_width, int sub_height, int num_block);

#endif
