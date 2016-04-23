#ifndef KERNELS_H
#define KERNELS_H

#include <stdio.h>
#include <stdlib.h>

#define SHARED_SIZE_LIMIT 1024

__constant__ float const_histogram[60];

__global__ void staticReverse(float *d, int n);

//Probably obsolete at this point
__global__ void gpuMeanShiftKernelForSubFrame(unsigned char * g_idata, float *g_odata, int * readyArray, int input_length, int blockCount,int width, int xOffset, int yOffset);

__device__ void warpReduce(volatile float* shared_M00, volatile float* shared_M1x, volatile float* shared_M1y, int tid);

void setConstantMemoryHistogram(float * histogram);

// Below are my working kernels for now

__global__ void gpuBlockReduce(unsigned char *g_idata, float *g_odata, int subframe_length, int blockCount, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset);

__global__ void  gpuFinalReduce(float * g_odata, int * cx, int *cy, int * row_offset, int * col_offset, int sub_width, int sub_height, int num_block);


// Testing kernel for single kernel convergence

__global__ void gpuSingleKernelMeanShift(unsigned char *g_idata, float *g_odata, int * readyArray, int subframe_length, int blockCount, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset, int * cxy);

#endif
