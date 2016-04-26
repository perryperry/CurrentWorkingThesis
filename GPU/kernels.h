#ifndef KERNELS_H
#define KERNELS_H

#include <stdio.h>
#include <stdlib.h>

#define SHARED_SIZE_LIMIT 1024
#define HIST_BUCKETS 60
//Histogram bucket size is 60, but give it 120 for objects (for now)
__constant__ float const_histogram[120];

__global__ void staticReverse(float *d, int n);

__device__ void warpReduce(volatile float* shared_M00, volatile float* shared_M1x, volatile float* shared_M1y, int tid);

void setConstantMemoryHistogram(float * histogram, int num_objects);

// Below are my working kernels for now

__global__ void gpuBlockReduce(int obj_id, unsigned char *g_idata, float *g_odata, int subframe_length, int blockCount, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset);

__global__ void  gpuFinalReduce(int obj_id, float * g_odata, int * cx, int *cy, int * row_offset, int * col_offset, int sub_width, int sub_height, int num_block);


// Testing kernel for single kernel convergence

__global__ void gpuSingleKernelMeanShift(unsigned char *g_idata, float *g_odata, int * readyArray, int subframe_length, int blockCount, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset, int * cxy);







#endif
