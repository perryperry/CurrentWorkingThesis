#ifndef KERNELS_H
#define KERNELS_H

#include <stdio.h>
#include <stdlib.h>
#include <math_functions.h>

#define SHARED_SIZE_LIMIT 1024
#define HIST_BUCKETS 60
#define FRAME_WIDTH 1080
#define FRAME_HEIGHT 720

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


/****************************** MULTI OBJECT KERNELS BELOW **************************************/
__device__ int gpuBlockStart(int * d_obj_block_ends, int num_objects);

__global__ void gpuMultiObjectBlockReduce(int * d_obj_block_ends,
                                          int num_objects,
                                          unsigned char *g_idata,
                                          float *g_odata,
                                          int * subframe_length,
                                          int num_block,
                                          int abs_width,
                                          int * sub_widths,
                                          int * row_offset,
                                          int * col_offset,
                                          bool * converged);


__global__ void gpuMultiObjectFinalReduce(int * d_obj_block_ends,
                                          int num_objects,
                                          float *g_odata,
                                          int * finalX,
                                          int * finalY,
                                          int * cx,
                                          int * cy,
                                          int * prevX,
                                          int * prevY,
                                          int * row_offset,
                                          int * col_offset,
                                          int * sub_widths,
                                          int * sub_heights,
                                          int * sub_lengths,
                                          int num_block,
                                          bool * converged,
                                          bool adapt_window);

__device__ int gpuCalcObjID(int * d_obj_block_ends, int num_objects);

__global__ void gpuCamShiftMultiObjectFinalReduce(int * d_obj_block_ends,
                                                  int num_objects,
                                                  float *g_odata,
                                                  int * cx,
                                                  int * cy,
                                                  int * prevX,
                                                  int * prevY,
                                                  int * subframe_length,
                                                  int * row_offset,
                                                  int * col_offset,
                                                  int * sub_widths,
                                                  int * sub_heights,
                                                  int num_block,
                                                  bool * converged);

__host__ __device__ int gpuDistance(int x1, int y1, int x2, int y2);




/****************************** Dynamic parallelism BELOW **************************************/

__global__ void dynamicCamShiftLaunchKernel(int num_obj,
                                           unsigned char * frame,
                                           int frame_total,
                                           int frame_width,
                                           int * block_ends,
                                           float * g_out,
                                           int * subframe_totals,
                                           int * subframe_widths,
                                           int * subframe_heights,
                                           int * row_offset,
                                           int * col_offset,
                                           int * finalX,
                                           int * finalY,
                                           int * cx,
                                           int * cy,
                                           int * prevX,           // previous x coordinates of objects
                                           int * prevY,           // previous y coordinates of objects
                                           bool * converged,
                                           bool adapt_window);

__host__ __device__ bool objectsConverged(int num_objects,
                                 bool * obj_converged);

#endif
