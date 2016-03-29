#ifndef GPUMERGE_H
#define GPUMERGE_H

#include <stdio.h>
#include <stdlib.h>


#define SHARED_SIZE_LIMIT 1024

__constant__ float c_hist[60];

__global__ void staticReverse(float *d, int n);

__global__ void gpuSummationReduce(float *in, float *out, int n);

__global__ void gpuBackProjectKernel(float * d_hist, int * d_hueArray, int hueArrayLength , float * d_backproj, int width, int xOffset, int yOffset);

#endif
