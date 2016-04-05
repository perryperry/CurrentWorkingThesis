#include <stdio.h>
#include "gpuMerge.h"

__global__ void staticReverse(float *d, int n)
{
  __shared__ float s[60];

  int t = threadIdx.x;
  int tr = n-t-1;

  if(t < 60)
    s[t] = d[t];

  __syncthreads();

  if(t < 60)
    d[t] = s[tr];
}


__global__ void gpuBackProjectKernel(float * d_hist, int * d_hueArray, int hueArrayLength, float * d_M00, float * d_M1x, float * d_M1y, int width, int xOffset, int yOffset)
{
  
  __shared__ float sharedHistogram[60];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  int col = 0;
  int row = 0;
  float probability = 0;

  if(tid < 60)
  {
    sharedHistogram[tid] = d_hist[tid];
  }

  __syncthreads();

  if(i < hueArrayLength)
  {
      col = i % width;
      row = i / width;

      probability = sharedHistogram[ d_hueArray[i] / 3 ];

      d_M00[i] = probability; //for M00

      d_M1x[i] = ((float)(col + xOffset)) * probability; //for M1x

      d_M1y[i] = ((float)(row + yOffset)) * probability; //for M1y
  }

}//end kernel

__global__ void gpuSummationReduce(float * M00_in, float * M00_out, float * M1x_in, float * M1x_out, float * M1y_in, float * M1y_out, int n)
{
    __shared__ float shared_M00[64];
    __shared__ float shared_M1x[64];
    __shared__ float shared_M1y[64];

    // load shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    shared_M00[tid] = (i < n) ? M00_in[i] : 0;
    shared_M1x[tid] = (i < n) ? M1x_in[i] : 0;
    shared_M1y[tid] = (i < n) ? M1y_in[i] : 0;

    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if (( tid % ( 2 * s )) == 0)
        {
             //bigger number stored in low index
           shared_M00[tid] += shared_M00[tid + s];
           shared_M1x[tid] += shared_M1x[tid + s];
           shared_M1y[tid] += shared_M1y[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) 
    {
      M00_out[blockIdx.x] = shared_M00[0];
      M1x_out[blockIdx.x] = shared_M1x[0];
      M1y_out[blockIdx.x] = shared_M1y[0];
    }
}