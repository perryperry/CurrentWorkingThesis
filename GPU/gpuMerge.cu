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


__global__ void gpuBackProjectKernel(float * d_hist, int * d_hueArray, float * d_backproj, int hueLength)
{
  
  __shared__ float sharedHistogram[60];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < 60)
  {
    sharedHistogram[tid] = d_hist[tid];
  }

  __syncthreads();

  if(i < hueLength)
  {
    //printf("t == %d and its hue == %d\n", t, d_hueArray[tid]);
      d_backproj[i] = sharedHistogram[ d_hueArray[i] / 3 ];
  }

}//end kernel


__global__ void gpuSummationReduce(float *in, float *out, int n)
{
    extern __shared__ float sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if (( tid % ( 2 * s )) == 0)
        {
             //bigger number stored in low index
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) out[blockIdx.x] = sdata[0];
}