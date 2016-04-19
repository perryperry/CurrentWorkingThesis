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


__global__ void gpuBackProjectKernel(float * d_hist, unsigned char * d_hueArray, int hueArrayLength, float * d_M00, float * d_M1x, float * d_M1y, int width, int xOffset, int yOffset)
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
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (( tid % ( 2 * stride )) == 0)
        {
             //bigger number stored in low index
           shared_M00[tid] += shared_M00[tid + stride ];
           shared_M1x[tid] += shared_M1x[tid + stride ];
           shared_M1y[tid] += shared_M1y[tid + stride ];
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



__global__ void bpTestKernel(unsigned char * d_hueArray, int * d_converted, int hueArrayLength)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  __syncthreads();

  if(i < hueArrayLength)
  {
    d_converted[i] = (int) d_hueArray[i];
  }

}//end kernel







/**********************************************************************************************************/

//New improved kernels below



__global__ void gpuMeanShiftKernelForSubFrame(unsigned char *g_idata, float *g_odata, int * readyArray, int input_length, int blockCount, int width, int xOffset, int yOffset)
{
  __shared__ float shared_M00[1024];
  __shared__ float shared_M1x[1024];
  __shared__ float shared_M1y[1024];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
  unsigned int col = 0;
  unsigned int row = 0;

  shared_M00[tid] = (i < input_length) ? const_histogram[ g_idata[i] / 3 ] : 0;

  if(i < input_length)
  {
      col = i % width;
      row = i / width;

      shared_M1x[tid] = ((float)(col + xOffset)) * shared_M00[tid]; //const_histogram[ g_idata[i] / 3 ];//
      shared_M1y[tid] = ((float)(row + yOffset)) * shared_M00[tid]; //const_histogram[ g_idata[i]/ 3 ];//
  }
  else
  {
       shared_M1x[tid] = 0;
       shared_M1y[tid] = 0;
  }

    __syncthreads();

    for (unsigned int s=blockDim.x/2; s > 32; s >>= 1) 
    { 
      if (tid < s)
      {
        shared_M00[tid] += shared_M00[tid + s]; 
        shared_M1x[tid] += shared_M1x[tid + s]; 
        shared_M1y[tid] += shared_M1y[tid + s]; 
      }
      __syncthreads(); 
    }

    if(tid < 32){
      /*warpReduce(shared_M00, tid);
      warpReduce(shared_M1x, tid);
      warpReduce(shared_M1y, tid);*/
       warpReduce(shared_M00, shared_M1x, shared_M1y, tid);
    }

    // write result for this block to global mem
    if (tid == 0) {
      g_odata[blockIdx.x] = shared_M00[0]; 
      g_odata[blockIdx.x + blockCount] = shared_M1x[0]; 
      g_odata[blockIdx.x + (2 * blockCount)] = shared_M1y[0]; 


      readyArray[blockIdx.x] = 1;
    }

    if( blockIdx.x == 0 && tid < blockCount ) // summation of global out across blocks
    {
      int index = 0;
      int M1yOffset = 2 * blockCount;

      while(atomicAdd(&readyArray[tid], 0) == 0);

      shared_M00[tid] = g_odata[tid];
      shared_M1x[tid] = g_odata[tid + blockCount];
      shared_M1y[tid] = g_odata[tid + M1yOffset];

      __syncthreads(); 

      if(tid == 0)
      {
        float M00 = 0.0;
        float M1x = 0.0;
        float M1y = 0.0;

        for(index = 0; index < blockCount; index ++)
        {
          M00 += shared_M00[index];
          M1x += shared_M1x[index];
          M1y += shared_M1y[index];
        }

        g_odata[0] = M00;
        g_odata[blockCount] = M1x;
        g_odata[M1yOffset] = M1y;
      }
    }
}

__device__ void warpReduceSingleMatrix(volatile float* sdata, int tid) 
{ 
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16]; 
  sdata[tid] += sdata[tid + 8]; 
  sdata[tid] += sdata[tid + 4]; 
  sdata[tid] += sdata[tid + 2]; 
  sdata[tid] += sdata[tid + 1];
}

__device__ void warpReduce(volatile float* shared_M00, volatile float* shared_M1x, volatile float* shared_M1y, int tid) 
{ 
  shared_M00[tid] += shared_M00[tid + 32];
  shared_M00[tid] += shared_M00[tid + 16]; 
  shared_M00[tid] += shared_M00[tid + 8]; 
  shared_M00[tid] += shared_M00[tid + 4]; 
  shared_M00[tid] += shared_M00[tid + 2]; 
  shared_M00[tid] += shared_M00[tid + 1];

  shared_M1x[tid] += shared_M1x[tid + 32];
  shared_M1x[tid] += shared_M1x[tid + 16]; 
  shared_M1x[tid] += shared_M1x[tid + 8]; 
  shared_M1x[tid] += shared_M1x[tid + 4]; 
  shared_M1x[tid] += shared_M1x[tid + 2]; 
  shared_M1x[tid] += shared_M1x[tid + 1];

  shared_M1y[tid] += shared_M1y[tid + 32];
  shared_M1y[tid] += shared_M1y[tid + 16]; 
  shared_M1y[tid] += shared_M1y[tid + 8]; 
  shared_M1y[tid] += shared_M1y[tid + 4]; 
  shared_M1y[tid] += shared_M1y[tid + 2]; 
  shared_M1y[tid] += shared_M1y[tid + 1];
}


void setConstantMemoryHistogram(float * histogram)
{
    cudaDeviceReset();
    cudaMemcpyToSymbol(const_histogram, histogram, sizeof(float) * 60);
}



//Kernel for meanshift of sub-window based on entire Frame

/*
Finding sub-matrix index in greater matrix:

( row_offset * absolute_matrix_width ) + col_offset + sub_col + ( absolute_matrix_width * sub_row ) == absolute_index

The reduce kernel you have has the calculation to get sub_col and sub_row based on sub_width and sub_absolute_index.
So given the thread id (i.e. the sub_absolute_index), you can get those values and plug them in to the above for the 
absolute index in the overall picture frame char *. 
*/

__global__ void gpuMeanShiftKernelForEntireFrame(unsigned char *g_idata, float *g_odata, int subframe_length, int blockCount, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset, int * cxy)
{
  __shared__ float shared_M00[1024];
  __shared__ float shared_M1x[1024];
  __shared__ float shared_M1y[1024];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
  unsigned int sub_col = 0;
  unsigned int sub_row = 0;
  unsigned int absolute_index = 0;

  if(i < subframe_length) 
  {
      sub_col = i % sub_width;
      sub_row = i / sub_width;

      absolute_index = (row_offset[0] * abs_width) + col_offset[0] + sub_col + (abs_width * sub_row);
      
      shared_M00[tid] = const_histogram[ g_idata[absolute_index] / 3 ];
      shared_M1x[tid] = ((float)(sub_col + col_offset[0])) * shared_M00[tid];
      shared_M1y[tid] = ((float)(sub_row + row_offset[0])) * shared_M00[tid];
  }
  else
  {
      shared_M00[tid] = 0;
      shared_M1x[tid] = 0;
      shared_M1y[tid] = 0;
  }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) 
    { 
      if (tid < s)
      {
        shared_M00[tid] += shared_M00[tid + s]; 
        shared_M1x[tid] += shared_M1x[tid + s]; 
        shared_M1y[tid] += shared_M1y[tid + s]; 
      }
      __syncthreads(); 
    }

    if(tid < 32){
       warpReduce(shared_M00, shared_M1x, shared_M1y, tid);
    }
     __syncthreads();
    // write result for this block to global mem
    if (tid == 0) {
      g_odata[blockIdx.x] = shared_M00[0]; 
      g_odata[blockIdx.x + blockCount] = shared_M1x[0]; 
      g_odata[blockIdx.x + (2 * blockCount)] = shared_M1y[0]; 
    }
}


__global__ void gpuFinalReduce(float * g_odata, int * cxy, int * row_offset, int * col_offset, int sub_width, int sub_height, int num_block)
{
    extern __shared__ float shared_sum[];
    //unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

    if(i < num_block){

      shared_sum[i] = g_odata[i];
      shared_sum[i + num_block] = g_odata[i + num_block];
      shared_sum[i + num_block + num_block] = g_odata[i + num_block + num_block];

    }
    __syncthreads();

    if(i == 0)
    {
      int ind = 0;
      float M00 = 0.0;
      float M1x = 0.0;
      float M1y = 0.0;
      int newX = 0;
      int newY = 0;

      for(ind = 0; ind < num_block; ind ++)
      {
          M00 += shared_sum[ind];
          M1x += shared_sum[ind + num_block];
          M1y += shared_sum[ind + num_block + num_block];
      }
      newX = M1x / M00;
      newY = M1y / M00;

      col_offset[0] = newX - (sub_width / 2);
      row_offset[0] = newY - (sub_height / 2);
      if(col_offset[0] < 0) col_offset[0] = 0;
      if(row_offset[0] < 0) row_offset[0] = 0;

      cxy[0] = newX;
      cxy[1] = newY;
      //printf("\n\n\n\nMOO in gpuFinalReduce: M00:%f M1x:%f M1y: %f\n", M00, M1x, M1y);
    }
}





