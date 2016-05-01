#include "kernels.h"

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
/**********************************************************************************************************/

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


void setConstantMemoryHistogram(float * histogram, int num_objects)
{
    cudaMemcpyToSymbol(const_histogram, histogram, sizeof(float) * 60 * num_objects);
}

//************** WORKING KERNELS ********************//

//Kernel for meanshift of sub-window based on entire Frame

/*
Finding sub-matrix index in greater matrix:

( row_offset * absolute_matrix_width ) + col_offset + sub_col + ( absolute_matrix_width * sub_row ) == absolute_index

The reduce kernel you have has the calculation to get sub_col and sub_row based on sub_width and sub_absolute_index.
So given the thread id (i.e. the sub_absolute_index), you can get those values and plug them in to the above for the 
absolute index in the overall picture frame char *. 
*/

__global__ void gpuBlockReduce(int obj_id, unsigned char *g_idata, float *g_odata, int subframe_length, int blockCount, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset)
{
  __shared__ float shared_M00[1024];
  __shared__ float shared_M1x[1024];
  __shared__ float shared_M1y[1024];
  __shared__ int shared_hist_offset[1];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
  unsigned int sub_col = 0;
  unsigned int sub_row = 0;
  unsigned int absolute_index = 0;

  if(tid == 0){
        shared_hist_offset[0] =  obj_id * HIST_BUCKETS;
  }
  __syncthreads();
  if(i < subframe_length) 
  {
      
      sub_col = i % sub_width;
      sub_row = i / sub_width;

      absolute_index = (row_offset[obj_id] * abs_width) + col_offset[obj_id] + sub_col + (abs_width * sub_row);
      
      shared_M00[tid] = const_histogram[ (g_idata[absolute_index] / 3) + shared_hist_offset[0]];
      shared_M1x[tid] = ((float)(sub_col + col_offset[obj_id])) * shared_M00[tid];
      shared_M1y[tid] = ((float)(sub_row + row_offset[obj_id])) * shared_M00[tid];
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
__global__ void gpuFinalReduce(int obj_id, float * g_odata, int * cx, int * cy, int * row_offset, int * col_offset, int sub_width, int sub_height, int num_block)
{
    extern __shared__ float shared_sum[];
    //unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

    if(i < num_block){

      shared_sum[i] = g_odata[i]; //M00
      shared_sum[i + num_block] = g_odata[i + num_block]; //M1x
      shared_sum[i + num_block + num_block] = g_odata[i + num_block + num_block];//M1y

    }
    __syncthreads();

    if(i == 0)
    {
      int ind = 0;
      double M00 = 0.0;
      double M1x = 0.0;
      double M1y = 0.0;
      int newX = 0;
      int newY = 0;

      for(ind = 0; ind < num_block; ind ++)
      {
          M00 += shared_sum[ind];
          M1x += shared_sum[ind + num_block];
          M1y += shared_sum[ind + num_block + num_block];
      }
      newX = (int) ((int)M1x / (int)M00);
      newY = (int) ((int)M1y / (int)M00);

      col_offset[obj_id] = newX - (sub_width / 2);
      row_offset[obj_id] = newY - (sub_height / 2);

      if(col_offset[obj_id] < 0) col_offset[obj_id] = 0;
      if(row_offset[obj_id] < 0) row_offset[obj_id] = 0;

      cx[obj_id] = newX;
      cy[obj_id] = newY;
    // printf("\nIn gpuFinalReduce: M00:%lf M1x:%lf M1y: %lf, NewX: %d NewY: %d \n", M00, M1x, M1y, newX, newY);
    }
}


/************************************************************* MULTI-Object concurrent tracking kernels ***************************************************/

__global__ void gpuMultiObjectBlockReduce(int * d_obj_block_ends, int num_objects, unsigned char *g_idata, float *g_odata, int * subframe_length, int num_block, int abs_width, int * sub_widths, int * row_offset, int * col_offset)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 

  //*********************************************************************************/
  __shared__ float shared_M00[1024];
  __shared__ float shared_M1x[1024];
  __shared__ float shared_M1y[1024];
  __shared__ int shared_hist_offset[1];
  __shared__ int shared_obj_id[1];
  __shared__ int shared_thread_offset[1];

 // __shared__ int shared_block_start[1];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int sub_col = 0;
  unsigned int sub_row = 0;
  unsigned int absolute_index = 0;

  if(tid == 0)
  {
    shared_obj_id[0] = gpuCalcObjID(d_obj_block_ends, num_objects);
    shared_hist_offset[0] =  shared_obj_id[0] * HIST_BUCKETS;
    if(shared_obj_id[0] != 0 )
      shared_thread_offset[0] = d_obj_block_ends[shared_obj_id[0] - 1] * 1024;
    else
      shared_thread_offset[0] = 0;
  }
  __syncthreads();

    i -= shared_thread_offset[0];
  
    if(i < subframe_length[shared_obj_id[0]]) 
    {
        sub_col = i % sub_widths[shared_obj_id[0]];
        sub_row = i / sub_widths[shared_obj_id[0]];

        absolute_index = (row_offset[ shared_obj_id[0] ] * abs_width) + col_offset[ shared_obj_id[0] ] + sub_col + (abs_width * sub_row);
        
        shared_M00[tid] = const_histogram[ (g_idata[absolute_index] / 3) + shared_hist_offset[0]];
        shared_M1x[tid] = ((float)(sub_col + col_offset[shared_obj_id[0]])) * shared_M00[tid];
        shared_M1y[tid] = ((float)(sub_row + row_offset[shared_obj_id[0]])) * shared_M00[tid];
    }
    else 
    {
        shared_M00[tid] = 0;
        shared_M1x[tid] = 0;
        shared_M1y[tid] = 0;
    }


    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) 
    { 
      if (tid < stride)
      {
        shared_M00[tid] += shared_M00[tid + stride]; 
        shared_M1x[tid] += shared_M1x[tid + stride]; 
        shared_M1y[tid] += shared_M1y[tid + stride]; 
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
      g_odata[blockIdx.x + num_block] = shared_M1x[0]; 
      g_odata[blockIdx.x + (2 * num_block)] = shared_M1y[0]; 
    }
}


__device__ int gpuBlockStart(int * d_obj_block_ends, int num_objects)
{
  int block = blockIdx.x;
  int prevBlockEnd = 0;
  int i = 0;
  for(i = 0; i < num_objects; i ++)
  {
    if(block >= prevBlockEnd && block < d_obj_block_ends[i])
      return prevBlockEnd;
    prevBlockEnd = d_obj_block_ends[i];
  }
  return prevBlockEnd;
}

__global__ void gpuMultiObjectFinalReduce(int * d_obj_block_ends, int num_objects, float *g_odata, int * cx, int * cy, int * row_offset, int * col_offset, int * sub_widths, int * sub_heights, int num_block)
{
    extern __shared__ float shared_sum[];
   __shared__ int shared_block_dim[2]; //0 index == starting block, 1 index == end block
    unsigned int tid = threadIdx.x; 
    
     if(tid == 0){ //Calculate the starting block 
       shared_block_dim[0] = (blockIdx.x > 0) ? d_obj_block_ends[blockIdx.x - 1]: 0;
       shared_block_dim[1] = d_obj_block_ends[blockIdx.x];
       // printf("*****Final reduce: Block %d, the shared obj id: %d, the shared block start: %d\n", blockIdx.x, blockIdx.x, shared_block_start[0]);
     }
     __syncthreads();
    
    if(tid >= shared_block_dim[0] && tid < shared_block_dim[1])
    {
      shared_sum[tid] = g_odata[tid]; //M00
      shared_sum[tid + num_block] = g_odata[tid + num_block]; //M1x
      shared_sum[tid + num_block + num_block] = g_odata[tid + num_block + num_block];//M1y
    }
    else
   {
     shared_sum[tid] = 0.0; //M00
     shared_sum[tid + num_block] = 0.0; //M1x
     shared_sum[tid + num_block + num_block] = 0.0;//M1y
   }
    __syncthreads();

    if(tid == 0)
    {
      int ind = 0;
      double M00 = 0.0;
      double M1x = 0.0;
      double M1y = 0.0;
      int newX = 0;
      int newY = 0;
      int newColOffset = 0;
      int newRowOffset = 0;

      for(ind = 0; ind < num_block; ind ++)
      {
          M00 += shared_sum[ind];
          M1x += shared_sum[ind + num_block];
          M1y += shared_sum[ind + num_block + num_block];
      }
      newX = (int) ((int)M1x / (int)M00);
      newY = (int) ((int)M1y / (int)M00);

      newColOffset =  newX - (sub_widths[blockIdx.x] / 2);
      newRowOffset = newY - (sub_heights[blockIdx.x] / 2);

      col_offset[blockIdx.x] = (newColOffset > 0) ? newColOffset : 0;
      row_offset[blockIdx.x] = (newRowOffset > 0) ? newRowOffset : 0;

      cx[blockIdx.x] = newX;
      cy[blockIdx.x] = newY;
  //  printf("\nIn gpuFinalReduce Block %d: M00:%lf M1x:%lf M1y: %lf, NewX: %d NewY: %d \n", blockIdx.x, M00, M1x, M1y, newX, newY);
  }
}

//Based on block id, determine if entire block of threads belong to calculation for a given object
//Threads in the block don't need to be aware of the block boundaries specifically
__device__ int gpuCalcObjID(int * d_obj_block_ends, int num_objects)
{
  int block = blockIdx.x;
  int prevBlockEnd = 0;
  int i = 0;
  for(i = 0; i < num_objects; i ++)
  {
    if(block >= prevBlockEnd && block < d_obj_block_ends[i])
      return i;
    prevBlockEnd = d_obj_block_ends[i];
  }
  return num_objects - 1;
}









































































































































































































/*********************** TESTING KERNEL ***********************************/
//Seeing if we can do the entire convergence within a single kernel

__global__ void gpuSingleKernelMeanShift(unsigned char *g_idata, float *g_odata, int * readyArray, int subframe_length, int blockCount, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset, int * cxy)
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
       //warpReduceSingleMatrix(shared_M00, tid);
    }
     __syncthreads();
    // write result for this block to global mem
    if (tid == 0) {
      g_odata[blockIdx.x] = shared_M00[0]; 
      g_odata[blockIdx.x + blockCount] = shared_M1x[0]; 
      g_odata[blockIdx.x + (2 * blockCount)] = shared_M1y[0]; 

      readyArray[blockIdx.x] = 1;
    }
    __syncthreads();

    if( blockIdx.x == 0 && tid < blockCount ) // summation of global out across blocks
    {
      int index = 0;
      int M1yOffset = 2 * blockCount;

    //  while(atomicAdd(&readyArray[tid], 0) == 0);

      while( atomicCAS(&readyArray[tid], 1, 0) == 0);

      shared_M00[tid] = g_odata[tid];
      shared_M1x[tid] = g_odata[tid + blockCount];
      shared_M1y[tid] = g_odata[tid + M1yOffset];

      __syncthreads(); 

      if(tid == 0)
      {
        float M00 = 0.0;
        float M1x = 0.0;
        float M1y = 0.0;

        int newX = 0;
        int newY = 0;

        for(index = 0; index < blockCount; index ++)
        {
          M00 += shared_M00[index];
          M1x += shared_M1x[index];
          M1y += shared_M1y[index];
        }

       // g_odata[0] = M00;
        //g_odata[blockCount] = M1x;
        //g_odata[M1yOffset] = M1y;
        newX = M1x / M00;
        newY = M1y / M00;

        cxy[0] = newX;
        cxy[1] = newY;
        col_offset[0] = newX - (sub_width / 2);
        row_offset[0] = newY - (sub_height / 2);
        if(col_offset[0] < 0) col_offset[0] = 0;
        if(row_offset[0] < 0) row_offset[0] = 0;
        
      // printf("Inside GPU MeanShift ---> M00 = %f M1x = %f M1y = %f\n", M00, M1x, M1y);
     //  printf("Inside GPU MeanShift ---> centroid (%d, %d)  topX, topY (%d,%d)\n", newX, newY, col_offset[0], row_offset[0]  );
      }
}
}
