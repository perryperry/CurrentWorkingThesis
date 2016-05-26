#include "dynamicCamShift.h"

__host__ void setConstantMemoryHistogram(float * histogram)
{
    cudaMemcpyToSymbol(const_histogram, histogram, sizeof(float) * HIST_BUCKETS * MAX_OBJECTS);
}

/********************************************************/

__device__ unsigned int gpuDistance(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    unsigned int distx = (x2 - x1) * (x2 - x1);
    unsigned int disty = (y2 - y1) * (y2 - y1);
    
    double dist = sqrt((float)(distx + disty));
    
    return (unsigned int) dist;
}

//https://gist.github.com/yoggy/8999625
//http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
__global__ void gpuBGRtoHue(unsigned char * bgr, unsigned char * hueArray, int total)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 

	unsigned int bgrIndex = 3 * i;

  if( i < total){

	float b, g, r; 

	b = (float) bgr[bgrIndex    ]  / 255.0f;
	g = (float) bgr[bgrIndex + 1] / 255.0f;
	r = (float) bgr[bgrIndex + 2] / 255.0f;

  float hue; // h:0-360.0, s:0.0-1.0, v:0.0-1.0
    
   float max = fmaxf(r, fmaxf(g, b));
  float min = fminf(r, fminf(g, b));
    //v = max;

    if (max == 0.0f) {
        hue = 0;
        //s = 0;
    }
    else if (max - min == 0.0f) {
        hue = 0;
        //s = 0;
    }
    else { 
       //s = (max - min) / max;
        if (max == r) {
            hue = 60 * ((g - b) / (max - min)) + 0;
        }
        else if (max == g) {
            hue = 60 * ((b - r) / (max - min)) + 120;
        }
        else {
            hue = 60 * ((r - g) / (max - min)) + 240;
        }
    }
    
    if (hue < 0) hue += 360.0f;
    
   hueArray[i] = (unsigned char)(hue / 2);   // Hue --> 0-180

 }
}

/********************************************************/

__device__ bool converged(unsigned int num_objects)
{
  unsigned int obj_cur;
  unsigned int total = 0;
  for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
  {
    if(g_converged[obj_cur]) //object has not converged yet
      total ++;
  }
  if(total == num_objects) //All objects have finished converging
    return true;
  else
    return false;
}

/********************************************************/
//Based on block id, determine if entire block of threads belong to calculation for a given object
//Threads in the block don't need to be aware of the block boundaries specifically
__device__ unsigned int getObjID(unsigned int * block_ends, unsigned int num_objects)
{
  unsigned int block = blockIdx.x;
  unsigned int prevBlockEnd = 0;
  unsigned int i = 0;
  for(i = 0; i < num_objects; i ++)
  {
    if(block >= prevBlockEnd && block < block_ends[i])
      return i;
    prevBlockEnd = block_ends[i];
  }
  return num_objects - 1;
}

/********************************************************/

__device__ unsigned int getBlockStart(unsigned int * block_ends, unsigned int num_objects)
{
  unsigned int block = blockIdx.x;
  unsigned int prevBlockEnd = 0;
  unsigned int i = 0;
  for(i = 0; i < num_objects; i ++)
  {
    if(block >= prevBlockEnd && block < block_ends[i])
      return prevBlockEnd;
    prevBlockEnd = block_ends[i];
  }
  return prevBlockEnd;
}

/********************************************************/

__global__ void dynamicCamShiftMain(
unsigned int num_objects,
unsigned char * frame,
unsigned int frame_total,
unsigned int frame_width,
unsigned int * block_ends,
unsigned int * cx,
unsigned int * cy,
unsigned int * prevX,
unsigned int * prevY,
unsigned int * row_offset,
unsigned int * col_offset,
unsigned int * sub_widths,
unsigned int * sub_heights,
unsigned int * sub_totals,
bool adjust_window)                  
{
    unsigned int num_block = 0;
    unsigned int obj_cur = 0; 
   
    for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
    {

       //printf("INSIDE OF KERNEL (%d) --> cx: %d cy: %d\n", obj_cur, cx[obj_cur], cy[obj_cur]);
      prevX[obj_cur] = 0; 
      prevY[obj_cur] = 0;
      g_converged[obj_cur] = false;
      //g_sub_totals[obj_cur] = sub_widths[obj_cur] * sub_heights[obj_cur];
      num_block += ceil( (float) sub_totals[obj_cur] / (float) TILE_WIDTH);
      block_ends[obj_cur] = num_block;
    }

    dim3 block(TILE_WIDTH, 1, 1);
    dim3 grid(num_block, 1, 1);

    if(num_block <= TILE_WIDTH)
    {
      while( ! converged(num_objects) )
      {
          blockReduce<<<grid, block>>>(
          num_objects, 
          num_block,
          frame, 
          frame_width, 
          sub_widths,
          sub_totals,
          block_ends, 
          row_offset,
          col_offset);

          cudaDeviceSynchronize();
         
          finalReduce<<< num_objects, num_block, num_block * 3 * sizeof(float) >>>( 
          cx, 
          cy,
          prevX,
          prevY, 
          sub_widths, 
          sub_heights,
          sub_totals,
          row_offset,
          col_offset,
          block_ends,
          num_block,
          adjust_window);

         // 
          
          if(adjust_window)
          {
            cudaDeviceSynchronize();
            num_block = 0;
            for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
            {
                num_block += ceil( (float) sub_totals[obj_cur] / (float)TILE_WIDTH);
                block_ends[obj_cur] = num_block;
            }
            grid = dim3(num_block, 1, 1);
         }
      }
    }
    else
    {
       printf("Error: invalid number of blocks: %d > %d\n", num_block, TILE_WIDTH);
    }
}


/********************************************************/

__device__ void warpReduce(
volatile float* shared_M00, 
volatile float* shared_M1x, 
volatile float* shared_M1y, 
unsigned int tid) 
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

/********************************************************/

__global__ void blockReduce(
unsigned int num_objects,
unsigned int num_block, 
unsigned char * frame,  
unsigned int frame_width, 
unsigned int * sub_widths,
unsigned int * sub_totals,
unsigned int * block_ends,
unsigned int * row_offset,
unsigned int * col_offset)
{
  __shared__ float shared_M00[1024];
  __shared__ float shared_M1x[1024];
  __shared__ float shared_M1y[1024];
  __shared__ unsigned int shared_hist_offset[1];
  __shared__ unsigned int shared_obj_id[1];
  __shared__ unsigned int shared_thread_offset[1];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
  unsigned int tid = threadIdx.x;
  unsigned int sub_col = 0;
  unsigned int sub_row = 0;
  unsigned int absolute_index = 0;

  if(tid == 0)
  {
    shared_obj_id[0] = getObjID(block_ends, num_objects);
    shared_hist_offset[0] =  shared_obj_id[0] * HIST_BUCKETS;

    if(shared_obj_id[0] != 0 )
      shared_thread_offset[0] = block_ends[shared_obj_id[0] - 1] * 1024;
    else
      shared_thread_offset[0] = 0;
  }
  __syncthreads();

  if(  g_converged[shared_obj_id[0]] )
    return;

  i -= shared_thread_offset[0];

  if(i < sub_totals[shared_obj_id[0]]) 
  {
      sub_col = i % sub_widths[shared_obj_id[0]];
      sub_row = i / sub_widths[shared_obj_id[0]];

      absolute_index = (row_offset[ shared_obj_id[0] ] * frame_width) + col_offset[ shared_obj_id[0] ] + sub_col + (frame_width * sub_row);
      
      shared_M00[tid] = const_histogram[ (frame[absolute_index] / 3) + shared_hist_offset[0]];
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
    g_output[blockIdx.x] = shared_M00[0]; 
    g_output[blockIdx.x + num_block] = shared_M1x[0]; 
    g_output[blockIdx.x + (2 * num_block)] = shared_M1y[0]; 
  }
} 

/********************************************************/

__global__ void finalReduce(
unsigned int * cx, 
unsigned int * cy,
unsigned int * prevX,
unsigned int * prevY,  
unsigned int * sub_widths, 
unsigned int * sub_heights,
unsigned int * sub_totals,
unsigned int * row_offset,
unsigned int * col_offset,
unsigned int * block_ends,
int num_block,
bool adapt_window)
{
    extern __shared__ float shared_sum[];

   __shared__ int shared_block_dim[2]; //0 index == starting block, 1 index == end block
    unsigned int tid = threadIdx.x; 

    if( g_converged[blockIdx.x] )
      return;
    
     if(tid == 0){ //Calculate the starting block 
       shared_block_dim[0] = (blockIdx.x > 0) ? block_ends[blockIdx.x - 1]: 0;
       shared_block_dim[1] = block_ends[blockIdx.x];
     }
     __syncthreads();
    
    if(tid >= shared_block_dim[0] && tid < shared_block_dim[1])
    {
      shared_sum[tid] = g_output[tid]; //M00
      shared_sum[tid + num_block] = g_output[tid + num_block]; //M1x
      shared_sum[tid + num_block + num_block] = g_output[tid + num_block + num_block];//M1y
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

     // for(ind = shared_block_dim[0]; ind < shared_block_dim[1]; ind ++)
      for(ind = 0; ind < num_block; ind ++)
      {
          M00 += shared_sum[ind];
          M1x += shared_sum[ind + num_block];
          M1y += shared_sum[ind + num_block + num_block];
      }
      newX = (int) ((int) M1x / (int) M00);
      newY = (int) ((int) M1y / (int) M00);

      cx[blockIdx.x] = newX;
      cy[blockIdx.x] = newY;

      newColOffset =  newX - (sub_widths[blockIdx.x] / 2);
      newRowOffset = newY - (sub_heights[blockIdx.x] / 2);

      if( gpuDistance(newX, newY, prevX[blockIdx.x], prevY[blockIdx.x]) <= 1 )
      {
        g_converged[blockIdx.x] = true;
        return;
      }
      else
      {
          prevX[blockIdx.x] = cx[blockIdx.x];
          prevY[blockIdx.x] = cy[blockIdx.x];
      }

      if( adapt_window )
      {
          adaptWindow(M00, newX, newY, &newRowOffset, &newColOffset, sub_widths, sub_heights, sub_totals, blockIdx.x);
      }

      col_offset[blockIdx.x] = (newColOffset > 0) ? newColOffset : 0;
      row_offset[blockIdx.x] = (newRowOffset > 0) ? newRowOffset : 0;
  }  
}

/********************************************************/

__device__ void adaptWindow(
double M00, 
int newX,
int newY,
int * newRowOffset, 
int * newColOffset, 
unsigned int * sub_widths, 
unsigned int * sub_heights, 
unsigned int * sub_totals,
int objectID)
{
	int width = ceil(2.0 * sqrt(M00));
	if(width < 20){
		width = 200;
	}
	int height = ceil(width * 1.1);
	*newColOffset =  newX - (width / 2); // x
	*newRowOffset = newY - (height / 2); // y
	int bottomRightX = *newColOffset + width;
	int bottomRightY = *newRowOffset + height;
	if(bottomRightX > FRAME_WIDTH - 1)
	{
	  width = FRAME_WIDTH - *newColOffset - 1;
	}
	if(bottomRightY > FRAME_HEIGHT - 1)
	{
	  height = FRAME_HEIGHT - *newRowOffset - 1;
	}
	sub_widths [objectID] = width;
	sub_heights[objectID] = height;
	sub_totals [objectID] = width * height;
}
