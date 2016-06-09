#include "dynamicCamShift.h"

//##################################################################
// Set the search window to the size of the entire frame.
// Used to measure the timecost of processing the entire frame.
// Future use 
//##################################################################
__device__ unsigned int setWindowToEntireFrame(
  unsigned int obj, 
  unsigned int ** topX, 
  unsigned int ** topY,
  unsigned int ** bottomX, 
  unsigned int ** bottomY, 
  unsigned int ** width, 
  unsigned int ** height, 
  unsigned int ** length)
{
    (*topX)[obj]       = 0;
    (*topY)[obj]       = 0;
    (*bottomX)[obj]    = FRAME_WIDTH - 1;
    (*bottomY)[obj]    = FRAME_HEIGHT - 1;
    (*width)[obj]      = FRAME_WIDTH - 1;
    (*height)[obj]     = FRAME_HEIGHT - 1;
    (*length)[obj]     = (FRAME_WIDTH - 1) * (FRAME_HEIGHT - 1);
    return FRAME_TOTAL;
}

//##################################################################
// Rounds the number to the next highest power of 2 
// For the final reduction of the blocks' results
//##################################################################
__device__ unsigned long roundToPow2(unsigned long v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

//##################################################################
// sets the static constant memory histogram
//##################################################################
__host__ void setConstantMemoryHistogram(float * histogram)
{
    cudaMemcpyToSymbol(const_histogram, histogram, sizeof(float) * HIST_BUCKETS * MAX_OBJECTS);
}

//##################################################################
// Calculates the distance between points
//##################################################################
__device__ unsigned int gpuDistance(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    unsigned int distx = (x2 - x1) * (x2 - x1);
    unsigned int disty = (y2 - y1) * (y2 - y1);
    
    double dist = sqrt((float)(distx + disty));
    
    return (unsigned int) dist;
}

//##################################################################
//https://gist.github.com/yoggy/8999625
//http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
//##################################################################
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

//##################################################################
// Unfolded reduction outside of the loop,
// because it does not need the __syncThreads(), the threads in the
// warp execute at the same time and are strided avoiding bank issue.
//##################################################################
__device__ void warpReduce(
volatile float* shared_M00, 
volatile float* shared_M1x, 
volatile float* shared_M1y, 
unsigned int tid,
unsigned int blocksize) 
{ 
    if (blocksize >= 64) shared_M00[tid] += shared_M00[tid + 32];
    if (blocksize >= 32) shared_M00[tid] += shared_M00[tid + 16]; 
    if (blocksize >= 16) shared_M00[tid] += shared_M00[tid + 8]; 
    if (blocksize >= 8)  shared_M00[tid] += shared_M00[tid + 4]; 
    if (blocksize >= 4)  shared_M00[tid] += shared_M00[tid + 2]; 
    if (blocksize >= 2)  shared_M00[tid] += shared_M00[tid + 1];

    if (blocksize >= 64) shared_M1x[tid] += shared_M1x[tid + 32];
    if (blocksize >= 32) shared_M1x[tid] += shared_M1x[tid + 16]; 
    if (blocksize >= 16) shared_M1x[tid] += shared_M1x[tid + 8]; 
    if (blocksize >= 8)  shared_M1x[tid] += shared_M1x[tid + 4]; 
    if (blocksize >= 4)  shared_M1x[tid] += shared_M1x[tid + 2]; 
    if (blocksize >= 2)  shared_M1x[tid] += shared_M1x[tid + 1];

    if (blocksize >= 64) shared_M1y[tid] += shared_M1y[tid + 32];
    if (blocksize >= 32) shared_M1y[tid] += shared_M1y[tid + 16]; 
    if (blocksize >= 16) shared_M1y[tid] += shared_M1y[tid + 8]; 
    if (blocksize >= 8)  shared_M1y[tid] += shared_M1y[tid + 4]; 
    if (blocksize >= 4)  shared_M1y[tid] += shared_M1y[tid + 2]; 
    if (blocksize >= 2)  shared_M1y[tid] += shared_M1y[tid + 1];
}


//##################################################################
// Either resizes the window based on the first moment
// and/or checks for corners out of bounds, 
// shrinks the window size, if out of bounds. 
//##################################################################
__device__ void adjustWindow(
bool shouldAdjust,
double M00, 
int newCX,
int newCY,
int * newTopX, 
int * newTopY, 
int * newBottomX, 
int * newBottomY,
unsigned int * sub_widths, 
unsigned int * sub_heights, 
unsigned int * sub_totals,
int obj_cur)
{
  int width = sub_widths[obj_cur], height = sub_heights[obj_cur];

  if(shouldAdjust)
  {
  	 width = ceil(2.0 * sqrt(M00));
     if(width < 20)
      width = 200;
     height = ceil(width * 1.1);
  }
  
  *newTopX =  newCX - (width / 2); // x
  *newTopY = newCY - (height / 2); // y

  if(*newTopX < 0)
  {
      *newTopX = 0;
      //width = *newBottomX;
  }
  if(*newTopY < 0)
  {
      *newTopY = 0;
      //height = *newBottomY;
  }

  *newBottomX = *newTopX + width;
  *newBottomY = *newTopY + height;

  if(*newBottomX > FRAME_WIDTH - 1)
  {
    width = FRAME_WIDTH - *newTopX - 1;
    *newBottomX = FRAME_WIDTH - 1;
  }

  if(*newBottomY > FRAME_HEIGHT - 1)
  {

    //printf("%s --> %d \n", "BEFORE", sub_heights[obj_cur]);
    height = FRAME_HEIGHT - *newTopY - 1;
     *newBottomY = FRAME_HEIGHT - 1;
   // printf("%s --> %d \n", "THIS HAPPEN?", height);
  }

  sub_widths [obj_cur] = width;
  sub_heights[obj_cur] = height;
  sub_totals [obj_cur] = width * height;
}

__global__ void dynamicCamShiftMain(
unsigned char * frame,
unsigned int frame_total,
unsigned int * cx,
unsigned int * cy,
unsigned int * topX,
unsigned int * topY,
unsigned int * bottomX,
unsigned int * bottomY,
unsigned int * sub_widths,
unsigned int * sub_heights,
unsigned int * sub_totals,
bool adjust_window)                  
{
    unsigned int obj_cur = threadIdx.x, prevCX = 0, prevCY; 
    int newTopX = 0, newTopY = 0, newBottomX = 0, newBottomY = 0;
    //printf("Row: %d Col: %d Width: %d Hieght: %d Total: %d\n", topY[obj_cur], topX[obj_cur], sub_widths[obj_cur], sub_heights[obj_cur], sub_totals[obj_cur]);
    unsigned int count = 0;
    unsigned int num_block_roundedup = 0; //num_block rounded to nearest pow of 2 for final reduce
    unsigned int num_block = ceil( (float) sub_totals[obj_cur] / (float) TILE_WIDTH);
    float * output = (obj_cur == 0) ? g_output1 : g_output2;
   
    dim3 block(TILE_WIDTH, 1, 1);
    dim3 grid(num_block, 1, 1);

    if(num_block <= TILE_WIDTH)
    {
     while( 1 )
    {
          prevCX = cx[obj_cur]; 
          prevCY = cy[obj_cur];

          blockReduce<<<grid, block>>>(
          obj_cur, 
          num_block,
          frame, 
          sub_widths,
          sub_totals, 
          topY,
          topX,
          output);

          num_block_roundedup = roundToPow2(num_block);

          cudaDeviceSynchronize();
         
          finalReduce<<< 1, num_block_roundedup>>>( output, num_block );

          cudaDeviceSynchronize();

          float M00 = 0.0f, M1x = 0.0f, M1y = 0.0f;

          M00 = output[0];
          M1x = output[1];
          M1y = output[2];

          unsigned int newCX = (int) (M1x /  M00);
          unsigned int newCY = (int) (M1y /  M00);
          cx[obj_cur] = newCX;
          cy[obj_cur] = newCY;

          adjustWindow(adjust_window, M00, newCX, newCY, &newTopX, &newTopY, &newBottomX, &newBottomY ,sub_widths, sub_heights, sub_totals, obj_cur);
          
          topX[obj_cur] = newTopX;
          topY[obj_cur] = newTopY;
          bottomX[obj_cur] = newBottomX;
          bottomY[obj_cur] = newBottomY;

          num_block = ceil( (float) sub_totals[obj_cur] / (float) TILE_WIDTH);
          block = dim3(TILE_WIDTH, 1, 1);
          grid = dim3(num_block, 1, 1);

        if(gpuDistance(cx[obj_cur], cy[obj_cur], prevCX, prevCY) <= 1 )
          break;

        count ++;

        if( count > LOST_OBJECT )
        {
            printf("The GPU kernel has lost the object! --> %d\n", count);
            setWindowToEntireFrame(obj_cur, &topY, &topX, &bottomX, &bottomY, &sub_widths, &sub_heights, &sub_totals);
            break;
        }

      }//end while

    } //ennd if

  } //end kernel

/********************************************************/

__global__ void blockReduce(
unsigned int obj_cur,
unsigned int num_block, 
unsigned char * frame,  
unsigned int * sub_widths,
unsigned int * sub_totals,
unsigned int * topY,
unsigned int * topX, 
float * output)
{
  __shared__ float shared_M00[1024];
  __shared__ float shared_M1x[1024];
  __shared__ float shared_M1y[1024];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
  unsigned int tid = threadIdx.x;
  unsigned int sub_col = 0;
  unsigned int sub_row = 0;
  unsigned int absolute_index = 0;
  unsigned int hist_offset = obj_cur * HIST_BUCKETS;


  if(i < sub_totals[obj_cur]) 
  {
      sub_col = i % sub_widths[obj_cur];
      sub_row = i / sub_widths[obj_cur];

      absolute_index = (topY[ obj_cur ] * FRAME_WIDTH ) + topX[ obj_cur ] + sub_col + ( FRAME_WIDTH * sub_row);
      
      shared_M00[tid] = const_histogram[ (frame[absolute_index] / 3) + hist_offset ];
      shared_M1x[tid] = ((float)(sub_col + topX[obj_cur])) * shared_M00[tid];
      shared_M1y[tid] = ((float)(sub_row + topY[obj_cur])) * shared_M00[tid];
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
     warpReduce(shared_M00, shared_M1x, shared_M1y, tid, blockDim.x);
  }
   __syncthreads(); 
  // write result for this block to global mem
  if (tid == 0) {
    output[blockIdx.x] = shared_M00[0]; 
    output[blockIdx.x + num_block] = shared_M1x[0]; 
    output[blockIdx.x + num_block + num_block] = shared_M1y[0]; 
  }
} 

/********************************************************/

__global__ void finalReduce(float * output, unsigned int num_block)
{
  unsigned int tid = threadIdx.x; 

  __shared__ float shared_M00[1024];
  __shared__ float shared_M1x[1024];
  __shared__ float shared_M1y[1024];

    if(tid < num_block)
    {
      shared_M00[tid] = output[tid]; //M00
      shared_M1x[tid] = output[tid + num_block]; //M1x
      shared_M1y[tid] = output[tid + num_block + num_block];//M1y
    }
   else
   {
     shared_M00[tid] = 0.0; //M00
     shared_M1x[tid] = 0.0; //M1x
     shared_M1y[tid] = 0.0;//M1y
   }
    __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) 
  { 
    if(tid < stride)
    {
      shared_M00[tid] += shared_M00[tid + stride]; 
      shared_M1x[tid] += shared_M1x[tid + stride]; 
      shared_M1y[tid] += shared_M1y[tid + stride]; 
    }
    __syncthreads(); 
  }

  if(tid < 32){
     warpReduce(shared_M00, shared_M1x, shared_M1y, tid, blockDim.x);
  }
 __syncthreads(); 
  if(tid == 0)
  {
     output[0] = shared_M00[tid]; 
     output[1] = shared_M1x[tid]; 
     output[2] = shared_M1y[tid]; 
  }
}