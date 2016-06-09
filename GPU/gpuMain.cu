#include "gpuMain.h"
#include "dynamicCamShift.h"
#include "deviceProperties.h"

float launchGPU_BGR_to_Hue(unsigned char * bgr, d_struct ds, int total)
{
  cudaError_t err; 
  unsigned int num_block = 0;

  num_block += ceil( (float) total / (float) 1024);
  dim3 block(1024, 1, 1);
  dim3 grid(num_block, 1, 1);

  cudaEvent_t launch_begin, launch_end;

  float time = 0;
  cudaEventCreate(&launch_begin);
  cudaEventCreate(&launch_end);
  cudaEventRecord(launch_begin,0);

  if(( err =  cudaMemcpy(ds.d_bgr, bgr, sizeof(unsigned char) * total * 3, cudaMemcpyHostToDevice)) != cudaSuccess)
      printf("Is this the error? %s\n", cudaGetErrorString(err));

  gpuBGRtoHue<<< grid, block >>>(ds.d_bgr, ds.d_frame, total);

  cudaEventRecord(launch_end,0);
  cudaEventSynchronize(launch_end);
  cudaEventElapsedTime(&time, launch_begin, launch_end);

  return time;
}

//This function was taken from http://cuda-programming.blogspot.com/2013/01/how-to-query-to-devices-in-cuda-cc.html
void printDeviceProperties()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (unsigned int i = 0; i < devCount; i++)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
    printf("---------------End of Device Properties-------------\n\n");
}

void timeMemoryTransfer()
{
	float time = 0;
	cudaError_t err; 
  printf("Enter the number of input to test memory transfer time: ");
	unsigned int num;
  scanf("%d", &num);
	unsigned int * h_array = (unsigned int * )malloc(sizeof(unsigned int) * num);
	unsigned int * d_array;
	if(( err = cudaMalloc((void **)&d_array, num * sizeof(unsigned int))) != cudaSuccess)
          printf("%s\n", cudaGetErrorString(err));

  cudaEvent_t launch_begin, launch_end; 

	cudaEventCreate(&launch_begin);
  cudaEventCreate(&launch_end);
  cudaEventRecord(launch_begin,0);

  err =  cudaMemcpy(h_array, d_array, sizeof(unsigned int) * num, cudaMemcpyDeviceToHost);

 	cudaDeviceSynchronize();
  cudaEventRecord(launch_end,0);
  cudaEventSynchronize(launch_end);
  cudaEventElapsedTime(&time, launch_begin, launch_end);
  printf("Time to transfer from gpu to cpu %d elements: %f\n", num, time);
  free(h_array);
  cudaFree(d_array);
}

//wrapper function because constant memory must be in the same file that accesses it, linking issue
void mainConstantMemoryHistogramLoad(float * histogram, unsigned int num_objects)
{
  cudaDeviceReset();
  setConstantMemoryHistogram(histogram);
}

float initDeviceStruct(
unsigned int num_objects, 
d_struct * ds, 
unsigned char * frame, 
unsigned int frameLength, 
unsigned int * cx, 
unsigned int * cy, 
unsigned int * col_offset, 
unsigned int * row_offset, 
unsigned int * sub_lengths, 
unsigned int * sub_widths, 
unsigned int * sub_heights)
{
    cudaError_t err;
    unsigned int * d_cx;
    unsigned int * d_cy;
    unsigned int * d_prevX;
    unsigned int * d_prevY;
    unsigned int * d_col_offset;
    unsigned int * d_row_offset;
    unsigned int * d_sub_widths;
    unsigned int * d_sub_heights;
    unsigned int * d_sub_lengths;
    unsigned char * d_frame;
    unsigned char * d_bgr;


    cudaEvent_t launch_begin, launch_end; 
    float time = 0.0f;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    cudaEventRecord(launch_begin,0);


    if(( err = cudaMalloc((void **)&d_frame, frameLength * sizeof(unsigned char))) != cudaSuccess)
          printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMalloc((void **)&d_bgr, sizeof(unsigned char) * frameLength * 3)) != cudaSuccess )
          printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMalloc((void **)&d_cx, sizeof(unsigned int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMemcpy(d_cx, cx, sizeof(unsigned int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
         printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMalloc((void **)&d_cy, sizeof(unsigned int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMemcpy(d_cy, cy, sizeof(unsigned int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
         printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMalloc((void **)&d_prevX, sizeof(int) * num_objects)) != cudaSuccess) 
        printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMemset(d_prevX, 0, sizeof(int) * num_objects)) != cudaSuccess)
         printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMalloc((void **)&d_prevY, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMemset(d_prevY, 0, sizeof(int) * num_objects)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMalloc((void **)&d_sub_lengths, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMemcpy(d_sub_lengths, sub_lengths, sizeof(int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    if((err = cudaMalloc((void **)&d_sub_widths, sizeof(unsigned int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMemcpy(d_sub_widths, sub_widths, sizeof(unsigned int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMalloc((void **)&d_row_offset, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMemcpy(d_row_offset, row_offset, sizeof(int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    if((err = cudaMalloc((void **)&d_col_offset, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMemcpy(d_col_offset, col_offset, sizeof(int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    if(( err = cudaMalloc((void **)&d_sub_heights, sizeof(unsigned int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    
    if(( err = cudaMemcpy(d_sub_heights, sub_heights, sizeof(unsigned int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    (*ds).d_bgr = d_bgr;
    (*ds).d_frame = d_frame;
    (*ds).d_cx = d_cx;
    (*ds).d_cy = d_cy;
    (*ds).d_prevX = d_prevX;
    (*ds).d_prevY = d_prevY;
    (*ds).d_col_offset = d_col_offset;
    (*ds).d_row_offset = d_row_offset;
    (*ds).d_sub_lengths = d_sub_lengths;
    (*ds).d_sub_widths = d_sub_widths;
    (*ds).d_sub_heights = d_sub_heights;

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    cudaEventElapsedTime(&time, launch_begin, launch_end);

  return time;
}

void freeDeviceStruct(d_struct * ds)
{
    cudaFree((*ds).d_bgr);
    cudaFree((*ds).d_frame);
    cudaFree((*ds).d_cx);
    cudaFree((*ds).d_cy);
    cudaFree((*ds).d_prevX);
    cudaFree((*ds).d_prevY);
    cudaFree((*ds).d_row_offset);
    cudaFree((*ds).d_col_offset);
    cudaFree((*ds).d_sub_lengths);
    cudaFree((*ds).d_sub_widths);
    cudaFree((*ds).d_sub_heights);
}


float gpuCamShift(
d_struct ds, 
unsigned int num_objects, 
unsigned char * frame, 
unsigned int frame_length, 
unsigned int frame_width, 
unsigned int ** sub_widths, 
unsigned int ** sub_heights, 
unsigned int ** cx, 
unsigned int ** cy, 
bool adapt_window)
{
    cudaEvent_t launch_begin, launch_end;
    cudaError_t err; 
    float time = 0;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    cudaEventRecord(launch_begin,0);
    //Copy new frame into device memory
   // cudaMemcpy(ds.d_frame, frame, frame_length * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    dynamicCamShiftMain<<< 1 , num_objects >>>(num_objects,
                                    ds.d_frame,
                                    frame_length,
                                    frame_width,
                                    ds.d_cx,
                                    ds.d_cy,
                                    ds.d_prevX,
                                    ds.d_prevY,
                                    ds.d_row_offset,
                                    ds.d_col_offset,
                                    ds.d_sub_widths,
                                    ds.d_sub_heights,
                                    ds.d_sub_lengths,
                                    adapt_window);

   // cudaDeviceSynchronize();

    if(( err =  cudaMemcpy(*cx, ds.d_cx, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    if(( err =  cudaMemcpy(*cy, ds.d_cy, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err)); 

    if(adapt_window)
    {
      if(( err =  cudaMemcpy(*sub_widths, ds.d_sub_widths, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

      if(( err =  cudaMemcpy(*sub_heights, ds.d_sub_heights, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    cudaEventElapsedTime(&time, launch_begin, launch_end);
    
    return time;
}