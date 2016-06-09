//###########################################################################################################################
// Matthew Perry
// EWU ID: 00681703
//
// Bridge file between nvcc compiled dynamicCamShift.cu file with kernels and main.cpp compiled with g++ and OpenCV.
// Allocates the memory used in the computation and display of the gpu version's search window,
// Launches kernels for the gpu computation.
//###########################################################################################################################

#include "gpuMain.h"
#include "dynamicCamShift.h"
#include "deviceProperties.h"

//###########################################################################################################################
// Allocate the host memory for the region of interest (roi) search window
//###########################################################################################################################
h_roi * initHostROI(int num_objects)
{
  h_roi * roi     = (h_roi *) malloc(sizeof(h_roi));
  roi->h_cx       = (unsigned int *) malloc(sizeof(unsigned int) * num_objects);
  roi->h_cy       = (unsigned int *) malloc(sizeof(unsigned int) * num_objects);
  roi->h_topX     = (unsigned int *) malloc(sizeof(unsigned int) * num_objects);
  roi->h_topY     = (unsigned int *) malloc(sizeof(unsigned int) * num_objects);
  roi->h_bottomX  = (unsigned int *) malloc(sizeof(unsigned int) * num_objects);
  roi->h_bottomY  = (unsigned int *) malloc(sizeof(unsigned int) * num_objects);
  return roi;
}

//###########################################################################################################################
// Free the host memory for the region of interest (roi) search window
//###########################################################################################################################
void freeHostROI(h_roi * roi)
{
  free(roi->h_cx);
  free(roi->h_cy); 
  free(roi->h_topX); 
  free(roi->h_topY);
  free(roi->h_bottomX);
  free(roi->h_bottomY); 
  free(roi);
}

//###########################################################################################################################
// Launches the kernel for parallel conversion from the BGR frame to the Hue frame in device memory
//###########################################################################################################################
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

//###########################################################################################################################
// This function launches a kernel
// who's algorithm was taken from http://cuda-programming.blogspot.com/2013/01/how-to-query-to-devices-in-cuda-cc.html
//
// Displays each online graphics card's architectural information 
//###########################################################################################################################
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

//###########################################################################################################################
// Test to time how long device to host memory transfer takes
//###########################################################################################################################
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

//###########################################################################################################################
// wrapper function because constant memory must be in the same file that accesses it, solves a linking issue
//###########################################################################################################################
void mainConstantMemoryHistogramLoad(float * histogram, unsigned int num_objects)
{
  cudaDeviceReset(); //makes sure previous runs memory is not still in device memory
  setConstantMemoryHistogram(histogram);
}


//###########################################################################################################################
// Allocate the device memory initially required and for persistent use across processing frames
//###########################################################################################################################
float initDeviceStruct(
unsigned int num_objects, 
d_struct * ds, 
h_roi * roi,
unsigned char * frame, 
unsigned int frameLength,  
unsigned int * sub_lengths, 
unsigned int * sub_widths, 
unsigned int * sub_heights)
{
    cudaError_t err;
    unsigned int * d_cx;
    unsigned int * d_cy;
    unsigned int * d_topX;
    unsigned int * d_topY;
    unsigned int * d_bottomX;
    unsigned int * d_bottomY;
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

    //######################################### VIDEO HUE FRAME ###############################################################
    if(( err = cudaMalloc((void **)&d_frame, frameLength * sizeof(unsigned char))) != cudaSuccess)
          printf("%s\n", cudaGetErrorString(err));

    //######################################### VIDEO BGR FRAME ###############################################################
    if(( err = cudaMalloc((void **)&d_bgr, sizeof(unsigned char) * frameLength * 3)) != cudaSuccess )
          printf("%s\n", cudaGetErrorString(err));

    //######################################### CENTROID X COORDINATE ##########################################################
    if(( err = cudaMalloc((void **)&d_cx, sizeof(unsigned int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    if(( err = cudaMemcpy(d_cx, roi->h_cx, sizeof(unsigned int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
         printf("%s\n", cudaGetErrorString(err));

    //######################################### CENTROID Y COORDINATE ##########################################################
    if(( err = cudaMalloc((void **)&d_cy, sizeof(unsigned int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    if(( err = cudaMemcpy(d_cy, roi->h_cy, sizeof(unsigned int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
         printf("%s\n", cudaGetErrorString(err));

    //######################################### SEARCH WINDOW LENGTH ##########################################################
    if(( err = cudaMalloc((void **)&d_sub_lengths, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    if(( err = cudaMemcpy(d_sub_lengths, sub_lengths, sizeof(int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    //######################################### TOP Y ###############################################################
    if(( err = cudaMalloc((void **)&d_topY, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    if(( err = cudaMemcpy(d_topY, roi->h_topY , sizeof(int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    //######################################### TOP X ###############################################################
    if((err = cudaMalloc((void **)&d_topX, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    if(( err = cudaMemcpy(d_topX, roi->h_topX, sizeof(int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    //######################################### BOTTOM Y ###############################################################
    if((err = cudaMalloc((void **)&d_bottomY, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    if(( err = cudaMemcpy(d_bottomY, roi->h_bottomY , sizeof(int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    //######################################### BOTTOM X ###############################################################
    if((err = cudaMalloc((void **)&d_bottomX, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    if(( err = cudaMemcpy(d_bottomX, roi->h_bottomX, sizeof(int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    //######################################### SUB WIDTHS ###############################################################

     if((err = cudaMalloc((void **)&d_sub_widths, sizeof(unsigned int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    if(( err = cudaMemcpy(d_sub_widths, sub_widths, sizeof(unsigned int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    //######################################### SUB HEIGHTS ###############################################################
    if(( err = cudaMalloc((void **)&d_sub_heights, sizeof(unsigned int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    if(( err = cudaMemcpy(d_sub_heights, sub_heights, sizeof(unsigned int) * num_objects, cudaMemcpyHostToDevice)) != cudaSuccess)
           printf("%s\n", cudaGetErrorString(err));

    (*ds).d_bgr = d_bgr;
    (*ds).d_frame = d_frame;
    (*ds).d_cx = d_cx;
    (*ds).d_cy = d_cy;
    (*ds).d_topX = d_topX;
    (*ds).d_topY = d_topY;
    (*ds).d_bottomX = d_bottomX;
    (*ds).d_bottomY = d_bottomY;
    (*ds).d_sub_lengths = d_sub_lengths;
    (*ds).d_sub_widths = d_sub_widths;
    (*ds).d_sub_heights = d_sub_heights;

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    cudaEventElapsedTime(&time, launch_begin, launch_end);

  return time;
}

//###########################################################################################################################
// Frees the device memory at the end of the program
//###########################################################################################################################
void freeDeviceStruct(d_struct * ds)
{
    cudaFree((*ds).d_bgr);
    cudaFree((*ds).d_frame);
    cudaFree((*ds).d_cx);
    cudaFree((*ds).d_cy);
    cudaFree((*ds).d_topY);
    cudaFree((*ds).d_topX);
    cudaFree((*ds).d_bottomY);
    cudaFree((*ds).d_bottomX);
    cudaFree((*ds).d_sub_lengths);
    cudaFree((*ds).d_sub_widths);
    cudaFree((*ds).d_sub_heights);
}

//###########################################################################################################################
// Launches the parent threads of a dynamically parallel CAMShift, each parent thread controls a different object's tracking
//###########################################################################################################################
float gpuCamShift(
d_struct ds,
h_roi * roi, 
unsigned int num_objects, 
unsigned char * frame, 
unsigned int frame_length, 
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
    
    dynamicCamShiftMain<<< 1 , num_objects >>>(
                                    ds.d_frame,
                                    frame_length,
                                    ds.d_cx,
                                    ds.d_cy,
                                    ds.d_topX,
                                    ds.d_topY,
                                    ds.d_bottomX,
                                    ds.d_bottomY,
                                    ds.d_sub_widths,
                                    ds.d_sub_heights,
                                    ds.d_sub_lengths,
                                    adapt_window);

   //Transfer back the final results for this frame
    if(( err =  cudaMemcpy(roi->h_cx, ds.d_cx, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    if(( err =  cudaMemcpy(roi->h_cy, ds.d_cy, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err)); 

   if(( err =  cudaMemcpy(roi->h_topY, ds.d_topY, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    if(( err =  cudaMemcpy(roi->h_topX, ds.d_topX, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err)); 

    if(( err =  cudaMemcpy(roi->h_bottomY, ds.d_bottomY, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    if(( err =  cudaMemcpy(roi->h_bottomX, ds.d_bottomX, sizeof(unsigned int) * num_objects, cudaMemcpyDeviceToHost)) != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    cudaEventElapsedTime(&time, launch_begin, launch_end);
    
    return time;
}