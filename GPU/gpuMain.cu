#include "gpuMain.h"
#include "kernels.h"

#define LEVELS 5
#define THRESH 2

float * fillArray(int n)
{
   int i;

   float *ret = (float *) malloc(sizeof(float) * n );

    for( i = 0; i < n ; i++ ) {
      ret[i] = (float) 3.3;
   }

   return ret;
}

void printArray(float *arr, int n){

   int i;

   for(i = 0; i < n; i ++)
      printf("%f ", arr[i]);

   printf("\n");
}

float cpuReduce(float * h_in, int n)
{
   double total = 0.0;

	 int i;

    for(i = 0; i < n; i ++)
        total += (double) h_in[i];
   // printf("CPU---> %fn", total);
    return total;
}

void reverseIt(float * histogram)
{
	const int n = 60;
  	float d[n];

  	float *d_hist;
  	cudaMalloc(&d_hist, n * sizeof(float)); 

	  cudaMemcpy(d_hist, histogram, n*sizeof(float), cudaMemcpyHostToDevice);


	  staticReverse<<< 1,64 >>>(d_hist, n);
    
	  cudaMemcpy(d, d_hist, n*sizeof(float), cudaMemcpyDeviceToHost);

    printf("*************************************************\n");

	 for (int i = 0; i < n; i++) {
      printf("%d) %f\n",i, d[i]);
    }

    cudaFree(d_hist);
}



/*
  This function and it's kernel are probably obsolete at this point, but leaving them in just in case.

  Obsolete because it requires a hueFrame of only the search window hues and not the entire frame.
  Limited, in that the frame must be reset between iterations, I'd have to remove the cudaMallocs and reuse a statically sized
  search window to make use of this. It also would perclude the opportunity of extending for multiple objects
*/

int launchMeanShiftKernelForSubFrame(unsigned char * hueFrame, int hueFrameLength, int width, int xOffset, int yOffset, int * cx, int * cy)
{
  printf("\nInside Launching GPU MeanShift...\n");

 
   unsigned char * d_in;

    cudaError_t err = cudaMalloc((void **)&d_in, hueFrameLength * sizeof(unsigned char)); 
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_in, hueFrame, hueFrameLength * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaEvent_t launch_begin, launch_end;
    int tile_width = 1024;
    int num_block = ceil( (float) hueFrameLength / (float) tile_width);
    dim3 block(tile_width, 1, 1);
    dim3 grid(num_block, 1, 1);

    //Make d_out 3 times the block size to store M00, M1x, M1y values at a stride of num_block
    float * d_out;
    err = cudaMalloc((void **)&d_out, 3 * num_block * sizeof(float)); 
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

      int * readyArray;
    err = cudaMalloc((void **)&readyArray, num_block * sizeof(int)); 
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));
      cudaMemset(readyArray, 0, sizeof(int) * num_block ); 

    //Make h_out 3 times the block size to store M00, M1x, M1y values at a stride of num_block
    float * h_out = (float *) malloc(3 * num_block * sizeof(float));

  printf("Num_block: %d vs tile_width %d\n", num_block, tile_width);


    if(num_block <= tile_width){

     cudaEventCreate(&launch_begin);
     cudaEventCreate(&launch_end);

     cudaEventRecord(launch_begin,0);

    gpuMeanShiftKernelForSubFrame<<< grid, block >>>(d_in, d_out, readyArray, hueFrameLength, num_block, width, xOffset, yOffset);
      

    err =  cudaMemcpy(h_out, d_out, 3 * num_block * sizeof(float), cudaMemcpyDeviceToHost);


     cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    printf("GPU time cost in milliseconds for improved meanshift kernel with atomic add: %f\n", time);
    printf("improved meanshift kernel with atomic add total: M00 = %f M1x = %f M1y = %f \n", h_out[0], h_out[num_block], h_out[num_block * 2]);

    //cpuReduce(h_out, num_block);

   //  printArray(h_out, num_block);

  }
  else
    printf("Cannot launch kernel: num_block (%d) > tile_width (%d)\n", num_block, tile_width);


    cudaFree(d_out);
    cudaFree(readyArray);
    free(h_out);
    cudaFree(d_in);

    return 1;
}


//***********************************************************************************************//
// Below launches new improved kernel stuff

//wrapper function because constant memory must be in the same file that accesses it, linking issue
void mainConstantMemoryHistogramLoad(float * histogram)
{
  cudaDeviceReset();
  setConstantMemoryHistogram(histogram);
}

void initDeviceStruct(d_struct * ds, unsigned char * frame, int frameLength, int * cx, int * cy, int * col_offset, int * row_offset)
{
    cudaError_t err; 
    int * d_cx;
    int * d_cy;
    int * d_col_offset;
    int * d_row_offset;
    unsigned char * d_frame;

    if(( err = cudaMalloc((void **)&d_frame, frameLength * sizeof(unsigned char))) != cudaSuccess)
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_frame, frame, frameLength * sizeof(unsigned char), cudaMemcpyHostToDevice);  
    if((err = cudaMalloc((void **)&d_cx, sizeof(int))) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_cx, cx, sizeof(int), cudaMemcpyHostToDevice);
    if((err = cudaMalloc((void **)&d_cy, sizeof(int))) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_cy, cy, sizeof(int), cudaMemcpyHostToDevice);
    if((err = cudaMalloc((void **)&d_row_offset, sizeof(int))) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_row_offset, row_offset, sizeof(int), cudaMemcpyHostToDevice);
    if((err = cudaMalloc((void **)&d_col_offset, sizeof(int))) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_col_offset, col_offset, sizeof(int), cudaMemcpyHostToDevice);

    (*ds).d_frame = d_frame;
    (*ds).d_cx = d_cx;
    (*ds).d_cy = d_cy;
    (*ds).d_col_offset = d_col_offset;
    (*ds).d_row_offset = d_row_offset;
}

void freeDeviceStruct(d_struct * ds)
{
    cudaFree((*ds).d_frame);
    cudaFree((*ds).d_cx);
    cudaFree((*ds).d_cy);
    cudaFree((*ds).d_row_offset);
    cudaFree((*ds).d_col_offset);
}


float launchTwoKernelReduction(d_struct ds, unsigned char * frame, int frameLength, int subFrameLength, int abs_width, int sub_width, int sub_height, int * cx, int * cy, bool shouldPrint)
{
    float time = 0;
    // printf("\nInside Launching GPU MeanShift for entire frame...\n");
    cudaEvent_t launch_begin, launch_end;
    int tile_width = 1024;
    int num_block = ceil( (float) subFrameLength / (float) tile_width);
    dim3 block(tile_width, 1, 1);
    dim3 grid(num_block, 1, 1);
    unsigned int dynamic_sharedMem_size = 3 * num_block * sizeof(float);

    cudaError_t err; 

    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    cudaEventRecord(launch_begin,0);
    int * h_cx = (int *) malloc(sizeof(int));
    int * h_cy = (int *) malloc(sizeof(int));
    h_cx[0] = -1;
    h_cy[0] = -1;
    //Make d_out 3 times the block size to store M00, M1x, M1y values at a stride of num_block
    float * d_out;
    if((err = cudaMalloc((void **)&d_out, 3 * num_block * sizeof(float)))!= cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    //Make h_out 3 times the block size to store M00, M1x, M1y values at a stride of num_block
    float * h_out = (float *) malloc(3 * num_block * sizeof(float));

     int prevX = 0;
     int prevY = 0;

    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    cudaEventRecord(launch_begin,0);


    err = cudaMemcpy(ds.d_frame, frame, frameLength * sizeof(unsigned char), cudaMemcpyHostToDevice);

    if(num_block <= tile_width)
    {

    while(prevX != h_cx[0] && prevY != h_cy[1]){

      prevX = h_cx[0];
      prevY = h_cy[0];

      gpuBlockReduce<<< grid, block >>>(ds.d_frame, d_out, subFrameLength, num_block, abs_width, sub_width, sub_height, ds.d_row_offset, ds.d_col_offset);
      gpuFinalReduce<<< 1, num_block, dynamic_sharedMem_size >>>(d_out, ds.d_cx, ds.d_cy, ds.d_row_offset, ds.d_col_offset, sub_width, sub_height, num_block);

      err =  cudaMemcpy(h_cx, ds.d_cx, sizeof(int), cudaMemcpyDeviceToHost);
      err =  cudaMemcpy(h_cy, ds.d_cy, sizeof(int), cudaMemcpyDeviceToHost);
     
      if(shouldPrint)
     	printf("PrevX vs NewX(%d, %d) and PrevY vs NewY(%d, %d)\n", prevX, h_cx[0], prevY, h_cy[0]);

    }
    cudaDeviceSynchronize();
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    cudaEventElapsedTime(&time, launch_begin, launch_end);
  }
  else
    printf("Cannot launch kernel: num_block (%d) > tile_width (%d)\n", num_block, tile_width);

    *cx = h_cx[0];
    *cy = h_cy[0];

    cudaFree(d_out);
    free(h_out);
    free(h_cx);
    free(h_cy);

    return time;
}
