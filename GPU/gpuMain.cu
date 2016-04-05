
#include "timing.c"
#include "gpuMerge.h"

#define LEVELS 5
#define MAXDRET 102400
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

void usage()
{
   printf("Usage: ./progName blockWidth numElementsInput p \n");
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

//Backprojects M00, M1x, M1y as double pointers, in preparation for reduce summation kernel
void gpuBackProjectMain(int * hueArray, int hueLength, float * histogram, int width, int xOffset, int yOffset, float ** h_M00, float ** h_M1x, float ** h_M1y)
{
    cudaMemcpyToSymbol(c_hist, histogram, sizeof(float) * 60);

    int tile_width = 64;
    int num_block = ceil(hueLength / (float) tile_width);
    dim3 block(tile_width, 1, 1);
    dim3 grid(num_block, 1, 1);

    const int histogramLength = 60;
    float *d_hist;
    cudaMalloc(&d_hist, histogramLength * sizeof(float)); 
    cudaMemcpy(d_hist, histogram, histogramLength * sizeof(float), cudaMemcpyHostToDevice);

    int * d_hueArray;
    cudaMalloc(&d_hueArray, hueLength * sizeof(int)); 
    cudaMemcpy(d_hueArray, hueArray, hueLength * sizeof(int), cudaMemcpyHostToDevice);

    float * d_M00; //device back projected histogram
    float * d_M1x;
    float * d_M1y;

    cudaMalloc(&d_M00, hueLength * sizeof(float)); 
    cudaMemset(d_M00, 0, sizeof(float) * hueLength);
    cudaMalloc(&d_M1x, hueLength * sizeof(float)); 
    cudaMemset(d_M1x, 0, sizeof(float) * hueLength);
    cudaMalloc(&d_M1y, hueLength * sizeof(float)); 
    cudaMemset(d_M1y, 0, sizeof(float) * hueLength);

    //Was trying this grid, block, tile_width * sizeof(float) below

    gpuBackProjectKernel<<<ceil(hueLength / (float) 64), 64>>>(d_hist, d_hueArray, hueLength, d_M00, d_M1x, d_M1y, width, xOffset, yOffset);

    cudaMemcpy(*h_M00, d_M00, hueLength * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(*h_M1x, d_M1x, hueLength * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(*h_M1y, d_M1y, hueLength * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_hist);
    cudaFree(d_hueArray);
    cudaFree(d_M00);
    cudaFree(d_M1x);
    cudaFree(d_M1y);
}


int gpuReduceMain(int blockWidth, float * M00, float * M1x, float * M1y, int length)
{
   int tile_width = blockWidth;

   float *h_M00_out, *d_M00_in, *d_M00_out;
   float *h_M1x_out, *d_M1x_in, *d_M1x_out;
   float *h_M1y_out, *d_M1y_in, *d_M1y_out;

   // set up host memory
   h_M00_out = (float *) malloc(MAXDRET * sizeof(float));
   h_M1x_out = (float *) malloc(MAXDRET * sizeof(float));
   h_M1y_out = (float *) malloc(MAXDRET * sizeof(float));

   memset(h_M00_out, 0, MAXDRET * sizeof(float));
   memset(h_M1x_out, 0, MAXDRET * sizeof(float));
   memset(h_M1y_out, 0, MAXDRET * sizeof(float));

   int num_block = ceil(length / (float)tile_width);
   dim3 block(tile_width, 1, 1);
   dim3 grid(num_block, 1, 1);

   // allocate storage for the device
   cudaMalloc((void**)&d_M00_in, sizeof(float) * length);
   cudaMalloc((void**)&d_M00_out, sizeof(float) * MAXDRET);
   cudaMemset(d_M00_out, 0, sizeof(float) * MAXDRET);

   cudaMalloc((void**)&d_M1x_in, sizeof(float) * length);
   cudaMalloc((void**)&d_M1x_out, sizeof(float) * MAXDRET);
   cudaMemset(d_M1x_out, 0, sizeof(float) * MAXDRET);

   cudaMalloc((void**)&d_M1y_in, sizeof(float) * length);
   cudaMalloc((void**)&d_M1y_out, sizeof(float) * MAXDRET);
   cudaMemset(d_M1y_out, 0, sizeof(float) * MAXDRET);

   // copy input to the device
   cudaMemcpy(d_M00_in, M00, sizeof(float) * length, cudaMemcpyHostToDevice);
   cudaMemcpy(d_M1x_in, M1x, sizeof(float) * length, cudaMemcpyHostToDevice);
   cudaMemcpy(d_M1y_in, M1y, sizeof(float) * length, cudaMemcpyHostToDevice);

   // time the kernel launches using CUDA events
   cudaEvent_t launch_begin, launch_end;
   cudaEventCreate(&launch_begin);
   cudaEventCreate(&launch_end);

   //----------------------time many kernel launches and take the average time--------------------
   
   int num_in = length, num_out = ceil((float)length / tile_width);
   int launch = 1;

   // record a CUDA event immediately before and after the kernel launch
   cudaEventRecord(launch_begin,0);

   while( 1 )
   {
       if(launch % 2 == 1) // odd launch
           gpuSummationReduce<<< grid, block, tile_width * sizeof(float) >>>(d_M00_in, d_M00_out, d_M1x_in, d_M1x_out, d_M1y_in, d_M1y_out, num_in);
       else
           gpuSummationReduce<<< grid, block, tile_width * sizeof(float) >>>(d_M00_out, d_M00_in, d_M1x_out, d_M1x_in, d_M1y_out, d_M1y_in, num_in);

       cudaDeviceSynchronize();

       // if the number of local max returned by kernel is greater than the threshold,
       // we do reduction on GPU for these returned local maxes for another pass,
       // until, num_out < threshold

       if(num_out >= THRESH)
       {
           num_in = num_out;
           num_out = ceil((float) num_out / tile_width);
           grid.x = num_out; //change the grid dimension in x direction
       }
       else //copy the ouput of last lauch back to host
       {
           if(launch % 2 == 1)
           {
              cudaMemcpy(h_M00_out, d_M00_out, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
              cudaMemcpy(h_M1x_out, d_M1x_out, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
              cudaMemcpy(h_M1y_out, d_M1y_out, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
           }
           else
           {
              cudaMemcpy(h_M00_out, d_M00_in, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
              cudaMemcpy(h_M1x_out, d_M1x_in, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
              cudaMemcpy(h_M1y_out, d_M1y_in, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
           }

           break;
       }

       launch ++;
   }//end of while

  cudaEventRecord(launch_end,0);
  cudaEventSynchronize(launch_end);

  // measure the time spent in the kernel
  float time = 0;
  cudaEventElapsedTime(&time, launch_begin, launch_end);

  printf("Done! GPU time cost in second: %f\n", time / 1000);
  printf("The output array from device is: M00 --> %f M1x --> %f M1y --> %f\n", h_M00_out[0], h_M1x_out[0], h_M1y_out[0]);
   
  //------------------------ now time the sequential code on CPU------------------------------

  clock_t now, then;
  float cpuTotal = 0;

  // timing on CPU
  then = clock();
  cpuTotal = cpuReduce(M00, length);
  now = clock();

  // measure the time spent on CPU
  time = timeCost(then, now);

  printf(" done. CPU time cost in second: %f\n", time);
  printf("CPU finding total is %f\n", cpuTotal);

  //--------------------------------clean up-----------------------------------------------------
  cudaEventDestroy(launch_begin);
  cudaEventDestroy(launch_end);

  // deallocate device memory
  cudaFree(d_M00_in);
  cudaFree(d_M00_out);
  cudaFree(d_M1x_in);
  cudaFree(d_M1x_out);
  cudaFree(d_M1y_in);
  cudaFree(d_M1y_out);

  free(h_M00_out);
  free(h_M1x_out);
  free(h_M1y_out);

  return 0;
}

