
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

    printf("CPU---> %lfn", total);
    return total;
}

void usage()
{
   printf("Usage: ./progName blockWidth numElementsInput p \n");
}


void reverseIt(float * histogram)
{
	   const int n = 60;
  	float a[n], r[n], d[n];

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


float * gpuBackProjectMain(int * hueArray, int hueLength, float * histogram, int width, int xOffset, int yOffset)
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

    float * d_backproj; //device back projected histogram

    cudaMalloc(&d_backproj, hueLength * sizeof(float)); 
    cudaMemset(d_backproj, 0, sizeof(float) * hueLength);

    float * h_backproj; // host back projected histogram

    h_backproj = (float *) malloc(hueLength * sizeof(float));


    //Was trying this grid, block, tile_width * sizeof(float) below

    gpuBackProjectKernel<<<ceil(hueLength / (float) 64), 64>>>(d_hist, d_hueArray,  hueLength, d_backproj, width, xOffset, yOffset);

    cudaMemcpy(h_backproj, d_backproj, hueLength * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_hist);
    cudaFree(d_hueArray);
    cudaFree(d_backproj);

    return h_backproj;
}











int gpuMain(int blockWidth, char p, unsigned char * hueArray, int length)
{
  int shouldPrint = 0;

  if(p == 'p')
    shouldPrint = 1;

   int tile_width = blockWidth;

   if ( ! tile_width )
   {
       printf("Wrong argument passed in for blockWidth!\n");
       exit(-1);
   }
   int n = length; //size of 1D input array

   if ( ! n )
   {
       printf("Wrong argument passed in for size of input array!\n");
       exit(-1);
   }

   // set up host memory
   float *h_in, *h_out, *d_in, *d_out;

   //int sizeDout[LEVELS]; //we can have at most 5 levels of kernel launch

   h_out = (float *)malloc(MAXDRET * sizeof(float));

   memset(h_out, 0, MAXDRET * sizeof(float));

   //generate input data from random generator
   h_in = fillArray(n);

   cpuReduce(h_in, n);

   if( ! h_in || ! h_out )
   {
       printf("Error in host memory allocation!\n");
       exit(-1);
   }

   int num_block = ceil(n / (float)tile_width);
   dim3 block(tile_width, 1, 1);
   dim3 grid(num_block, 1, 1);

   // allocate storage for the device
   cudaMalloc((void**)&d_in, sizeof(float) * n);
   cudaMalloc((void**)&d_out, sizeof(float) * MAXDRET);
   cudaMemset(d_out, 0, sizeof(float) * MAXDRET);

   // copy input to the device
   cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);

   // time the kernel launches using CUDA events
   cudaEvent_t launch_begin, launch_end;
   cudaEventCreate(&launch_begin);
   cudaEventCreate(&launch_end);


   //print out original array
    if(shouldPrint)
    {
        printf("The input array is:\n");
        printArray(h_in, n);

    }

   //----------------------time many kernel launches and take the average time--------------------
   
   float average_simple_time = 0;
   int num_in = n, num_out = ceil((float)n / tile_width);
   int launch = 1;

   printf("Timing simple GPU implementation… \n");
   
       // record a CUDA event immediately before and after the kernel launch
       cudaEventRecord(launch_begin,0);
    
       while( 1 )
       {
           if(launch % 2 == 1) // odd launch
               gpuSummationReduce<<<grid, block, tile_width * sizeof(float)>>>(d_in, d_out, num_in);
           else
               gpuSummationReduce<<< grid, block, tile_width * sizeof(float) >>>(d_out, d_in, num_in);

           cudaDeviceSynchronize();

           // if the number of local max returned by kernel is greater than the threshold,
           // we do reduction on GPU for these returned local maxes for another pass,
           // until, num_out < threshold

           if(num_out >= THRESH)
           {
               num_in = num_out;
               num_out = ceil((float)num_out / tile_width);
               grid.x = num_out; //change the grid dimension in x direction
               //cudaMemset(d_in, 0, n * sizeof(int));//reset d_in, used for output of next iteration
           }
           else
           {
               //copy the ouput of last lauch back to host,
               if(launch % 2 == 1)
                  cudaMemcpy(h_out, d_out, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
               else
                  cudaMemcpy(h_out, d_in, sizeof(float) * num_out, cudaMemcpyDeviceToHost);

               break;
           }



           launch ++;
       }//end of while

       cudaEventRecord(launch_end,0);
       cudaEventSynchronize(launch_end);

       // measure the time spent in the kernel
       float time = 0;
       cudaEventElapsedTime(&time, launch_begin, launch_end);

       average_simple_time += time;
 
 
  printf(" done! GPU time cost in second: %f\n", average_simple_time / 1000);
  printf(" done! GPU time cost in second: %f\n", time / 1000);

      printf("The output array from device is:\n");
      printArray(h_out, num_out);


  //------------------------ now time the sequential code on CPU------------------------------

  // time many multiplication calls and take the average time
  float average_cpu_time = 0;
  clock_t now, then;

  float cpuTotal = 0;

  printf("Timing CPU implementation…\n");

 




    // timing on CPU
    then = clock();
    cpuTotal = cpuReduce(h_in, n);
    now = clock();






    // measure the time spent on CPU
   time = 0;
    time = timeCost(then, now);

    average_cpu_time += time;
 
  //average_cpu_time /= num_cpu_test;
  printf(" done. CPU time cost in second: %f\n", average_cpu_time);
  printf(" done. CPU time cost in second: %f\n", time);

  //if (shouldPrint)
      printf("CPU finding total is %.1f\n", cpuTotal);

  //--------------------------------clean up-----------------------------------------------------
  cudaEventDestroy(launch_begin);
  cudaEventDestroy(launch_end);

  // deallocate device memory
  cudaFree(d_in);
  cudaFree(d_out);

  free(h_in);
  free(h_out);


//exit(0);

  return 0;
}

