
#include "timing.c"
#include "gpuMain.h"
#include "gpuMerge.h"

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

//Backprojects M00, M1x, M1y as double pointers, in preparation for reduce summation kernel
void gpuBackProjectMain(unsigned char * hueArray, int hueLength, float * histogram, int width, int xOffset, int yOffset, float ** h_M00, float ** h_M1x, float ** h_M1y)
{
    int tile_width = 64;
    int num_block = ceil(hueLength / (float) tile_width);
    dim3 block(tile_width, 1, 1);
    dim3 grid(num_block, 1, 1);

    const int histogramLength = 60;
    float *d_hist;
    cudaError_t err = cudaMalloc((void **)&d_hist, histogramLength * sizeof(float)); 
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

   err = cudaMemcpy(d_hist, histogram, histogramLength * sizeof(float), cudaMemcpyHostToDevice);
   if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    unsigned char * d_hueArray;
    err = cudaMalloc((void **)&d_hueArray, hueLength * sizeof(unsigned char)); 
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));


    err = cudaMemcpy(d_hueArray, hueArray, hueLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    float * d_M00; //device back projected histogram
    float * d_M1x;
    float * d_M1y;

    err = cudaMalloc((void **)&d_M00, hueLength * sizeof(float)); 
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    err = cudaMalloc((void **)&d_M1x, hueLength * sizeof(float)); 
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    err = cudaMalloc((void **)&d_M1y, hueLength * sizeof(float)); 
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));
    
    gpuBackProjectKernel<<<ceil(hueLength / (float) 64), 64>>>(d_hist, d_hueArray, hueLength, d_M00, d_M1x, d_M1y, width, xOffset, yOffset);

    err = cudaMemcpy(*h_M00, d_M00, hueLength * sizeof(float), cudaMemcpyDeviceToHost);

    if(err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
    }

   err = cudaMemcpy(*h_M1x, d_M1x, hueLength * sizeof(float), cudaMemcpyDeviceToHost);

   if(err != cudaSuccess)
   {
        printf("%s\n", cudaGetErrorString(err));
   }

  err =  cudaMemcpy(*h_M1y, d_M1y, hueLength * sizeof(float), cudaMemcpyDeviceToHost);


   if(err != cudaSuccess)
   {
        printf("%s\n", cudaGetErrorString(err));
   }

    cudaFree(d_hist);
    cudaFree(d_hueArray);
    cudaFree(d_M00);
    cudaFree(d_M1x);
    cudaFree(d_M1y);
}


int gpuReduceMain(int blockWidth, float * M00, float * M1x, float * M1y, int length, int * xc, int * yc)
{
   int tile_width = blockWidth;

   float *h_M00_out, *d_M00_in, *d_M00_out;
   float *h_M1x_out, *d_M1x_in, *d_M1x_out;
   float *h_M1y_out, *d_M1y_in, *d_M1y_out;

   // set up host memory
   h_M00_out = (float *) malloc(length * sizeof(float)); //MAXDRET
   h_M1x_out = (float *) malloc(length * sizeof(float)); //MAXDRET
   h_M1y_out = (float *) malloc(length * sizeof(float)); //MAXDRET

  // memset(h_M00_out, 0, length * sizeof(float)); //MAXDRET
  // memset(h_M1x_out, 0, length * sizeof(float)); //MAXDRET
   //memset(h_M1y_out, 0, length * sizeof(float)); //MAXDRET

   int num_block = ceil(length / (float)tile_width);
   dim3 block(tile_width, 1, 1);
   dim3 grid(num_block, 1, 1);

   // allocate storage for the device
   cudaMalloc((void**)&d_M00_in, sizeof(float) * length);
   cudaMalloc((void**)&d_M00_out, sizeof(float) * length ); //MAXDRET
   //cudaMemset(d_M00_out, 0, sizeof(float) * length ); //MAXDRET

   cudaMalloc((void**)&d_M1x_in, sizeof(float) * length);
   cudaMalloc((void**)&d_M1x_out, sizeof(float) * length); //MAXDRET
  // cudaMemset(d_M1x_out, 0, sizeof(float) * length); //MAXDRET

   cudaMalloc((void**)&d_M1y_in, sizeof(float) * length);
   cudaMalloc((void**)&d_M1y_out, sizeof(float) * length); //MAXDRET
  // cudaMemset(d_M1y_out, 0, sizeof(float) * length); //MAXDRET

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

 printf("Done! GPU time cost in second: %f\n", time );
 // printf("From GPU: M00 --> %f M1x --> %f M1y --> %f\n", h_M00_out[0], h_M1x_out[0], h_M1y_out[0]);


  //Calculate centroid

  if( h_M00_out[0] > 0){//Can't divide by zero...
        
        *xc = (int) (h_M1x_out[0] /  h_M00_out[0]);
        *yc = (int) (h_M1y_out[0] /  h_M00_out[0]);
        
      //  printf("Inside GPU MeanShift ---> centroid (%d, %d)\n", *xc, *yc);
    }
   printf("**********THIS BETTER BE SO! M00 = %f M1x = %f M1y = %f **************\n", h_M00_out[0], h_M1x_out[0], h_M1y_out[0]);
  //------------------------ now time the sequential code on CPU------------------------------





  clock_t now, then;
  float cpuTotal = 0;

  // timing on CPU
  then = clock();
  cpuTotal = cpuReduce(M00, length);
  now = clock();

  // measure the time spent on CPU
  time = timeCost(then, now);

  printf(" done. CPU time cost in second: %f\n", time * 1000);
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


//Backprojects M00, M1x, M1y as double pointers, in preparation for reduce summation kernel
void bpTest(unsigned char * hueArray, int ** convertedArray, int hueLength)
{
	  int index = 40499;
    int s = sizeof(unsigned char);
    printf("Size of unsigned char == %d, hueLength == %d\n", s, hueLength);

    int tile_width = 64;
    int num_block = ceil(hueLength / (float) tile_width);
    dim3 block(tile_width, 1, 1);
    dim3 grid(num_block, 1, 1);

    unsigned char * d_hueArray;

    printf("Checking hueArray--> %d\n", hueArray[index]);

    cudaError_t err = cudaMalloc((void **) &d_hueArray, hueLength * sizeof(unsigned char)); 

    printf("Checking hueArray--> %d\n", hueArray[index]);
    
    if(err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
    }

    printf("Checking hueArray--> %d\n", hueArray[index]);

    cudaMemcpy(d_hueArray, hueArray, hueLength * sizeof(unsigned char), cudaMemcpyHostToDevice);

    printf("Checking hueArray--> %d\n", hueArray[index]);

    int * d_converted;

    cudaMalloc((void **)&d_converted, hueLength * sizeof(int)); 
   //  cudaMemset(d_converted, 0, sizeof(int) * hueLength);

   bpTestKernel<<<ceil(hueLength / (float) 64), 64>>>(d_hueArray, d_converted, hueLength);

   cudaMemcpy(*convertedArray, d_converted, hueLength * sizeof(int), cudaMemcpyDeviceToHost);
    
   //	cudaMemcpy(hueArray, d_hueArray, hueLength * sizeof(unsigned char), cudaMemcpyDeviceToHost);


   cudaFree(d_converted);
    cudaFree(d_hueArray);

}


//***********************************************************************************************//
// Below launches new improved kernel stuff

void mainConstantMemoryHistogramLoad(float * histogram)
{
  setConstantMemoryHistogram(histogram);
}

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

  //  printf("Num_block: %d vs tile_width %d\n", num_block, tile_width);


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



































