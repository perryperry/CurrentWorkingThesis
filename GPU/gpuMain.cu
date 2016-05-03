#include "gpuMain.h"
#include "kernels.h"


void timeMemoryTransfer()
{
	float time = 0;
	cudaError_t err; 
	int num = 4;
	int * h_array = (int * )malloc(sizeof(int) * num);
	int * d_array;
	if(( err = cudaMalloc((void **)&d_array, num * sizeof(int))) != cudaSuccess)
          printf("%s\n", cudaGetErrorString(err));

   cudaEvent_t launch_begin, launch_end;
      

	cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    cudaEventRecord(launch_begin,0);

     err =  cudaMemcpy(h_array, d_array, sizeof(int) * num, cudaMemcpyDeviceToHost);

 	cudaDeviceSynchronize();
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    cudaEventElapsedTime(&time, launch_begin, launch_end);



    printf("Time to transfer from gpu to cpu %d elements: %f\n", num, time);




      free(h_array);
      cudaFree(d_array);
}

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

//wrapper function because constant memory must be in the same file that accesses it, linking issue
void mainConstantMemoryHistogramLoad(float * histogram, int num_objects)
{
  cudaDeviceReset();
  setConstantMemoryHistogram(histogram, num_objects);
}

int initDeviceStruct(int num_objects, d_struct * ds, int * obj_block_ends, unsigned char * frame, int frameLength, int * cx, int * cy, int * col_offset, int * row_offset, int * sub_lengths, int * sub_widths, int * sub_heights)
{


    cudaError_t err;
    float * d_out; 
    int * d_cx;
    int * d_cy;
    int * d_col_offset;
    int * d_row_offset;
    int * d_obj_block_ends;
    int * d_sub_widths;
    int * d_sub_lengths;
    int * d_sub_heights;
    unsigned char * d_frame;

    int num_block = 0;
    int obj_cur = 0;
    int tile_width = 1024;

    for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
    {
       num_block += ceil( (float) sub_lengths[obj_cur] / (float) tile_width);
       obj_block_ends[obj_cur] = num_block;
         printf("SUB LENGTHS: %d sub_widths: %d sub_heights: %d Combined: %d\n", sub_lengths[obj_cur], sub_widths[obj_cur], sub_heights[obj_cur], sub_widths[obj_cur] * sub_heights[obj_cur]);

    }


    if(( err = cudaMalloc((void **)&d_frame, frameLength * sizeof(unsigned char))) != cudaSuccess)
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_frame, frame, frameLength * sizeof(unsigned char), cudaMemcpyHostToDevice); 

    //Make d_out 3 times the block size to store M00, M1x, M1y values at a stride of num_block
    if((err = cudaMalloc((void **)&d_out, 3 * num_block * sizeof(float)))!= cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    if((err = cudaMalloc((void **)&d_cx, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_cx, cx, sizeof(int) * num_objects, cudaMemcpyHostToDevice);

    if((err = cudaMalloc((void **)&d_cy, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_cy, cy, sizeof(int) * num_objects, cudaMemcpyHostToDevice);

    if((err = cudaMalloc((void **)&d_row_offset, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_row_offset, row_offset, sizeof(int) * num_objects, cudaMemcpyHostToDevice);

    if((err = cudaMalloc((void **)&d_col_offset, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_col_offset, col_offset, sizeof(int) * num_objects, cudaMemcpyHostToDevice);

    if((err = cudaMalloc((void **)&d_sub_lengths, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_sub_lengths, sub_lengths, sizeof(int) * num_objects, cudaMemcpyHostToDevice);

    if((err = cudaMalloc((void **)&d_sub_widths, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_sub_widths, sub_widths, sizeof(int) * num_objects, cudaMemcpyHostToDevice);

     if((err = cudaMalloc((void **)&d_sub_heights, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_sub_heights, sub_heights, sizeof(int) * num_objects, cudaMemcpyHostToDevice);

    if((err = cudaMalloc((void **)&d_obj_block_ends, sizeof(int) * num_objects)) != cudaSuccess) 
          printf("%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_obj_block_ends, obj_block_ends, num_objects * sizeof(int), cudaMemcpyHostToDevice);
   
    (*ds).d_frame = d_frame;
    (*ds).d_out = d_out;
    (*ds).d_cx = d_cx;
    (*ds).d_cy = d_cy;
    (*ds).d_col_offset = d_col_offset;
    (*ds).d_row_offset = d_row_offset;
    (*ds).d_sub_lengths = d_sub_lengths;
    (*ds).d_sub_widths = d_sub_widths;
    (*ds).d_sub_heights = d_sub_heights;
    (*ds).d_obj_block_ends = d_obj_block_ends;

    return num_block;
}

void freeDeviceStruct(d_struct * ds)
{
    cudaFree((*ds).d_frame);
    cudaFree((*ds).d_out);
    cudaFree((*ds).d_cx);
    cudaFree((*ds).d_cy);
    cudaFree((*ds).d_row_offset);
    cudaFree((*ds).d_col_offset);
    cudaFree((*ds).d_sub_lengths);
    cudaFree((*ds).d_sub_widths);
    cudaFree((*ds).d_sub_heights);
    cudaFree((*ds).d_obj_block_ends);
}

//For single object tracking, but capable of toggle between tracking objects based on obj_id in main with its dimensions
float launchTwoKernelReduction(int obj_id, int num_objects, d_struct ds, unsigned char * frame, int frameLength, int subFrameLength, int abs_width, int sub_width, int sub_height, int ** cx, int ** cy, bool shouldPrint)
{
    float time = 0;
    // printf("\nInside Launching GPU MeanShift for entire frame...\n");
    cudaEvent_t launch_begin, launch_end;
    int tile_width = 1024;
    int num_block = ceil( (float) subFrameLength / (float) tile_width);
    dim3 block(tile_width, 1, 1);
    dim3 grid(num_block, 1, 1);
    unsigned int dynamic_sharedMem_size = 3 * num_block * sizeof(float);


    printf("num_block: %d\n", num_block);



    cudaError_t err; 

    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    cudaEventRecord(launch_begin,0);
    int * h_cx = (int *) malloc(sizeof(int) * num_objects);
    int * h_cy = (int *) malloc(sizeof(int) * num_objects);
    h_cx[obj_id] = (*cx)[obj_id];
    h_cy[obj_id] = (*cy)[obj_id];
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

      while(gpuDistance(prevX, prevY, h_cx[obj_id], h_cy[obj_id]) > 1){
        if(shouldPrint)
          printf("PrevX vs NewX(%d, %d) and PrevY vs NewY(%d, %d)\n", prevX, h_cx[obj_id], prevY, h_cy[obj_id]);
        prevX = h_cx[obj_id];
        prevY = h_cy[obj_id];

        gpuBlockReduce<<< grid, block >>>(obj_id, ds.d_frame, d_out, subFrameLength, num_block, abs_width, sub_width, sub_height, ds.d_row_offset, ds.d_col_offset);
        gpuFinalReduce<<< 1, num_block, dynamic_sharedMem_size >>>(obj_id, d_out, ds.d_cx, ds.d_cy, ds.d_row_offset, ds.d_col_offset, sub_width, sub_height, num_block);

        err =  cudaMemcpy(h_cx, ds.d_cx, sizeof(int) * num_objects, cudaMemcpyDeviceToHost);
        err =  cudaMemcpy(h_cy, ds.d_cy, sizeof(int) * num_objects, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    cudaEventElapsedTime(&time, launch_begin, launch_end);
    if(shouldPrint)
          printf("***** GPU FINISHED FRAME: PrevX vs NewX(%d, %d) and PrevY vs NewY(%d, %d)\n", prevX, h_cx[obj_id], prevY, h_cy[obj_id]);

  }
  else
    printf("Cannot launch kernel: num_block (%d) > tile_width (%d)\n", num_block, tile_width);

    (*cx)[obj_id] = h_cx[obj_id];
    (*cy)[obj_id] = h_cy[obj_id];

    cudaFree(d_out);
    free(h_out);
    free(h_cx);
    free(h_cy);

    if(shouldPrint)
      printf("Finished GPU Frame\n");
    return time;
}

int gpuDistance(int x1, int y1, int x2, int y2)
{
    int distx = (x2 - x1) * (x2 - x1);
    int disty = (y2 - y1) * (y2 - y1);
    
    double dist = sqrt(distx + disty);
    
    return (int) dist;
}


/******************************************** Multi-object tracking below ****************************************************/

//Notes for the morning:
//Draw out the mapping of it and trace through how it should look.
//None of this is tested. You need to also add the appropriate arrays in main and call this function from main. That hasn't been done yet.
//Also, I'd pass in the obj_converged array to the kernels, so that blocks not needed don't execute. (Down the road you can make that dynamic maybe...)


float launchMultiObjectTwoKernelReduction(int num_objects, int num_block, d_struct ds, unsigned char * frame, int frameLength, int frame_width, int ** cx, int ** cy, bool shouldPrint)
{
    int obj_cur = 0;	
	  cudaError_t err; 
    float time = 0;
    int tile_width = 1024;
    cudaEvent_t launch_begin, launch_end;

    int prevX[num_objects];
    int prevY[num_objects];
    bool * obj_converged = (bool *) malloc(sizeof(bool) * num_objects);

    for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
    {
    	prevX[obj_cur] = 0; 
   		prevY[obj_cur] = 0;
   		obj_converged[obj_cur] = false;
    }

    dim3 block(tile_width, 1, 1);
    dim3 grid(num_block, 1, 1);

    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    cudaEventRecord(launch_begin,0);

    //Copy new frame into device memory
    err = cudaMemcpy(ds.d_frame, frame, frameLength * sizeof(unsigned char), cudaMemcpyHostToDevice);

    
  if(num_block <= tile_width)
  {

     while(  ! gpuMultiObjectConverged(num_objects, *cx, *cy, prevX, prevY, &obj_converged, shouldPrint) )
     {

      	for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
      	{
        	prevX[obj_cur] = (*cx)[obj_cur];
        	prevY[obj_cur] = (*cy)[obj_cur];
        }

        gpuMultiObjectBlockReduce<<< grid, block >>>(ds.d_obj_block_ends, num_objects, ds.d_frame, ds.d_out, ds.d_sub_lengths, num_block, frame_width, ds.d_sub_widths, ds.d_row_offset, ds.d_col_offset);
        gpuMultiObjectFinalReduce<<< num_objects, num_block, num_block * 3 * sizeof(float) >>>(ds.d_obj_block_ends, num_objects, ds.d_out, ds.d_cx, ds.d_cy, ds.d_row_offset, ds.d_col_offset, ds.d_sub_widths, ds.d_sub_heights, num_block);
     
        err =  cudaMemcpy(*cx, ds.d_cx, sizeof(int) * num_objects, cudaMemcpyDeviceToHost);
        err =  cudaMemcpy(*cy, ds.d_cy, sizeof(int) * num_objects, cudaMemcpyDeviceToHost);
   }
    cudaDeviceSynchronize();
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    cudaEventElapsedTime(&time, launch_begin, launch_end);
    
    if(shouldPrint)
    	 for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
          	printf("***** GPU FINISHED FRAME: Prev->(%d, %d) and New ->(%d, %d)\n", prevX[obj_cur], prevY[obj_cur], (*cx)[obj_cur],  (*cy)[obj_cur]);
   
  }
  else
    printf("Cannot launch kernel: num_block (%d) > tile_width (%d)\n", num_block, tile_width);
	
    free(obj_converged);
    if(shouldPrint)
      printf("Finished GPU Frame\n");
    return time;
}

bool gpuMultiObjectConverged(int num_objects, int * cx, int * cy, int * prevX, int * prevY, bool ** obj_converged, bool shouldPrint)
{
	int obj_cur;
	int total = 0;
	for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
	{
		if(!(*obj_converged)[obj_cur]) //object has not converged yet
		{
 			if(gpuDistance(prevX[obj_cur], prevY[obj_cur], cx[obj_cur], cy[obj_cur]) <= 1) //has converged
 			{
 				(*obj_converged)[obj_cur] = true; //set to converged
 				total ++;
 			}
		}
		else
			total ++;
		if(shouldPrint)
			printf("PrevX vs NewX(%d, %d) and PrevY vs NewY(%d, %d)\n", prevX[obj_cur], cx[obj_cur], prevY[obj_cur], cy[obj_cur]);
	}
	if(total == num_objects) //All objects have finished converging
		return true;
	else
		return false;
}





float launchMultiObjectTwoKernelCamShift(int num_objects, int num_block,  int * obj_block_ends, d_struct ds, unsigned char * frame, int frameLength, int frame_width, int ** cx, int ** cy, int **sub_widths, int ** sub_heights, int * sub_lengths, bool shouldPrint)
{
    int obj_cur = 0;  
    cudaError_t err; 
    float time = 0;
    int tile_width = 1024;
    cudaEvent_t launch_begin, launch_end;

    int prevX[num_objects];
    int prevY[num_objects];
    bool * obj_converged = (bool *) malloc(sizeof(bool) * num_objects);


    for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
    {
      prevX[obj_cur] = 0; 
      prevY[obj_cur] = 0;
      obj_converged[obj_cur] = false;
    }

    dim3 block(tile_width, 1, 1);
    dim3 grid(num_block, 1, 1);

    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    cudaEventRecord(launch_begin,0);

    //Copy new frame into device memory
    err = cudaMemcpy(ds.d_frame, frame, frameLength * sizeof(unsigned char), cudaMemcpyHostToDevice);

    
  if(num_block <= tile_width)
  {

     while(  ! gpuMultiObjectConverged(num_objects, *cx, *cy, prevX, prevY, &obj_converged, shouldPrint))
     {

        for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
        {
          prevX[obj_cur] = (*cx)[obj_cur];
          prevY[obj_cur] = (*cy)[obj_cur];
        }

        gpuMultiObjectBlockReduce<<< grid, block >>>(ds.d_obj_block_ends, num_objects, ds.d_frame, ds.d_out, ds.d_sub_lengths, num_block, frame_width, ds.d_sub_widths, ds.d_row_offset, ds.d_col_offset);
        gpuCamShiftMultiObjectFinalReduce<<< num_objects, num_block, num_block * 3 * sizeof(float) >>>(ds.d_obj_block_ends, num_objects, ds.d_out, ds.d_cx,  ds.d_cy, ds.d_sub_lengths, ds.d_row_offset, ds.d_col_offset, ds.d_sub_widths, ds.d_sub_heights, num_block);
     
        err =  cudaMemcpy(*cx, ds.d_cx, sizeof(int) * num_objects, cudaMemcpyDeviceToHost);
        err =  cudaMemcpy(*cy, ds.d_cy, sizeof(int) * num_objects, cudaMemcpyDeviceToHost);
        err =  cudaMemcpy(sub_lengths, ds.d_sub_lengths, sizeof(int) * num_objects, cudaMemcpyDeviceToHost);
        num_block = 0;

        for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
        {
          num_block += ceil( (float) sub_lengths[obj_cur] / (float) tile_width);
          
           obj_block_ends[obj_cur] = num_block;

         //printf("SUB LENGTHS: %d num_block: %d\n", sub_lengths[obj_cur], num_block);
        }
       // printf("*********NEXT Iteration***************\n");
        err =  cudaMemcpy(ds.d_obj_block_ends, obj_block_ends, sizeof(int) * num_objects, cudaMemcpyHostToDevice);

        grid = dim3(num_block, 1, 1);

   }
    cudaDeviceSynchronize();
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    cudaEventElapsedTime(&time, launch_begin, launch_end);
    
    if(shouldPrint)
       for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
            printf("***** GPU FINISHED FRAME: Prev->(%d, %d) and New ->(%d, %d)\n", prevX[obj_cur], prevY[obj_cur], (*cx)[obj_cur],  (*cy)[obj_cur]);
   

    err =  cudaMemcpy(*sub_widths, ds.d_sub_widths, sizeof(int) * num_objects, cudaMemcpyDeviceToHost);
    err =  cudaMemcpy(*sub_heights, ds.d_sub_heights, sizeof(int) * num_objects, cudaMemcpyDeviceToHost);

  }
  else
    printf("Cannot launch kernel: num_block (%d) > tile_width (%d)\n", num_block, tile_width);
  
    free(obj_converged);
    if(shouldPrint)
      printf("Finished GPU Frame\n");
    return time;
}
























