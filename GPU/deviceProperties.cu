// CUDA Device Query
// All code in this file taken from http://cuda-programming.blogspot.com/2013/01/how-to-query-to-devices-in-cuda-cc.html

#include <stdio.h>
#include "deviceProperties.h"

//Get number of streaming processor cores 
int get_SP_Cores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128; 
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf(CYNA "Major revision number:");printf(YELLOW "         %d\n",  devProp.major);
    printf(CYNA "Minor revision number:");printf(YELLOW "         %d\n",  devProp.minor);
    printf(CYNA "Name:");printf(YELLOW "                          %s\n",  devProp.name);
    printf(CYNA "Total global memory:");printf(YELLOW "           %lu\n",  devProp.totalGlobalMem);
    printf(CYNA "Total shared memory per block:");printf(YELLOW " %lu\n",  devProp.sharedMemPerBlock);
    printf(CYNA "Total registers per block:");printf(YELLOW "     %d\n",  devProp.regsPerBlock);
    printf(CYNA "Warp size:");printf(YELLOW "                     %d\n",  devProp.warpSize);
    printf(CYNA "Maximum memory pitch:");printf(YELLOW "          %lu\n",  devProp.memPitch); // Maximum pitch in bytes allowed by memory copies
    printf(CYNA "Maximum threads per block:");printf(YELLOW "     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i){
    printf(CYNA "Maximum dimension %d of block:", i);
    printf(YELLOW "  %d\n", devProp.maxThreadsDim[i]);
  }
    for (int i = 0; i < 3; ++i){
      printf(CYNA "Maximum dimension %d of grid:", i);
      printf(YELLOW "   %d\n", devProp.maxGridSize[i]);
    }
    printf(CYNA "Clock rate:");printf(YELLOW "                    %d\n",  devProp.clockRate);
    printf(CYNA "Total constant memory:");printf(YELLOW "         %lu\n",  devProp.totalConstMem); //in bytes
    printf(CYNA "Texture alignment:");printf(YELLOW "             %lu\n",  devProp.textureAlignment);
    printf(CYNA "Concurrent copy and execution:");printf(YELLOW " %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf(CYNA "streaming multiprocessors:");printf(YELLOW "     %d\n",  devProp.multiProcessorCount); //streaming multi-processors? (i.e., sm's)
    printf(CYNA "Streaming processor cores:");printf(YELLOW "     %d\n",  get_SP_Cores(devProp));
    printf(CYNA "Kernel execution timeout:");printf(YELLOW "      %s\n" RESET,  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

