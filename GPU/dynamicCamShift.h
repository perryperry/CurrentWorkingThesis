#ifndef DYNAMICCAMSHIFT_H
#define DYNAMICCAMSHIFT_H

#include <stdio.h>
#include <stdlib.h>
#include <math_functions.h>

#define MAX_OBJECTS 3
#define TILE_WIDTH 1024
//1024 * 1024 * 3 <--(max blocks, threads per block, 3 moments of camshift)
#define MAX_OUTPUT TILE_WIDTH * 3
#define SHARED_SIZE_LIMIT 1024
#define HIST_BUCKETS 60
#define FRAME_WIDTH 1080
#define FRAME_HEIGHT 720
#define FRAME_TOTAL FRAME_WIDTH * FRAME_HEIGHT
#define LOST_OBJECT 20
__constant__ float const_histogram[HIST_BUCKETS * MAX_OBJECTS];

/************** Static Device Memory ********************/

__device__ float g_output1[MAX_OUTPUT];
__device__ float g_output2[MAX_OUTPUT];

__device__ bool  g_converged[MAX_OBJECTS];

/********************************************************/

__host__ void setConstantMemoryHistogram(float * histogram);

/********************************************************/

__device__ unsigned int gpuDistance(
unsigned int x1,
unsigned int y1,
unsigned int x2,
unsigned int y2);

/********************************************************/

__device__ bool converged(unsigned int num_objects);

/********************************************************/

__device__ unsigned int getObjID(unsigned int * block_ends, unsigned int num_objects);

/********************************************************/

__device__ unsigned int getBlockStart(unsigned int * block_ends, unsigned int num_objects);

/********************************************************/

__device__ void warpReduce(
volatile float* shared_M00,
volatile float* shared_M1x,
volatile float* shared_M1y,
unsigned int tid,
unsigned int num_block);

/********************************************************/

__global__ void gpuBGRtoHue(unsigned char * bgr, unsigned char * hueArray, int total);

/********************************************************/

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
bool adjust_window);

/********************************************************/

__global__ void blockReduce(
unsigned int obj_cur,
unsigned int num_block,
unsigned char * frame,
unsigned int * sub_widths,
unsigned int * sub_totals,
unsigned int * row_offset,
unsigned int * col_offset,
float * output);

/********************************************************/

__global__ void finalReduce(float * output, unsigned int num_block);

/********************************************************/

__device__ void adjustWindow(
double M00,
int newX,
int newY,
int * newRowOffset,
int * newColOffset,
unsigned int * sub_widths,
unsigned int * sub_heights,
unsigned int * sub_totals,
int objectID);

#endif