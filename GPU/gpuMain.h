#ifndef GPUMAIN_H
#define GPUMAIN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <math.h>

struct device_struct
{
    unsigned int *  d_cx;
    unsigned int *  d_cy;
    unsigned int *  d_topX;
    unsigned int *  d_topY;
    unsigned int *  d_bottomX;
    unsigned int *  d_bottomY;
    unsigned int *  d_sub_widths;
    unsigned int *  d_sub_heights;
    unsigned int *  d_sub_lengths;
    unsigned int *  d_block_ends;
    unsigned char * d_frame;
    unsigned char * d_bgr;
}; typedef struct device_struct d_struct;

struct host_struct
{
    unsigned int *  h_cx;
    unsigned int *  h_cy;
    unsigned int *  h_topX;
    unsigned int *  h_topY;
    unsigned int *  h_bottomX;
    unsigned int *  h_bottomY;
}; typedef struct host_struct h_roi;

h_roi * initHostROI(int num_objects);

void freeHostROI(h_roi * roi);

float  launchGPU_BGR_to_Hue(unsigned char * bgr, d_struct ds, int total);

void printDeviceProperties();

void timeMemoryTransfer();

float initDeviceStruct(
unsigned int num_objects,
d_struct * ds,
h_roi * roi,
unsigned char * frame,
unsigned int frameLength,
unsigned int * subFrameLengths,
unsigned int * sub_widths,
unsigned int * sub_heights);

void freeDeviceStruct(d_struct * ds);

void mainConstantMemoryHistogramLoad(float * histogram, unsigned int num_objects);

float gpuCamShift(
d_struct ds,
h_roi * roi,
unsigned int num_objects,
unsigned char * frame,
unsigned int frame_length,
bool adjust_window);



#endif

