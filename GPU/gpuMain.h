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
    unsigned int *  d_prevX;
    unsigned int *  d_prevY;
    unsigned int *  d_col_offset;
    unsigned int *  d_row_offset;
    unsigned int *  d_sub_widths;
    unsigned int *  d_sub_heights;
    unsigned int *  d_sub_lengths;
    unsigned int *  d_block_ends;
    unsigned char * d_frame;
    unsigned char * d_bgr;
}; typedef struct device_struct d_struct;


float  launchGPU_BGR_to_Hue(unsigned char * bgr, d_struct ds, int total);

void printDeviceProperties();

void timeMemoryTransfer();

unsigned int initDeviceStruct(
unsigned int num_objects,
d_struct * ds,
unsigned int * block_ends,
unsigned char * frame,
unsigned int frameLength,
unsigned int * cx,
unsigned int * cy,
unsigned int * col_offset,
unsigned int * row_offset,
unsigned int * subFrameLengths,
unsigned int * sub_widths,
unsigned int * sub_heights);

void freeDeviceStruct(d_struct * ds);

void mainConstantMemoryHistogramLoad(float * histogram, unsigned int num_objects);

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
bool adjust_window);



#endif

