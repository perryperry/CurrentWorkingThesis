#ifndef GPUMAIN_H
#define GPUMAIN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <math.h>

struct device_struct {
    float * d_out;
    int * d_cx;
    int * d_cy;
    int * d_col_offset;
    int * d_row_offset;
    int * d_sub_lengths;
    int * d_sub_widths;
    int * d_sub_heights;
    int * d_obj_block_ends;
    unsigned char * d_frame;
}; typedef struct device_struct d_struct;

void timeMemoryTransfer();

int initDeviceStruct(int num_objects, d_struct * ds, int * obj_block_ends, unsigned char * frame, int frameLength, int * cx, int * cy, int * col_offset, int * row_offset, int * subFrameLengths, int * sub_widths, int * sub_heights);

void freeDeviceStruct(d_struct * ds);

void reverseIt(float * histogram);

float * fillArray(int n);

void printArray(float *arr, int n);

float cpuReduce(float * h_in, int n);

void mainConstantMemoryHistogramLoad(float * histogram, int num_objects);

float launchTwoKernelReduction(int obj_id, int num_objects, d_struct ds, unsigned char * frame, int frameLength, int subFrameLength, int abs_width, int sub_width, int sub_height, int ** cx, int ** cy, bool shouldPrint);

int gpuDistance(int x1, int y1, int x2, int y2);

/******************************************** Multi-object tracking below ****************************************************/

float launchMultiObjectTwoKernelReduction(int num_objects, int num_block, d_struct ds, unsigned char * frame, int frameLength, int frame_width, int ** cx, int ** cy, bool shouldPrint);

bool gpuMultiObjectConverged(int num_objects, int * cx, int * cy, int * prevX, int * prevY, bool ** obj_converged, bool shouldPrint);


float launchMultiObjectTwoKernelCamShift(int num_objects, int * num_block,  int * obj_block_ends, d_struct ds, unsigned char * frame, int frameLength, int frame_width, int ** cx, int ** cy, int ** sub_widths, int ** sub_heights, int * sub_lengths, bool shouldPrint);

#endif

