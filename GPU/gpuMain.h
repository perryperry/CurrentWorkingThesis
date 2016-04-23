#ifndef GPUMAIN_H
#define GPUMAIN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <math.h>

//#include "gpuReductionKernels.h"

struct device_struct {
    int * d_cx;
    int * d_cy;
    int * d_col_offset;
    int * d_row_offset;
    unsigned char * d_frame;
}; typedef struct device_struct d_struct;

void initDeviceStruct(d_struct * ds, unsigned char * frame, int frameLength, int * cx, int * cy, int * col_offset, int * row_offset);

void freeDeviceStruct(d_struct * ds);

void reverseIt(float * histogram);

int gpuReduceMain(int blockWidth, float * M00, float * M1x, float * M1y, int length, int * xc, int * yc);

float * fillArray(int n);

void printArray(float *arr, int n);

float cpuReduce(float * h_in, int n);

void gpuBackProjectMain(unsigned char * hueArray, int hueLength, float * histogram, int width, int xOffset, int yOffset, float ** h_M00, float ** h_M1x, float ** h_M1y);


void bpTest(unsigned char * hueArray, int ** convertArray, int hueLength);

void mainConstantMemoryHistogramLoad(float * histogram);

int launchMeanShiftKernelForSubFrame(unsigned char * hueFrame, int hueFrameLength, int width, int xOffset, int yOffset, int * cx, int * cy);

float launchMeanShiftKernelForEntireFrame(unsigned char * hueFrame, int hueFrameLength, int hueSubFrameLength, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset, int * cx, int * cy);

float launchSingleKernelEntireFrame(unsigned char * hueFrame, int hueFrameLength, int hueSubFrameLength, int abs_width, int sub_width, int sub_height, int * row_offset, int * col_offset, int * cx, int * cy);


float testThat(d_struct ds, unsigned char * frame, int frameLength, int subFrameLength, int abs_width, int sub_width, int sub_height, int * cx, int * cy, bool shouldPrint);



#endif

