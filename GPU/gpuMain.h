#ifndef GPUMAIN_H
#define GPUMAIN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <math.h>

//#include "gpuReductionKernels.h"

void reverseIt(float * histogram);

int gpuReduceMain(int blockWidth, float * M00, float * M1x, float * M1y, int length, int * xc, int * yc);

float * fillArray(int n);

void printArray(float *arr, int n);

float cpuReduce(float * h_in, int n);

void gpuBackProjectMain(unsigned char * hueArray, int hueLength, float * histogram, int width, int xOffset, int yOffset, float ** h_M00, float ** h_M1x, float ** h_M1y);


void bpTest(unsigned char * hueArray, int ** convertArray, int hueLength);

void mainConstantMemoryHistogramLoad(float * histogram);

int launchMeanShiftKernelForSubFrame(unsigned char * hueFrame, int hueFrameLength, int width, int xOffset, int yOffset, int * cx, int * cy);

#endif

