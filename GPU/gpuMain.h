#ifndef GPUMAIN_H
#define GPUMAIN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <math.h>

void reverseIt(float * histogram);

int gpuMain(int blockWidth, char p, unsigned char * hueArray, int length);

float * fillArray(int n);

void printArray(float *arr, int n);

float cpuReduce(float * h_in, int n);

void usage();

float * gpuBackProjectMain(int * hueArray, int hueLength, float * histogram);

#endif

