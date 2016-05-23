//
//  SerialCamShift.hpp
//  ThesisSerialCamShift
//
//  Created by Matthew Perry on 1/28/16.
//  Copyright © 2016 Matthew Perry. All rights reserved.
//

#ifndef UcharSerialCamShift_hpp
#define UcharSerialCamShift_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "RegionOfInterest.hpp"
#include <ctime>
#include <ratio>
#include <chrono>
#include <math.h>

using namespace std::chrono;
#define BUCKET_WIDTH 3
#define BUCKETS 60

class SerialCamShift
{

public:
    void createHistogram(unsigned char * entireHueArray, unsigned int step, RegionOfInterest cpu_objects[], float ** histogram, unsigned int num_objects);

    void printHistogram(float * histogram, unsigned int length);
    
    void backProjectHistogram(unsigned char * hsv, unsigned int step, Mat * frame, RegionOfInterest roi, float * histogram);
    
    float cpuMeanShift(unsigned char * hueArray, unsigned int step, RegionOfInterest cpu_objects, unsigned int obj_index, float * histogram, bool shouldPrint, unsigned int * cpu_cx, unsigned int * cpu_cy);
    
    float cpuCamShift(unsigned char * hueArray, unsigned int step, RegionOfInterest cpu_objects, unsigned int obj_index, float * histogram, bool shouldPrint, unsigned int * cpu_cx, unsigned int * cpu_cy, unsigned int * width, unsigned int * height, unsigned int hueLength, float * angle);
};

unsigned int distance(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

#endif /* SerialCamShift_hpp */
