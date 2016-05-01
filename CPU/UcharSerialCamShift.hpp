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
    void createHistogram(unsigned char * entireHueArray, int step, RegionOfInterest cpu_objects[], float ** histogram, int num_objects);

    void printHistogram(float * histogram, int length);
    
    void backProjectHistogram(unsigned char * hsv, int step, Mat * frame, RegionOfInterest roi, float * histogram);
    
    float cpu_entireFrameMeanShift(unsigned char * hueArray, int step, RegionOfInterest cpu_objects, int obj_index, float * histogram, bool shouldPrint, int * cpu_cx, int * cpu_cy);
    
    float cpuCamShift(unsigned char * hueArray, int step, RegionOfInterest cpu_objects, int obj_index, float * histogram, bool shouldPrint, int * cpu_cx, int * cpu_cy, int * width, int * height);
};

int distance(int x1, int y1, int x2, int y2);

#endif /* SerialCamShift_hpp */
