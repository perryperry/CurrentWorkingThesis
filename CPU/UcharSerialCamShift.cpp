////
//  SerialCamShift.cpp
//  ThesisSerialCamShift
//
//  Created by Matthew Perry on 1/28/16.
//  Copyright © 2016 Matthew Perry. All rights reserved.
//
#include "UcharSerialCamShift.hpp"
#define PI 3.14159265

void SerialCamShift::createHistogram(unsigned char * entireHueArray, unsigned int step, RegionOfInterest cpu_objects[], float ** histogram, unsigned int num_objects)
{
    unsigned int hue = 0;
    float total_pixels = 0;
    unsigned int index = 0;
    unsigned int obj_offset = 0;

    for(unsigned int obj_cur = 0; obj_cur < num_objects; obj_cur++)
    {
        obj_offset = obj_cur * BUCKETS; //offset to the next object's segment of the histogram
        total_pixels = (float) cpu_objects[obj_cur].getTotalPixels();
        //printf("OFFSET: %d\n",obj_offset);
        for(unsigned int col = cpu_objects[obj_cur].getTopLeftX(); col < cpu_objects[obj_cur].getBottomRightX(); col++)
        {
            for(unsigned int row = cpu_objects[obj_cur].getTopLeftY(); row < cpu_objects[obj_cur].getBottomRightY(); row++)
            {
                hue = entireHueArray[ step * row + col ];
                index = hue / BUCKET_WIDTH;
                (*histogram)[index + obj_offset] ++;
            }
        }
        for(unsigned int i = obj_offset; i < BUCKETS + obj_offset; i++)//convert to probability
        {
            (*histogram)[i] /= total_pixels;
        }
    }
}

void SerialCamShift::backProjectHistogram(unsigned char * hsv, unsigned int step, Mat * frame, RegionOfInterest roi, float * histogram)
{
    unsigned int hue = 0, count = 0;
     
    for(unsigned int col = roi.getTopLeftX(); col < roi.getBottomRightX();col++)
    {
        for(unsigned int row = roi.getTopLeftY(); row < roi.getBottomRightY();row++)
        {
            int hue = hsv[step * row + col];
        
            (*frame).at<Vec3b>( row, col)[0] = (int) (255.0 * ( histogram[hue / BUCKET_WIDTH]));
            (*frame).at<Vec3b>( row, col)[1] = (int) (255.0 * ( histogram[hue / BUCKET_WIDTH]));
            (*frame).at<Vec3b>( row, col)[2] = (int) (255.0 * ( histogram[hue / BUCKET_WIDTH]));
        }
    }
}

void SerialCamShift::printHistogram(float * histogram, unsigned int length)
{
    printf("********** PRINTING HISTOGRAM **********\n");
    unsigned int i = 0;
    for(i =0; i < length; i++)
    {
        printf("%d) %f, ", i, histogram[i]);
        if(i % 10 == 0)
            printf("\n");
        
        if(i == 59)
            printf("\n################### NEXT OBJECT ##################\n");
            
    }
     printf("\n********** FINISHED PRINTING HISTOGRAM **********\n");
}

unsigned int distance(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    unsigned int distx = (x2 - x1) * (x2 - x1);
    unsigned int disty = (y2 - y1) * (y2 - y1);
    
    double dist = sqrt(distx + disty);
   
    return (int) dist;
}

float SerialCamShift::cpuCamShift(
unsigned char * hueArray,
unsigned int step,
RegionOfInterest roi,
unsigned int obj_index,
float * histogram,
bool shouldPrint,
unsigned int * cpu_cx,
unsigned int * cpu_cy,
unsigned int * width,
unsigned int * height,
unsigned int hueLength, //frame total
float * angle,
bool adjustWindow)
{
    high_resolution_clock::time_point time1;
    high_resolution_clock::time_point time2;
    float M00 = 0.0, M1x = 0.0, M1y = 0.0, M2x = 0.0, M2y = 0.0, probability = 0.0, ratio = 0.0;
    unsigned int hue = 0, prevX = 0, prevY = 0, cx = 0, cy = 0, maxHue = 0;
  
    time1 = high_resolution_clock::now();
    
    Point topLeft = roi.getTopLeft();
    Point bottomRight = roi.getBottomRight();
    
    unsigned int obj_offset = obj_index * BUCKETS; //offset to the next object's segment of the histogram

    while(distance(prevX, prevY, cpu_cx[0], cpu_cy[0]) > 1)
    {
        M00 = 0.0;
        M1x = 0.0;
        M1y = 0.0;
        M2x = 0.0;
        M2y = 0.0;
        prevX = cpu_cx[0];
        prevY = cpu_cy[0];
     
        for(unsigned int col = roi.getTopLeftX(); col < roi.getBottomRightX();col++)
        {
            for(unsigned int row = roi.getTopLeftY(); row < roi.getBottomRightY();row++)
            {
                if(step * row + col < hueLength)
                {
                    hue = hueArray[step * row + col];
                    probability = histogram[(hue / BUCKET_WIDTH) + obj_offset];
                    M00 += probability;
                    M1x += ((float)col ) * probability;
                    M1y += ((float)row) * probability;
                    M2x += ((float)col * col ) * probability;
                    M2y += ((float)row * row ) * probability;
                }
                else{
                    printf("Problem: %d %d\n", roi.getBottomRightX(), roi.getBottomRightY());
                }
            }
        }
        if(M00 > 0){//Can't divide by zero...
            
            cx = (int)(M1x / M00);
            cy = (int)(M1y / M00);
            cpu_cx[0] = cx;
            cpu_cy[0] = cy;
            roi.setCentroid(Point( cpu_cx[0], cpu_cy[0] ));
     
            
            if(adjustWindow){
                *width = ceil(2 * sqrt(M00));
            
                if(*width < 20)
                    *width = 200;
            
                *height = ceil(*width * 1.1);
                roi.setWidthHeight(*width, *height);
            }
        }
        else
        {
            printf("Divided by zero, that's a problem... ");
            printf("Let's see: obj_offset: %d M00: %f\n", obj_offset, M00);
            return 1000000.0;//return an unrealistically large number to show something went wrong
        }
        
        if(shouldPrint){
            printf("Inside CPU MeanShift ---> M00 = %lf M1x = %lf M1y = %lf \n", M00, M1x, M1y);
            printf("Inside CPU MeanShift ---> centroid:(%d, %d), topX:%d, topY:%d\n",  cpu_cx[0],  cpu_cy[0], roi.getTopLeftX(), roi.getTopLeftY());
        }
        
    }//end of converging
    
  /*  float numerator = 2 * (( (M1x * M1y) / M00) - (cx * cy));
    float denomator = ((M2y / M00) - (cy * cy)) - ((M2x / M00) - (cx * cx)) ;
    
   *angle = atan(numerator / denomator) / 2.0;
    
    *angle = *angle * 180 / PI;
    
    
    printf("Angle: %f\n", *angle);*/
    
    
    
    
    
    if(shouldPrint)
        printf("************* CPU FINISHED A FRAME FOR OBJECT %d ***********\n", obj_index);
    time2 = high_resolution_clock::now();
    auto cpu_duration = duration_cast<duration<double>>( time2 - time1 ).count();
    return (float)(cpu_duration * 1000.0); //convert to milliseconds
}