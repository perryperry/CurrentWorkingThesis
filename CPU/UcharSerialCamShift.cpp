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
RegionOfInterest ** roi,
unsigned int obj_index,
float * histogram,
bool shouldPrint,
unsigned int hueLength, //frame total
float * angle,
bool adjustWindow)
{
    unsigned int width = 0, height = 0;
    high_resolution_clock::time_point time1;
    high_resolution_clock::time_point time2;
    double M00 = 0.0, M1x = 0.0, M1y = 0.0, M2x = 0.0, M2y = 0.0, probability = 0.0, ratio = 0.0;
    unsigned int hue = 0, prevX = 0, prevY = 0, cx = 0, cy = 0, maxHue = 0, count = 0;
    unsigned int LOST_OBJECT = 20;
    time1 = high_resolution_clock::now();

    unsigned int obj_offset = obj_index * BUCKETS; //offset to the next object's segment of the histogram
    
    while(1)
    {
       // (*roi)[obj_index].setWindowToFullFrame();
        width  = (*roi)[obj_index]._width;
        height = (*roi)[obj_index]._height;
        M00 = 0.0;
        M1x = 0.0;
        M1y = 0.0;
        M2x = 0.0;
        M2y = 0.0;
        prevX = cx;
        prevY = cy;
     
        for(unsigned int col = (*roi)[obj_index].getTopLeftX(); col < (*roi)[obj_index].getBottomRightX();col++)
        {
            for(unsigned int row = (*roi)[obj_index].getTopLeftY(); row < (*roi)[obj_index].getBottomRightY();row++)
            {
                if(step * row + col < hueLength)
                {
                    hue = hueArray[step * row + col];
                    probability = histogram[(hue / BUCKET_WIDTH) + obj_offset];
                    M00 += probability;
                    M1x += ((float)col ) * probability;
                    M1y += ((float)row) * probability;
                    //M2x += ((float)col * col ) * probability;
                   // M2y += ((float)row * row ) * probability;
                }
                else{
                    printf("Error in cpuCAMShift, out of bounds: %d %d\n", (*roi)[obj_index].getBottomRightX(),  (*roi)[obj_index].getBottomRightY());
                }
            }
        }
        if(M00 > 0){//Can't divide by zero...
            
            cx = (int)(M1x / M00);
            cy = (int)(M1y / M00);
            
            if(adjustWindow)
            {
                width = ceil(2 * sqrt(M00));
            
                if(width < 20)
                    width = 200;
                height = ceil(width * 1.1);
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
            printf("Inside CPU MeanShift ---> centroid:(%d, %d), topX:%d, topY:%d\n",  cx,  cy,  (*roi)[obj_index].getTopLeftX(),  (*roi)[obj_index].getTopLeftY());
        }
        (*roi)[obj_index].setCentroid(Point(cx,cy));
        (*roi)[obj_index].setCorners(Point(cx, cy), width, height);
        
        if(count > LOST_OBJECT)
        {
            (*roi)[obj_index].setWindowToFullFrame();
            count = 0;
        }
        if( distance(prevX, prevY, cx, cy) <= 1)
            break;
        count++;
    }//end of converging

    if(shouldPrint)
        printf("************* CPU FINISHED A FRAME FOR OBJECT %d ***********\n", obj_index);
    time2 = high_resolution_clock::now();
    auto cpu_duration = duration_cast<duration<double>>( time2 - time1 ).count();
    return (float)(cpu_duration * 1000.0); //convert to milliseconds
}