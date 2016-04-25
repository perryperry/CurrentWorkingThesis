//
//  SerialCamShift.cpp
//  ThesisSerialCamShift
//
//  Created by Matthew Perry on 1/28/16.
//  Copyright © 2016 Matthew Perry. All rights reserved.
//
#include <fstream>
#include <iostream>
#include "UcharSerialCamShift.hpp"


void SerialCamShift::createHistogramFullTest(unsigned char * entireHueArray, int step, RegionOfInterest cpu_objects[], float ** histogram, int num_objects)
{
    unsigned int hue = 0;
    float total_pixels = 0;
    unsigned int index = 0;
    unsigned int obj_offset = 0;

    for(unsigned int obj_cur = 0; obj_cur < num_objects; obj_cur++)
    {
        obj_offset = obj_cur * BUCKETS; //offset to the next object's segment of the histogram
        total_pixels = (float) cpu_objects[obj_cur].getTotalPixels();
        printf("OFFSET: %d\n",obj_offset);
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

void SerialCamShift::backProjectHistogram(unsigned char * hsv, int step, Mat * frame, RegionOfInterest roi, float * histogram)
{
    int hue = 0, count = 0;
    for(int col = 0; col < roi._width; col ++)
    {
        for(int row = 0; row < roi._height; row++)
        {
            hue = hsv[roi._width * row + col];
            (*frame).at<Vec3b>( row + roi.getTopLeftY(), col + roi.getTopLeftX() )[0] = (int) (255 * histogram[hue / BUCKET_WIDTH]);
            (*frame).at<Vec3b>( row + roi.getTopLeftY(), col + roi.getTopLeftX() )[1] = (int) (255 * histogram[hue / BUCKET_WIDTH]);
            (*frame).at<Vec3b>( row + roi.getTopLeftY(), col + roi.getTopLeftX() )[2] = (int) (255 * histogram[hue / BUCKET_WIDTH]);
        }
    }
}

void SerialCamShift::printHistogram(float * histogram, int length)
{
    printf("********** PRINTING HISTOGRAM **********\n");
    int i = 0;
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

void SerialCamShift::createHistogram(unsigned char * hueArray, RegionOfInterest roi, float ** histogram, int num_objects)
{
    int hue = 0;
    int width = roi._width;
    int height = roi._height;
    float total_pixels = (float) roi.getTotalPixels();
    
    for(int row = 0; row < height; row ++)
    {
        for(int col = 0; col < width; col++)
        {
            hue = hueArray[ width * row + col ];
            int index = hue / BUCKET_WIDTH;
            (*histogram)[index] ++;
        }
    }
    for(int i =0; i < BUCKETS; i++)//convert to probability
    {
        (*histogram)[i] /= total_pixels;
    }
   // cout << "END OF HISTOGRAM **********" << endl;
}

bool SerialCamShift::subMeanShift(unsigned char * hueArray, RegionOfInterest * roi, float * histogram, int * prevX, int * prevY)
{
  //  FILE * pFileTXT;
   // pFileTXT = fopen ("sub.txt","a");
    
    double M00 = 0.0, M1x = 0.0, M1y = 0.0;
    int xc = 0;
    int yc = 0;
    int hue = 0;
    float probability = 0.0;
    int yOffset = (*roi).getTopLeftY();
    int xOffset = (*roi).getTopLeftX();
    int width = (*roi)._width;
    int height = (*roi)._height;
    int count = 0;
    
    for(int row = 0; row < height; row ++)
    {
        for(int col = 0; col < width; col++)
        {
            hue = hueArray[ width * row + col ];
            probability = histogram[hue / BUCKET_WIDTH];
            M00 += probability;
            
            M1x += ((float)(col + xOffset)) * probability;
            M1y += ((float)(row + yOffset)) * probability;
            
            // fprintf (pFileTXT, "%d\n", hue);
            
         //   cout << "HUE AT (" << row + xOffset << ", " << col + yOffset << ") is " << hue << endl;
        }
         //printf("NEW--> %d %d %d %d\n", hue, (int)M00, (int)M1x,(int) M1y);
    }
 // printf("Inside CPU MeanShift ---> M00 = %lf M1x = %lf M1y = %lf \n", M00, M1x, M1y);
   //  fclose (pFileTXT);
    if(M00 > 0){//Can't divide by zero...
        
        xc = (int) (M1x / M00);
        yc = (int) (M1y / M00);
        (*roi).setCentroid(Point(xc, yc));
        
     // printf("Inside CPU MeanShift ---> centroid (%d, %d)  topX, topY (%d,%d)\n", xc, yc, (*roi).getTopLeftX(), (*roi).getTopLeftY());
    }

    if(*prevX - xc < 1 && *prevX - xc > -1  && *prevY - yc < 1 && *prevY - yc > -1)
    {
        (*roi).setCentroid(Point(xc, yc));
        return false;
    }
    else
    {
        (*roi).setCentroid(Point(xc, yc));
        *prevX = xc;
        *prevY = yc;
        return true;
    }
}

/*float SerialCamShift::cpu_entireFrameMeanShift(unsigned char * hueArray, int step, RegionOfInterest roi, float * histogram, bool shouldPrint, int * cpu_cx, int * cpu_cy)
{
   // FILE * pFileTXT;
  ///  pFileTXT = fopen ("entire.txt","a");
    high_resolution_clock::time_point time1;
    high_resolution_clock::time_point time2;
    double M00 = 0.0, M1x = 0.0, M1y = 0.0;
    int xc = 0;
    int yc = 0;
    int hue = 0;
    float probability = 0.0;
    bool converging = true;
    
    int prevX = 0;
    int prevY = 0;
    
   time1 = high_resolution_clock::now();
    
    Point topLeft = roi.getTopLeft();
    Point bottomRight = roi.getBottomRight();

    while(converging)
    {
        prevX = roi.getCenterX();
        prevY = roi.getCenterY();
        
        for(int col = roi.getTopLeftX(); col < roi.getBottomRightX();col++)
        {
            for(int row = roi.getTopLeftY(); row < roi.getBottomRightY();row++)
            {
                hue = hueArray[ step * row + col ];
                probability = histogram[hue / BUCKET_WIDTH];
                M00 += probability;
                M1x += ((float)col) * probability;
                M1y += ((float)row) * probability;
                 // fprintf (pFileTXT, "%d\n", hue);
            }
             //printf("OTHER NEW--> %d %d %d %d\n", hue, (int)M00, (int)M1x,(int) M1y);
        }
        
        if(M00 > 0){//Can't divide by zero...
            xc = (int)((int)M1x / (int)M00);
            yc = (int)((int)M1y / (int)M00);
            roi.setCentroid(Point(xc, yc));
        }
        else
            return 0.0;
 
        if(distance(prevX, prevY, xc, yc) <= 1)
            converging = false;
        else
        {
            prevX = xc;
            prevY = yc;
        }
        
        if(shouldPrint){
            printf("Inside CPU MeanShift ---> M00 = %lf M1x = %lf M1y = %lf \n", M00, M1x, M1y);
            printf("Inside CPU MeanShift ---> centroid:(%d, %d), topX:%d, topY:%d\n", xc, yc, roi.getTopLeftX(), roi.getTopLeftY());
        }
        M00 = 0.0;
        M1x = 0.0;
        M1y = 0.0;
        
    }//end of converging
    // fclose (pFileTXT);
      if(shouldPrint)
          printf("************* CPU FINISHED A FRAME *********** \n");
    time2 = high_resolution_clock::now();
    auto cpu_duration = duration_cast<duration<double>>( time2 - time1 ).count();
    
    *cpu_cx = xc;
    *cpu_cy = yc;
    return (float)(cpu_duration * 1000.0); //convert to milliseconds
 }*/

int distance(int x1, int y1, int x2, int y2)
{
    int distx = (x2 - x1) * (x2 - x1);
    int disty = (y2 - y1) * (y2 - y1);
    
    double dist = sqrt(distx + disty);
    
    return (int) dist;
}

float SerialCamShift::cpu_entireFrameMeanShift(unsigned char * hueArray, int step, RegionOfInterest roi, int obj_index, float * histogram, bool shouldPrint, int * cpu_cx, int * cpu_cy)
{
    high_resolution_clock::time_point time1;
    high_resolution_clock::time_point time2;
    double M00 = 0.0, M1x = 0.0, M1y = 0.0;
    int hue = 0;
    float probability = 0.0;
    bool converging = true;
    
    int prevX = 0;
    int prevY = 0;
    
    time1 = high_resolution_clock::now();
    
    Point topLeft = roi.getTopLeft();
    Point bottomRight = roi.getBottomRight();
    int obj_offset = obj_index * BUCKETS; //offset to the next object's segment of the histogram
   
    while(converging)
    {
        prevX = roi.getCenterX();
        prevY = roi.getCenterY();
        
        for(int col = roi.getTopLeftX(); col < roi.getBottomRightX();col++)
        {
            for(int row = roi.getTopLeftY(); row < roi.getBottomRightY();row++)
            {
                hue = hueArray[ step * row + col ];
                probability = histogram[(hue / BUCKET_WIDTH) + obj_offset];
                M00 += probability;
                M1x += ((float)col) * probability;
                M1y += ((float)row) * probability;
            }
        }
        
        if(M00 > 0){//Can't divide by zero...
            *cpu_cx = (int)((int)M1x / (int)M00);
            *cpu_cy = (int)((int)M1y / (int)M00);
            roi.setCentroid(Point(*cpu_cx, *cpu_cy));
        }
        else
        {
            printf("Divided by zero, that's a problem...");
            printf("Let's see: obj_offset: %d M00: %lf\n", obj_offset, M00);
            return 1000000.0;//return an unrealistically large number to show something went wrong
        }
        if(distance(prevX, prevY,*cpu_cx, *cpu_cy) <= 1)
            converging = false;
        else
        {
            prevX = *cpu_cx;
            prevY = *cpu_cy;
        }
        
        if(shouldPrint){
            printf("Inside CPU MeanShift ---> M00 = %lf M1x = %lf M1y = %lf \n", M00, M1x, M1y);
            printf("Inside CPU MeanShift ---> centroid:(%d, %d), topX:%d, topY:%d\n", *cpu_cx, *cpu_cy, roi.getTopLeftX(), roi.getTopLeftY());
        }
        M00 = 0.0;
        M1x = 0.0;
        M1y = 0.0;
        
    }//end of converging
    if(shouldPrint)
        printf("************* CPU FINISHED A FRAME *********** \n");
    time2 = high_resolution_clock::now();
    auto cpu_duration = duration_cast<duration<double>>( time2 - time1 ).count();
    return (float)(cpu_duration * 1000.0); //convert to milliseconds
}
