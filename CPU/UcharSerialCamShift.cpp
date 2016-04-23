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
    }
     printf("\n********** FINISHED PRINTING HISTOGRAM **********\n");
}

void SerialCamShift::createHistogram(unsigned char * hueArray, RegionOfInterest roi, float ** histogram)
{
    //cout << "Creating Histogram"<< endl;
    int hue = 0;
    int width = roi._width;
    int height = roi._height;
    float total_pixels = (float) roi.getTotalPixels();
    
   // cout << "Total pixels: " << total_pixels << endl;
    
    for(int row = 0; row < height; row ++)
    {
        for(int col = 0; col < width; col++)
        {
            hue = hueArray[ width * row + col ];
           // cout << "hue: " << hue << " ";
            int index = hue / BUCKET_WIDTH;
           // cout << "index: " << index << endl;
            
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

float SerialCamShift::cpu_entireFrameMeanShift(unsigned char * hueArray, int step, RegionOfInterest * roi, float * histogram, bool shouldPrint)
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
    
    Point topLeft = (*roi).getTopLeft();
    Point bottomRight = (*roi).getBottomRight();

    while(converging)
    {
        prevX = (*roi).getCenterX();
        prevY = (*roi).getCenterY();
        
        for(int col = (*roi).getTopLeftX(); col < (*roi).getBottomRightX();col++)
        {
            for(int row = (*roi).getTopLeftY(); row < (*roi).getBottomRightY();row++)
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
            (*roi).setCentroid(Point(xc, yc));
        }
        else
            return 0.0;
        //if(prevX - xc < 1 && prevX - xc > -1  && prevY - yc < 1 && prevY - yc > -1)
        if(prevX == xc && prevY == yc)
            converging = false;
        else
        {
            prevX = xc;
            prevY = yc;
        }
        
        if(shouldPrint){
            printf("Inside CPU MeanShift ---> M00 = %lf M1x = %lf M1y = %lf \n", M00, M1x, M1y);
            printf("Inside CPU MeanShift ---> centroid:(%d, %d), topX:%d, topY:%d\n", xc, yc, (*roi).getTopLeftX(), (*roi).getTopLeftY());
        }
        M00 = 0.0;
        M1x = 0.0;
        M1y = 0.0;
        
    }//end of converging
    // fclose (pFileTXT);
    time2 = high_resolution_clock::now();
    auto cpu_duration = duration_cast<duration<double>>( time2 - time1 ).count();
    return (float)(cpu_duration * 1000.0); //convert to milliseconds
 }