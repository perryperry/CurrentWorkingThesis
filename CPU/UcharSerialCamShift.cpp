//
//  SerialCamShift.cpp
//  ThesisSerialCamShift
//
//  Created by Matthew Perry on 1/28/16.
//  Copyright © 2016 Matthew Perry. All rights reserved.
//

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
 
    count ++;
    int total  = 0;
    
    for(int row = 0; row < height; row ++)
    {
        for(int col = 0; col < width; col++)
        {
            hue = hueArray[ width * row + col ];
            total += hue;
            
            probability = histogram[hue / BUCKET_WIDTH];
            M00 += probability;
            
            M1x += ((float)(col + xOffset)) * probability;
            M1y += ((float)(row + yOffset)) * probability;
            
         //   cout << "HUE AT (" << row + xOffset << ", " << col + yOffset << ") is " << hue << endl;
        }
        
    }
  // printf("Inside CPU MeanShift ---> M00 = %lf M1x = %lf M1y = %lf \n", M00, M1x, M1y);
    
    if(M00 > 0){//Can't divide by zero...
        
        xc = (int) (M1x / M00);
        yc = (int) (M1y / M00);
        (*roi).setCentroid(Point(xc, yc));
        
      //  printf("Inside CPU MeanShift ---> centroid (%d, %d)\n", xc, yc);
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