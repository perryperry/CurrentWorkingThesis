//
//  main.cpp
//  ThesisSerialCamShift
//
//  Created by Matthew Perry on 1/27/16.
//  Copyright © 2016 Matthew Perry. All rights reserved.
//

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/core/core.hpp"

#include "GPU/gpuMain.h"
#include "CPU/RegionOfInterest.hpp"
#include "CPU/UcharSerialCamShift.hpp"

#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

#define OUTPUTFILENAME "out.mov"
#define MAXTHREADS 3

void parameterCheck(int argCount)
{
    if(argCount != 3)
    {
        cout << "Usage: </path/to/videofile> </path/to/window/file>" << endl;
        exit(-1);
    }
}

unsigned char * parseSubHueData(Mat hsvMat, RegionOfInterest roi)
{
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    Mat subframe = hueMatrix(Rect(roi.getTopLeftX(), roi.getTopLeftY(), roi._width, roi._height)).clone();
    cout << subframe.total() << " <----  Smaller T O T A L \n";
    cout << hueMatrix.total() << " <----  Larger T O T A L \n" << endl;
    return (unsigned char *) subframe.data;
}

int main(int argc, const char * argv[])
{
    bool shouldCPU = false;
    int prevX = 0;
    int prevY = 0;
    
    high_resolution_clock::time_point t1;
    high_resolution_clock::time_point t2;
    
    int blockWidth = 64;
    int numElementsInput = 555555;
    char p = 'n';
    
    parameterCheck(argc);
    
    VideoCapture cap(argv[1]);
    
    int x, y, x2, y2;
    
    ifstream infile(argv[2]);
    
    //Read in windows from input file
    while (infile >> x >> y >> x2 >> y2)
    {
        cout << x <<  " " << y << " " << x2 << " " << y2 << endl;
    }
    
    //open output VideoWriter
    
    VideoWriter outputVideo = VideoWriter();
    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));//get codec type in int form
    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
    Size size = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    outputVideo.open(OUTPUTFILENAME, ex, cap.get(CV_CAP_PROP_FPS), size, true);
    
    if (! outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write."<< endl;
        exit(-1);
    }
    else{
        cout << "#########################################" << endl;
        cout << "Input frame resolution: Width=" << size.width << "  Height=" << size.height
        << " of nr#: " << cap.get(CV_CAP_PROP_FRAME_COUNT) << endl;
        cout << "Input codec type: " << EXT << endl;
        cout << "#########################################" << endl;
    }
    
    //end open output VideoWriter
    
    int i = 0;
 
    SerialCamShift camShift;
    
    Mat frame, hsv;
    
    float * histogram = (float *) calloc(sizeof(float), BUCKETS);

    cap.read(frame);
    
    RegionOfInterest roi(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
 
    cvtColor(frame, hsv, CV_RGB2HSV);
    
   // outputVideo.write(hsv);//just testing

    int step = 0;
    
    t1 = high_resolution_clock::now();
    
    unsigned char * hueArray = parseSubHueData(hsv, roi);
    

  /* for(int row = 0; row < roi._height; row ++)
    {
        for(int col = 0; col < roi._width; col ++)
        {
            printf("%d ", hueArray[row * roi._width + col]);
        }
        
        printf("\n");
    }*/
    
    
    
     camShift.createHistogram(hueArray, roi, &histogram);
    camShift.printHistogram(histogram, BUCKETS);

     bool go = true;

    /*while(go){
        prevX = roi.getCenterX();
        prevY = roi.getCenterY();
        cout << prevX << " AND PREVY " << prevY << endl;
        hueArray = parseSubHueData(hsv, roi);
        go = camShift.subMeanShift(hueArray, &roi, histogram, &prevX, &prevY);
    }*/
    
    
    
    //This is a test comparison of CPU and GPU
    int * convertedHue = (int * ) malloc(sizeof(int) * roi.getTotalPixels());
    
    for(int index = 0; index < roi.getTotalPixels(); index ++ )
    {
        convertedHue[index] = (int) hueArray[index];
    }
    
    
    
    
        camShift.subMeanShift(hueArray, &roi, histogram, &prevX, &prevY);
    
    float * bp = gpuBackProjectMain(convertedHue, roi.getTotalPixels(), histogram);
    
    float tot = 0.0;
    
   cout << "TOTAL PIXELS ---> " << roi.getTotalPixels() << endl;
    
    for(int index = 0; index < roi.getTotalPixels(); index ++ )
    {
     //cout << index << ")" << bp[index] << endl;
        tot += bp[index];
    }
    
    
    cout << "PLEASE WORK! -----> " << tot << endl;
    
    free(bp);
    free(convertedHue);
    //Endtest comparison of CPU and GPU

    
    
   // reverseIt(histogram);
    
    printf("*****************************\n");

    
    
    if(shouldCPU)
    {
    
        int length = roi._height * roi._width;
    
    
     

   
    
     while(cap.read(frame)){
     
         cvtColor(frame, hsv, CV_RGB2HSV);
         hueArray = parseSubHueData(hsv, roi);
         
         //t1 = high_resolution_clock::now();
        // t2 = high_resolution_clock::now();
         //auto duration2 = duration_cast<microseconds>( t2 - t1 ).count();
        
         
       // cout << "Duration of preparing using entire frame: " << duration2 / 1000.0 << endl;
         
         
     // camShift.meanshift(hueArray, step, &roi, histogram);
    

        go = true;
         
       
         
         while(go){
             cout << "top" << endl;
             prevX = roi.getCenterX();
             prevY = roi.getCenterY();
             cout << prevX << " AND PREVY " << prevY << endl;
             go = camShift.subMeanShift(hueArray, &roi, histogram, &prevX, &prevY);
             cout << "Going in" << endl;
             hueArray = parseSubHueData(hsv, roi);
             cout << "made it" << endl;
       }
    
         cout << "GOT OUT" << endl;
         
       camShift.backProjectHistogram(hueArray, frame.step, &frame, roi, histogram);
         
         cout << "MOVE ON " << endl;
            roi.drawROI(&frame);
          //  roi.printROI();
           outputVideo.write(frame);
            
     }//end while
  
	cout << endl << "****END OF CPU Serial MeanShift****" << endl;
 

    }//end shouldCPU

    outputVideo.release();
    
   free(histogram);
    
   return 0;
}
