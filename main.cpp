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


float * buildIt()
{
    float * array = (float *) calloc(sizeof(float), 12345678);

    int i = 0;
    
    for(i = 0; i < 12345678; i ++)
    {
        array[i] = 0.17;
    }
    return array;
}

void parameterCheck(int argCount)
{
    if(argCount != 3)
    {
        cout << "Usage: </path/to/videofile> </path/to/window/file>" << endl;
        exit(-1);
    }
}


// This version assumes an already hsv converted matrix the size of the entire picture frame.
// It then splits the matrix into the hue channel and gets the subframe from it.
// But this runtime was 1.4 to 2.1 ms, and it requires the very slow hsv conversion of the entire picture frame.
// Do not use this!
void parseSubHueData(Mat hsvMat, RegionOfInterest roi, Mat * subframe)
{
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    *subframe = hueMatrix(Rect(roi.getTopLeftX(), roi.getTopLeftY(), roi._width, roi._height)).clone();
}


// runtime is .4 to 1.2 ms
// Takes the full picture frame 'frame', and first creates the subFrame, then does the hsv conversion
// Stores the subFrame hue channel in 'subHueFrame'.
// This avoids having to do the hsv conversion on the entire picture frame which takes 220 to 260 ms
void convertToSubHue(Mat frame, RegionOfInterest roi, Mat * subHueFrame)
{
    Mat hsv;
    Mat subFrame = frame(Rect(roi.getTopLeftX(), roi.getTopLeftY(), roi._width, roi._height)).clone();
    cvtColor(subFrame, hsv, CV_RGB2HSV);
    std::vector<cv::Mat> hsv_channels;
    split(hsv, hsv_channels);
    *subHueFrame = hsv_channels[0];
}

int main(int argc, const char * argv[])
{
    high_resolution_clock::time_point time1;
    high_resolution_clock::time_point time2;
    
    bool shouldProcessVideo = false;
    bool shouldCPU = true;
    bool shouldGPU = true;
    bool cpu_continue = true;
    bool gpu_continue = true;
    bool shouldBackProjectFrame = false;
    
    int prevX = 0;
    int prevY = 0;
    
    int gpu_xc = 0;
    int gpu_yc = 0;
    int gpu_prevX = 0;
    int gpu_prevY = 0;
    
  
   // auto duration;
    //auto gpu_duration;
    
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
    
    /******************************************************************************************************************/
    
    int i = 0;
 
    SerialCamShift camShift;
    
    Mat frame, hsv;
    Mat subHueFrame;
    float * histogram = (float *) calloc(sizeof(float), BUCKETS);

    cap.read(frame);

	Mat testIt;
 cvtColor(frame, testIt, CV_RGB2HSV);


	printf("\n **************** Lets see how many: %d \n", (int)testIt.total());
    
    RegionOfInterest roi(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    RegionOfInterest gpu_roi(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));

  
   /* time1 = high_resolution_clock::now();
    
    
    cvtColor(frame, hsv, CV_RGB2HSV);
    
    
    time2 = high_resolution_clock::now();
    
    
    auto duration2 = duration_cast<microseconds>( time2 - time1 ).count();
    
    
    cout << "Duration of preparing using entire frame: " << duration2 / 1000.0 << endl;

    
    
    
    
    
    
    
    
   // outputVideo.write(hsv);//just testing
    
   // time1 = high_resolution_clock::now();
    
   
    

    
    
    time1 = high_resolution_clock::now();
    parseSubHueData(hsv, gpu_roi, &subHueFrame);
    time2 = high_resolution_clock::now();
    duration2 = duration_cast<microseconds>( time2 - time1 ).count();
    
    
    cout << "Duration of preparing using entire frame without converting to hsv: " << duration2 / 1000.0 << endl;*/
 
    
    time1 = high_resolution_clock::now();
    
    convertToSubHue(frame, gpu_roi, &subHueFrame);
    
    time2 = high_resolution_clock::now();
   auto duration2 = duration_cast<microseconds>( time2 - time1 ).count();
    
    
    cout << "Duration of preparing using subframe with converting to hsv: " << duration2 / 1000.0 << endl;
    
    
    
    
    unsigned char * hueArray = (unsigned char * ) subHueFrame.data;
    
     camShift.createHistogram(hueArray, roi, &histogram);
   // camShift.printHistogram(histogram, BUCKETS);

    /******************************************************************************************************************/
    //This is a test comparison of CPU and GPU
    int hueLength = gpu_roi.getTotalPixels();
    
    float * M00 = (float *) malloc(hueLength * sizeof(float));
    float * M1x = (float *) malloc(hueLength * sizeof(float));
    float * M1y = (float *) malloc(hueLength * sizeof(float));
    
    int xOffset = gpu_roi.getTopLeftX();
    int yOffset = gpu_roi.getTopLeftY();
    
    camShift.subMeanShift(hueArray, &roi, histogram, &prevX, &prevY);
    
    gpuBackProjectMain(hueArray, gpu_roi.getTotalPixels(), histogram, gpu_roi._width, xOffset, yOffset, &M00, &M1x ,&M1y);
    gpuReduceMain(64, M00, M1x, M1y, gpu_roi.getTotalPixels(), &gpu_xc, &gpu_yc);
    
    double tot = 0.0;
    
    printf("CENTROID FROM GPU: (%d, %d)\n", gpu_xc, gpu_yc);

    //Endtest comparison of CPU and GPU
    
    /******************************************************************************************************************/
    printf("*****************************\n");

    
    
    
    /******************************************************************************************************************/
    
    //Testing new and improved kernel
    
    int testX = 0, testY = 0;
    
  //  camShift.printHistogram(histogram, BUCKETS);

    
    
    
    
    mainConstantMemoryHistogramLoad(histogram);

  launchMeanShiftKernelForSubFrame(hueArray, gpu_roi.getTotalPixels(), gpu_roi._width, xOffset, yOffset, &testX, &testY);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    /******************************************************************************************************************/
    
    
    
    if( shouldProcessVideo )
    {
        int length = roi._height * roi._width;
    
        while(cap.read(frame))
        {
     
         //cvtColor(frame, hsv, CV_RGB2HSV);
        // hueArray = parseSubHueData(hsv, roi);
    
        //CPU MeanShift until Convergence
        cpu_continue = true;
            
         time1 = high_resolution_clock::now();
         while( shouldCPU && cpu_continue )
         {
            // hueArray = parseSubHueData(hsv, roi);
             convertToSubHue(frame, roi, &subHueFrame);
             hueArray = (unsigned char *) subHueFrame.data;
             prevX = roi.getCenterX();
             prevY = roi.getCenterY();
             cpu_continue = camShift.subMeanShift(hueArray, &roi, histogram, &prevX, &prevY);
         }
         time2 = high_resolution_clock::now();
        auto cpu_duration = duration_cast<microseconds>( time2 - time1 ).count();
            
 
        //GPU MeanShift until Convergence
        gpu_continue = true;
        
        time1 = high_resolution_clock::now();
        while( shouldGPU && gpu_continue )
        {
            // hueArray = parseSubHueData(hsv, gpu_roi);
            convertToSubHue(frame, gpu_roi, &subHueFrame);
            hueArray = (unsigned char *) subHueFrame.data;
            gpu_prevX = gpu_roi.getCenterX();
            gpu_prevY = gpu_roi.getCenterY();
            xOffset = gpu_roi.getTopLeftX();
            yOffset = gpu_roi.getTopLeftY();
            
            gpuBackProjectMain(hueArray, gpu_roi.getTotalPixels(), histogram, gpu_roi._width, xOffset, yOffset, &M00, &M1x ,&M1y);
            
            gpuReduceMain(64, M00, M1x, M1y, gpu_roi.getTotalPixels(), &gpu_xc, &gpu_yc);
            
            // printf("CENTROID FROM GPU: (%d, %d)\n", gpu_xc, gpu_yc);
            
            gpu_roi.setCentroid(Point(gpu_xc, gpu_yc));
            
            if(gpu_prevX - gpu_xc < 1 && gpu_prevX - gpu_xc > -1  && gpu_prevY - gpu_yc < 1 && gpu_prevY - gpu_yc > -1)
            {
                gpu_continue = false;
            }
            
            gpu_prevX = gpu_xc;
            gpu_prevY = gpu_yc;
        }
        time2 = high_resolution_clock::now();
        auto gpu_duration = duration_cast<microseconds>( time2 - time1 ).count();
            
      

            
            
      //   cout << "CPU time: " << cpu_duration / 1000.0 <<  " GPU time: " << gpu_duration / 1000.0 << endl;
         //   cout << "CPU centroid (" << roi.getCenterX() << ", " << roi.getCenterY() << ") GPU centroid (" << gpu_roi.getCenterX() << ", " << gpu_roi.getCenterY() << ") " << endl;
 
            
        if(shouldBackProjectFrame)
            camShift.backProjectHistogram(hueArray, frame.step, &frame, roi, histogram);
       
        if(shouldCPU)
            roi.drawCPU_ROI(&frame);
            
        if(shouldGPU)
            gpu_roi.drawGPU_ROI(&frame);
            
        //  roi.printROI();
        outputVideo.write(frame);
            
     }//end while
  
        cout << endl << "**** END of MeanShift ****" << endl;
    
    }// end of shouldProcessVideo


    outputVideo.release();
    
    free(histogram);
    
    //free GPU data structures
    free(M00);
    free(M1x);
    free(M1y);
    
   return 0;
}
