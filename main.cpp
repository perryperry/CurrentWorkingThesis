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

unsigned char * parseSubHueData(Mat hsvMat, RegionOfInterest roi)
{
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    Mat subframe = hueMatrix(Rect(roi.getTopLeftX(), roi.getTopLeftY(), roi._width, roi._height)).clone();
   // cout << subframe.total() << " <----  Smaller T O T A L \n";
   // cout << hueMatrix.total() << " <----  Larger T O T A L \n" << endl;
    return (unsigned char *) subframe.data;
}

int main(int argc, const char * argv[])
{
    bool shouldCPU = false;
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
    
    /******************************************************************************************************************/
    
    int i = 0;
 
    SerialCamShift camShift;
    
    Mat frame, hsv;
    
    float * histogram = (float *) calloc(sizeof(float), BUCKETS);

    cap.read(frame);
    
    RegionOfInterest roi(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    RegionOfInterest gpu_roi(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    
    cvtColor(frame, hsv, CV_RGB2HSV);
    
   // outputVideo.write(hsv);//just testing
    
    t1 = high_resolution_clock::now();
    
    unsigned char * hueArray = parseSubHueData(hsv, roi);
    
     camShift.createHistogram(hueArray, roi, &histogram);
   // camShift.printHistogram(histogram, BUCKETS);

    

    /******************************************************************************************************************/
    //This is a test comparison of CPU and GPU
    int * convertedHue = (int * ) malloc(sizeof(int) * roi.getTotalPixels());
    
    for(int index = 0; index < gpu_roi.getTotalPixels(); index ++ )
    { //Convert to ints for the kernel...
        convertedHue[index] = (int) hueArray[index];
    }
    
    int hueLength = gpu_roi.getTotalPixels();
    
    float * M00 = (float *) malloc(hueLength * sizeof(float));
    float * M1x = (float *) malloc(hueLength * sizeof(float));
    float * M1y = (float *) malloc(hueLength * sizeof(float));
    
    int xOffset = gpu_roi.getTopLeftX();
    int yOffset = gpu_roi.getTopLeftY();
    
    
    camShift.subMeanShift(hueArray, &roi, histogram, &prevX, &prevY);
    
    // Histogram BackProjected array
    gpuBackProjectMain(convertedHue, gpu_roi.getTotalPixels(), histogram, gpu_roi._width, xOffset, yOffset, &M00, &M1x ,&M1y);
    
   // float * bp = buildIt();
    
    cout  << "************* Running GPU Reduction ******************\n";
   gpuReduceMain(64, M00, M1x, M1y, roi.getTotalPixels(), &gpu_xc, &gpu_yc);
    
    double tot = 0.0;
    
    printf("CENTROID FROM GPU: (%d, %d)\n", gpu_xc, gpu_yc);
    
   /* for(int index = 0; index < roi.getTotalPixels(); index ++ )
    {
        tot += M00[index];
    }*/
    
  //  cout << "Precision concerns: " << tot << endl;
   // printf("M00 after GPU backprojection and sequential summation -----> %lf\n", tot);
    
    
    
    //Endtest comparison of CPU and GPU
    
    /******************************************************************************************************************/
   // reverseIt(histogram);
    
    printf("*****************************\n");


        int length = roi._height * roi._width;
    
        while(cap.read(frame))
        {
     
         cvtColor(frame, hsv, CV_RGB2HSV);
        // hueArray = parseSubHueData(hsv, roi);
         
         //t1 = high_resolution_clock::now();
        // t2 = high_resolution_clock::now();
         //auto duration2 = duration_cast<microseconds>( t2 - t1 ).count();
        
         
       // cout << "Duration of preparing using entire frame: " << duration2 / 1000.0 << endl;
    
        cpu_continue = true;
         
         while( shouldCPU && cpu_continue )
         {
             hueArray = parseSubHueData(hsv, roi);
             prevX = roi.getCenterX();
             prevY = roi.getCenterY();
             cpu_continue = camShift.subMeanShift(hueArray, &roi, histogram, &prevX, &prevY);
         }
        
            
         gpu_continue = true;
            
            
         while( shouldGPU && gpu_continue )
         {
             hueArray = parseSubHueData(hsv, gpu_roi);
             
             for(int index = 0; index < gpu_roi.getTotalPixels(); index ++ )
             { //Convert to ints for the kernel...
                 convertedHue[index] = (int) hueArray[index];
             }
             
             gpu_prevX = gpu_roi.getCenterX();
             gpu_prevY = gpu_roi.getCenterY();
             xOffset = gpu_roi.getTopLeftX();
             yOffset = gpu_roi.getTopLeftY();
             
             gpuBackProjectMain(convertedHue, gpu_roi.getTotalPixels(), histogram, gpu_roi._width, xOffset, yOffset, &M00, &M1x ,&M1y);
         
             gpuReduceMain(64, M00, M1x, M1y, gpu_roi.getTotalPixels(), &gpu_xc, &gpu_yc);
              printf("CENTROID FROM GPU: (%d, %d)\n", gpu_xc, gpu_yc);
             gpu_roi.setCentroid(Point(gpu_xc, gpu_yc));
             
             if(gpu_prevX - gpu_xc < 1 && gpu_prevX - gpu_xc > -1  && gpu_prevY - gpu_yc < 1 && gpu_prevY - gpu_yc > -1)
             {
                 gpu_continue = false;
             }
             
             gpu_prevX = gpu_xc;
             gpu_prevY = gpu_yc;
         }
            
        if(shouldBackProjectFrame)
            camShift.backProjectHistogram(hueArray, frame.step, &frame, roi, histogram);
       
        if(shouldCPU)
            roi.drawCPU_ROI(&frame);
            
        if(shouldGPU)
            gpu_roi.drawGPU_ROI(&frame);
            
        //  roi.printROI();
        outputVideo.write(frame);
            
     }//end while
  
	cout << endl << "****END OF CPU Serial MeanShift****" << endl;
 


    outputVideo.release();
    
   free(histogram);
    
    //free GPU data structures
    free(M00);
    free(M1x);
    free(M1y);
    free(convertedHue);
    
   return 0;
}
