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
#include <pthread.h>

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

int convertToHueArray(Mat frame, unsigned char ** hueArray, int * step)
{
    Mat hsvMat;
    cvtColor(frame, hsvMat, CV_RGB2HSV);
    //printf("\n **************** Lets see how many: %d, abs_width: %d or this width?: %d\n", (int)hsvMat.total(), size.width, hsvMat.cols);
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    *hueArray = (unsigned char * ) hueMatrix.data;
    *step = hueMatrix.step;
    return hueMatrix.total();
}

void convertToSubHue(Mat frame, RegionOfInterest roi, Mat * subHueFrame)
{
    Mat hsv;
    Mat subFrame = frame(Rect(roi.getTopLeftX(), roi.getTopLeftY(), roi._width, roi._height)).clone();
    cvtColor(subFrame, hsv, CV_RGB2HSV);
    std::vector<cv::Mat> hsv_channels;
    split(hsv, hsv_channels);
    *subHueFrame = hsv_channels[0];
}

void * test(void * data)
{
    char * str;
    str = (char * )data;
    printf("%s\n", str);
}

void menu(bool * processVid, bool * cpu, bool * gpu, bool * print, bool * bp)
{
    int answer = 0;
    printf("GPU vs CPU meanshift menu:\n");
    cout << "Should process entire video (0/1):";
    scanf("%d", &answer);
    *processVid = answer;
    cout <<"Should run cpu version (0/1):";
    scanf("%d", &answer);
    *cpu = answer;
    cout << "Should run gpu version (0/1):";
    scanf("%d", &answer);
    *gpu = answer;
    cout << "Should print intermediate output (0/1):";
    scanf("%d", &answer);
    *print = answer;
    cout << "Should backproject (0/1):";
    scanf("%d", &answer);
    *bp = answer;
}


int main(int argc, const char * argv[])
{
    bool shouldProcessVideo = false;
    bool shouldCPU = false;
    bool shouldGPU = false;
    bool shouldBackProjectFrame = false;
    bool shouldPrint = false;
    
    parameterCheck(argc);
    menu(&shouldProcessVideo, &shouldCPU, &shouldGPU, &shouldPrint, &shouldBackProjectFrame);

    d_struct * ds = (d_struct *) malloc(sizeof(d_struct));
    
    float gpu_time_cost = 0.0;
    float cpu_time_cost = 0;
    
    high_resolution_clock::time_point time1;
    high_resolution_clock::time_point time2;
    
    int gpu_xc = 0;
    int gpu_yc = 0;
    int gpu_prevX = 0;
    int gpu_prevY = 0;
    int * row_offset = (int *) malloc(sizeof(int));
    int * col_offset = (int *) malloc(sizeof(int));
    SerialCamShift camShift;
    
    Mat frame, hsv;
    Mat subHueFrame;
    float * histogram = (float *) calloc(sizeof(float), BUCKETS);

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
    
    int totalFrames = 0;
    
    if (! outputVideo.isOpened()){
        cout  << "Could not open the output video for write."<< endl;
        exit(-1);
    }
    else{
        totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    }
    
    //end open output VideoWriter
    
     /*************************************** Testing Pthread *********************************************/
    
    
  /*   pthread_t thread1, thread2;
     pthread_create(&thread1, NULL, test, (void *)"Keeping it trillion from thread 1.\n");
     pthread_create(&thread2, NULL, test, (void *)"Keeping it trillion from thread 2.\n");
    
     pthread_join(thread1, NULL);*/

    /*************************************** End Testing Pthread *********************************************/

    
        cap.read(frame);

        RegionOfInterest cpu_roi(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
        RegionOfInterest gpu_roi(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
      
       // time1 = high_resolution_clock::now();
        
        convertToSubHue(frame, gpu_roi, &subHueFrame);
        
       // time2 = high_resolution_clock::now();
       // auto duration2 = duration_cast<microseconds>( time2 - time1 ).count();
        
       // cout << "Duration of preparing using subframe with converting to hsv: " << duration2 / 1000.0 << endl;
    
    /************************************* First Frame initialize and process ****************************************/
    unsigned char * hueArray = (unsigned char * ) subHueFrame.data;
    camShift.createHistogram(hueArray, cpu_roi, &histogram);
    mainConstantMemoryHistogramLoad(histogram);
    // camShift.printHistogram(histogram, BUCKETS);
    
    int cx = gpu_roi.getCenterX(), cy = gpu_roi.getCenterY();
    int step;
    unsigned char * entireHueArray;
    int totalHue = convertToHueArray(frame, &entireHueArray, &step);
        
    row_offset[0] = gpu_roi.getTopLeftY();
    col_offset[0] = gpu_roi.getTopLeftX();
    
   if(shouldCPU){
        cpu_time_cost += camShift.cpu_entireFrameMeanShift(entireHueArray, step, &cpu_roi, histogram, shouldPrint);
        cpu_roi.drawCPU_ROI(&frame);
    }
    if(shouldGPU)
    {
        initDeviceStruct(ds, entireHueArray, totalHue, &cx, &cy, col_offset, row_offset);
        launchTwoKernelReduction(*ds, entireHueArray, totalHue, gpu_roi.getTotalPixels(), step, gpu_roi._width, gpu_roi._height, &cx, &cy, shouldPrint);
        gpu_roi.setCentroid(Point(cx, cy));
        gpu_roi.drawGPU_ROI(&frame);
    }
    /**************************************** Process the rest of the video ***********************************/
    if( shouldProcessVideo )
    {
        outputVideo.write(frame);
        while(cap.read(frame))
        {
            totalHue = convertToHueArray(frame, &entireHueArray, &step);
            /******************************** CPU MeanShift until Convergence ***************************************/
            if(shouldCPU)
                cpu_time_cost += camShift.cpu_entireFrameMeanShift(entireHueArray, step, &cpu_roi, histogram, shouldPrint);
            /******************************** GPU MeanShift until Convergence **********************************************/
            if(shouldGPU)
            {
                gpu_time_cost += launchTwoKernelReduction(*ds, entireHueArray, totalHue, gpu_roi.getTotalPixels(), step, gpu_roi._width, gpu_roi._height, &cx, &cy, shouldPrint);
                gpu_roi.setCentroid(Point(cx, cy));
                gpu_roi.drawGPU_ROI(&frame);
            }
            /******************************** Write to Output Video *******************************************/
            if(shouldBackProjectFrame){
                if(shouldCPU)
                    camShift.backProjectHistogram(hueArray, frame.step, &frame, cpu_roi, histogram);
                if(shouldGPU)
                    camShift.backProjectHistogram(hueArray, frame.step, &frame, gpu_roi, histogram);
            }
            if(shouldCPU)
                cpu_roi.drawCPU_ROI(&frame);
            if(shouldGPU)
                gpu_roi.drawGPU_ROI(&frame);
            outputVideo.write(frame);
            
        }//end while
        printf("GPU average time cost in milliseconds: %f\n", gpu_time_cost / ((float) totalFrames));
        printf("CPU average time cost in milliseconds: %f\n", cpu_time_cost / ((float) totalFrames));
    }// end if shouldProcessVideo
    
    //clean-up
    outputVideo.release();
    free(histogram);
    free(row_offset);
    free(col_offset);
    freeDeviceStruct(ds);
	free(ds);   
    return 0;
}
