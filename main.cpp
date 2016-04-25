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

int calculateHueArrayLength(Mat frame, int * step)
{
    Mat hsvMat;
    cvtColor(frame, hsvMat, CV_RGB2HSV);
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    int total = hueMatrix.total();
    *step = hueMatrix.step;
    return total;
}

int convertToHueArray(Mat frame, unsigned char ** hueArray)
{
    Mat hsvMat;
    cvtColor(frame, hsvMat, CV_RGB2HSV);
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    int i = 0;
    for(i = 0; i < hueMatrix.total(); i ++)
    {
        (*hueArray)[i]=(unsigned char) hueMatrix.data[i];
    }
}

//For pthread testing
void * test(void * data)
{
    char * str;
    str = (char * )data;
    printf("%s\n", str);
}

void menu(int * num_objects, bool * processVid, bool * cpu, bool * gpu, bool * print, bool * bp)
{
    int answer = 0;
    printf("GPU vs CPU meanshift menu:\n");
    cout << "Number of objects:";
    scanf("%d", &answer);
    *num_objects = answer;
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
    
    if(*num_objects == 0)
        exit(-1);
}

int main(int argc, const char * argv[])
{
    bool shouldProcessVideo = false;
    bool shouldCPU = false;
    bool shouldGPU = false;
    bool shouldBackProjectFrame = false;
    bool shouldPrint = false;
    int num_objects = 1;
   
    parameterCheck(argc);
    menu(&num_objects, &shouldProcessVideo, &shouldCPU, &shouldGPU, &shouldPrint, &shouldBackProjectFrame);
    VideoCapture cap(argv[1]);
    int x, y, x2, y2, obj_cur = 0;
    
    ifstream infile(argv[2]);
    
    RegionOfInterest cpu_objects[num_objects];
    RegionOfInterest gpu_objects[num_objects];
    
    //Read in windows from input file
    while (infile >> x >> y >> x2 >> y2)
    {
        if(obj_cur > num_objects)
        {
            cout << "ERROR: Too many lines in input file for number of objects to track!" << endl;
            exit(-1);
        }
        cout << x <<  " " << y << " " << x2 << " " << y2 << endl;
        
        cpu_objects[obj_cur].init(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
        gpu_objects[obj_cur].init(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
        obj_cur++;
    }
    obj_cur = 0; //reset the current object index
    d_struct * ds = (d_struct *) malloc(sizeof(d_struct));
    
    float gpu_time_cost = 0.0;
    float cpu_time_cost = 0;
    
    high_resolution_clock::time_point time1;
    high_resolution_clock::time_point time2;
    
    int gpu_xc = 0;
    int gpu_yc = 0;
    int cpu_cx = 0;
    int cpu_cy = 0;
    int gpu_prevX = 0;
    int gpu_prevY = 0;
    int step = 0; //The width of the entire frame
    int * row_offset = (int *) malloc(sizeof(int));
    int * col_offset = (int *) malloc(sizeof(int));
    SerialCamShift camShift;
    
    Mat frame, hsv;
    Mat subHueFrame;
    
    
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

    
    
    
    /************************************* First Frame initialize and process ****************************************/
    cap.read(frame);
    float * histogram = (float *) calloc(sizeof(float), BUCKETS * num_objects);
    int totalHue = calculateHueArrayLength(frame, &step);
    unsigned char * entireHueArray = (unsigned char *) malloc(sizeof(unsigned char) * totalHue);
    convertToHueArray(frame, &entireHueArray);

    camShift.createHistogram(entireHueArray, step, cpu_objects, &histogram, num_objects);

    mainConstantMemoryHistogramLoad(histogram, num_objects);
    
    //for testing
   // if(! shouldProcessVideo )
   // {
        
        //camShift.printHistogram(histTEST, BUCKETS * num_objects);
       // camShift.printHistogram(histogram, BUCKETS * num_objects);
       
        for(obj_cur = 0; obj_cur < num_objects; obj_cur++){
            cpu_objects[obj_cur].drawCPU_ROI(&frame);
            gpu_objects[obj_cur].drawGPU_ROI(&frame);
        }
        outputVideo.write(frame);
   // }
    
  
    
     /*
       if(shouldCPU){
          for(obj_cur = 0; obj_cur < num_objects; obj_cur++){
           
               cpu_cx = cpu_objects[obj_cur].getCenterX();
               cpu_cy = cpu_objects[obj_cur].getCenterY();
               cpu_time_cost += camShift.cpu_entireFrameMeanShift(entireHueArray, step, cpu_objects[obj_cur], obj_cur, histogram, shouldPrint, &cpu_cx, &cpu_cy);
               cpu_objects[obj_cur].setCentroid(Point(cpu_cx, cpu_cy));
               cpu_objects[obj_cur].drawCPU_ROI(&frame);
           }
           printf("**************************************** After first one ***********************************\n");

           for(obj_cur = 0; obj_cur < num_objects; obj_cur++){
               printf("OBJECT %d: (%d, %d)\n", obj_cur, cpu_objects[obj_cur].getCenterX(), cpu_objects[obj_cur].getCenterY());
           }
        }
        if(shouldGPU)
        {
           initDeviceStruct(ds, entireHueArray, totalHue, &cx, &cy, col_offset, row_offset);
           
            gpu_time_cost += launchTwoKernelReduction(*ds, entireHueArray, totalHue, gpu_objects[obj_cur].getTotalPixels(), step, gpu_objects[obj_cur]._width, gpu_objects[obj_cur]._height, &cx, &cy, shouldPrint);
           gpu_objects[obj_cur].setCentroid(Point(cx, cy));
           gpu_objects[obj_cur].drawGPU_ROI(&frame);
        }
        printf("**************************************** Process the rest of the video ***********************************\n");
    
        outputVideo.write(frame);
    */
    
    
  if( shouldProcessVideo )
  {
      row_offset[0] = gpu_objects[obj_cur].getTopLeftY();
      col_offset[0] = gpu_objects[obj_cur].getTopLeftX();
      int cx = gpu_objects[obj_cur].getCenterX(), cy = gpu_objects[obj_cur].getCenterY();
      
      initDeviceStruct(ds, entireHueArray, totalHue, &cx, &cy, col_offset, row_offset); //gpu device struct for kernel memory re-use

        while(cap.read(frame))
        {
            totalHue = convertToHueArray(frame, &entireHueArray);
            /******************************** CPU MeanShift until Convergence ***************************************/
            if(shouldCPU)
            {
               for(obj_cur = 0; obj_cur < num_objects; obj_cur++){
                    cpu_cx = cpu_objects[obj_cur].getCenterX();
                    cpu_cy = cpu_objects[obj_cur].getCenterY();
                    cpu_time_cost += camShift.cpu_entireFrameMeanShift(entireHueArray, step, cpu_objects[obj_cur], obj_cur, histogram, shouldPrint, &cpu_cx, &cpu_cy);
                    cpu_objects[obj_cur].setCentroid(Point(cpu_cx, cpu_cy));
                    cpu_objects[obj_cur].drawCPU_ROI(&frame);
              }
            }
            /******************************** GPU MeanShift until Convergence **********************************************/
            if(shouldGPU)
            {
                gpu_time_cost += launchTwoKernelReduction(*ds, entireHueArray, totalHue, gpu_objects[obj_cur].getTotalPixels(), step, gpu_objects[obj_cur]._width, gpu_objects[obj_cur]._height, &cx, &cy, shouldPrint);
                gpu_objects[obj_cur].setCentroid(Point(cx, cy));
                gpu_objects[obj_cur].drawGPU_ROI(&frame);
                gpu_objects[obj_cur].drawGPU_ROI(&frame);
            }
            /******************************** Write to Output Video *******************************************/
            if(shouldBackProjectFrame){
               /* if(shouldCPU)
                    camShift.backProjectHistogram(hueArray, frame.step, &frame, cpu_objects[obj_cur], histogram);
                if(shouldGPU)
                    camShift.backProjectHistogram(hueArray, frame.step, &frame, gpu_objects[obj_cur], histogram);*/
            }
               
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
    free(entireHueArray);
    
    printf("Program exited successfully\n");
    return 0;
}
