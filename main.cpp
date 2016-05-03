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

void parameterCheck(int argCount)
{
    if(argCount != 3)
    {
        cout << "Usage: </path/to/videofile> </path/to/window/file>" << endl;
        exit(-1);
    }
}

int calculateHueArrayLength(Mat frame, int * step, int * mat_rows, int * mat_cols)
{
    Mat hsvMat;
    cvtColor(frame, hsvMat, CV_RGB2HSV);
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    int total = hueMatrix.total();
    *step = hueMatrix.step;
    *mat_rows = hueMatrix.rows;
    *mat_cols = hueMatrix.cols;
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
    return hueMatrix.total();
}

void printHueSum(unsigned char * entireHueArray, int hueLength)
{
    long total = 0;
    for(int i =0; i < hueLength; i++)
    {
        total += (int) entireHueArray[i];
    }
    printf("Total Hue Sum in CPU: %ld\n", total);
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
    
    //timeMemoryTransfer();
    //exit(1);
    
    
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
    
    float gpu_time_cost = 0.0f;
    float cpu_time_cost = 0.0f;
    
    high_resolution_clock::time_point time1;
    high_resolution_clock::time_point time2;
    
    int * gpu_cx = (int *) malloc(sizeof(int) * num_objects); //centroid x for gpu
    int * gpu_cy = (int *) malloc(sizeof(int) * num_objects); //centroid y for gpu
    
    //For multi-object test
    int * obj_block_ends = (int *) malloc(sizeof(int) * num_objects); //ending block of each object in kernel
    int * subFrameLengths = (int *) malloc(sizeof(int) * num_objects);;
    int * sub_widths = (int *) malloc(sizeof(int) * num_objects);;
    int * sub_heights = (int *) malloc(sizeof(int) * num_objects);
    
    for(obj_cur = 0; obj_cur < num_objects; obj_cur++){
       subFrameLengths[obj_cur] = gpu_objects[obj_cur].getTotalPixels();
        printf("In Main: %d\n", subFrameLengths[obj_cur]);
       sub_widths[obj_cur] = gpu_objects[obj_cur]._width;
       sub_heights[obj_cur] = gpu_objects[obj_cur]._height;
    }
    
    
    int cpu_cx = 0; //centroid x for cpu
    int cpu_cy = 0; //centroid x for cpu
    int step = 0; //The width of the entire frame
    int mat_rows = 0, mat_cols = 0;
    int * gpu_row_offset = (int *) malloc(sizeof(int) * num_objects); //for gpu
    int * gpu_col_offset = (int *) malloc(sizeof(int) * num_objects); //for gpu
    SerialCamShift camShift;
    Mat frame, hsv;
    
    /*************************************** Open Output VideoWriter *********************************************/
    //Code for opening VideoWriter taken from http://docs.opencv.org/3.1.0/d7/d9e/tutorial_video_write.html#gsc.tab=0
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

     /*************************************** Testing Pthread *********************************************/
    
  /*   pthread_t thread1, thread2;
     pthread_create(&thread1, NULL, test, (void *)"Keeping it trillion from thread 1.\n");
     pthread_create(&thread2, NULL, test, (void *)"Keeping it trillion from thread 2.\n");
    
     pthread_join(thread1, NULL);*/
    
    /************************************* First Frame initialize and process ****************************************/
    cap.read(frame);
    float * histogram = (float *) calloc(sizeof(float), BUCKETS * num_objects);
    int hueLength = calculateHueArrayLength(frame, &step, &mat_rows, &mat_cols);
    
    printf("Mat: rows: %d cols: %d\n", mat_rows, mat_cols);
    unsigned char * entireHueArray = (unsigned char *) malloc(sizeof(unsigned char) * hueLength);
    convertToHueArray(frame, &entireHueArray);

    camShift.createHistogram(entireHueArray, step, cpu_objects, &histogram, num_objects);

    mainConstantMemoryHistogramLoad(histogram, num_objects);

    //For the initial frame, just render the initial search windows' positions
    for(obj_cur = 0; obj_cur < num_objects; obj_cur++){
       cpu_objects[obj_cur].drawCPU_ROI(&frame);
        gpu_objects[obj_cur].drawGPU_ROI(&frame);
        //load gpu starting values as well
        gpu_row_offset[obj_cur] = gpu_objects[obj_cur].getTopLeftY();
        gpu_col_offset[obj_cur] = gpu_objects[obj_cur].getTopLeftX();
        gpu_cx[obj_cur] = gpu_objects[obj_cur].getCenterX();
        gpu_cy[obj_cur] = gpu_objects[obj_cur].getCenterY();
    }
    outputVideo.write(frame);
    
    
    float ratio = (float) cpu_objects[0]._height / (float) cpu_objects[0]._width;

    
  if( shouldProcessVideo )
  {
      int num_block = initDeviceStruct(num_objects, ds, obj_block_ends, entireHueArray, hueLength, gpu_cx , gpu_cy, gpu_col_offset, gpu_row_offset, subFrameLengths, sub_widths, sub_heights); //gpu device struct for kernel memory re-use

   while(cap.read(frame))
     {
           hueLength = convertToHueArray(frame, &entireHueArray);
           //printHueSum(entireHueArray, hueLength);
            /******************************** CPU MeanShift until Convergence ***************************************/
            if(shouldCPU)
            {
               for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
               {
                    cpu_cx = cpu_objects[obj_cur].getCenterX();
                    cpu_cy = cpu_objects[obj_cur].getCenterY();
                   // cpu_time_cost += camShift.cpuMeanShift(entireHueArray, step, cpu_objects[obj_cur], obj_cur, histogram, shouldPrint, &cpu_cx, &cpu_cy);
                   
                  int width = 0, height = 0;
                   
                   //Cam Shift test
                   
                 cpu_time_cost += camShift.cpuCamShift(entireHueArray, step, cpu_objects[obj_cur], obj_cur, histogram, shouldPrint, &cpu_cx, &cpu_cy, &width, &height, hueLength);
                   
                    cpu_objects[obj_cur].setCentroid(Point(cpu_cx, cpu_cy));
                   cpu_objects[obj_cur].setWidthHeight(width, height);
                    cpu_objects[obj_cur].drawCPU_ROI(&frame);
              }
            }
            /******************************** GPU MeanShift until Convergence **********************************************/
            if(shouldGPU)
            {
               /* obj_cur = 1;
                gpu_time_cost += launchTwoKernelReduction(obj_cur, num_objects, *ds, entireHueArray, hueLength, gpu_objects[obj_cur].getTotalPixels(), step, gpu_objects[obj_cur]._width, gpu_objects[obj_cur]._height, &gpu_cx , &gpu_cy, shouldPrint);
                gpu_objects[obj_cur].setCentroid(Point(gpu_cx[obj_cur], gpu_cy[obj_cur]));
                gpu_objects[obj_cur].drawGPU_ROI(&frame);*/
                
                
                
                
                
                
                
                //Multi-object MeanShift
                
                //gpu_time_cost += launchMultiObjectTwoKernelReduction( num_objects, num_block, *ds, entireHueArray, hueLength, step, &gpu_cx, &gpu_cy, shouldPrint);

                
              //Multi-Object CamShift
              gpu_time_cost += launchMultiObjectTwoKernelCamShift(num_objects, &num_block, obj_block_ends, *ds, entireHueArray, hueLength, step, &gpu_cx, &gpu_cy, &sub_widths, &sub_heights, subFrameLengths, shouldPrint);
                
                for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
                {
                    gpu_objects[obj_cur].setCentroid(Point(gpu_cx[obj_cur], gpu_cy[obj_cur]));
                     //Multi-object MeanShift
                    gpu_objects[obj_cur].setWidthHeight(sub_widths[obj_cur], sub_heights[obj_cur]);
                    
                    gpu_objects[obj_cur].drawGPU_ROI(&frame);
                }
                
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
        float gpu_average_time_cost = gpu_time_cost / ((float) totalFrames);
        float cpu_average_time_cost = cpu_time_cost / ((float) totalFrames);
        printf("GPU average time cost in milliseconds: %f\n", gpu_average_time_cost);
        printf("CPU average time cost in milliseconds: %f\n", cpu_average_time_cost);
        if(shouldGPU)
            printf("Speed-up: %f\n", cpu_average_time_cost/ gpu_average_time_cost);
    }// end if shouldProcessVideo
    
    //clean-up
    outputVideo.release();
    free(histogram);
    free(gpu_row_offset);
    free(gpu_col_offset);
    freeDeviceStruct(ds);
	free(ds);
    free(entireHueArray);
    free(gpu_cx);
    free(gpu_cy);
    
    
    free(subFrameLengths);
    free(sub_widths);
    free(sub_heights);
    free(obj_block_ends);
    
    
    printf("Program exited successfully\n");
    return 0;
}
