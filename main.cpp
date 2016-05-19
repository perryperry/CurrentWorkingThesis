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

//prunsigned int colors
#define RESET "\033[0m"
#define RED "\x1B[31m"
#define GREEN "\x1B[32m"
#define YELLOW "\x1B[33m"
#define BLUE "\x1B[34m"
#define MAGETA "\x1B[35m"
#define CYNA "\x1B[36m"
#define WHITE "\x1B[37m"

using namespace cv;
using namespace std;
using namespace std::chrono;

#define OUTPUTFILENAME "out.mov"
#define MAXTHREADS 3

void parameterCheck(unsigned int argCount)
{
    if(argCount != 3)
    {
        cout << RED "Usage: </path/to/videofile> </path/to/window/file>" RESET << endl;
        exit(-1);
    }
}

unsigned int calculateHueArrayLength(Mat frame, unsigned int * step, unsigned int * mat_rows, unsigned int * mat_cols)
{
    Mat hsvMat;
    cvtColor(frame, hsvMat, CV_RGB2HSV);
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    unsigned int total = hueMatrix.total();
    *step = hueMatrix.step;
    *mat_rows = hueMatrix.rows;
    *mat_cols = hueMatrix.cols;
    return total;
}

unsigned int convertToHueArray(Mat frame, unsigned char ** hueArray)
{
    Mat hsvMat;
    cvtColor(frame, hsvMat, CV_RGB2HSV);
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    unsigned int i = 0;
    for(i = 0; i < hueMatrix.total(); i ++)
    {
        (*hueArray)[i]=(unsigned char) hueMatrix.data[i];
    }
    return hueMatrix.total();
}

void printHueSum(unsigned char * entireHueArray, unsigned int hueLength)
{
    long total = 0;
    for(unsigned int i =0; i < hueLength; i++)
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

void menu(unsigned int * num_objects, bool * deviceProp, bool * cpu, bool * gpu, bool * print, bool * windowAdjust)
{
    unsigned int answer = 0;
    cout << YELLOW "\n##############################################################\n";
    printf(  "GPU vs CPU meanshift menu:\n");
    cout << CYNA "Should display device properties," << RED " SKIP" << CYNA << " processing video " << RED "(0/1):" RESET;
    scanf("%d", &answer);
    *deviceProp = answer;
    
    if(*deviceProp) //Won't process video, skip the rest of the menu options
        return;
    
    cout << CYNA "Number of objects:" RED;
    scanf("%d", &answer);
    *num_objects = answer;
    cout <<CYNA "Should run cpu version" << RED " (0/1):";
    scanf("%d", &answer);
    *cpu = answer;
    cout << CYNA "Should run gpu version" << RED " (0/1):";
    scanf("%d", &answer);
    *gpu = answer;
    cout << CYNA "Should print intermediate output" << RED " (0/1):";
    scanf("%d", &answer);
    *print = answer;
    cout << CYNA "Should adjust window size" << RED " (0/1):";
    scanf("%d", &answer);
    *windowAdjust = answer;
    cout << YELLOW "##############################################################\n\n" RESET;

    if(*num_objects == 0)
        exit(-1);
}

int main(int argc, const char * argv[])
{
    //timeMemoryTransfer();
    //exit(1);
    
    bool shouldAdjustWindowSize = false;
    bool shouldDisplayDeviceProperties = false;
    bool shouldCPU = false;
    bool shouldGPU = false;
    bool shouldBackProjectFrame = false;
    bool shouldPrint = false;
    unsigned int num_objects = 1;
   
    parameterCheck(argc);
    menu(&num_objects, &shouldDisplayDeviceProperties, &shouldCPU, &shouldGPU, &shouldPrint, &shouldAdjustWindowSize);
    
    if( shouldDisplayDeviceProperties )
        printDeviceProperties();
    else //Process Video
    {
    
        VideoCapture cap(argv[1]);
        unsigned int x, y, x2, y2, obj_cur = 0;
        
        ifstream infile(argv[2]);
        
        RegionOfInterest cpu_objects[num_objects];
        RegionOfInterest gpu_objects[num_objects];
        
        //Read in windows from input file
        while (infile >> x >> y >> x2 >> y2)
        {
            if(obj_cur > num_objects){
                cout << RED "ERROR: Too many lines in input file for number of objects to track!" RESET<< endl;
                exit(-1);
            }
            printf("Initializes search window #%d: (%d, %d) to (%d, %d)\n", obj_cur, x, y, x2, y2);
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
        
        unsigned int * gpu_cx = (unsigned int *) malloc(sizeof(int) * num_objects); //centroid x for gpu
        unsigned int * gpu_cy = (unsigned int *) malloc(sizeof(int) * num_objects); //centroid y for gpu
        
        //For multi-object test
        unsigned int * obj_block_ends = (unsigned int *) malloc(sizeof(int) * num_objects); //ending block of each object in kernel
        unsigned int * subFrameLengths = (unsigned int *) malloc(sizeof(int) * num_objects);;
        unsigned int * sub_widths = (unsigned int *) malloc(sizeof(int) * num_objects);;
        unsigned int * sub_heights = (unsigned int *) malloc(sizeof(int) * num_objects);
        
        for(obj_cur = 0; obj_cur < num_objects; obj_cur++){
           subFrameLengths[obj_cur] = gpu_objects[obj_cur].getTotalPixels();
           sub_widths[obj_cur] = gpu_objects[obj_cur]._width;
           sub_heights[obj_cur] = gpu_objects[obj_cur]._height;
        }
        
        
        unsigned int cpu_cx = 0; //centroid x for cpu
        unsigned int cpu_cy = 0; //centroid x for cpu
        unsigned int step = 0; //The width of the entire frame
        unsigned int mat_rows = 0, mat_cols = 0;
        unsigned int * gpu_row_offset = (unsigned int *) malloc(sizeof(int) * num_objects); //for gpu
        unsigned int * gpu_col_offset = (unsigned int *) malloc(sizeof(int) * num_objects); //for gpu
        SerialCamShift camShift;
        Mat frame, hsv;
        
        /*************************************** Open Output VideoWriter *********************************************/
        //Code for opening VideoWriter taken from http://docs.opencv.org/3.1.0/d7/d9e/tutorial_video_write.html#gsc.tab=0
        VideoWriter outputVideo = VideoWriter();
        unsigned int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));//get codec type in unsigned int form
        // Transform from unsigned int to char via Bitwise operators
        char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
        Size size = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
        outputVideo.open(OUTPUTFILENAME, ex, cap.get(CV_CAP_PROP_FPS), size, true);
        
        unsigned int totalFrames = 0;
        
        if (! outputVideo.isOpened()){
            cout  << "Could not open the output video for write."<< endl;
            exit(-1);
        }
        else{
            totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
        }

         /*************************************** Testing Pthread *********************************************/
        
        /* pthread_t thread1, thread2;
         pthread_create(&thread1, NULL, test, (void *)"Keeping it trillion from thread 1.\n");
         pthread_create(&thread2, NULL, test, (void *)"Keeping it trillion from thread 2.\n");
        
         pthread_join(thread1, NULL);*/
        
        /************************************* First Frame initialize and process ****************************************/
        cap.read(frame);
        float * histogram = (float *) calloc(sizeof(float), BUCKETS * num_objects);
        unsigned int hueLength = calculateHueArrayLength(frame, &step, &mat_rows, &mat_cols);
        
        printf("Mat: rows: %d cols: %d\n", mat_rows, mat_cols);
        unsigned char * entireHueArray = (unsigned char *) malloc(sizeof(unsigned char) * hueLength);
        convertToHueArray(frame, &entireHueArray);

        camShift.createHistogram(entireHueArray, step, cpu_objects, &histogram, num_objects);

        mainConstantMemoryHistogramLoad(histogram, num_objects);

        //For the initial frame, just render the initial search windows' positions
        for(obj_cur = 0; obj_cur < num_objects; obj_cur++){
           cpu_objects[obj_cur].drawCPU_ROI(&frame, obj_cur);
            gpu_objects[obj_cur].drawGPU_ROI(&frame, obj_cur);
            //load gpu starting values as well
            gpu_row_offset[obj_cur] = gpu_objects[obj_cur].getTopLeftY();
            gpu_col_offset[obj_cur] = gpu_objects[obj_cur].getTopLeftX();
            gpu_cx[obj_cur] = gpu_objects[obj_cur].getCenterX();
            gpu_cy[obj_cur] = gpu_objects[obj_cur].getCenterY();
        }
        outputVideo.write(frame);
  
        //gpu device struct for kernel memory re-use
        unsigned int num_block = initDeviceStruct(num_objects,
                                                 ds,
                                                 obj_block_ends,
                                                 entireHueArray,
                                                 hueLength,
                                                 gpu_cx,
                                                 gpu_cy,
                                                 gpu_col_offset,
                                                 gpu_row_offset,
                                                 subFrameLengths,
                                                 sub_widths,
                                                 sub_heights);

       while(cap.read(frame))
       {
            hueLength = convertToHueArray(frame, &entireHueArray);
            /******************************** CPU MeanShift until Convergence ***************************************/
            if(shouldCPU)
            {
               for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
               {
                    cpu_cx = cpu_objects[obj_cur].getCenterX();
                    cpu_cy = cpu_objects[obj_cur].getCenterY();
                   
                    if( shouldAdjustWindowSize )
                    {
                       unsigned int width = 0, height = 0;
                       cpu_time_cost += camShift.cpuCamShift(entireHueArray, step, cpu_objects[obj_cur], obj_cur, histogram, shouldPrint, &cpu_cx, &cpu_cy, &width, &height, hueLength);
                       cpu_objects[obj_cur].setWidthHeight(width, height);
                   }
                   else
                   {
                       cpu_time_cost += camShift.cpuMeanShift(entireHueArray, step, cpu_objects[obj_cur], obj_cur, histogram, shouldPrint, &cpu_cx, &cpu_cy);
                   }
                   cpu_objects[obj_cur].setCentroid(Point(cpu_cx, cpu_cy));
                   cpu_objects[obj_cur].drawCPU_ROI(&frame, obj_cur);
              }
            }
            /******************************** GPU MeanShift until Convergence **********************************************/
            if(shouldGPU)
            {
                gpu_time_cost += gpuCamShift(
                                            *ds,
                                            num_objects,
                                            entireHueArray,
                                            hueLength,
                                            step,
                                            &sub_widths,
                                            &sub_heights,
                                            &gpu_cx,
                                            &gpu_cy,
                                            shouldAdjustWindowSize);
             
                for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
                {
                    //printf("OUTSIDE OF KERNEL (%d) --> cx: %d cy: %d WIDTH: %d HEIGHT: %d\n", obj_cur, gpu_cx[obj_cur], gpu_cy[obj_cur], sub_widths[obj_cur], sub_heights[obj_cur]);
                    gpu_objects[obj_cur].setROI(Point(gpu_cx[obj_cur], gpu_cy[obj_cur]), sub_widths[obj_cur], sub_heights[obj_cur]);
                    gpu_objects[obj_cur].drawGPU_ROI(&frame, obj_cur);
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
        printf(RED "GPU average time cost in milliseconds: "); printf(YELLOW "%f\n", gpu_average_time_cost);
        printf(RED "CPU average time cost in milliseconds: "); printf(YELLOW "%f\n", cpu_average_time_cost);
        if(shouldGPU)
            printf(RED "Speed-up: "); printf( YELLOW "%f\n", cpu_average_time_cost/ gpu_average_time_cost);
        
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
        
    }//end should not display device properties
    
    printf(MAGETA "Program exited successfully\n" RESET);
    return 0;
}
