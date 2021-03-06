//
//  main.cpp
//  ThesisSerialCamShift
//
//  Created by Matthew Perry on 1/27/16.
//  Copyright © 2016 Matthew Perry. All rights reserved.
//


#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/core/core.hpp"

#include "GPU/gpuMain.h"
#include "CPU/RegionOfInterest.hpp"
#include "CPU/UcharSerialCamShift.hpp"

#include <chrono>
#include <pthread.h>
#include <iostream>
#include <fstream>
#include <string>
//print colors
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

#include <stdio.h>

void writeHistogram(float * histogram, unsigned int num_objects )
{
    string hist_filename;
    printf("Enter the histogram output file name:");
    cin >> hist_filename;
    int i = 0;
    float line;
    ofstream myfile( hist_filename );
    if (myfile.is_open())
    {
        for(i = 0; i < num_objects * BUCKETS; i++)
        {
           myfile << histogram[i] << endl;
        }
        myfile.close();
    }
    else cout << "Unable to open file";
}

void loadHistogram(float * histogram, unsigned int num_objects )
{
    string hist_filename;
    printf("Enter the histogram input file name:");
    cin >> hist_filename;
    int i = 0;
    float line;
    ifstream myfile( hist_filename );
    if (myfile.is_open())
    {
        while ( myfile >> line )
        {
            histogram[i++] = line;
            if(i == BUCKETS * num_objects)
                break;
        }
        myfile.close();
    }
    else cout << "Unable to open file";
}

Mat removeBackGround(Mat frame, Ptr<BackgroundSubtractor> mog2)
{
    Mat fgMaskMOG2; //fore ground mask
    Mat foregroundImg;
    //update the background model
    mog2->apply(frame, fgMaskMOG2, 0);
    //convert mask from Grayscale to BGR
    cvtColor(fgMaskMOG2,fgMaskMOG2, COLOR_GRAY2BGR);
    // Blackout foreground
    foregroundImg = Scalar::all(0);
    
    frame.copyTo(foregroundImg, fgMaskMOG2);
    return foregroundImg;
}


void parameterCheck(unsigned int argCount)
{
    if(argCount != 3)
    {
        cout << RED "Usage: </path/to/videofile> </path/to/window/file>" RESET << endl;
        exit(-1);
    }
}

float cpuConvertBGR_To_Hue(Mat frame, unsigned char ** hueArray, unsigned int * step)
{
    high_resolution_clock::time_point time1;
    high_resolution_clock::time_point time2;
    
    time1 = high_resolution_clock::now();
    
    Mat hsvMat;
    cvtColor(frame, hsvMat, CV_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    split(hsvMat, hsv_channels);
    Mat hueMatrix = hsv_channels[0];
    *step = hueMatrix.step;
    memcpy(*hueArray, hueMatrix.data, sizeof(unsigned char) * hueMatrix.total());
    
    time2 = high_resolution_clock::now();
    auto cpu_duration = duration_cast<duration<double>>( time2 - time1 ).count();
    
    return (float)(cpu_duration * 1000.0);
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

void menu(unsigned int * num_objects, bool * deviceProp, bool * cpu, bool * gpu, bool * bgRemove, bool * windowAdjust, bool * histLoad)
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
    cout << CYNA "Should adjust window size" << RED " (0/1):";
    scanf("%d", &answer);
    *windowAdjust = answer;
    cout << CYNA "Should background remove" << RED " (0/1):";
    scanf("%d", &answer);
    *bgRemove = answer;
    cout << CYNA "Should load histogram file" << RED " (0/1):";
    scanf("%d", &answer);
    *histLoad = answer;
    
    cout << YELLOW "##############################################################\n\n" RESET;

    if(*num_objects == 0)
        exit(-1);
}

int main(int argc, const char * argv[])
{
    //timeMemoryTransfer();
    //exit(1);
    
    Ptr<BackgroundSubtractor> mog2 = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
    
    bool shouldAdjustWindowSize = false;
    bool shouldDisplayDeviceProperties = false;
    bool shouldBackgroundRemove = false;
    bool shouldCPU = false;
    bool shouldGPU = false;
    bool shouldBackProjectFrame = true;
    bool shouldPrint = false;
    bool shouldLoadHistogram = false;
    unsigned int num_objects = 1;
   
    parameterCheck(argc);
    menu(&num_objects, &shouldDisplayDeviceProperties, &shouldCPU, &shouldGPU, &shouldBackgroundRemove, &shouldAdjustWindowSize, &shouldLoadHistogram);
    
    if( shouldDisplayDeviceProperties )
        printDeviceProperties();
    else //Process Video
    {
        VideoCapture cap(argv[1]);
        unsigned int x, y, x2, y2, obj_cur = 0;
        
        ifstream infile(argv[2]);
        
        RegionOfInterest * cpu_objects = (RegionOfInterest * ) malloc(sizeof(RegionOfInterest) * num_objects);
        RegionOfInterest gpu_objects[num_objects];
        
        
        h_roi * h_roi_gpu = initHostROI(num_objects);
        
        
        //Read in windows from input file
        while (infile >> x >> y >> x2 >> y2)
        {
            if(obj_cur >= num_objects){
                break;
            }
            printf("Initializes search window #%d: (%d, %d) to (%d, %d)\n", obj_cur, x, y, x2, y2);
            cpu_objects[obj_cur].init(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
            gpu_objects[obj_cur].init(Point(x,y), Point(x2,y2), cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
            
            h_roi_gpu->h_topX[obj_cur] = x;
            h_roi_gpu->h_topY[obj_cur] = y;
            h_roi_gpu->h_bottomX[obj_cur] = x2;
            h_roi_gpu->h_bottomY[obj_cur] = y2;
            
            h_roi_gpu->h_cx[obj_cur] = (x2 + x) / 2;
            h_roi_gpu->h_cy[obj_cur] = (y2 + y) / 2;

            obj_cur++;
        }
        obj_cur = 0; //reset the current object index
        d_struct * ds = (d_struct *) malloc(sizeof(d_struct));
        
 
        
        
        
        
        
        
        printf("%d vs %d and %d vs %d \n", h_roi_gpu->h_cx[0],gpu_objects[0].getCenterX(), h_roi_gpu->h_cy[0], gpu_objects[0].getCenterY());
        
        
        
        
        
        
        float gpu_time_cost = 0.0f;
        float cpu_time_cost = 0.0f;
        float cpu_bgr_to_hue_time = 0.0f;
        float gpu_bgr_to_hue_time = 0.0f;
        
        unsigned int * gpu_cx = (unsigned int *) malloc(sizeof(int) * num_objects); //centroid x for gpu
        unsigned int * gpu_cy = (unsigned int *) malloc(sizeof(int) * num_objects); //centroid y for gpu
        
        //For multi-object
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
        Mat bg_removed_frame;
        
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

        /************************************* First Frame initialize and process ****************************************/
        cap.read(frame);
  
        float * histogram = (float *) calloc(sizeof(float), BUCKETS * num_objects);
        unsigned int hueLength = frame.total();
        unsigned char * entireHueArray = (unsigned char *) malloc(sizeof(unsigned char) * hueLength);
        cpuConvertBGR_To_Hue(frame, &entireHueArray, &step);

    
        //holds the bgr data from the entire frame, for gpu conversion to hue
       unsigned char * bgr = (unsigned char *) malloc(sizeof(unsigned char ) * frame.total() * 3);
    
        if( shouldLoadHistogram )
        {
            loadHistogram(histogram, num_objects);
         
            for(int i = 0; i < num_objects * BUCKETS ; i++)
                cout << histogram[i] << endl;
        }
        else
        {
            camShift.createHistogram(entireHueArray, step, cpu_objects, &histogram, num_objects);
            writeHistogram(histogram, num_objects);
        }
        
        if(shouldGPU)
            mainConstantMemoryHistogramLoad(histogram, num_objects);

            //For the initial frame, just render the initial search windows' positions
            for(obj_cur = 0; obj_cur < num_objects; obj_cur++){
               cpu_objects[obj_cur].drawCPU_ROI(&frame, obj_cur, 0);
                gpu_objects[obj_cur].drawGPU_ROI(&frame, obj_cur, 0);
                //load gpu starting values as well
                gpu_row_offset[obj_cur] = gpu_objects[obj_cur].getTopLeftY();
                gpu_col_offset[obj_cur] = gpu_objects[obj_cur].getTopLeftX();
                gpu_cx[obj_cur] = gpu_objects[obj_cur].getCenterX();
                gpu_cy[obj_cur] = gpu_objects[obj_cur].getCenterY();
            }
            
      
            if(shouldGPU)//gpu device struct for kernel memory re-use
                initDeviceStruct(num_objects,
                                 ds,
                                 h_roi_gpu,
                                 entireHueArray,
                                 hueLength,
                                 subFrameLengths,
                                 sub_widths,
                                 sub_heights);
        
        outputVideo.write(frame);
        
        int frame_count = 1;
        float cpu_angle = 0;
     
       while(cap.read(frame))
       {
           
           if( shouldBackgroundRemove )
           {
               bg_removed_frame = removeBackGround(frame, mog2);
               cpu_bgr_to_hue_time += cpuConvertBGR_To_Hue(bg_removed_frame, &entireHueArray, &step);
           }
           else
               cpu_bgr_to_hue_time += cpuConvertBGR_To_Hue(frame, &entireHueArray, &step);
           
            /******************************** CPU MeanShift until Convergence ***************************************/
            if(shouldCPU)
            {
               for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
               {
                    cpu_time_cost += camShift.cpuCamShift(
                                                         entireHueArray,
                                                         step,
                                                         &cpu_objects,
                                                         obj_cur,
                                                         histogram,
                                                         shouldPrint,
                                                         hueLength,
                                                         &cpu_angle,
                                                         shouldAdjustWindowSize
                                                         );
                   
                   cpu_objects[obj_cur].drawCPU_ROI(&frame, obj_cur, cpu_angle);
              }
            }
            /******************************** GPU MeanShift until Convergence **********************************************/
            if(shouldGPU)
            {
                memcpy(bgr, frame.data, sizeof(unsigned char ) * frame.total() * 3);
                gpu_bgr_to_hue_time += launchGPU_BGR_to_Hue(bgr, *ds, hueLength);
                gpu_time_cost += gpuCamShift(
                                            *ds,
                                            h_roi_gpu,
                                            num_objects,
                                            entireHueArray,
                                            hueLength,
                                            shouldAdjustWindowSize
                                            );
             
                for(obj_cur = 0; obj_cur < num_objects; obj_cur++)
                {
                    gpu_objects[obj_cur].setROI(Point(gpu_cx[obj_cur], gpu_cy[obj_cur]), sub_widths[obj_cur], sub_heights[obj_cur]);
                    //gpu_objects[obj_cur].drawGPU_ROI(&frame, obj_cur, 0);
                    
                    gpu_objects[obj_cur].testDraw(&frame,
                                                  obj_cur,
                                                  Point( h_roi_gpu->h_topX[obj_cur], h_roi_gpu->h_topY[obj_cur]),
                                                  Point( h_roi_gpu->h_bottomX[obj_cur],h_roi_gpu->h_bottomY[obj_cur]),
                                                  Point( h_roi_gpu->h_cx[obj_cur], h_roi_gpu->h_cy[obj_cur]));
                }
                
            }
            /******************************** Write to Output Video *******************************************/
            if(shouldBackProjectFrame){
               // if(shouldCPU)
                    //camShift.backProjectHistogram(entireHueArray, step, &frame, cpu_objects[0], histogram);
           }
           if(shouldCPU || shouldGPU)
               printf(BLUE "Frame #%d processed\n", frame_count++);
            outputVideo.write(frame);
                
         }//end while processing video frames
        
        //Average time costs structures
        float gpu_average_time_cost = gpu_time_cost / ((float) totalFrames);
        float cpu_average_time_cost = cpu_time_cost / ((float) totalFrames);
        float cpu_average_bgr_hue_time =  cpu_bgr_to_hue_time / ((float) totalFrames);
        float gpu_average_bgr_hue_time =  gpu_bgr_to_hue_time / ((float) totalFrames);
        
        printf(YELLOW "\n************************  RESULTS  ************************\n");
        
        printf(GREEN    "CPU average bgr to hue conversion time cost in milliseconds: ");
        printf(YELLOW "%f\n", cpu_average_bgr_hue_time);
        printf(GREEN "GPU average bgr to hue conversion time cost in milliseconds: ");
        printf(YELLOW "%f\n", gpu_average_bgr_hue_time);
        printf(RED "BGR to Hue conversion Speed-up: ");
        printf( YELLOW "%f\n\n", cpu_average_bgr_hue_time / gpu_average_bgr_hue_time);
        
        
        printf(GREEN "CPU average CAMSHIFT computation time cost in milliseconds: ");
        printf(YELLOW "%f\n", cpu_average_time_cost);
        printf(GREEN "GPU average CAMSHIFT computation time cost in milliseconds: ");
        printf(YELLOW "%f\n", gpu_average_time_cost);
        printf(RED "CAMSHIFT computation Speed-up: ");
        printf( YELLOW "%f\n\n", cpu_average_time_cost/ gpu_average_time_cost);
            
        printf(RED "Total Speed-up: ");
        printf(YELLOW "%f\n", (cpu_average_time_cost + cpu_average_bgr_hue_time) / (gpu_average_time_cost + gpu_average_bgr_hue_time));
    
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
        free(bgr);
        free(cpu_objects);
        
        freeHostROI(h_roi_gpu);
        
    }//end should not display device properties
    
    printf(MAGETA "Program exited successfully\n" RESET);
    return 0;
}
