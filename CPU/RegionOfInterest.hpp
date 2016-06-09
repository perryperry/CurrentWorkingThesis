//
//  RegionOfInterest.hpp
//  ThesisSerialCamShift
//
//  Created by Matthew Perry on 1/28/16.
//  Copyright © 2016 Matthew Perry. All rights reserved.
//

#ifndef RegionOfInterest_hpp
#define RegionOfInterest_hpp

#include <stdio.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/core/core.hpp"



using namespace cv;
using namespace std;

#define THICKNESS 2

class RegionOfInterest{
public:
    int _width;
    int _height;
    RegionOfInterest();
    void init(Point topR, Point botL, int fWidth, int fHeight);
    void setWidthHeight(int width, int height);
    void drawCPU_ROI(Mat * frame, int obj_num, float angle);
    void drawGPU_ROI(Mat * frame, int obj_num, float angle);
    void setCentroid(Point centroid);
    
    
    void setWindowToFullFrame();
    void setCorners(Point centroid, int width, int height);
    
    
    void setROI(Point centroid, int width, int height);
    Point getTopLeft();
    Point getBottomRight();
    int getTotalPixels();
    int getFrameWidth();
    int getFrameHeight();
    void printROI();
    Point _centroid;
    int getCenterX();
    int getCenterY();
    int getBottomRightX();
    int getBottomRightY();
    int getTopLeftX();
    int getTopLeftY();
    void testDraw(Mat * frame, int object_num, Point top, Point bottom, Point center);
private:
    
    Point _topLeft;
    Point _bottomRight;
    int _frameWidth;
    int _frameHeight;
    Point calcBottomRight(Point topLeft, int w, int h);
    Point calcTopLeft(Point centroid, int ww, int wh);
    Point calcCentroid(Point topLeft, int w, int h);
    
};
#endif /* RegionOfInterest_hpp */
