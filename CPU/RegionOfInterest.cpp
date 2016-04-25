//
//  RegionOfInterest.cpp
//  ThesisSerialCamShift
//
//  Created by Matthew Perry on 1/28/16.
//  Copyright © 2016 Matthew Perry. All rights reserved.
//

#include "RegionOfInterest.hpp"

//default constructor, but requires a call to init to get appropriate values
RegionOfInterest::RegionOfInterest()
{
   /* _topLeft = Point(0,0);
    _bottomRight = Point(100,100);
    _width = _bottomRight.x -  _topLeft.x;
    _height = _bottomRight.y -  _topLeft.y;
    _frameWidth = 100;
    _frameHeight = 100;
    _centroid = calcCentroid(_topLeft, _width, _height);*/
}
void RegionOfInterest::init(Point topL, Point botR, int frameWidth, int frameHeight)
{
    _topLeft = topL;
    _bottomRight = botR;
    _width = botR.x - topL.x;
    _height = botR.y - topL.y;
    _frameWidth = frameWidth;
    _frameHeight = frameHeight;
    _centroid = calcCentroid(topL, _width, _height);
}

int RegionOfInterest::getTotalPixels(){
    return _width * _height;
}

void RegionOfInterest::setCentroid(Point centroid)
{
    _centroid = centroid;
    _topLeft = calcTopLeft(centroid, _width, _height);
    _bottomRight = calcBottomRight(_topLeft, _width, _height);
}

void RegionOfInterest::setROI(Point centroid, int width, int height)
{
    _centroid = centroid;
    _width =  width;
    _height =  height;
    _topLeft = calcTopLeft(centroid, width, height);
    _bottomRight = calcBottomRight(_topLeft, width, height);
}

Point RegionOfInterest::calcCentroid(Point topLeft, int w, int h)
{
    int cx = topLeft.x + (w / 2);
    int cy = topLeft.y + (h / 2);
    
    return Point(cx, cy);
    
}


Point RegionOfInterest::calcTopLeft(Point centroid, int ww, int wh)
{
    int topLeft_x = centroid.x - (ww / 2);
    int topLeft_y = centroid.y - (wh / 2);
    if(topLeft_x < 0) topLeft_x = 0;
    if(topLeft_y < 0) topLeft_y = 0;
    
    return Point(topLeft_x, topLeft_y);
    
}

Point RegionOfInterest::calcBottomRight(Point topLeft, int ww, int wh)
{
    int bottomRight_x = topLeft.x + ww;
    int bottomRight_y = topLeft.y + wh;
    if(bottomRight_x > _frameWidth - 1) bottomRight_x = _frameWidth- 1;
    if(bottomRight_y > _frameHeight - 1) bottomRight_y = _frameHeight -1;
    
    return Point(bottomRight_x, bottomRight_y);
}


int RegionOfInterest::getCenterX(){
    return  _centroid.x;
}

int RegionOfInterest::getCenterY(){
    return  _centroid.y;
}

int RegionOfInterest::getFrameHeight(){
    return _frameHeight;
}

int RegionOfInterest::getFrameWidth(){
    return _frameWidth;
}

Point RegionOfInterest::getTopLeft()
{
    return _topLeft;
}

int RegionOfInterest::getTopLeftX()
{
    return _topLeft.x;
}

int RegionOfInterest::getTopLeftY()
{
    return _topLeft.y;
}

Point RegionOfInterest::getBottomRight()
{
    return _bottomRight;
}

int RegionOfInterest::getBottomRightX()
{
    return _bottomRight.x;
}

int RegionOfInterest::getBottomRightY()
{
    return _bottomRight.y;
}

void RegionOfInterest::printROI()
{
    printf("******* PRINTING ROI ***********\n");
    printf("TopLeft --> (%d, %d) BottomRight(%d, %d)\n", _topLeft.x, _topLeft.y, _bottomRight.x, _bottomRight.y);
    printf("******* FINISHED PRINTING ROI ***********\n");
}

void RegionOfInterest::drawCPU_ROI(Mat * frame)
{
    rectangle(*frame, _topLeft, _bottomRight, Scalar( 0, 0, 255 ), THICKNESS + 4, 8, 0);
    circle( *frame, _centroid, 5.0, Scalar( 0, 255, 255 ), -1, 8, 0 );
}

void RegionOfInterest::drawGPU_ROI(Mat * frame)
{
    rectangle(*frame, _topLeft, _bottomRight, Scalar( 255, 0, 0 ), THICKNESS, 8, 0);
    circle( *frame, _centroid, 5.0, Scalar( 0, 255, 255 ), -1, 8, 0 );
}
