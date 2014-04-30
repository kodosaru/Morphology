//
//  Harris.cpp
//  CountObjects
//
//  Created by Don Johnson on 4/29/14.
//  Copyright (c) 2014 Donald Johnson. All rights reserved.
//

#include "Harris.h"

/**
 * @function cornerHarris_Demo.cpp
 * @brief Demo code for detecting corners using Harris-Stephens method
 * @author OpenCV team
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "Settings.h"

using namespace cv;
using namespace std;

void cornerHarris_demo(Mat& src, int thresh, int max_thresh)
{
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros(src.size(), CV_32FC1 );
    
    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    
    /// Detecting corners
    cornerHarris( src, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
    
    /// Normalizing
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    
    /// Drawing a circle around corners
    for( int j = 0; j < dst_norm.rows ; j++ )
    { for( int i = 0; i < dst_norm.cols; i++ )
    {
        if( (int) dst_norm.at<float>(j,i) > thresh )
        {
            circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
        }
    }
    }
    /// Showing the result
    if(SHOW_WIN)
    {
        string winName="Corners Detected";
        namedWindow(winName,WINDOW_NORMAL);
        imshow(winName, dst_norm_scaled );
    }
    if(WAIT_WIN)
        waitKey();
}

