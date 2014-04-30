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

using namespace cv;
using namespace std;

/// Global variables
Mat src_gray;
int thresh;
int max_thresh;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo( int, void* );

/**
 * @function main
 */
int harris(Mat& src)
{
    /// Load source image and convert it to gray
    //src = imread( argv[1], 1 );
    //cvtColor( src, src_gray, COLOR_BGR2GRAY );
    src.copyTo(src_gray);
    
    /// Create a window and a trackbar
    //namedWindow( source_window, WINDOW_AUTOSIZE );
    Size sz=src.size();
    //createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo, &sz );
    //imshow( source_window, src );
    
    cornerHarris_demo( 0, &sz );
    
    //waitKey(0);
    return(0);
}

/**
 * @function cornerHarris_demo
 * @brief Executes the corner detection and draw a circle around the possible corners
 */
void cornerHarris_demo( int, void* userData )
{
    Size sz = *((Point*)&userData);
    cout << sz << endl;
    
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( sz, CV_32FC1 );
    
    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    
    /// Detecting corners
    cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
    
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
    namedWindow( corners_window, WINDOW_AUTOSIZE );
    imshow( corners_window, dst_norm_scaled );
}

