//
//  Settings.h
//  CountObjects
//
//  Created by Don Johnson on 4/29/14.
//  Copyright (c) 2014 Donald Johnson. All rights reserved.
//

#ifndef CountObjects_Settings_h
#define CountObjects_Settings_h

#define FILLED -1
#define NUM_HU 8
#define WAIT_WIN 0
#define SHOW_WIN 1
#define FULL_STAT 0

extern cv::Mat src_gray;
extern int thresh;
extern int max_thresh;
extern cv::RNG rng;

/// Function header
void thresh_callback(int, void* );


#endif
