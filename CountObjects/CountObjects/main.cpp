//
//  main.cpp
//  CountObjects
//
//  Created by Don Johnson on 4/26/14.
//  Copyright (c) 2014 Donald Johnson. All rights reserved.
//

#include <iostream>
#include "FloodFillMethods.h"
#include "Stack.hpp"
#include "CountObjectsMethods.h"
#include "KMeansMethods.h"
#include "Moments.h"
#include "Settings.h"
#include <sstream>

using namespace cv;
using namespace std;



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
#include "main.h"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[])
{
    // Input arguments
    // 0 program name
    // 1 full input file name with extension
    // 2 output file name without extension used to save intermiediate products
    // 3 boolen whether to save k-means state: background classes and class palette
    // 4 maximum number k-means classes to allow
    
    string inputDataDir="/Users/donj/workspace/Morphology/Data/Input/";
    string outputDataDir="/Users/donj/workspace/Morphology/Data/Output/";
    
    if(argc!=5)
    {
        cout<<"Incorrect number of program arguments"<<endl;
        return 1;
    }
    
    // Read in program arguments
    string inputFullFileName=argv[1];
    string outputFileName=argv[2];
    Mat input;
    if(readInImage(input, inputDataDir, inputFullFileName, outputDataDir, outputFileName, 1.0/3.0))
    {
        cout<<"Unable to read input file "<<inputDataDir+inputFullFileName<<endl;
        return 1;        
    }
    bool bSaveState;
    if(strcmp(argv[3],"true")==0)
        bSaveState=true;
    else if(strcmp(argv[3],"false")==0)
        bSaveState=false;
    else
    {
        cout<<"You must specify whether to save the state of the k-means classification"<<endl;
    }
    int maxClusters=atoi(argv[4]);
    
    // Perform k-means classfication
    int clusterCount=0;
    if(bSaveState)
    {
        if(kMeansCustom(bSaveState, outputDataDir, outputFileName, maxClusters, clusterCount))
            return 1;
    }
    else
    {
        clusterCount=maxClusters;
        if( classifyUsingSavedCenters(bSaveState, outputDataDir, outputFileName, maxClusters, clusterCount))
            return 1;
    }

    // Set up flood fill segmentation
    char cn[256];
    sprintf(cn,"%s%s%s%d%s",outputDataDir.c_str(),outputFileName.c_str(),"Binary",clusterCount,".png");
    cout<<"Path to binary image file: "<<cn<<endl;
    Mat binary=imread(cn,CV_LOAD_IMAGE_GRAYSCALE);
    
    // Create data structures to save region and blob information
    unsigned short nRegions=0;
    vector<vector<PIXEL>*> regionLists(USHRT_MAX);
    unsigned short  nBlobs=0;
    vector<vector<PIXEL>*> blobLists(USHRT_MAX);
    constructRegionBlobLists(regionLists, blobLists);
    
    // Create image of flood-fill regions
    Mat regions(binary.size(),CV_16UC1);
    
    // Segment the binary object into regions
    floodFill(binary, regions, nRegions, regionLists);
    
    // Extract a list of object candiate blobs and the list of pixels in each
    extractblobs(regions, clusterCount, nRegions, regionLists, nBlobs, blobLists, outputDataDir, outputFileName);
    
    // Put blobs in their own images
    vector<Mat> blobImages;
    PIXEL blobPix;
    destroyAllWindows();
    for(int i=0;i<nBlobs;i++)
    {
        blobImages.push_back(Mat(regions.rows,regions.cols,CV_8U));
        blobImages[i]=Scalar(0,0,0,0);
        cout<<"size of blob "<<i<<" is "<<blobLists[i]->size()<<endl;
        for(int j=0;j<blobLists[i]->size();j++)
        {
            blobPix=(*blobLists[i])[j];
            if( blobPix.val[0])
                blobImages[i].at<uchar>(blobPix.pt)=255;
        }
        stringstream ss;
        ss << i;
        string sVal = ss.str();
        string winName = "Blob["+sVal+"]";
        namedWindow(winName,WINDOW_NORMAL);
        imshow("Blob"+sVal,blobImages[i]);
        
        Mat src_gray;
        RNG rng(12345);
        int thresh=100;
        int max_thresh = 255;
        
        /// Calculate contours
        winName = "Blob["+sVal+"] Contour";
        calculateContours(blobImages[i], rng, thresh, max_thresh, i, winName);
        
        //if(WAIT_WIN)
            waitKey();
    }
    destroyAllWindows();

    // Read in region file
    sprintf(cn,"%s%s%s%d%s",outputDataDir.c_str(),outputFileName.c_str(),"Regions",clusterCount,".png");
    Mat tempRegions=imread(cn,CV_8UC1);
    
    // Read in Hu invariants reference vectors and classify blobs
    vector<HUREF> references;
    readInReferences(references, outputDataDir+"references.txt");
    
    // Calculate centroids of blobs
    for(int i=0;i<nBlobs;i++)
    {
        if(blobLists[i]!=nullptr)
        {
            Scalar labelColor;
            Point centroid(xbar(*blobLists[i]), ybar(*blobLists[i]));
            stringstream ss;
            ss << i;
            string sVal = ss.str();
            vector<PIXEL> blob=*blobLists[i];
            if (blob[0].val[0]<=128)
                labelColor=CV_RGB(180,180,180);
            else
                labelColor=CV_RGB(80,80,80);
            circle(tempRegions, centroid, 5, labelColor, FILLED);
            putText(tempRegions, sVal, Point(centroid.x+4,centroid.y-1),FONT_HERSHEY_SIMPLEX, 0.45, labelColor,1);
            
            // Begin blob statistics and moments
            cout<<endl<<"Blob["<<i<<"]"<<endl;
            cout<<"Centroid of blob: ("<<centroid.x<<","<<centroid.y<<")"<<endl;
            
            // Draw major axis
            Mat *orient = orientation(*blobLists[i]);
            double angle = orient->at<double>(0,0);
            cout<<"Orientation: "<<angle*180/M_PI<<" deg"<<endl;
            Scalar lineColor = CV_RGB(0,0,0);
            int d = 40;
            int deltaX = int(d * tan(angle) + 0.5);
            circle(tempRegions, Point(centroid.x+deltaX,centroid.y-d), 2, lineColor, FILLED);
            circle(tempRegions, Point(centroid.x-deltaX,centroid.y+d), 2, lineColor, FILLED);
            line(tempRegions, Point(centroid.x+deltaX,centroid.y-d), Point(centroid.x-deltaX,centroid.y+d), lineColor);
            orient->release();
            
        }
        else
        {
            cout<<"Blob "<<i<<" has a null pointer"<<endl;
        }
    }
    sprintf(cn,"%s%s%s%d%s",outputDataDir.c_str(),outputFileName.c_str(),"Blobs",clusterCount,".png");
    imwrite(cn,tempRegions);
    namedWindow("Moments, Centroid and Major Axis",WINDOW_NORMAL);
    imshow("Moments, Centroid and Major Axis",tempRegions);
    if(WAIT_WIN)
        waitKey();
    
    // Calculate remaining statistics
    for(int i=0;i<nBlobs;i++)
    {
        if(blobLists[i]!=nullptr)
        {
            if(FULL_STAT)
                blobStatistics(blobLists,i);

            
            // Hu moments
            cout<<endl;
            vector<vector<double>> huMoments(nBlobs);
            for(int j=1;j<=NUM_HU;j++)
            {
                huMoments[i].push_back(Hui(*blobLists[i],j));
            }
            cout<<"Hu Invarients:";
            for(int j=0;j<NUM_HU;j++)
                printf(" %0.6f",huMoments[i][j]);
            cout<<endl;
            
            // Classify blobs based on Hu invariants
            classifyObject(references, i, huMoments[i]);
        }
        else
        {
            cout<<"Blob "<<i<<" has a null pointer"<<endl;
        }
        
        
    } // End of blobs


    destroyRegionBlobLists(regionLists, blobLists);

    waitKey();
    
    return 0;
}
