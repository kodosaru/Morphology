//
//  CountObjectsMethods.cpp
//  CountObjects
//
//  Created by Don Johnson on 4/26/14.
//  Copyright (c) 2014 Donald Johnson. All rights reserved.
//

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <fstream>
#include <sstream>
//Boost
#define BOOST_FILESYSTEM_NO_DEPRECATED
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
// Mine
#include "CountObjectsMethods.h"
#include "FloodFillMethods.h"
#include "Settings.h"

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

void constructRegionBlobLists(vector<vector<PIXEL>*>& regionLists, vector<cv::vector<PIXEL>*>& blobLists)
{
    for(unsigned short i=0;i<USHRT_MAX;i++)
        regionLists[i]=new vector<PIXEL>;
    
    for(unsigned short i=0;i<USHRT_MAX;i++)
        blobLists[i]=new vector<PIXEL>;
}

void destroyRegionBlobLists(vector<vector<PIXEL>*>& regionLists, vector<vector<PIXEL>*>& blobLists)
{
    // Destory region and blob lists
    for(int i=0;i<USHRT_MAX;i++)
    {
        if(regionLists[i]!=nullptr)
        {
            regionLists[i]->erase(regionLists[i]->begin(),regionLists[i]->end());
            regionLists[i]->~vector();
        }
        if(blobLists[i]!=nullptr)
        {
            blobLists[i]->erase(blobLists[i]->begin(),blobLists[i]->end());
            blobLists[i]->~vector();
        }
    }
}

void extractblobs(Mat& regions, int clusterCount, unsigned short& nRegion, vector<vector<PIXEL>*>& regionLists, unsigned short& nBlobs, vector<vector<PIXEL>*>& blobLists, std::string outputDataDir, std::string outputFileName)
{
    // Display region map
    Mat tempRegions(regions.size(),CV_8UC1);
    regions.convertTo(tempRegions, CV_8UC1);
    
    PIXEL blobPix;
    nBlobs=0;
    
    cout<<"Before removing very small and large areas"<<endl;
    for(int i=0;i<nRegion;i++)
    {
        cout<<"Region "<<i+1<<" has "<<regionLists[i]->size()<<" pixels"<<endl;
    }
    
    for(int i=0;i<nRegion;i++)
    {
        if(regionLists[i]->size() < MIN_REGION || regionLists[i]->size() > MAX_REGION)
        {
            for(int j=0;j<regionLists[i]->size();j++)
            {
                blobPix=(*regionLists[i])[j];
                //printf("Object pixel(%d,%d)=%0.0f\n",blobPix.pt.x,blobPix.pt.y,blobPix.val[0]);
                blobPix.val[0]=0;
                blobPix.val[1]=0;
                blobPix.val[2]=0;
                blobPix.val[2]=0;
                setPixel_8UC1(blobPix,tempRegions);
            }
            regionLists[i]->erase(regionLists[i]->begin(),regionLists[i]->end());
            regionLists[i]->~vector();
            regionLists[i]=nullptr;
        }
        else
        {
            nBlobs++;
        }
    }
    for(int i=0,j=0;i<nRegion;i++)
    {
        if(regionLists[i]!=nullptr)
        {
            for(vector<PIXEL>::iterator it = regionLists[i]->begin(); it != regionLists[i]->end(); ++it) {
                blobLists[j]->push_back(*it);
            }
            j++;
        }
    }
    
    cout<<endl<<"After removing very small and large areas"<<endl;
    float beta=50.0;
    int incVal=(int)((UCHAR_MAX-beta)/nBlobs);
    for(int i=0;i<nBlobs;i++)
    {
        cout<<"Blob "<<i<<", originally region "<<(*blobLists[i])[0].val[0]<<", has "<<blobLists[i]->size()<<" pixels and assigned "<<(i+1)*incVal+beta<<" grayscale value"<<endl;
        
        for(int j=0;j<blobLists[i]->size();j++)
        {
            blobPix=(*blobLists[i])[j];
            blobPix.val[0]=(i+1)*incVal+beta;
            blobPix.val[1]=0;
            blobPix.val[2]=0;
            blobPix.val[3]=0;
            setPixel_8UC1(blobPix,tempRegions);
        }
    }
    
    char cn[256];
    sprintf(cn,"%s%s%s%d%s",outputDataDir.c_str(),outputFileName.c_str(),"Regions",clusterCount,".png");
    imwrite(cn,tempRegions);
    if(SHOW_WIN)
    {
        string winName="Extraction";
        namedWindow(winName,WINDOW_NORMAL);
        imshow(winName,tempRegions);
    }
    if(WAIT_WIN)
        waitKey();
}

int readInImage(Mat& image, string inputDataDir, string fullInputfileName, string outputDataDir, string outputFileName, float resizeFactor)
{
    Mat input = imread(inputDataDir+fullInputfileName);
    if (input.empty())
    {
        cout << "Cannot open input image" << endl;
        return 1;
    }
    resize(input,input,Size(input.cols * resizeFactor, input.rows * resizeFactor),0,0,INTER_LINEAR);
    if(SHOW_MAIN_WIN)
    {
        string winName="Input";
        namedWindow(winName,WINDOW_NORMAL);
        imshow(winName, input);
    }
    if(WAIT_WIN)
        waitKey();
    
    Mat ycrcb;
    cvtColor(input,ycrcb,CV_BGR2YCrCb);
    imwrite(outputDataDir + outputFileName + ".png",ycrcb);
    
    if(SHOW_WIN)
    {
        string winName="YCrCb";
        namedWindow(winName);
        imshow(winName, ycrcb);
    }
    if(WAIT_WIN)
        waitKey();
    
    return 0;
}

unsigned long listFiles(string targetPath, vector<string>& fileNames)
{
    boost::progress_timer t( std::clog );
    
    fs::path full_path( fs::initial_path<fs::path>() );
    full_path = fs::system_complete( fs::path( targetPath ) );
    
    unsigned long file_count = 0;
    unsigned long dir_count = 0;
    unsigned long other_count = 0;
    unsigned long err_count = 0;
    
    if ( !fs::exists( full_path ) )
    {
        cout << "\nNot found: " << full_path.string() << std::endl;
        return 1;
    }
    
    if ( fs::is_directory( full_path ) )
    {
        cout << "\nIn directory: "
        << full_path.string() << "\n\n";
        fs::directory_iterator end_iter;
        for ( fs::directory_iterator dir_itr( full_path );
             dir_itr != end_iter;
             ++dir_itr )
        {
            try
            {
                if ( fs::is_directory( dir_itr->status() ) )
                {
                    ++dir_count;
                    std::cout << dir_itr->path().filename() << " [directory]\n";
                }
                else if ( fs::is_regular_file( dir_itr->status() ) )
                {
                    ++file_count;
                    std::cout << dir_itr->path().filename() << "\n";
                    fileNames.push_back(dir_itr->path().filename().string());
                }
                else
                {
                    ++other_count;
                    std::cout << dir_itr->path().filename() << " [other]\n";
                }
                
            }
            catch ( const std::exception & ex )
            {
                ++err_count;
                std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
            }
        }
        std::cout << "\n" << file_count << " files\n"
        << dir_count << " directories\n"
        << other_count << " others\n"
        << err_count << " errors\n";
    }
    else // must be a file
    {
        ++file_count;
        std::cout << "\nFound: " << full_path.string() << "\n";
        fileNames.push_back(full_path.string());
    }
    return file_count;
}

double calculateContours(Mat& src_gray, vector<Vec4i>& hierarchy, vector<vector<Point>>& contours, RNG rng, int thresh, int max_thresh, int nBlob, int& maxBlob)
{
    Mat canny_output;
    
    // Blur image
    blur(src_gray, src_gray, Size(3,3) );
    
    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    /// Find max contour
    double area=0;
    maxBlob = -1;
    double maxArea = -DBL_MAX;
    long size = contours.size();
    for( int i = 0; i<size; i++ )
    {
        area = contourArea(contours[i],false);
        if(area>maxArea)
        {
            maxBlob=i;
            maxArea=area;
        }
    }

    /// Analyze contour
    string sbConvex;
    double length = arcLength(contours[maxBlob],true);
    bool bConvex = isContourConvex(contours[maxBlob]);
    sbConvex = bConvex ? "true" : "false";
    printf("Blob[%d] area: %0.0f arc length: %0.0f convex: %s\n",nBlob,maxArea,length,sbConvex.c_str());

    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing, contours, maxBlob, color, 2, 8, hierarchy, 0, Point() );
    
    /// Show in a win
    if(SHOW_WIN)
    {
        stringstream ss;
        ss << nBlob;
        string sVal = ss.str();
        string winName = "Blob["+sVal+"] Contour";
        namedWindow(winName, WINDOW_NORMAL);
        imshow(winName, drawing );
    }
    if (WAIT_WIN) {
        waitKey();
    }
    return length;
}

void calculateContoursPost(Mat& src_gray, vector<Vec4i>& hierarchy, vector<vector<Point>>& contours, RNG rng, int thresh, int max_thresh, int nBlob, int& maxBlob)
{
    Mat canny_output;
    
    // Blur image
    blur(src_gray, src_gray, Size(3,3) );
    
    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    /// Find max contour
    double area=0;
    maxBlob = -1;
    double maxArea = -DBL_MAX;
    long size = contours.size();
    for( int i = 0; i<size; i++ )
    {
        area = contourArea(contours[i],false);
        if(area>maxArea)
        {
            maxBlob=i;
            maxArea=area;
        }
    }
    
    /// Draw contours
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours(src_gray, contours, maxBlob, color, 2, 8, hierarchy, 0, Point() );
}

void drawObjectContours(Mat& dst, vector<vector<Point>>& contours, RNG rng, int nBlob, int maxBlob)
{
    vector<Vec4i> hierarchy;
    
    /// Draw contours
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
    drawContours(dst, contours, maxBlob, color, 2, 8, hierarchy, 0, Point());
}


bool is_file_exist(string fileName)
{
    fstream infile(fileName);
    return infile.good();
}

