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
#include <sstream>
#include <fstream>
#include <algorithm>

#define FILLED -1

using namespace cv;
using namespace std;

long split(const string &txt, vector<string> &strs, char ch)
{
    unsigned long pos = txt.find( ch );
    unsigned long initialPos = 0;
    strs.clear();
    
    // Decompose statement
    string  sTemp;
    while( pos != std::string::npos )
    {
        sTemp = txt.substr(initialPos, pos - initialPos);
        if(sTemp.size())
            strs.push_back(sTemp);
        initialPos = pos + 1;
        
        pos = txt.find( ch, initialPos );
    }
    
    // Add the last one
    long min = pos < txt.size() ? pos : txt.size();
    sTemp = txt.substr( initialPos, min - initialPos + 1 );
    if(sTemp.size())
        strs.push_back(sTemp);
    
    return strs.size();
}

struct huRef
{
    string objectDesc;
    vector<double> val;
};
typedef struct huRef HUREF;

void readInReferences(vector<HUREF>& references, string filePath)
{
    ifstream ref (filePath);
    cout <<filePath<<endl;
    string line;
    HUREF temp;
    vector<string> strs;
    bool bLabel;
	if (ref.is_open())
	{
		while (ref.good())
		{
            bLabel = true;
			getline(ref,line);
			cout << line << endl;
            split(line, strs, ' ');
            for(int i=0;i<strs.size();i++)
            {
                if(bLabel)
                {
                    bLabel = false;
                    temp.objectDesc = strs[i];
                }
                else{
                    temp.val.push_back(atof(strs[i].c_str()));
                    cout<<temp.objectDesc<<" Hu val: "<<temp.val[i-1]<<endl;
                }
            }
            references.push_back(temp);
            temp.val.clear();
		}
        ref.close();
	}
	else
	{
		cout << "Unable to open file " << filePath << endl;
	}
}

int classifyObject(vector<HUREF> references, vector<double> object)
{
    float tolerance = 1.0;
    
    if(references[0].val.size() != object.size())
    {
        cout<<"Unable to compute Euclidean distance in classifyObject() - vectors are of a different length";
        return -INT_MAX;
    }
    
    double minDist=DBL_MAX, tempDist,sum=0.0;
    int referenceNdx = -1;
    // Loop through all of the reference Hu variant vectors
    for(int j=0;j<references.size();j++)
    {
        sum = 0.0;
        // For each reference object vector, loop through the elements
        for(int k=0;k<references[j].val.size();k++)
        {
            sum += POW2(object[k] - references[j].val[k]);
        }
        tempDist = sqrt(sum);
        if(tempDist < tolerance && tempDist < minDist)
        {
            referenceNdx = j;
            minDist = tempDist;
        }
    }
    
    cout<<"Object's descriptor is "<<minDist<<"away from and closest to "<<references[referenceNdx].objectDesc<<"["<<referenceNdx<<"]'s descriptor"<<endl;
    return referenceNdx;
}

void readInReference(vector<vector<string>> reference)
{
    
}
using namespace std;
using namespace cv;

int main(int argc, const char * argv[])
{
    //vector<HUREF> dummy;
    //string path="/Users/donj/references.txt";
    //readInReferences(dummy, path);
    //return 0;
    
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
    imshow("Binary",binary);
    
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
    imshow("Blobs",tempRegions);
    
    // Calculate remaining statistics
    for(int i=0;i<nBlobs;i++)
    {
        if(blobLists[i]!=nullptr)
        {
            /*
             Mat *covar = covarianceMatrix(*blobLists[i]);
             cout<<"Covariance Matrix"<<endl;
             cout<<"| "<<covar->at<double>(0,0)<<" "<<covar->at<double>(0,1)<<"|"<<endl;
             cout<<"| "<<covar->at<double>(1,0)<<" "<<covar->at<double>(1,1)<<"|"<<endl;
             Mat eval, evec;
             eigen((*covar),eval,evec);
             // Printing out column order and taking negative of eigenvectors like MATLAB
             evec = -1.0 * evec;
             cout<<"Eigenvector Matrix"<<endl;
             cout<<"| "<<evec.at<double>(1,0)<<" "<<evec.at<double>(0,0)<<"|"<<endl;
             cout<<"| "<<evec.at<double>(1,1)<<" "<<evec.at<double>(0,1)<<"|"<<endl;
             cout<<"OpenCV Calc Eigenvalue Matrix"<<endl;
             cout<<"| "<<eval.at<double>(1,0)<<" "<<0<<"|"<<endl;
             cout<<"| "<<0<<" "<<eval.at<double>(0,0)<<"|"<<endl;
             evec = *eigenvalueMatrix(*blobLists[i]);
             cout<<"Moment Calc Eigenvalue Matrix"<<endl;
             cout<<"| "<<eval.at<double>(1,0)<<" "<<0<<"|"<<endl;
             cout<<"| "<<0<<" "<<eval.at<double>(0,0)<<"|"<<endl;
             cout<<"Eccentricity: "<<eccentricity(*blobLists[i])<<endl;
             covar->release();
             eigenvalueMatrix->release();

             printf("Raw Moments: %0.2f  %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f  %0.2f %0.2f %0.2f\n",Mij(*blobLists[i],0,0), Mij(*blobLists[i],1,0),  Mij(*blobLists[i],0,1),  Mij(*blobLists[i],2,0),  Mij(*blobLists[i],1,1),  Mij(*blobLists[i],0,2), Mij(*blobLists[i],3,0),  Mij(*blobLists[i],2,1), Mij(*blobLists[i],1,2),  Mij(*blobLists[i],0,3));
             printf("Central Moments: %0.2f  %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n",muij(*blobLists[i],2,0), muij(*blobLists[i],1,1), muij(*blobLists[i],0,2),muij(*blobLists[i],3,0), muij(*blobLists[i],2,1), muij(*blobLists[i],1,2), muij(*blobLists[i],0,3));
             
             printf("Normalized Moments: %0.2f  %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n",etaij(*blobLists[i],2,0), etaij(*blobLists[i],1,1), etaij(*blobLists[i],0,2),etaij(*blobLists[i],3,0), etaij(*blobLists[i],2,1), etaij(*blobLists[i],1,2), etaij(*blobLists[i],0,3));
             */
            
            // Hu moments
            vector<vector<double>> huMoments(nBlobs);
            for(int j=1;j<=8;j++)
            {
                huMoments[i].push_back(Hui(*blobLists[i],j));
            }
            cout<<"Hu Invarients:";
            for(int j=0;j<NUM_HU;j++)
                printf(" %0.2f",huMoments[i][j]);
            cout<<endl;
            
            // Classify blobs based on Hu invariants
            classifyObject(references, huMoments[i]);
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
