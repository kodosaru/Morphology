//
//  Moments.cpp
//  CountObjects
//
//  Created by Don Johnson on 4/26/14.
//  Copyright (c) 2014 Donald Johnson. All rights reserved.
//
//  From http://en.wikipedia.org/wiki/Image_moment

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Moments.h"
#include "FloodFillMethods.h"
#include "math.h"
#include "Settings.h"
#include <fstream>
#include <algorithm>
#include <iomanip>

using namespace cv;
using namespace std;

void blobStatistics(vector<vector<PIXEL>*>& blobLists, int nBlob)
{
    Mat *covar = covarianceMatrix(*blobLists[nBlob]);
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
    evec = *eigenvalueMatrix(*blobLists[nBlob]);
    
    cout<<"Moment Calc Eigenvalue Matrix"<<endl;
    cout<<"| "<<eval.at<double>(1,0)<<" "<<0<<"|"<<endl;
    cout<<"| "<<0<<" "<<eval.at<double>(0,0)<<"|"<<endl;
    
    cout<<"Eccentricity: "<<eccentricity(*blobLists[nBlob])<<endl;
    covar->release();
    
    printf("Raw Moments: %0.2f  %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f  %0.2f %0.2f %0.2f\n",Mij(*blobLists[nBlob],0,0), Mij(*blobLists[nBlob],1,0),  Mij(*blobLists[nBlob],0,1),  Mij(*blobLists[nBlob],2,0),  Mij(*blobLists[nBlob],1,1),  Mij(*blobLists[nBlob],0,2), Mij(*blobLists[nBlob],3,0),  Mij(*blobLists[nBlob],2,1), Mij(*blobLists[nBlob],1,2),  Mij(*blobLists[nBlob],0,3));
    printf("Central Moments: %0.2f  %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n",muij(*blobLists[nBlob],2,0), muij(*blobLists[nBlob],1,1), muij(*blobLists[nBlob],0,2),muij(*blobLists[nBlob],3,0), muij(*blobLists[nBlob],2,1), muij(*blobLists[nBlob],1,2), muij(*blobLists[nBlob],0,3));
    
    printf("Normalized Moments: %0.2f  %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n",etaij(*blobLists[nBlob],2,0), etaij(*blobLists[nBlob],1,1), etaij(*blobLists[nBlob],0,2),etaij(*blobLists[nBlob],3,0), etaij(*blobLists[nBlob],2,1), etaij(*blobLists[nBlob],1,2), etaij(*blobLists[nBlob],0,3));
}

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
            if(line.size() == 0)
                break;
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

string classifyObject(vector<HUREF> references, int nObject, vector<double> object, vector<int>& inventory)
{
    // Actually using square of distance to avoid taking square root
    
    // The reference vector is two elements bigger than the object vector because it include area and perimeter length
    if(references[0].val.size() != object.size())
    {
        cout<<"Unable to compute Euclidean distance in classifyObject() - reference and object vectors are of a different length"<<endl;;
        return "FAIL";
    }
    
    double minDist=DBL_MAX, tempDist;
    int referenceNdx = -1;
    if(references[0].val.size() != (NUM_HU_INVARIANTS + 2))
    {
        cout<<"The number of statistics being compared is greater than the references provided"<<endl;
        return "FAIL";
    }
    // Loop through all of the reference Hu variant vectors
    for(int j=0;j<NUM_DESCRIPTORS;j++)
    {
        tempDist = 0.0;
        // For each reference object vector, loop through the elements
        for(int k=0;k<NUM_HU_INVARIANTS;k++)
        {
            //printf("object[%d]: %0.2f references[%d].val[%d]: %0.2f\n",k,object[k],j,k,references[j].val[k]);
            tempDist += POW2(object[k] - references[j].val[k]);
        }
        if(tempDist < minDist)
        {
            referenceNdx = j;
            minDist = tempDist;
        }
    }
    double area = object[8];
    cout<<"Object's["<<nObject<<"] area is "<<setprecision(4)<<area<<fixed<<endl;
    double length = object[9];
    cout<<"Object's["<<nObject<<"] perimeter length is "<<setprecision(4)<<length<<fixed<<endl;
    double compactness = object[8]/object[9];
    cout<<"Object's["<<nObject<<"] compactness is "<<setprecision(4)<<compactness<<fixed<<endl;
    double hu34Sum = object[2]+object[3];
    cout<<"Object's["<<nObject<<"] sum of Hu 3+4 is "<<setprecision(4)<<hu34Sum<<fixed<<endl;

    if(minDist > CLASSIFIER_TOLERANCE)
    {
           referenceNdx = -1;
    }

    string finalClassification;
    if(referenceNdx != -1)
    {
        // If preliminary classification based on Hu invariants is fork or knife, use compactness to make final decision
        if(referenceNdx == FORK || referenceNdx == KNIFE)
        {
            if(area > KNIFE_AREA_BOUNDARY)
            {
                finalClassification="Knife";
                referenceNdx=KNIFE;
                inventory[KNIFE]++;
            }
            else
            {
                if(hu34Sum > FORK_HU34_SUM_BOUNDARY)
                {
                    finalClassification="Fork";
                    referenceNdx=FORK;
                    inventory[FORK]++;
                }
                else
                {
                    finalClassification="Unknown";
                    referenceNdx=UNKNOWN;
                    inventory[UNKNOWN]++;
                }
            }
        }
        // Object is a spoon
        else
        {
            inventory[SPOON]++;
            finalClassification=references[referenceNdx].objectDesc;
        }
    }
    // Unknown is the catch-all classification
    else
    {
        finalClassification="Unknown";
        referenceNdx=UNKNOWN;
        inventory[UNKNOWN]++;
    }
    cout<<"Object's["<<nObject<<"] descriptor is Hu invariant distance "<<setprecision(2)<<minDist<<" away from and closest to "<<finalClassification<<"'s descriptor"<<endl;

    return finalClassification;
}

// Orientaton Angle
Mat* orientation(vector<PIXEL> v, int nBlob)
{
    Mat *m=new Mat(2,1,CV_64FC1);
    double mup20 = muPrimeij(v, 2, 0);
    double mup02 = muPrimeij(v, 0, 2);
    if(mup20 == mup02)
    {
        cout<<"Orientation cannot be computed because muPrime20 equals muPrime02"<<endl;
        m->at<double>(0,0) = -DBL_MAX;
        m->at<double>(1,0) = -DBL_MAX;
        return m;
    }
    // range -90 to 90 deg
    double rad = 0.5 * atan( (2.0 * muPrimeij(v, 1, 1)) / (mup20 - mup02) );
    //if(rad > 0)
        //rad+=M_PI/2;
    double deg = 180.0/M_PI * m->at<double>(0,0);
    m->at<double>(0,0) = rad;
    m->at<double>(1,0) = deg;
    return m;
}

// Eccentricity
double eccentricity(vector<PIXEL> v)
{
    Mat m = *eigenvalueMatrix(v);
    double lambda2 = m.at<double>(0,0);
    double lambda1 = m.at<double>(1,0);
    
    return sqrt(1.0 - lambda2/lambda1);
}

// Eigenvalue Matrix
Mat* eigenvalueMatrix(vector<PIXEL> v)
{
    Mat *m=new Mat(2,1,CV_64FC1);
    double leftTerm = muPrimeij(v, 2, 0) + muPrimeij(v, 0, 2);
    leftTerm /= 2.0;
    double rightTerm = sqrt(4.0 * POW2(muPrimeij(v, 1, 1)) + POW2(muPrimeij(v, 2, 0) + muPrimeij(v, 0, 2)));
    rightTerm /= 2.0;
    m->at<double>(0,0) = leftTerm - rightTerm;
    m->at<double>(1,0) = leftTerm + rightTerm;

    return m;
}

// Covariance Matrix
Mat* covarianceMatrix(vector<PIXEL> v)
{
    Mat *m=new Mat(2,2,CV_64FC1);
    m->at<double>(0,0) = muPrimeij(v, 2, 0);
    m->at<double>(0,1) = muPrimeij(v, 1, 1);
    m->at<double>(1,0) = muPrimeij(v, 1, 1);
    m->at<double>(1,1) = muPrimeij(v, 0, 2);
    return m;
}

// Central moments divided by sum of the pixel values M00
double muPrimeij(vector<PIXEL> v, long i, long j)
{
    return muij(v, i, j) / muij(v, 0, 0);
}

// Scale invariant moments
double etaij(vector<PIXEL> v, long i, long j)
{
    return muij(v, i, j) / pow(muij(v, 0, 0), 1 + (i + j)/2.0);
}

// Raw moments
double Mij(vector<PIXEL> v, long i, long j)
{
    double sum=0.0;
    
    //M00
    if(i==0 && j==0)
    {
        return v.size();
    }
    //M01
    else if(i==0 && j==1)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW1((*it).pt.y);
        }
        
    }
    //M10
    else if(i==1 && j==0)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW1((*it).pt.x);
        }
        
    }
    //M11
    else if(i==1 && j==1)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW1((*it).pt.x) * POW1((*it).pt.y);
        }
        
    }
    //M20
    else if(i==2 && j==0)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW2((*it).pt.x);
        }
        
    }
    //M02
    else if(i==0 && j==2)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW2((*it).pt.y);
        }
        
    }
    //M21
    else if(i==2 && j==1)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW2((*it).pt.x) * POW1((*it).pt.y);
        }
        
    }
    //M12
    else if(i==1 && j==2)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW1((*it).pt.x) * POW2((*it).pt.y);
        }
        
    }
    //M30
    else if(i==3 && j==0)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW3((*it).pt.x);
        }
        
    }
    //M03
    else if(i==0 && j==3)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW3((*it).pt.y);
        }
        
    }
    else
    {
        cout<<"Invalid raw moment("<<i<<","<<j<<")"<<endl;
        return -DBL_MAX;
    }
    return sum;
}

double xbar(vector<PIXEL> v)
{
    return Mij(v, 1, 0)/Mij(v, 0, 0);
}

double ybar(vector<PIXEL> v)
{
    return Mij(v, 0, 1)/Mij(v, 0, 0);
}

// Central moments
double muij(vector<PIXEL> v, long i, long j)
{
    //mu00
    if(i==0 && j==0)
    {
        return Mij(v, 0, 0);
        
    }
    //mu01
    else if(i==0 && j==1)
    {
        return 0.0;
        
    }
    //mu10
    else if(i==1 && j==0)
    {
        return 0.0;
    }
    //mu11
    else if(i==1 && j==1)
    {
        return (Mij(v, 1, 1) - xbar(v) * Mij(v, 0, 1));

    }
    //mu20
    else if(i==2 && j==0)
    {
        return Mij(v, 2, 0) - xbar(v) * Mij(v, 1, 0);
    }
    //mu02
    else if(i==0 && j==2)
    {
        return Mij(v, 0, 2) - ybar(v) * Mij(v, 0, 1);
    }
    //mu21
    else if(i==2 && j==1)
    {
        return Mij(v, 2, 1) - 2.0 * xbar(v) * Mij(v, 1, 1) -  ybar(v) * Mij(v, 2, 0) + 2.0 * POW2(xbar(v)) * Mij(v, 0, 1);
    }
    //mu12
    else if(i==1 && j==2)
    {
        return Mij(v, 1, 2) - 2.0 * ybar(v) * Mij(v, 1, 1) -  xbar(v) * Mij(v, 0, 2) + 2.0 * POW2(ybar(v)) * Mij(v, 1, 0);
    }
    //mu30
    else if(i==3 && j==0)
    {
        return Mij(v, 3, 0) - 3.0 * xbar(v) * Mij(v, 2, 0) +  2.0 * POW2(xbar(v)) * Mij(v, 1, 0);
    }
    //mu03
    else if(i==0 && j==3)
    {
        return Mij(v, 0, 3) - 3.0 * ybar(v) * Mij(v, 0, 2) +  2.0 * POW2(ybar(v)) * Mij(v, 0, 1);
    }
    else
    {
        cout<<"Invalid central moment("<<i<<","<<j<<")"<<endl;
        return -DBL_MAX;
    }
}

// Hu Translation, scale, and rotation invarient moments (+I8 recommended by Flusser & Suk)
double Hui(vector<PIXEL> v, long i)
{
    // Calculate normalized moments upfront so that we are not doing extra work
    double n02 = etaij(v, 0, 2);
    double n03 = etaij(v, 0, 3);
    double n11 = etaij(v, 1, 1);
    double n12 = etaij(v, 1, 2);
    double n20 = etaij(v, 2, 0);
    double n21 = etaij(v, 2, 1);
    double n30 = etaij(v, 3, 0);
    
    // Make the formuli below a little simpler with these partial solutions
    double ap = n20 + n02;
    double am = n20 - n02;
    double bp = n30 + n12;
    double cp = n21 + n03;
    double dm = n30 - 3.0 * n12;
    double em = 3.0 * n21 - n03;
    double fp = n03 + n21;
 
    switch (i) {
        case 1:
            return ap;
            break;
            
        case 2:
            return POW2(am) + 4.0 * POW2(n11);
            break;
            
        case 3:
            return POW2(dm) + POW2(em);
            break;
            
        case 4:
            return POW2(bp) + POW2(cp);
            break;
            
        case 5:
            return dm * bp * ( POW2(bp) - 3 * POW2(cp) ) + em * cp * ( 3.0 * POW2(bp) - POW2(cp) );
            break;

        case 6:
            return am * ( POW2(bp) - POW2(cp) ) + 4 * n11 * bp * cp;
            break;

        case 7:
            return em * bp * ( POW2(bp) - 3.0 * POW2(cp) ) - dm * cp * ( 3.0 * POW2(bp) - POW2(cp) );
            break;

        case 8:
            return n11 * ( POW2(bp) + POW2(fp) ) - ap * bp * fp;
            break;
            
        default:
            cout<<"Invalid Hu moment "<<i<<endl;
            return -DBL_MAX;
            break;
    }

}

// Raw moments which include pixel intensity in calculations
double MijIxy(vector<PIXEL> v, long i, long j)
{
    double sum=0.0;
    
    //M00
    if(i==0 && j==0)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += (*it).val[0];
        }
        
    }
    //M01
    else if(i==0 && j==1)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW1((*it).pt.y) * (*it).val[0];
        }
        
    }
    //M10
    else if(i==1 && j==0)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW1((*it).pt.x) * (*it).val[0];
        }
        
    }
    //M11
    else if(i==1 && j==1)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW1((*it).pt.x) * POW1((*it).pt.y) * (*it).val[0];
        }
        
    }
    //M20
    else if(i==2 && j==0)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW2((*it).pt.x) * (*it).val[0];
        }
        
    }
    //M02
    else if(i==0 && j==2)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW2((*it).pt.y) * (*it).val[0];
        }
        
    }
    //M21
    else if(i==2 && j==1)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW2((*it).pt.x) * POW1((*it).pt.y) * (*it).val[0];
        }
        
    }
    //M12
    else if(i==1 && j==2)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW1((*it).pt.x) * POW2((*it).pt.y) * (*it).val[0];
        }
        
    }
    //M30
    else if(i==3 && j==0)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW3((*it).pt.x) * (*it).val[0];
        }
        
    }
    //M03
    else if(i==0 && j==3)
    {
        for(vector<PIXEL>::iterator it = v.begin(); it != v.end(); ++it) {
            sum += POW3((*it).pt.y) * (*it).val[0];
        }
        
    }
    else
    {
        cout<<"Invalid raw moment("<<i<<","<<j<<")"<<endl;
        return -DBL_MAX;
    }
    return sum;
}

