#pragma once
#include <highgui.h>
#include <ml.h>
#include <cv.h>
#include <stdio.h>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "opencv2/core/core.hpp"
#include "features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <core/types_c.h>
#include <math.h>
#include <atlstr.h>
#include "Affine2D.h"
using namespace cv;
using namespace std;

class LKTracking
{
public:
	int pre_image;
	CvCapture* capture_Video;
	CvCapture* capture_Camera;
	static const int MAX_CORNERS = 1500;
	double ransacReprojThreshold;
	double confidence;
public:
	LKTracking(void);
	//LKTracking(int pre_Img,char* video_path,int camera_flag,int ransacR,int conf );
	~LKTracking(void);

public:
	//void setParameters(int pre_Img,char* video_path,int camera_flag,int ransacR,int conf);
	IplImage* getTargetFeaturePoint(IplImage* grayPre,IplImage* grayNext);
	void optimal_LKpyr(IplImage* grayA,IplImage* grayB,IplImage* grayC,CvSize img_sz,int win_size);
	int estimateAffine2D(InputArray _from, InputArray _to,OutputArray _out, OutputArray _inliers,double param1, double param2);

private:
	int count_num;
};

