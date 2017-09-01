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
#include "features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <core/types_c.h>
#include <math.h>
using namespace cv;
using namespace std;

class Affine2D
{
protected:
	CvRNG rng;
	int modelPoints;
	CvSize modelSize;
	int maxBasicSolutions;
	bool checkPartialSubsets;
public:
	int runKernel( const CvMat* m1, const CvMat* m2, CvMat* model );
	bool runRANSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
		CvMat* mask, double threshold,double confidence=0.99, int maxIters=2000 );
	bool getSubset( const CvMat* m1, const CvMat* m2,
		CvMat* ms1, CvMat* ms2, int maxAttempts=1000 );
	bool checkSubset( const CvMat* ms1, int count);
	int findInliers( const CvMat* m1, const CvMat* m2,
		const CvMat* model, CvMat* error,CvMat* mask, double threshold );
	void computeReprojError( const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error);
	Affine2D(void);
	~Affine2D(void);
};

