#include "LKTracking.h"
#include "Affine2D.h"

LKTracking::LKTracking(void)
{
}


LKTracking::~LKTracking(void)
{
}

int LKTracking::estimateAffine2D(InputArray _from, InputArray _to,OutputArray _out, OutputArray _inliers,double param1, double param2)
{
	Mat from = _from.getMat(), to = _to.getMat();
	int count = from.checkVector(2, CV_32F);

	CV_Assert( count >= 0 && to.checkVector(2, CV_32F) == count );

	_out.create(2, 3, CV_64F);
	Mat out = _out.getMat();

	_inliers.create(count, 1, CV_8U, -1, true);
	Mat inliers = _inliers.getMat();
	inliers = Scalar::all(1);

	Mat dFrom, dTo;
	from.convertTo(dFrom, CV_64F);
	to.convertTo(dTo, CV_64F);

	CvMat F2x3 = out;
	CvMat mask  = inliers;
	CvMat m1 = dFrom;
	CvMat m2 = dTo;

	const double epsilon = numeric_limits<double>::epsilon();        
	param1 = param1 <= 0 ? 3 : param1;
	param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;

	return Affine2D().runRANSAC(&m1, &m2, &F2x3, &mask, param1, param2 );    
}

IplImage* LKTracking::getTargetFeaturePoint(IplImage* grayPre,IplImage* grayNext)
{

	IplConvKernel* kernel_Dilate = cvCreateStructuringElementEx(6,6,2,2,CV_SHAPE_ELLIPSE);
	IplConvKernel* kernel_Erode  = cvCreateStructuringElementEx(2,2,1,1,CV_SHAPE_RECT);

	CvSize img_sz = cvGetSize( grayPre );  //frame size
	
	int win_size = 5; 
	IplImage * grayA= cvCreateImage( img_sz, IPL_DEPTH_8U, 1 );
	IplImage * grayB = cvCreateImage( img_sz, IPL_DEPTH_8U, 1 );

	int distanceThreshold = 15;
	int targetSizeThreshold = 20;	

	if (grayPre->nChannels == 3)
	{
		cvCvtColor(grayPre,grayA,CV_BGR2GRAY);
		cvCvtColor(grayNext,grayB,CV_BGR2GRAY);
	}

	cvCopy(grayPre,grayA);
	cvCopy(grayNext,grayB);

	/*cvAbsDiff(grayA,grayB,grayA);
	cvShowImage("Result",grayA);
	cvWaitKey();*/

	IplImage * grayC= cvCreateImage( img_sz, IPL_DEPTH_8U, 1 );
	grayC = cvCloneImage(grayA);

	optimal_LKpyr(grayA,grayB,grayC,img_sz,win_size);

	//cvMorphologyEx(grayC,grayC,NULL,kernel_Dilate,CV_MOP_CLOSE,1); 
	cvMorphologyEx(grayC,grayC,NULL,kernel_Erode,CV_MOP_OPEN,1);  //进一步进行形态学闭运算
		
	//cvShowImage("Result",grayC);

	//char file_name[256];
	//sprintf(file_name,"result/%d.jpg",count_num++);
	//cvSaveImage(file_name,grayC);

	//cvWaitKey(1);	
	
	cvReleaseImage(&grayA);
	cvReleaseImage(&grayB);
	
	return grayC;
}


void LKTracking::optimal_LKpyr(IplImage* grayA,IplImage* grayB,IplImage* grayC,CvSize img_sz,int win_size)//a是第一针。B是第5真，c是a变化后的图
{
	int corner_count = MAX_CORNERS;  
	IplConvKernel* kernel_Dilate = cvCreateStructuringElementEx(5,5,2,2,CV_SHAPE_ELLIPSE);
	IplConvKernel* kernel_Erode  = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_RECT);
	CvPoint2D32f * cornersA      = new CvPoint2D32f[ MAX_CORNERS+1000 ];                //feature point of imageA
	CvPoint2D32f * cornersB      = new CvPoint2D32f[ MAX_CORNERS+1000 ];                //feature point of imageB
	char* features_found         = new char[ MAX_CORNERS+1000 ];                          //mask the founded features
	float* feature_errors        = new float[ MAX_CORNERS+1000 ];
	IplImage * eig_image         = cvCreateImage(img_sz, IPL_DEPTH_32F, 1); 
	IplImage * tmp_image         = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);
	IplImage * grayD             = cvCreateImage( img_sz, IPL_DEPTH_8U, 1 );
	Mat map_matrix = Mat(2,3,CV_32FC1);
	CvScalar s = cvScalarAll(49);
	/*get the Feature point of imageA*/
//	cvShowImage( "Raw Image", grayA );
	cvGoodFeaturesToTrack(
		grayA,
		eig_image,
		tmp_image,
		cornersA,
		&corner_count,
		0.01,
		5.0,
		0,
		3,
		0,
		0.04);
	cvFindCornerSubPix(
		grayA,
		cornersA,
		corner_count,
		cvSize(win_size, win_size),
		cvSize(- 1, - 1),
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03)
		);
	CvSize pyr_sz = cvSize(grayA->width + 8, grayB->height / 3);
	IplImage * pyrA = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);
	IplImage * pyrB = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);
	//*LK optimal flow*/
	cvCalcOpticalFlowPyrLK(
		grayA,
		grayB,
		pyrA,
		pyrB,
		cornersA,
		cornersB,
		corner_count,
		cvSize(win_size, win_size),
		5,
		features_found,
		feature_errors,
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3),
		0);
	//*count the matched feature point*/
	Mat matchPointSetA = Mat(corner_count,1,CV_32FC2);
	Mat matchPointSetB = Mat(corner_count,1,CV_32FC2);
	for (int i = 0; i < corner_count; i++)
	{
		if (features_found[i] == 0 || feature_errors[i] > 550)
		{ 
			//printf("Error is %f\n", feature_errors[i]);
			continue;
		}
		CvPoint p0 = cvPoint(
			cvRound(cornersA[i].x),
			cvRound(cornersA[i].y)
			);
		CvPoint p1 = cvPoint(
			cvRound(cornersB[i].x),
			cvRound(cornersB[i].y)
			);
		matchPointSetA.at<Vec2f>(i,0) = Vec2f(p0.x,p0.y);
		matchPointSetB.at<Vec2f>(i,0) = Vec2f(p1.x,p1.y);
	}
      

	Mat tempMask = Mat( 1, corner_count, CV_8U );
	int ok = estimateAffine2D(matchPointSetA,matchPointSetB,map_matrix,tempMask,ransacReprojThreshold,confidence);
	CvMat map_matrix_new = map_matrix;
	////showTargetPoint(matchPointSetA,background,tempMask);
	cvWarpAffine(grayA,grayC,&map_matrix_new,CV_INTER_LINEAR,s);  //仿射变换

	//warpPerspective(Mat(grayA),Mat(grayC),H,Size(1242,375),1,0);
	cvAbsDiff(grayB,grayC,grayD);  //差分操作

	//cvNamedWindow("warpAffine");
	//cvShowImage("src",grayB);
	//cvShowImage("warpAffine",grayC);
	
	//加窗求图像在窗内最小值，目的为边缘抑制

	cvThreshold(grayD,grayC,40,255,CV_THRESH_BINARY);
	
	//cvMorphologyEx(grayD,grayC,NULL,kernel_Dilate,CV_MOP_CLOSE,1);  //进一步进行形态学闭运算
	//morphologyEx()
	


	IplImage * tmp_imageMask = cvCreateImage(img_sz, IPL_DEPTH_8U, 1);
	IplImage * tmp_imageMaskFinal = cvCreateImage(img_sz, IPL_DEPTH_8U, 1);

	for(int i=0;i<img_sz.width*img_sz.height;i++)
	{
		tmp_imageMask->imageData[i]=1;
		tmp_imageMaskFinal->imageData[i]=0;
	}
	cvWarpAffine(tmp_imageMask,tmp_imageMaskFinal,&map_matrix_new,CV_INTER_LINEAR,s);  //仿射变换

	//cvShowImage("grayC",grayC);

	/*cout<<tmp_imageMaskFinal->imageSize<<"  "<<grayC->imageSize<<endl;
	IplImage * graymask;
	graymask = cvCloneImage(grayC);*/
	for(int i=0;i<img_sz.width*img_sz.height;i++)
	{ 
		if (tmp_imageMaskFinal->imageData[i]==0)
			//graymask->imageData[i]=0;
		    grayC->imageData[i]=0;
	}

	//cvAbsDiff(graymask,grayC,graymask);

	cvReleaseImage(&tmp_imageMask);
	cvReleaseImage(&tmp_imageMaskFinal);
	// 	cvConvert(grayC,src);
	// 	cvAnd(dst,src,dst1);
	// 	cvGetImage(dst1,grayC);
	////cvMorphologyEx(grayC,grayA,NULL,kernel_Erode,CV_MOP_TOPHAT,1);  //进一步进行形态学“礼帽”操作
	////cvMorphologyEx(grayA,grayC,grayB,kernel_Erode,CV_MOP_GRADIENT,1);  //进一步进行形态学梯度操作
	cvReleaseStructuringElement(&kernel_Dilate);
	cvReleaseStructuringElement(&kernel_Erode);
	delete[] feature_errors;
	delete[] features_found;
	delete[] cornersB;
	delete[] cornersA;
	tempMask.release();
	cvReleaseData(&map_matrix_new);
	map_matrix.release();
	matchPointSetA.release();
	matchPointSetB.release();
	cvReleaseImage(&grayD);
	cvReleaseImage(&pyrA);
	cvReleaseImage(&pyrB);
	cvReleaseImage(&eig_image);
	cvReleaseImage(&tmp_image);
}



