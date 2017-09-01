/*
Copyright 2012. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libviso2.
Authors: Andreas Geiger

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

/*
  Documented C++ sample code of stereo visual odometry (modify to your needs)
  To run this demonstration, download the Karlsruhe dataset sequence
  '2010_03_09_drive_0019' from: www.cvlibs.net!
  Usage: ./viso2 path/to/sequence/2010_03_09_drive_0019
*/
#include "demo_matching_flow.h"
#include "LKTracking.h"

#define  inlier_threshold 0.00002

//计算基本矩阵 F，极线约束
//--------------------------------------------------------------------------------------------------------------------//
vector<int32_t> getRandomSample(int32_t N,int32_t num) {

	// init sample and totalset
	vector<int32_t> sample;
	vector<int32_t> totalset;

	// create vector containing all indices
	for (int32_t i=0; i<N; i++)
		totalset.push_back(i);

	// add num indices to current sample
	sample.clear();
	for (int32_t i=0; i<num; i++) {
		int32_t j = rand()%totalset.size();
		sample.push_back(totalset[j]);
		totalset.erase(totalset.begin()+j);
	}

	// return sample
	return sample;
}

void fundamentalMatrix(const vector<Matcher::p_match> &p_matched,const vector<int32_t> &active,Matrix &F) 
{

	// number of active p_matched
	int32_t N = active.size();

	// create constraint matrix A
	Matrix A(N,9);
	for (int32_t i=0; i<N; i++) {
		Matcher::p_match m = p_matched[active[i]];
		A.val[i][0] = m.u1c*m.u1p;
		A.val[i][1] = m.u1c*m.v1p;
		A.val[i][2] = m.u1c;
		A.val[i][3] = m.v1c*m.u1p;
		A.val[i][4] = m.v1c*m.v1p;
		A.val[i][5] = m.v1c;
		A.val[i][6] = m.u1p;
		A.val[i][7] = m.v1p;
		A.val[i][8] = 1;
	}

	// compute singular value decomposition of A
	Matrix U,W,V;
	A.svd(U,W,V);

	// extract fundamental matrix from the column of V corresponding to the smallest singular value
 	F = Matrix::reshape(V.getMat(0,8,8,8),3,3);

	// enforce rank 2
	F.svd(U,W,V);
	W.val[2][0] = 0;
	F = U*Matrix::diag(W)*~V;
	//maxF = F;
}

bool normalizeFeaturePoints(vector<Matcher::p_match> &p_matched,Matrix &Tp,Matrix &Tc) {

	// shift origins to centroids
	double cpu=0,cpv=0,ccu=0,ccv=0;
	for (vector<Matcher::p_match>::iterator it = p_matched.begin(); it!=p_matched.end(); it++) {
		cpu += it->u1p;
		cpv += it->v1p;
		ccu += it->u1c;
		ccv += it->v1c;
	}
	cpu /= (double)p_matched.size();
	cpv /= (double)p_matched.size();
	ccu /= (double)p_matched.size();
	ccv /= (double)p_matched.size();
	for (vector<Matcher::p_match>::iterator it = p_matched.begin(); it!=p_matched.end(); it++) {
		it->u1p -= cpu;
		it->v1p -= cpv;
		it->u1c -= ccu;
		it->v1c -= ccv;
	}

	// scale features such that mean distance from origin is sqrt(2)
	double sp=0,sc=0;
	for (vector<Matcher::p_match>::iterator it = p_matched.begin(); it!=p_matched.end(); it++) {
		sp += sqrt(it->u1p*it->u1p+it->v1p*it->v1p);
		sc += sqrt(it->u1c*it->u1c+it->v1c*it->v1c);
	}
	if (fabs(sp)<1e-10 || fabs(sc)<1e-10)
		return false;
	sp = sqrt(2.0)*(double)p_matched.size()/sp;
	sc = sqrt(2.0)*(double)p_matched.size()/sc;
	for (vector<Matcher::p_match>::iterator it = p_matched.begin(); it!=p_matched.end(); it++) {
		it->u1p *= sp;
		it->v1p *= sp;
		it->u1c *= sc;
		it->v1c *= sc;
	}

	// compute corresponding transformation matrices
	double Tp_data[9] = {sp,0,-sp*cpu,0,sp,-sp*cpv,0,0,1};
	double Tc_data[9] = {sc,0,-sc*ccu,0,sc,-sc*ccv,0,0,1};
	Tp = Matrix(3,3,Tp_data);
	Tc = Matrix(3,3,Tc_data);

	// return true on success
	return true;
}

vector<int32_t> getInlier (vector<Matcher::p_match> &p_matched,Matrix &F) {

	// extract fundamental matrix
	double f00 = F.val[0][0]; double f01 = F.val[0][1]; double f02 = F.val[0][2];
	double f10 = F.val[1][0]; double f11 = F.val[1][1]; double f12 = F.val[1][2];
	double f20 = F.val[2][0]; double f21 = F.val[2][1]; double f22 = F.val[2][2];

	// loop variables
	double u1,v1,u2,v2;
	double x2tFx1;
	double Fx1u,Fx1v,Fx1w;
	double Ftx2u,Ftx2v;

	// vector with inliers
	vector<int32_t> inliers;

	// for all matches do
	for (int32_t i=0; i<(int32_t)p_matched.size(); i++) {

		// extract matches
		u1 = p_matched[i].u1p;
		v1 = p_matched[i].v1p;
		u2 = p_matched[i].u1c;
		v2 = p_matched[i].v1c;

		// F*x1
		Fx1u = f00*u1+f01*v1+f02;
		Fx1v = f10*u1+f11*v1+f12;
		Fx1w = f20*u1+f21*v1+f22;

		// F'*x2
		Ftx2u = f00*u2+f10*v2+f20;
		Ftx2v = f01*u2+f11*v2+f21;

		// x2'*F*x1
		x2tFx1 = u2*Fx1u+v2*Fx1v+Fx1w;

		// sampson distance
		double d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);

		// check threshold
		if (fabs(d)<inlier_threshold)
			inliers.push_back(i);
	}

	// return set of all inliers
	return inliers;
}

vector<int32_t> estimateMotion (vector<Matcher::p_match> p_matched) {

	vector<int32_t> inliers;
	// get number of matches
	int32_t N = p_matched.size();
	if (N<10)
		return vector<int32_t>();

	// normalize feature points and return on errors
	Matrix Tp,Tc;
	vector<Matcher::p_match> p_matched_normalized = p_matched;
	if (!normalizeFeaturePoints(p_matched_normalized,Tp,Tc))
		return vector<int32_t>();

	// initial RANSAC estimate of F
	Matrix E,F;
	inliers.clear();
	for (int32_t k=0;k<2000;k++) {

		// draw random sample set
		vector<int32_t> active = getRandomSample(N,8);

		// estimate fundamental matrix and get inliers
		fundamentalMatrix(p_matched_normalized,active,F);
		vector<int32_t> inliers_curr = getInlier(p_matched_normalized,F);

		// update model if we are better
		if (inliers_curr.size()>inliers.size())
			inliers = inliers_curr;
	}

	// are there enough inliers?
	if (inliers.size()<10)
		return vector<int32_t>();

	// refine F using all inliers
	fundamentalMatrix(p_matched_normalized,inliers,F); 

	F = ~Tc*F*Tp;
	return inliers;
}
//----------------------------------------------------------------------------------------//

float getEffectiveAreas(Mat img, Rect rect)
{
	int legal_count = 0;

	for (int i=rect.y ; i<rect.y+rect.height; i++)
	{
		for (int j= rect.x; j<rect.x+rect.width; j++)
		{
			if (img.at<uchar>(i,j) > 0)
			{
				legal_count++;
			}
		}
	}
	//cvtColor(img,img,CV_GRAY2BGR);
	//rectangle(img,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);

	float leagl_ratio = 1.0*legal_count/rect.area();
	//imshow("img1",img);
	//waitKey();
	return leagl_ratio;
}

vector<Rect> findCandidates(Mat original_img)
{
	Mat image;
	original_img.copyTo(image);

	vector <Rect> candidate;
	candidate.clear();
	/************find countours to obtain the moving area********/
	CvMemStorage *mem_storage= NULL;
	if(mem_storage == NULL)
	{
		mem_storage = cvCreateMemStorage(0);
	}
	else
	{
		cvClearMemStorage(mem_storage);
	}
	//CvMemStorage *mem_storage = NULL;
	CvSeq *first_contour = NULL, *c=NULL;

	int Nc=cvFindContours(&IplImage(image),mem_storage,&first_contour,sizeof(CvContour),CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
	//cvShowImage("countour",grayC);
	int count = 0;
	if(Nc!=0)
	{
		count = 0;
		for(c=first_contour;c!=NULL;c=c->h_next) 
		{

			CvRect rect_c=cvBoundingRect(c);

			float effective_area_ratio = getEffectiveAreas(original_img,rect_c);

			if ((rect_c.width < 5 || rect_c.height < 5) ||(rect_c.width >600 || rect_c.height >600) || (1.0*rect_c.width/rect_c.height > 4) || (1.0*rect_c.width/rect_c.height<0.2))
			{
			continue;
			}
			else if ((effective_area_ratio < 0.35))
			{
				continue;
			}
			else
			{
				//cout<<"effective_area_ratio: "<<effective_area_ratio<<endl;
				count++;	
				candidate.push_back(rect_c);

				CvPoint pt1,pt2;
				pt1.x=rect_c.x;
				pt1.y=rect_c.y;
				pt2.x=rect_c.x+rect_c.width;
				pt2.y=rect_c.y+rect_c.height;		
				//cvRectangle(pre_videoFrame[j],pt1,pt2,cvScalar(0,0,255));

			}
		}
		//cout<<"contour number"<<"  "<<count<<endl;
	}

	//cout<<candidate.size()<<endl;
	return candidate;
}

vector<Rect> findCandidatesbyBFS(Mat image)
{
	Mat input_img;
	image.copyTo(input_img);

	int mImgCols = input_img.cols;
	int mImgRows = input_img.rows;

	bool visitFlag[768][1024] = {0};
	int dist[8][2] = {{-1,0},{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1}};

	queue<int> q;
	int r,c;

	Rect rect;
	vector<Rect> candidate;
	candidate.clear();


	for (int i=0; i<mImgRows; i++)
	{
		for (int j=0; j<mImgCols; j++)
		{
			//cout<<(int)input_img.at<uchar>(i,j)<<endl;
			if ((int)input_img.at<uchar>(i,j) == 255  && !visitFlag[i][j])
			{
				q.push(j+i*mImgCols);
				visitFlag[i][j] = true;

				int max_row = i;
				int max_col = j;
				int min_row = i;
				int min_col = j;

				int count  = 0;

				while (!q.empty())
				{
					int num = q.front();
					q.pop();

					int row = num/mImgCols;
					int col = num%mImgCols;

					max_row = max_row > row ? max_row : row; 
					max_col = max_col > col ? max_col : col; 
					min_row = min_row > row ? row : min_row; 
					min_col = min_col > col ? col : min_col; 

					for (int m=0; m<8; m++)
					{						
						r = row + dist[m][0];
						c = col + dist[m][1];

						if (r<0 || c<0 || r>=mImgRows || c>=mImgCols)
						{
							continue;
						}

						if ((int)input_img.at<uchar>(r,c) == 255 && !visitFlag[r][c])
						{				
							q.push(c + r*mImgCols);
							visitFlag[r][c] = true;
							count ++;
						}
					}
					
					/*if (count > MAX_POINT_COUNT)
					{
						while(!q.empty())
						{
							q.pop();
						}
						break;
					}*/
				}

				rect.x = min_col;
				rect.y = min_row;

				rect.width = max_col - min_col;
				rect.height = max_row - min_row;

				float width_height_ratio = rect.width > rect.height ? 1.0*rect.width/rect.height: 1.0*rect.height/rect.width;

				float effective_area_ratio = getEffectiveAreas(input_img,rect);

				if ((rect.width < 5 || rect.height < 5) || width_height_ratio > 4 || effective_area_ratio < 0.35)
				{
					continue;
				}

				candidate.push_back(rect);
			}
		}
	}
	return candidate;

	/*for (vector<Rect>::iterator it = candidate.begin(); it!=candidate.end(); it++)
	{
	rectangle(input_img,Point(it->x,it->y),Point(it->x + it->width, it->y + it->height),Scalar(255,255,255),2,8,0);
	}

	imshow("test",input_img);
	waitKey();*/
}

int estimateAffine2D(InputArray _from, InputArray _to,OutputArray _out, OutputArray _inliers,double param1, double param2)
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

vector<Point2f> extractFeaturePoints( Mat preImage, Rect rect) 
{
	double qualityLevel = 0.01;
	double minDistance = 1;//两角点间最小距离
	int blockSize = 3;//邻域尺寸
	bool useHarrisDetector = true;//是否使用Harris
	double k = 0.04;
	vector<Point2f> corners; 
	vector<Point2f> new_corners; 
	Point2f point;


	Mat toFind = preImage(Rect(rect.x,rect.y,rect.width,rect.height));
	/*imshow("t",toFind);
	waitKey();*/
	
	goodFeaturesToTrack(toFind,corners,50,qualityLevel,minDistance,Mat(),blockSize,useHarrisDetector,k);

	//至少需要3个特征点进行跟踪
	if (corners.size() <= 2)
	{
		for (int i=rect.y ; i<rect.y+rect.height; i++)
		{
			for (int j= rect.x; j<rect.x+rect.width; j++)
			{				
				point.x = rect.x;
				point.y = rect.y;
				new_corners.push_back(point);
			}
		}
	}
	else
	{
		for (vector<Point2f>::iterator its=corners.begin(); its!=corners.end(); its++)
		{
			point.x = its->x+rect.x;
			point.y = its->y+rect.y;
			new_corners.push_back(point);
		}
	}

    //cout<<"corners:    "<<new_corners.size()<<endl;
	return new_corners;
}

int belongToWhichObject(vector<Point2f> point, vector<Rect> candinate)
{
	int max_id = -1;
	int max_count = 0;
	//Rect max_rect;
	int id = 0;

	for (vector<Rect>::iterator it=candinate.begin(); it!=candinate.end(); it++)
	{
		int in_count = 0;
		for (int i=0; i<point.size(); i++)
		{
			int x = point.at(i).x;
			int y = point.at(i).y;
			
			if ((x>=it->x && x<=it->x+it->width) && (y>=it->y && y<=it->y+it->height))
			{
				in_count++;
			}
		}
		
		if (in_count > max_count)
		{
			max_count = in_count;
			max_id = id;
		}
		id++;
	}
	float max_ratio = 1.0*max_count/point.size();
	if (max_ratio > 0)
	{
		return max_id;
	}

	return -1;
}

void computeDenseFlow( Mat preImage, Mat currImage, Mat currOptflow, Mat &preOptflow,vector<trajectory> &tracker, int current_frame_no) 
{
	vector<Point2f> new_corners;
	vector<Point2f> corners; 
	trajectory cur_trajectory;

	vector<Rect> candinate_pre;
	vector<Rect> candinate_curr;

	vector<bool> hasID;
	vector<int> ID;

	Mat flow,flowImage,showImage;

	preImage.copyTo(flowImage);
	currImage.copyTo(showImage);

	cvtColor(flowImage,flowImage,CV_GRAY2BGR);

	char flow_name[256];
	char first_detect_name[256];
	sprintf(flow_name,"flow/%d.jpg",current_frame_no);
	sprintf(first_detect_name,"first_detect/%d.jpg",current_frame_no);

	hasID.clear();

	if (preOptflow.data)
	{
		candinate_pre.clear();
		candinate_curr.clear();

		candinate_pre = findCandidatesbyBFS(preOptflow);
		candinate_curr = findCandidatesbyBFS(currOptflow);

		//--------------init label---------------//
		for (int i=0; i<candinate_curr.size(); i++)
		{
			hasID.push_back(false);
		}
		//---------------------------------------//

		calcOpticalFlowFarneback(preImage, currImage, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

		//-----------------tracker init-----------------------------//
		if (tracker.empty())
		{
			for (vector<Rect>::iterator it=candinate_pre.begin(); it!=candinate_pre.end(); it++)
			{					
				//cout<<it->x<<"   "<<it->y<<endl;
				cur_trajectory.objectTrack.clear();
				cur_trajectory.id = tracker.size();
				cur_trajectory.objectTrack.push_back(*it);
				cur_trajectory.start_frame = current_frame_no;
				cur_trajectory.end_frame = current_frame_no;
				cur_trajectory.active = true;
				cur_trajectory.object_status = UNCERTAIN;
				cur_trajectory.outliners_count = 0;
				cur_trajectory.total_count = 0;
				tracker.push_back(cur_trajectory);
			}
		}
		else
		{
			for (vector<trajectory>::iterator it = tracker.begin(); it!=tracker.end(); it++)
			{
				if (it->active)
				{
					new_corners.clear();
					Rect rect_pre = it->objectTrack.back();
					corners = extractFeaturePoints(preImage,rect_pre);

					//computeSparseFlow(corners, new_corners,preImage, currImage, rect_pre);

					for (vector<Point2f>::iterator it=corners.begin(); it!=corners.end(); it++)
					{
						Point2f point;
						point.x = it->x+flow.at<Vec2f>(it->y,it->x)[0];
						point.y = it->y+flow.at<Vec2f>(it->y,it->x)[1];

						new_corners.push_back(point);
					}							

					for (int i=0; i<corners.size(); i++)
					{
						int x1p = corners.at(i).x;
						int y1p = corners.at(i).y;

						int x1c = new_corners.at(i).x;
						int y1c = new_corners.at(i).y;

						line(flowImage,Point(x1p,y1p),Point(x1c,y1c),Scalar(255,255,255),1,8,0);
						circle(flowImage,Point(x1p,y1p),1,Scalar(0,0,255),1,8,0);
						circle(flowImage,Point(x1c,y1c),1,Scalar(0,255,0),1,8,0);
					}

					int belong_id = belongToWhichObject(new_corners,candinate_curr);


					if (belong_id >= 0)
					{
						Rect belong = candinate_curr.at(belong_id);

						float area_pre = rect_pre.width*rect_pre.height;
						float area_cur = belong.width*belong.height;
						float area_ratio = area_pre > area_cur? area_cur/area_pre : area_pre/area_cur;

						if (area_ratio > 0.3)
						{
							it->end_frame = current_frame_no;
							it->objectTrack.push_back(belong);
							it->active = true;

							hasID.at(belong_id) = true;
						}
						else
						{
							it->active = false;
						}
						//rectangle(showImage,Point(belong.x,belong.y),Point(belong.x+belong.width,belong.y+belong.height),Scalar(255,255,255),2,8,0);					
					}
					else
					{
						it->active = false;
					}
				}
			}

			//----------------------------------更新Tracker,添加新的跟踪器---------------------------------------//
			for (int m=0; m<candinate_curr.size(); m++)
			{
				if(!hasID.at(m)) //没有匹配目标,则为新目标
				{
					cur_trajectory.objectTrack.clear();
					cur_trajectory.id = tracker.size();
					cur_trajectory.objectTrack.push_back(candinate_curr.at(m));
					cur_trajectory.start_frame = current_frame_no;
					cur_trajectory.end_frame = current_frame_no;
					cur_trajectory.active = true;
					cur_trajectory.object_status = UNCERTAIN;
					cur_trajectory.outliners_count = 0;
					cur_trajectory.total_count = 0;

					tracker.push_back(cur_trajectory);
				}
			}

		}
	}

	//for (vector<trajectory>::iterator it=tracker.begin(); it!=tracker.end(); it++)
	//{
	//	if (it->active)// && (it->end_frame - it->start_frame) >= 3)
	//	{
	//		Rect belong = it->objectTrack.back();

	//		char id[256];
	//		sprintf(id,"%d",it->id);
	//		//cout<<"in_id:   "<<id<<endl;
	//		//putText(showImage,id,Point(belong.x,belong.y),FONT_HERSHEY_SCRIPT_COMPLEX,0.6,Scalar(255,255,255),1,8,0);
	//		//rectangle(detect_img,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(255,255,255),2,8,0);					
	//		rectangle(showImage,Point(belong.x,belong.y),Point(belong.x+belong.width,belong.y+belong.height),Scalar(255,255,255),2,8,0);	
	//	}
	//	else
	//	{
	//		if (it->objectTrack.size() < 3)
	//		{
	//			//tracker.erase(it);
	//		}
	//	}
	//}						

	for (vector<Rect>::iterator it=candinate_pre.begin(); it!=candinate_pre.end(); it++)
	{
		rectangle(flowImage,Point(it->x,it->y),Point(it->x+it->width,it->y+it->height),Scalar(0,0,255),2,8,0);
	}

	cvtColor(showImage,showImage,CV_GRAY2BGR);
	for (vector<Rect>::iterator it=candinate_curr.begin(); it!=candinate_curr.end(); it++)
	{
		rectangle(showImage,Point(it->x,it->y),Point(it->x+it->width,it->y+it->height),Scalar(255,255,255),2,8,0);
		rectangle(flowImage,Point(it->x,it->y),Point(it->x+it->width,it->y+it->height),Scalar(0,255,0),2,8,0);
	}
	//---------------------------
	
	imshow("flow",flowImage);
	imshow("detect_in",showImage);

	if (true)
	{
		imwrite(flow_name,flowImage);
		imwrite(first_detect_name,showImage);
	}
	
} 

void warpAffineTransformation( vector<Matcher::p_match> &p_matched, Mat pre_img, Mat curr_img, Mat& diff_img ) 
{
	Mat matchPointSetA = Mat(p_matched.size(),1,CV_32FC2);
	Mat matchPointSetB = Mat(p_matched.size(),1,CV_32FC2);

	for (int i = 0; i < p_matched.size(); i++)
	{
		Point p0 = Point(
			cvRound(p_matched.at(i).u1p),
			cvRound(p_matched.at(i).v1p)
			);
		Point p1 =Point(
			cvRound(p_matched.at(i).u1p),
			cvRound(p_matched.at(i).v1p)
			);

		matchPointSetA.at<Vec2f>(i,0) = Vec2f(p0.x,p0.y);
		matchPointSetB.at<Vec2f>(i,0) = Vec2f(p1.x,p1.y);
	}	

	Mat temp_pre;
	Mat map_matrix = Mat(2,3,CV_32FC1);

	Mat tempMask = Mat( 1, p_matched.size(), CV_8U );
	int ok = estimateAffine2D(matchPointSetA,matchPointSetB,map_matrix,tempMask,0,0.99);

	Mat map_matrix_new ;
	map_matrix.copyTo(map_matrix_new);

	warpAffine(pre_img,temp_pre,map_matrix_new,pre_img.size(),1,0,Scalar(49,49,49));

	//warpPerspective(left_img,temp_left,H,Size(1280,720),1,0);
	absdiff(temp_pre,curr_img,diff_img);  //差分操作
	threshold(diff_img,diff_img,40,255,CV_THRESH_BINARY);
}

int frameToframesubstraction(Mat pre_img, Mat curr_img, Mat& diff_img,int isOutliner[][IMG_COLS]) {
	

	Matcher::parameters param_matcher;
	param_matcher.nms_n					 = 5;
	param_matcher.nms_tau				 = 50;
	param_matcher.match_binsize          = 50;  // matching bin width/height (affects efficiency only)
	param_matcher.match_radius           = 200; //matching radius (du/dv in pixels)
	param_matcher.match_disp_tolerance   = 1;   // du tolerance for stereo matches (in pixels)
	param_matcher.outlier_disp_tolerance = 5;   // outlier removal: disparity tolerance (in pixels)
	param_matcher.outlier_flow_tolerance = 5;   // outlier removal: flow tolerance (in pixels)
	param_matcher.multi_stage            = 1;   // 0=disabled,1=multistage matching (denser and faster)
	param_matcher.half_resolution        = 0;   // 0=disabled,1=match at half resolution, refine at full resolution
	param_matcher.refinement             = 2;   // refinement (0=none,1=pixel,2=subpixel)

	Matcher demo_matching(param_matcher);

	int i = 0;

	if (pre_img.empty() || curr_img.empty())
	{
		return 1;
	}

	int32_t width  = pre_img.cols;
	int32_t height = pre_img.rows;
	
	// convert input images to uint8_t buffer
	uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
	uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
	int32_t k=0;
	for (int32_t v=0; v<height; v++) {
		for (int32_t u=0; u<width; u++) {
			left_img_data[k]  = pre_img.at<uchar>(v,u);
			right_img_data[k] = curr_img.at<uchar>(v,u);
			k++;
		}
	}

	int32_t dims[] = {width,height,width};
	
	demo_matching.pushBack(left_img_data,dims,0);
	demo_matching.pushBack(right_img_data,dims,0);
	demo_matching.matchFeatures(0);
	vector<Matcher::p_match> p_matched = demo_matching.getMatches();
	
    //cout<<"total:   "<<p_matched.size()<<"   inliners:   "<<inliers.size()<<endl;
	//Mat absdiff_img;

	if (false)
	{
		warpAffineTransformation(p_matched, pre_img, curr_img, diff_img);
	}
	else
	{
		LKTracking* LKTrack = new LKTracking();
		IplImage* opt_img = LKTrack->getTargetFeaturePoint(&IplImage(pre_img),&IplImage(curr_img));
		diff_img = Mat(opt_img,true);
	}
	
	 vector<Point2f> marks;
    
  //  //按照仿射变换矩阵，计算变换后各关键点在新图中所对应的位置坐标。
  //  for (int x = 0; x<pre_img.rows; x++)
  //  {
		//for(int y=0; y<pre_img.cols; y++)
		//{
		//	
		//	Point2f p = Point2f(0, 0);
		//	p.x = rot_mat.ptr<double>(0)[0] * y + rot_mat.ptr<double>(0)[1] * x + rot_mat.ptr<double>(0)[2];
		//	p.y = rot_mat.ptr<double>(1)[0] * y + rot_mat.ptr<double>(1)[1] * x + rot_mat.ptr<double>(1)[2];
		//	marks.push_back(p);
		//}
  //  }



	//形态学运算
	//erode(diff_img, diff_img, Mat(2, 2, CV_8UC1), Point(-1, -1));  // You can use Mat(5, 5, CV_8UC1) here for less distortion
	dilate(diff_img, diff_img, Mat(4,4, CV_8UC1), Point(-1, -1));

	//vector<Rect> candinate = findCandidates(diff_img);
	
	//vector<Rect> candinate = findCandidatesbyBFS(diff_img);

	vector<int32_t> inliers;
	inliers.clear();
	
	inliers= estimateMotion(p_matched);
	//外点点标记
	bool *inLinerFlag = new bool[p_matched.size()];
	memset(inLinerFlag,false,sizeof(bool)*p_matched.size());
	memset(isOutliner,0,sizeof(int)*IMG_COLS*IMG_ROWS);

	for (vector<int32_t>::iterator it = inliers.begin(); it!=inliers.end(); it++)
	{
		int id = *it;
		inLinerFlag[id] = true;
	} 

	cvtColor(pre_img,pre_img,CV_GRAY2BGR);
	for (int i=0; i<p_matched.size(); i++)
	{
		int u = p_matched.at(i).u1c;
		int v = p_matched.at(i).v1c;

		if (inLinerFlag[i])
		{
			isOutliner[v][u] = -1; //外点
			//circle(pre_img,Point(u,v),2,Scalar(0,0,255),2,8,0);
		}
		else
		{
			isOutliner[v][u] = 1; //内点
			//circle(pre_img,Point(u,v),2,Scalar(0,255,0),2,8,0);
		}
	}

	
	//-----------------------------------------------------------------------------//

	/*for (vector<Rect>::iterator it = candinate.begin(); it!=candinate.end(); it++)
	{
	    rectangle(diff_img,Point(it->x,it->y),Point(it->x+it->width,it->y+it->height),Scalar(255,255,255),2,8,0);
	}*/

	/*imshow("result",diff_img);
	imshow("left",pre_img);*/
	//waitKey(5);

	delete inLinerFlag;
    return 0;
}