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
#include "demo_viso_mono.h"
#include "LKTracking.h"

#define IMG_ROWS 720
#define IMG_COLS 1280

typedef struct{
	int id;
	vector<Rect> objectTrack;
	int start_frame;
	int end_frame;
	bool active;
	int object_status;
	int outliners_count;
	int total_count;
}trajectory;


typedef struct{
	int id;
	Rect objectPosition;
	bool hasId;
}objectFrame;


int isOutliner[IMG_ROWS][IMG_COLS] = {0}; //-1 inliner, 0 unknow, 1 outliner
enum{UNCERTAIN,SURE};


Mat MatrixToMat(Matrix pose,int dims)
{
	FLOAT_LIBVISO *val = new FLOAT_LIBVISO[dims*dims];
	Mat result = Mat::zeros(dims,dims,CV_32FC1);
	pose.getData(val,0,0,dims-1,dims-1);
	int k = 0;

	for (int i=0; i<dims; i++)
	{
		for (int j=0; j<dims; j++)
		{
			result.at<float>(i,j) = val[k++];
		}
	}
	delete val;

	return result;
}

float getEffectiveArea(Mat img, Rect rect)
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

vector<Rect> findCandidate(Mat original_img)
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

			float effective_area_ratio = getEffectiveArea(original_img,rect_c);

			if ((rect_c.width < 5 || rect_c.height < 5) ||(rect_c.width >400 || rect_c.height >400) || (1.0*rect_c.width/rect_c.height > 4) || (1.0*rect_c.width/rect_c.height<0.2))
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

float getInternalPointsRatio(int isOutliner[][IMG_COLS],Rect rect,Mat temp)
{
	int outLiner_count = 0;
	int total_count = 0;

	
	for (int i=rect.y ; i<rect.y+rect.height; i++)
	{
		for (int j= rect.x; j<rect.x+rect.width; j++)
		{			
			if (temp.at<uchar>(i,j) > 0)
			{
				if (isOutliner[i][j] == 0)
				{				
					continue;
				}
				else if (isOutliner[i][j] == -1)
				{
					total_count++;
				}
				else if(isOutliner[i][j] == 1)
				{
					total_count++;
					outLiner_count++;
				}
			}
		}
	}

	if (total_count == 0)
	{
		return 0;
	}
	else if (outLiner_count < 2)
	{
		return 0;
	}
	else
	{
		return 1.0*outLiner_count/(1.0*total_count);
	}
	
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

int isBelongToThisObject(vector<Point2f> point,Rect rect)
{
	int in_count = 0;

	for (int i=0; i<point.size(); i++)
	{
		int x = point.at(i).x;
		int y = point.at(i).y;

		if ((x>=rect.x && x<=rect.x+rect.width) && (y>=rect.y && y<=rect.y+rect.height))
		{
			in_count++;
		}
	}
	float max_ratio = 1.0*in_count/point.size();
	if (max_ratio > 0.2)
	{
		return true;
	}
	return false;	
}

void computeSparseFlow( vector<Point2f> corners, vector<Point2f> &new_corners,Mat preImage, Mat currImage,Rect rect_pre ) 
{
	vector<uchar> status;
	vector<float> err;
	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
	
	cornerSubPix(preImage,corners,Size(10,10),Size(-1,-1),termcrit);

	calcOpticalFlowPyrLK(preImage,currImage,corners,new_corners,status,err,Size(5,5),3,termcrit,0,0.0001);
}

void computeDenseFlow( Mat preImage, Mat currImage, Mat currOptflow, Mat &preOptflow, Mat &flow,vector<trajectory> &tracker, int current_frame_no,int i) 
{
	vector<Point2f> new_corners;
	vector<Point2f> corners; 
	trajectory cur_trajectory;

	vector<Rect> candinate_pre;
	vector<Rect> candinate_curr;

	//vector<objectFrame> total_object_pre;
	//vector<objectFrame> total_object_curr;
	vector<bool> hasID;
	vector<int> ID;
	
	char disp_name[256];
	char flow_name[256];
	char low_name[256];
	sprintf(disp_name,"disp/disp_%d.jpg",i);
	sprintf(flow_name,"flow/flow_%d.jpg",i);
	sprintf(low_name,"low/low_%d.jpg",i);

	hasID.clear();

	LKTracking* LKTrack = new LKTracking();

	Mat showImage,flowImage;
	currImage.copyTo(showImage);
	preImage.copyTo(flowImage);

	cvtColor(flowImage,flowImage,CV_GRAY2BGR);

	if (preImage.data)
	{
		//-----------------------------第一步， 帧差-------------------------------------------------//
	    
		IplImage* opt_img = LKTrack->getTargetFeaturePoint(&IplImage(preImage),&IplImage(currImage));

		Mat temp = Mat(opt_img,true);

		imwrite(disp_name,temp);

		temp.copyTo(currOptflow);

	

		if (preOptflow.data)
		{
			candinate_pre.clear();
			candinate_curr.clear();

			candinate_pre = findCandidate(Mat(preOptflow));
			candinate_curr = findCandidate(Mat(currOptflow));

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
							
							if (area_ratio > 0.5)
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

			for (vector<trajectory>::iterator it=tracker.begin(); it!=tracker.end(); it++)
			{
				if (it->active)// && (it->end_frame - it->start_frame) >= 3)
				{
					Rect belong = it->objectTrack.back();

					char id[256];
					sprintf(id,"%d",it->id);
					cout<<"in_id:   "<<id<<endl;
					putText(showImage,id,Point(belong.x,belong.y),FONT_HERSHEY_SCRIPT_COMPLEX,0.6,Scalar(255,255,255),1,8,0);
					//rectangle(detect_img,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(255,255,255),2,8,0);					
					rectangle(showImage,Point(belong.x,belong.y),Point(belong.x+belong.width,belong.y+belong.height),Scalar(255,255,255),2,8,0);	
				}
				else
				{
					if (it->objectTrack.size() < 3)
					{
						tracker.erase(it);
					}
				}
			}						

			for (vector<Rect>::iterator it=candinate_pre.begin(); it!=candinate_pre.end(); it++)
			{
				rectangle(flowImage,Point(it->x,it->y),Point(it->x+it->width,it->y+it->height),Scalar(0,0,255),2,8,0);
			}

			for (vector<Rect>::iterator it=candinate_curr.begin(); it!=candinate_curr.end(); it++)
			{
				rectangle(flowImage,Point(it->x,it->y),Point(it->x+it->width,it->y+it->height),Scalar(0,255,0),2,8,0);
			}
			//---------------------------
		//}
		swap(preOptflow,currOptflow);

		imshow("flow",flowImage);
		imshow("detect_in",showImage);
		imshow("diff_img",Mat(opt_img));

		imwrite(flow_name,flowImage);
		imwrite(low_name,showImage);
	}
} 

void drawTrajectory( vector<trajectory>::iterator it, Mat detect_img, Scalar color ) 
{
	int x1,y1,x2,y2;
	for (int num=0; num<it->objectTrack.size(); num++)
	{
		if (num == 0)
		{
			x1 = it->objectTrack.at(num).x +  it->objectTrack.at(num).width/2;
			y1 = it->objectTrack.at(num).y +  it->objectTrack.at(num).height/2;
			continue;
		}


		x2 = it->objectTrack.at(num).x +  it->objectTrack.at(num).width/2;
		y2 = it->objectTrack.at(num).y +  it->objectTrack.at(num).height/2;

		// circle(detect_img,Point(x1,y1),2,color,2,8,0);

		line(detect_img,Point(x1,y1),Point(x2,y2),color,2,8,0);

		x1 = x2;
		y1 = y2;
	}
}

int demo_viso_mono() {

  float K_data[12] = {776.4192,0,641.8310,0,0,776.4192,359.3430,0,0,0,1,0};

  Mat K = Mat(3,4,CV_32FC1,K_data);

  VisualOdometryMono::parameters param;

  // calibration parameters for sequence 2010_03_09_drive_0019
  param.calib.f  = 776.4192; // focal length in pixels
  param.calib.cu = 641.8310; // principal point (u-coordinate) in pixels^
  param.calib.cv = 359.3430; // principal point (v-coordinate) in pixels
  param.height = 1.6;
  param.pitch  = -0.08;
  param.inlier_threshold = 0.00005;

  param.bucket.max_features = 1;
  param.bucket.bucket_width = 10;
  param.bucket.bucket_height = 10;

//  param.match = param_matcher;
 
  int firt_frame = 100;
  int last_frame = 2000;

  // init visual odometry
  VisualOdometryMono* viso = new VisualOdometryMono(param);

  // current pose (this matrix transforms a point from the current
  // frame's camera coordinates to the first frame's camera coordinates)
  Matrix pose = Matrix::eye(4);

  vector<Mat> p_total;//projection 
  vector<Matrix> pose_total;
  pose_total.push_back(pose);

  
  vector<trajectory> tracker;
  int trajectory_count = 0;

  bool replace = false;
  char left_img_file_name[256] = {0};
  //char opt_file_name[256] ={0};

  //帧差输入图像
  Mat preImage;
  Mat currImage;

  //光流输入图像
  Mat preOptflow;
  Mat currOptflow;

  Mat flow;

  Mat input_img;
  Mat show_img;
  Mat detect_img;
  //FILE* fp = fopen("xyz.txt","w");

  for (int32_t i=firt_frame; i<last_frame; i+=10) {

	      int h = i - firt_frame + 1;

          sprintf(left_img_file_name,"E:/2016cvpr_v.1/Data/data1026/15/15 %.4d.jpg",i);
		  //sprintf(left_img_file_name,"E:/2016cvpr_v.1/Data/2011_09_26_drive_0005_sync/image_00/data/%.10d.png",i);
		 // sprintf(opt_file_name,"E:/2016cvpr_v.1/code/v1_mwg/LKOptimal-mwg/LKOptimal/result/%d.jpg",i);
         
		  input_img = imread(left_img_file_name);
		  input_img.copyTo(show_img);
		  input_img.copyTo(detect_img);
				 
		 // Mat opt_img = imread(opt_file_name,0);
         //viso_mono(left_img,i, viso, pose);


		  int64 t = getTickCount();


		  if (input_img.channels() == 3)
		  {
			  cvtColor(input_img,input_img,CV_BGR2GRAY);
		  }	
		  
		  //----------------------------------帧差,计算稠密光流--------------------------//
		  input_img.copyTo(currImage);
		  
		  if (preImage.data)
		  {
			 // Mat dst;
			 // absdiff(preImage,preImage,dst);
			  computeDenseFlow(preImage, currImage, currOptflow, preOptflow, flow,tracker,h,i);
			 //  imshow("dst",dst);
		  }
		  t = getTickCount() - t;
		  printf("%f ms\n", t*1000/getTickFrequency());

		  swap(preImage,currImage);
		  //-----------------------------------------------------------------------------//

		  /* for (int i=0; i<flow.rows; i+=1)
		  {
		  for (int j=0; j<flow.cols; j+=1)
		  {
		  int v = i + flow.at<Vec2f>(i,j)[0];
		  int u = j + flow.at<Vec2f>(i,j)[1];

		  float dis = sqrt(1.0*(u -j)*(u-j) + (v-i)*(v-i));

		  if (dis > 5)
		  {
		  line(show_img,Point(j,i),Point(u,v),Scalar(255,255,255),2,8,0);
		  }							
		  }
		  }
		  imshow("show_img",show_img);*/
		  //-----------------------------------------------------------------------------//

		  try {
			  // image dimensions^
			  int32_t width  = input_img.cols;
			  int32_t height = input_img.rows;

			  // convert input images to uint8_t buffer
			  uint8_t* input_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
			  int32_t k=0;
			  for (int32_t v=0; v<height; v++) {
				  for (int32_t u=0; u<width; u++) {
					  input_img_data[k]  = input_img.at<uchar>(v,u);
					  k++;
				  }
			  }
			  // status
			  cout << "Processing: Frame: " << i;

			  // compute visual odometry
			  int32_t dims[] = {width,height,width};

			  bool vo_success_flag = viso->process(input_img_data,dims,replace);

			  if ( h <= 1)
			  {
				  continue;
			  }  

			  if (vo_success_flag) 
			  {
				  replace = 0;
				  //get match point 
				  vector<Matcher::p_match> p_match = viso->getMatches(); 

				  // on success, update current pose
				  pose = pose * Matrix::inv(viso->getMotion());
				  //cout<<"replace:  "<<replace<<endl;
				  // output some statistics
				  double num_matches = viso->getNumberOfMatches();
				  double num_inliers = viso->getNumberOfInliers();
				   cout << ", Matches: " << num_matches;
				   cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;

				   bool *outLinerFlag = new bool[p_match.size()];
				   memset(outLinerFlag,true,sizeof(bool)*p_match.size());
				   memset(isOutliner,0,sizeof(int)*IMG_COLS*IMG_ROWS);
				   				   
				  
				  //-----------------------------------标记内点外点--------------------------------------//	
				   vector<int32_t> index = viso->getInlierIndices();					
				   for (vector<int32_t>::iterator it = index.begin(); it!=index.end(); it++)
				   {
					   int id = *it;
					   outLinerFlag[id] = false;
				   } 

				  cvtColor(input_img,input_img,CV_GRAY2BGR);
				  for (int i=0; i<p_match.size(); i++)
				  {
					  int u = p_match.at(i).u1c;
					  int v = p_match.at(i).v1c;
					  
					  if (outLinerFlag[i])
					  {
						  isOutliner[v][u] = 1; //外点
						  circle(input_img,Point(u,v),2,Scalar(0,255,0),2,8,0);
					  }
					  else
					  {
						  isOutliner[v][u] = -1; //内点
						  circle(input_img,Point(u,v),2,Scalar(0,0,255),2,8,0);
					  }
				  }
				
				  //--------------------------------------------------------------------------------//
				  delete outLinerFlag;
			  } 
			  else 
			  {
				  pose_total.push_back(pose_total.back());
				  replace = 1;
				  cout << " ... failed!" << endl;
			  }

			  //----------------------------------判断运动目标------------------------------------//
			  RNG rng;

			  for (vector<trajectory>::iterator it=tracker.begin(); it!=tracker.end(); it++)
			  {
				  int icolor = (unsigned)rng;
				  Scalar color =  Scalar(icolor&0xFF, (icolor>>8)&0xFF, (icolor>>16)&0xFF);

				  if (it->active) //&& (it->end_frame - it->start_frame) >= 3)
				  {
					  if (it->object_status == SURE)
					  {
						  Rect rect = it->objectTrack.back();
						  char id[15];
						  sprintf(id,"%d",it->id);
						 // cout<<"out-draw:   "<<id<<endl;
						  //putText(detect_img,id,Point(rect.x,rect.y),FONT_HERSHEY_SCRIPT_COMPLEX,0.6,color,1,8,0);
						  rectangle(detect_img,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),color,2,8,0);
						 
						  //drawTrajectory(it, detect_img, color);

					  }
					  else
					  {
						  Rect rect = it->objectTrack.back();
						  for (int i=rect.y; i<rect.y+rect.height; i++)
						  {
							  for (int j=rect.x; j<rect.x+rect.width; j++)
							  {
								  if (preOptflow.at<uchar>(i,j) > 0)
								  {
									  if (isOutliner[i][j] == 0)
									  {				
										  continue;
									  }
									  else if (isOutliner[i][j] == -1)
									  {
										  it->total_count++;
									  }
									  else if(isOutliner[i][j] == 1)
									  {
										  it->total_count++;
										  it->outliners_count++;
									  }
								  }
							  }

							  float ratio = 1.0*it->outliners_count/it->total_count;

							  if (ratio > 0.5 && it->outliners_count >= 2)
							  {
								  it->object_status = SURE;
							  }
							  //  cout<<ratio<<endl;
							  if (ratio > 0.4 && (it->end_frame - it->start_frame) >= 2)
							  {							 
								  char id[15];
								  sprintf(id,"%d",it->id);
								  //  cout<<"out-draw:   "<<id<<endl;
								  //putText(detect_img,id,Point(rect.x,rect.y),FONT_HERSHEY_SCRIPT_COMPLEX,0.6,Scalar(255,255,255),1,8,0);
								  rectangle(detect_img,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),color,2,8,0);	
								  // drawTrajectory(it, detect_img, color);
							  }

						  }
					  }						  
				  }
				  // cout<<"outliners:  "<<it->outliners_count<<"    total_count:   "<<it->total_count<<endl;
			  }

			  char detect_out_name[256];
			  char inlier_name[256];
			  sprintf(detect_out_name,"detect_out/detect_%d.jpg",i);
			  sprintf(inlier_name,"inlier/inlier_%d.jpg",i);
			  imwrite(detect_out_name,detect_img);
			  imwrite(inlier_name,input_img);


			  imshow("detect_out",detect_img);
			  imshow("input_img",input_img);
			  waitKey(5);
			  // release uint8_t buffers
			  free(input_img_data);

			  // catch image read errors here
		  } catch (...) {
			  cerr << "ERROR: Couldn't read input files!" << endl;
		  }

		
		  
  }
  cout << "Demo complete! Exiting ..." << endl;

  // exit
  return 0;
}