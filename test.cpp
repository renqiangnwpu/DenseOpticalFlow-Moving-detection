#include "demo_matching_flow.h"
#include "demo_viso_mono.h"

int isOutliner[IMG_ROWS][IMG_COLS] = {0}; //-1 inliner, 0 unknow, 1 outliner

void drawFinalDetectResult( vector<trajectory> &tracker, Mat final_detect, Mat &preOptFlow ) 
{
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
				//cout<<"out-draw:   "<<id<<endl;
				//putText(final_detect,id,Point(rect.x,rect.y),FONT_HERSHEY_SCRIPT_COMPLEX,0.6,color,1,8,0);
				rectangle(final_detect,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);

				//drawTrajectory(it, detect_img, color);

			}
			else
			{
				Rect rect = it->objectTrack.back();

				int outliners_count = 0;
				int total_count = 0;

				for (int i=rect.y; i<rect.y+rect.height; i++)
				{
					for (int j=rect.x; j<rect.x+rect.width; j++)
					{
						if (preOptFlow.at<uchar>(i,j) > 0)
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
								outliners_count++;
							}
						}
					}

					float temp_ratio = 1.0*outliners_count/total_count;

					/*if (temp_ratio > 0)
					{
						cout<<temp_ratio<<endl;
					}*/

					it->total_count = it->total_count + total_count;
					it->outliners_count = it->outliners_count + outliners_count;

					float ratio = 1.0*it->outliners_count/it->total_count;

					//------------------------------------------------------------------------//
					if (ratio > 0.5 && outliners_count >= 7 && it->objectTrack.size() >= 3)
					{
						it->object_status = SURE;
						rectangle(final_detect,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);
					}

					if (temp_ratio > 0.5 && outliners_count >= 7 && it->objectTrack.size() >= 3)
					{
						it->object_status = SURE;
						rectangle(final_detect,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);
					}

					if (it->objectTrack.size() >= 6 && it->total_count <= 10)
					{
						it->object_status = SURE;
						rectangle(final_detect,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);
					}

					//  cout<<ratio<<endl;
					if (ratio > 0.5 && it->objectTrack.size() >= 7 && it->outliners_count >= 5)
					{
						it->object_status = SURE;

						char id[15];
						sprintf(id,"%d",it->id);
						//putText(final_detect,id,Point(rect.x,rect.y),FONT_HERSHEY_SCRIPT_COMPLEX,0.6,Scalar(255,255,255),1,8,0);
						rectangle(final_detect,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);
					}
					//------------------------------------------------------------------------//

				}
			}						  
		}
		else
		{
			//tracker.erase(it);
		}
	}
}


void drawMiddleDetectResult( vector<trajectory> tracker, Mat middle_detect) 
{
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
				//cout<<"out-draw:   "<<id<<endl;
				//putText(middle_detect,id,Point(rect.x,rect.y),FONT_HERSHEY_SCRIPT_COMPLEX,0.6,color,1,8,0);
				rectangle(middle_detect,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);

				//drawTrajectory(it, detect_img, color);

			}
			else
			{
				Rect rect = it->objectTrack.back();

			    if (it->objectTrack.size() >= 3)
			    {
					it->object_status = SURE;

					//char id[15];
					//sprintf(id,"%d",it->id);
					//putText(middle_detect,id,Point(rect.x,rect.y),FONT_HERSHEY_SCRIPT_COMPLEX,0.6,Scalar(255,255,255),1,8,0);
				    rectangle(middle_detect,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);
				}
			}
		  }
	
 	}
	
}

int main()
{	
	char pre_file_name[256];
	char curr_file_name[256];
	char next_file_name[256];

	char final_detect_name[256];
	char middle_detect_name[256];
	char outliers_name[256];
	char substraction_name[256];

	vector<trajectory> tracker;
	vector<trajectory> tracker_middle;
	int frame_no = 0;
	Mat preOptFlow,  currOptFlow;
	Mat diff_img_1,diff_img_2,diff_img;
	Mat middle_detect,final_detect;
	Mat outliers_inliers;
	
	for (int i=500; i<=800; i+=1)
	{
		sprintf(final_detect_name,"final_detect/%d.jpg",frame_no);
		sprintf(middle_detect_name,"middle_detect/%d.jpg",frame_no);
		sprintf(outliers_name,"outliers/%d.jpg",frame_no);
		sprintf(substraction_name,"disp/%d.jpg",frame_no);

		sprintf(pre_file_name,"H:/Data/航拍视频/Egtest/egtest01/frame%.5d.jpg",i);
		sprintf(curr_file_name,"H:/Data/航拍视频/Egtest/egtest01/frame%.5d.jpg",i+10);
		sprintf(next_file_name,"H:/Data/航拍视频/Egtest/egtest01/frame%.5d.jpg",i+20);

		//cout<<"L:   "<<left_file_name<<"     "<<"R:   "<<right_file_name<<endl;

		Mat pre_img = imread(pre_file_name,1);
		Mat curr_img = imread(curr_file_name,1);
		Mat next_img = imread(next_file_name,1);

		curr_img.copyTo(final_detect);
		curr_img.copyTo(outliers_inliers);
		curr_img.copyTo(middle_detect);
		

		if (curr_img.channels() == 3)
		{
			cvtColor(pre_img,pre_img,CV_RGB2GRAY);
			cvtColor(curr_img,curr_img,CV_RGB2GRAY);
			cvtColor(next_img,next_img,CV_RGB2GRAY);
		}

		int64 t = getTickCount();

		//两次帧差
		if (false)
		{
			frameToframesubstraction(next_img,curr_img,diff_img_2,isOutliner);
			frameToframesubstraction(pre_img,curr_img,diff_img_1,isOutliner);
			bitwise_or(diff_img_1,diff_img_2,diff_img);
		}
		else
		{
			frameToframesubstraction(pre_img,curr_img,diff_img,isOutliner);
		}
		
		diff_img.copyTo(currOptFlow);

		if (preOptFlow.data)
		{
			computeDenseFlow(pre_img, curr_img, currOptFlow, preOptFlow,tracker,frame_no++);
		}
		swap(preOptFlow,currOptFlow);

		tracker_middle.assign(tracker.begin(),tracker.end());

		drawMiddleDetectResult(tracker_middle,middle_detect);
		
		drawFinalDetectResult(tracker, final_detect, preOptFlow);

		for (int i=0; i<IMG_ROWS; i++)
		{
			for (int j=0; j<IMG_COLS; j++)
			{
				if (isOutliner[i][j] == 1)
				{
					circle(outliers_inliers,Point(j,i),2,Scalar(0,255,0),2,8,0);
				}
				else if (isOutliner[i][j] == -1)
				{
					circle(outliers_inliers,Point(j,i),2,Scalar(0,0,255),2,8,0);
				}

			}
		}

		if (true)
		{
			imwrite(final_detect_name,final_detect);
			imwrite(outliers_name,outliers_inliers);
			imwrite(substraction_name,diff_img_1);
			imwrite(middle_detect_name,middle_detect);
		}

		if (true)
		{
			vector<Rect> candinate = findCandidatesbyBFS(diff_img);
			for (vector<Rect>::iterator it = candinate.begin(); it!=candinate.end(); it++)
			{
		      rectangle(diff_img,Point(it->x,it->y),Point(it->x+it->width,it->y+it->height),Scalar(255,255,255),2,8,0);
			}

			imshow("outliers",outliers_inliers);
			imshow("diff_2",diff_img);
		}
		
		imshow("final",final_detect);
		imshow("middle",middle_detect);		
		waitKey(5);

		t = getTickCount() - t;
		printf("Frame: %d,  %f ms\n",i, t*1000/getTickFrequency());
	}


	for (vector<trajectory>::iterator it = tracker.begin(); it!=tracker.end(); it++)
	{
		int k= 0;
		if (it->object_status == SURE)
		{
			cout<<"draw object:  "<<k++<<endl;
			for (int i=it->start_frame; i<=it->end_frame; i++)
			{
			sprintf(final_detect_name,"final_detect/%d.jpg",i);
			sprintf(middle_detect_name,"middle_detect/%d.jpg",i);

			final_detect = imread(final_detect_name,1);
			middle_detect = imread(middle_detect_name,1);

			Rect rect = it->objectTrack.at(i-it->start_frame);

			rectangle(final_detect,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);
			rectangle(middle_detect,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,0,255),2,8,0);

			imwrite(final_detect_name,final_detect);
			imwrite(middle_detect_name,middle_detect);
			}
		}
	}


}