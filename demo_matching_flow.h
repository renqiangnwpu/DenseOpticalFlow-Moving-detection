#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <stdint.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "viso_stereo.h"
#include "Affine2D.h"
//#include <png++/png.hpp>

using namespace std;
using namespace cv;


#define IMG_ROWS 768
#define IMG_COLS 1024

#define MAX_POINT_COUNT 50*50

enum{UNCERTAIN,SURE};

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


void computeDenseFlow( Mat preImage, Mat currImage, Mat currOptflow, Mat &preOptflow,vector<trajectory> &tracker, int current_frame_no);
int frameToframesubstraction(Mat pre_img, Mat curr_img, Mat& diff_img,int isOutliner[][IMG_COLS]);

vector<Rect> findCandidatesbyBFS(Mat image);