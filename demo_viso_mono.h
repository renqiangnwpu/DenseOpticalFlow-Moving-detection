#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "viso_mono.h"
#include "Affine2D.h"
//#include <png++/png.hpp>

using namespace std;
using namespace cv;

int viso_mono( Mat left_img, int32_t i, VisualOdometryMono &viso, Matrix &pose );

int demo_viso_mono();
