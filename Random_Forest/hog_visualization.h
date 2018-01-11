#ifndef HOG_VISUALIZATION_H
#define HOG_VISUALIZATION_H

#include "..\include\dirent.h"

// for general functionality
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\core\types.hpp"
#include "opencv2\core\mat.hpp"
#include "opencv2\core.hpp"

// for HOGDescriptor class
#include "opencv2\objdetect.hpp"

extern void visualizeHOG(cv::Mat & img, 
						 std::vector<float> &feats, 
						 cv::HOGDescriptor & hog_detector, 
						 int scale_factor = 3);

/*
* img          - the image used for computing HOG descriptors. 
				 **Attention here the size of the image should be the same as 
				 the window size of your cv::HOGDescriptor instance **
* feats        - the hog descriptors you get after calling cv::HOGDescriptor::compute
* hog_detector - the instance of cv::HOGDescriptor you used
* scale_factor - scale the image *scale_factor* times larger for better visualization
*/

int get_data_and_labels(cv::String path, cv::Mat & features, cv::Mat & labels, cv::Size & roi_size);

#endif // !HOG_VISUALIZATION

