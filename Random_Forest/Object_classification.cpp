#include <iostream>
#include "..\include\dirent.h"
#include "opencv2\ml.hpp"
#include "hog_visualization.h"

int get_data_and_labels(cv::String path, cv::Mat & features, cv::Mat & labels, cv::Size & roi_size);

int main()
{
	// creating DTree
	cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
	cv::Mat features;
	cv::Mat labels;
	cv::Size roi(50, 50);

	int x = get_data_and_labels("D:\\M.Sc\\Semester_3\\Tracking and Detection in Computer Vision\\Exercises\\2\\data\\task2\\train", features, labels, roi);
	if (x == 0)
	{
		std::cout << "\n\nFeature vector size:   " << features.size << std::endl;
		std::cout << "\n\nLabel vector size:   " << labels.size << std::endl;
	}
	return 0;
}

int get_data_and_labels(cv::String path, cv::Mat & features, cv::Mat & labels, cv::Size & roi_size)
{
	DIR *dir, *subdir;
	struct dirent *ent, *ent_subdir;
	cv::Mat image, crop(roi_size, CV_32F), row;
	cv::Rect roi;
	std::string subdir_path, image_path;

	if ((dir = opendir(path.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			//printf("%s\n", ent->d_name);
			if ((std::string(ent->d_name) != std::string(".")) &&
				(std::string(ent->d_name) != std::string("..")))
			{
				subdir_path = path + "\\" + ent->d_name;
				if ((subdir = opendir(subdir_path.c_str())) != NULL) {
					while ((ent_subdir = readdir(subdir)) != NULL) {
						//printf("%s\n", ent_subdir->d_name);
						if ((std::string(ent_subdir->d_name) != std::string(".")) &&
							(std::string(ent_subdir->d_name) != std::string("..")))
						{
							image_path = subdir_path + "\\" + ent_subdir->d_name;
							image = cv::imread(image_path, 1);
							roi = cv::Rect(image.cols/2 - roi_size.width/2,
										   image.rows/2 - roi_size.height/2,
										   roi_size.width, roi_size.height);
							image(roi).copyTo(crop);
							// row includes the 3 channels of the original image
							// do we need all the channels?
							row = crop.reshape(1, 1);
							row.convertTo(row, CV_32F);
							features.push_back(row);
							labels.push_back(std::stoi(ent->d_name));
						}
					}

					closedir(subdir);
				}
			}
		}

		closedir(dir);
		return 0;
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
}