#include <iostream>
#include "..\include\dirent.h"
#include "opencv2\ml.hpp"
#include "hog_visualization.h"

int get_data_and_labels(cv::String path, cv::Mat & features, cv::Mat & labels);

int main()
{
	// creating DTree
	//std::cout << "\nchk-1" << std::endl;
	cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
	cv::Mat features;
	cv::Mat labels;
	//std::cout << "\nchk-2" << std::endl;
	//cv::waitKey(5000);
	int x = get_data_and_labels("D:\\M.Sc\\Semester_3\\Tracking and Detection in Computer Vision\\Exercises\\2\\data\\task2\\train", features, labels);
	//cv::waitKey(5000);
	return 0;
}

int get_data_and_labels(cv::String path, cv::Mat & features, cv::Mat & labels)
{
	DIR *dir, *subdir;
	struct dirent *ent, *ent_subdir;
	if ((dir = opendir(path.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			printf("%s\n", ent->d_name);
			if ((subdir = opendir(std::string(path + "\\" + ent->d_name).c_str())) != NULL) {
				while ((ent_subdir = readdir(subdir)) != NULL) {
					printf("%s\n", ent_subdir->d_name);
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