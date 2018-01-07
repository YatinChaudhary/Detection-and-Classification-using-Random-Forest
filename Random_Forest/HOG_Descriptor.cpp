#include <iostream>
#include "hog_visualization.h"

bool visualize_progress = true;

int HOG_Descriptor()
{
	/**********************************************************************
	READING THE ORIGINAL IMAGE AND CREATING DIFFERENT IMAGES USING
	BASIC IMAGE PROCESSING IN OPENCV
	**********************************************************************/

	cv::String image_path = "../../data/task1/obj1000.jpg";

	// Reading the original image
	cv::Mat original_image;
	try
	{
		original_image = cv::imread(image_path, 1);

		if (visualize_progress)
		{
			std::cout << original_image.size() << std::endl;
			cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Image", original_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during reading the original image: " << ex.what() << "\n\n";
	}


	// Converting the original image to grayscale
	cv::Mat grayscale_image;
	try
	{
		cv::cvtColor(original_image, grayscale_image, CV_RGB2GRAY);

		if (visualize_progress)
		{
			std::cout << grayscale_image.size() << std::endl;
			cv::namedWindow("Grayscale Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Grayscale Image", grayscale_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during converting the original image to grayscale: " << ex.what() << "\n\n";
	}


	// Expanding the original image
	cv::Mat expanded_image;
	try
	{
		cv::resize(original_image, expanded_image, cv::Size(), 2.0, 2.0);

		if (visualize_progress)
		{
			std::cout << expanded_image.size() << std::endl;
			cv::namedWindow("Expanded Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Expanded Image", expanded_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during expanding the original image: " << ex.what() << "\n\n";
	}


	// Compressing the original image
	cv::Mat compressed_image;
	try
	{
		cv::resize(original_image, compressed_image, cv::Size(), 0.5, 0.5);

		if (visualize_progress)
		{
			std::cout << compressed_image.size() << std::endl;
			cv::namedWindow("Compressed Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Compressed Image", compressed_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during compressing the original image: " << ex.what() << "\n\n";
	}


	// Rotating the original image
	cv::Mat rotated_image;
	try
	{
		cv::rotate(original_image, rotated_image, cv::ROTATE_90_CLOCKWISE);

		if (visualize_progress)
		{
			std::cout << rotated_image.size() << std::endl;
			cv::namedWindow("Rotated Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Rotated Image", rotated_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during rotating the original image: " << ex.what() << "\n\n";
	}


	// Flipping the original image
	cv::Mat flipped_image;
	try
	{
		cv::flip(original_image, flipped_image, 0);

		if (visualize_progress)
		{
			std::cout << flipped_image.size() << std::endl;
			cv::namedWindow("Flipped Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Flipped Image", flipped_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during flipping the original image: " << ex.what() << "\n\n";
	}


	/**********************************************************************
	CALCULATING HOG DESCRIPTORS OF ALL DIFFERENT IMAGES CREATED ABOVE
	**********************************************************************/
	
	//std::cout << "\n\nSize: " << original_image.rows << "  " << original_image.cols << std::endl;
	cv::Rect region(0, 0, 124, 104);
	cv::Mat cropped_image = original_image(region);
	cv::Size cellsize(8, 8);
	cv::Size blocksize(16, 16);
	cv::Size stridesize(4, 4);
	cv::Size winsize(cropped_image.cols, cropped_image.rows);
	
	cv::HOGDescriptor hog_cropped_image(winsize, blocksize, stridesize, cellsize, 9);
	std::vector<float> descriptors_cropped_image;

	try
	{
		hog_cropped_image.compute(cropped_image,
								   descriptors_cropped_image,
								   cv::Size(),
								   cv::Size(0, 0));	
		//cv::waitKey(0);
		visualizeHOG(cropped_image, descriptors_cropped_image, hog_cropped_image);
		cv::waitKey(3000);
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during calculating the HOG descriptor: " << ex.what() << "\n\n";
	}

	//cv::waitKey(0);
	return 0;
}