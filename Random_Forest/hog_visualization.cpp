#include "hog_visualization.h"

void visualizeHOG(cv::Mat & img, std::vector<float> &feats, cv::HOGDescriptor & hog_detector, int scale_factor) {

    cv::Mat visual_image;
    resize(img, visual_image, cv::Size(img.cols * scale_factor, img.rows * scale_factor));

    int n_bins = hog_detector.nbins;
    float rad_per_bin = 3.14 / (float) n_bins;
    cv::Size win_size = hog_detector.winSize;
    cv::Size cell_size = hog_detector.cellSize;
    cv::Size block_size = hog_detector.blockSize;
    cv::Size block_stride = hog_detector.blockStride;

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = win_size.width / cell_size.width;
    int cells_in_y_dir = win_size.height / cell_size.height;
    int n_cells = cells_in_x_dir * cells_in_y_dir;
    int cells_per_block = (block_size.width / cell_size.width) * (block_size.height / cell_size.height);

    int blocks_in_x_dir = (win_size.width - block_size.width) / block_stride.width + 1;
    int blocks_in_y_dir = (win_size.height - block_size.height) / block_stride.height + 1;
    int n_blocks = blocks_in_x_dir * blocks_in_y_dir;

    float ***gradientStrengths = new float **[cells_in_y_dir];
    int **cellUpdateCounter = new int *[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float *[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[n_bins];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < n_bins; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }


    // compute gradient strengths per cell
    int descriptorDataIdx = 0;


    for (int block_x = 0; block_x < blocks_in_x_dir; block_x++) {
        for (int block_y = 0; block_y < blocks_in_y_dir; block_y++) {
            int cell_start_x = block_x * block_stride.width / cell_size.width;
            int cell_start_y = block_y * block_stride.height / cell_size.height;

            for (int cell_id_x = cell_start_x;
                 cell_id_x < cell_start_x + block_size.width / cell_size.width; cell_id_x++)
                for (int cell_id_y = cell_start_y;
                     cell_id_y < cell_start_y + block_size.height / cell_size.height; cell_id_y++) {

                    for (int bin = 0; bin < n_bins; bin++) {
                        float val = feats.at(descriptorDataIdx++);
                        gradientStrengths[cell_id_y][cell_id_x][bin] += val;
                    }
                    cellUpdateCounter[cell_id_y][cell_id_x]++;
                }
        }
    }


    // compute average gradient strengths
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

            float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < n_bins; bin++) {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }


    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cell_size.width;
            int drawY = celly * cell_size.height;

            int mx = drawX + cell_size.width / 2;
            int my = drawY + cell_size.height / 2;

            rectangle(visual_image,
                      cv::Point(drawX * scale_factor, drawY * scale_factor),
                      cv::Point((drawX + cell_size.width) * scale_factor,
                                (drawY + cell_size.height) * scale_factor),
                      CV_RGB(100, 100, 100),
                      1);

            for (int bin = 0; bin < n_bins; bin++) {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                if (currentGradStrength == 0)
                    continue;

                float currRad = bin * rad_per_bin + rad_per_bin / 2;

                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float maxVecLen = cell_size.width / 2;
                float scale = scale_factor / 5.0; // just a visual_imagealization scale,

                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visual_imagealization
                line(visual_image,
                     cv::Point(x1 * scale_factor, y1 * scale_factor),
                     cv::Point(x2 * scale_factor, y2 * scale_factor),
                     CV_RGB(0, 0, 255),
                     1);

            }

        }
    }


    for (int y = 0; y < cells_in_y_dir; y++) {
        for (int x = 0; x < cells_in_x_dir; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    cv::imshow("HOG vis", visual_image);
    cv::waitKey(0);
    cv::imwrite("hog_vis.jpg", visual_image);

}


int get_data_and_labels(cv::String path, cv::Mat & features, cv::Mat & labels, cv::Size & roi_size)
{
	DIR *dir, *subdir;
	struct dirent *ent, *ent_subdir;
	std::string subdir_path, image_path;

	cv::Mat image, crop(roi_size, CV_32F), row;
	cv::Rect roi;
	cv::Size cellsize(8, 8);
	cv::Size blocksize(16, 16);
	cv::Size stridesize(16, 16);
	cv::Size winsize(roi_size.width, roi_size.height);

	cv::HOGDescriptor hog(winsize, blocksize, stridesize, cellsize, 9);
	std::vector<float> descriptor;

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
							cv::cvtColor(image, image, CV_RGB2GRAY);

							roi = cv::Rect(image.cols / 2 - roi_size.width / 2,
								image.rows / 2 - roi_size.height / 2,
								roi_size.width, roi_size.height);
							image(roi).copyTo(crop);
							hog.compute(crop, descriptor, cv::Size(), cv::Size(0, 0));
							// row includes the 3 channels of the original image
							// do we need all the channels?
							//row = crop.reshape(1, 1);
							//row.convertTo(row, CV_32F);
							row = cv::Mat(descriptor, true);
							row = row.reshape(1, 1);
							features.push_back(row);
							labels.push_back(std::stoi(ent->d_name));

							descriptor.clear();
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