#include "Random_Forest.h"

Random_Forest::Random_Forest(const int & n_trees = 20,
							 const int & cv_folds = 10,
							 const int & max_categories = 10,
							 const int & max_depth = 100,
							 const int & min_sample_count = 10)
	:
	n_trees_(n_trees),
	cv_folds_(cv_folds),
	max_categories_(max_categories),
	max_depth_(max_depth),
	min_sample_count_(min_sample_count)
{
}

void Random_Forest::create()
{
	assert((trees.size() == 0) && "Random forest is already created");

	try
	{
		for (int i = 0; i < n_trees_; i++)
		{
			cv::Ptr<cv::ml::DTrees> tree = cv::ml::DTrees::create();
			tree->setCVFolds(1);
			tree->setMaxCategories(10);
			tree->setMinSampleCount(1);
			tree->setMaxDepth(12);

			trees.push_back(tree);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "\nException during creating Random Forest: " << ex.what() << std::endl;
	}
}

void Random_Forest::train(int ratio)
{
	assert((trees.size() != 0) && "Random forest is empty");

	roi = cv::Size(48,48);
	int x = get_data_and_labels("D:\\M.Sc\\Semester_3\\Tracking and Detection in Computer Vision\\Exercises\\2\\data\\task2\\train", features, labels, roi);
	if (x)
	{
		try
		{
			int num_rows = static_cast<int>(ratio * features.rows);
			std::vector<int> indices;
			cv::Mat random_feature_rows, random_label_rows;
			
			for (int i = 0; i < trees.size(); i++)
			{
				for (int j = 0; j < features.rows; j++) { 
					indices.push_back(j); 
				}
				cv::randShuffle(indices);
				
				for (int j = 0; j < num_rows; j++) { 
					random_feature_rows.push_back(features.row(indices[j]));
					random_label_rows.push_back(labels.row(indices[j]));
				}

				cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(
					random_feature_rows,
					cv::ml::SampleTypes::ROW_SAMPLE,
					random_label_rows
				);
				trees[i]->train(data);

				random_feature_rows.release();
				random_label_rows.release();
				indices.clear();
			}
		}
		catch (const std::exception& ex)
		{
			std::cout << "\nException during Random Forest training: " << ex.what() << std::endl;
		}
	}
}

int Random_Forest::predict(cv::Mat & feature)
{
	int forest_size = trees.size();
	int prediction, mode = -1;
	std::vector<int> predictions(max_categories_, 0);

	try
	{
		for (int i = 0; i < forest_size; i++)
		{
			prediction = static_cast<int>(trees[i]->predict(feature));
			predictions[prediction] += 1;
		}
		mode = std::distance(predictions.begin(), std::max_element(predictions.begin(), predictions.end()));
	}
	catch (const std::exception& ex)
	{
		std::cout << "\nException during Random Forest prediction: " << ex.what() << std::endl;
	}
	
	return mode;
}

Random_Forest::~Random_Forest() {
	int forest_size = trees.size();
	for (int i = 0; i < forest_size; i++)
	{
		// delete all DTrees here
	}
}