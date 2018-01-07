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
	 
}

void Random_Forest::train()
{

}

int Random_Forest::predict(cv::Mat & feature)
{
	return 0;
}

Random_Forest::~Random_Forest() {}