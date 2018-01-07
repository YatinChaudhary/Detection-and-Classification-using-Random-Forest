#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "hog_visualization.h"

class Random_Forest {
public:
	Random_Forest(const int & n_trees,
				  const int & cv_folds,
				  const int & max_categories,
				  const int & max_depth,
				  const int & min_sample_count);

	void create();
	void train();
	int predict(cv::Mat & feature);

	~Random_Forest();

private:
	int n_trees_;
	int cv_folds_;
	int max_categories_;
	int max_depth_;
	int min_sample_count_;
};

#endif // !RANDOM_FOREST_H

