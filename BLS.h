#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace cv;

class BLS
{
public:
	BLS();
	~BLS();
	cv::Mat BLS::backgroundLaplacianSaliency(Mat img_color);
private:
	cv::Mat BLS::dtgs(Mat& img);
	cv::Mat BLS::defaultGaussianBlur(cv::Mat& src);
	cv::Mat BLS::localLaplacianSaliency(cv::Mat& img_color);
};




