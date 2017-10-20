#include "BLS.h"


BLS::BLS()
{
}

BLS::~BLS()
{
}

cv::Mat BLS::backgroundLaplacianSaliency(Mat img_color)
{
	cv::Mat salOut(img_color.rows, img_color.cols, CV_32F, Scalar::all(0));
	int sizeX, sizeY;
	sizeX = salOut.rows;
	sizeY = salOut.cols;
	resize(img_color, img_color, Size(sizeY, sizeX));
	cv::cvtColor(img_color, img_color, CV_LBGR2Lab);
	cv::GaussianBlur(img_color, img_color, cv::Size(5, 5), 0, 0);

	cv::Mat  salGlobal = dtgs(img_color);
	normalize(salGlobal, salGlobal, 0, 1, NORM_MINMAX);

	Mat src, gray, abs_dst;

	cv::Mat lap = localLaplacianSaliency(img_color);

	normalize(lap, lap, 0, 1, NORM_MINMAX);
	float* pSal;
	float* pSalLocal;
	float* pSalOut;
	Mat salLG(img_color.rows, img_color.cols, CV_32F, Scalar::all(0));
	for (int i = 0; i < salOut.rows; i++)
	{
		pSalOut = salOut.ptr<float>(i);
		pSal = salGlobal.ptr<float>(i);
		pSalLocal = lap.ptr<float>(i);
		for (int j = 0; j < salOut.cols; j++)
		{
			pSalOut[j] = (pSal[j] + powf(pSalLocal[j], 1.0)) / 2;
		}
	}
	return salOut;
}



inline cv::Mat BLS::dtgs(Mat & img)
{

	cv::Mat salOut(img.rows, img.cols, CV_32F, Scalar::all(0));
	int sizeX, sizeY;
	sizeX = salOut.rows;
	sizeY = salOut.cols;
	cv::Mat resized;

	uchar* pImg;

	float* psal;
	uchar* pImgColor;

	float sumB = 0.0;
	float sumG = 0.0, sumR = 0.0;
	float valNorm = 0.0;


	cv::Mat dt;
	cv::Mat canny;
	cv::Canny(img, canny, 100, 200);
	bitwise_not(canny, canny);
	distanceTransform(canny, dt, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	
	float* pdt;

	for (int y = 0; y < img.rows; y++)
	{
		pImg = img.ptr<uchar>(y);
		pdt = dt.ptr<float>(y);
		for (int x = 0; x < img.cols; x++)
		{
			valNorm += powf(pdt[x], 1);
			sumB += ((float)pImg[x * 3] / 255)*powf(pdt[x], 1);
			sumG += ((float)pImg[x * 3 + 1] / 255)*powf(pdt[x], 1);
			sumR += ((float)pImg[x * 3 + 2] / 255)*powf(pdt[x], 1);
		}
	}


	sumB = sumB / valNorm;
	sumG = sumG / valNorm;
	sumR = sumR / valNorm;

	float ltable_b[256];
	float ltable_g[256];
	float ltable_r[256];

	for (int graytone = 0; graytone < 256; graytone++)
	{
		ltable_b[graytone] = (((float)graytone) / 255 - sumB)*(((float)graytone) / 255 - sumB);
		ltable_g[graytone] = (((float)graytone) / 255 - sumG)*(((float)graytone) / 255 - sumG);
		ltable_r[graytone] = (((float)graytone) / 255 - sumR)*(((float)graytone) / 255 - sumR);

	}




	for (int i = 0; i < salOut.rows; i++)
	{

		pImg = img.ptr<uchar>(i);
		pdt = dt.ptr<float>(i);
		psal = salOut.ptr<float>(i);
		for (int j = 0; j < salOut.cols; j++)
		{
			psal[j] = ((ltable_b[pImg[j * 3]] + ltable_g[pImg[j * 3 + 1]] + ltable_r[pImg[j * 3 + 2]]) / 3);
		}
	}

	return salOut;
}

inline cv::Mat BLS::defaultGaussianBlur(cv::Mat & src)
{
	cv::Mat dst(src.rows, src.cols, CV_32F, Scalar::all(0));
	float blurSigma = src.cols*0.045;
	//float blurSigma = src.cols*0.02;
	int kSize = round(blurSigma * 4);
	if (kSize % 2 == 0)
		kSize--;
	cv::GaussianBlur(src, dst, cv::Size(kSize, kSize), blurSigma, blurSigma);
	return dst;
}

inline cv::Mat BLS::localLaplacianSaliency(cv::Mat & img_color)
{
	cv::Mat localSalL;
	cv::Mat localSalA;
	cv::Mat localSalB;
	cv::Mat localSal(img_color.rows, img_color.cols, CV_32F, Scalar::all(0));

	cv::Mat imgChannels[3];
	//vector<cv::Mat> imgChannels;
	split(img_color, imgChannels);

	// Apply Laplace function
	Laplacian(imgChannels[0], localSal, CV_32F, 3, 1, 0, BORDER_DEFAULT);
	Laplacian(imgChannels[1], localSalA, CV_32F, 3, 1, 0, BORDER_DEFAULT);
	Laplacian(imgChannels[2], localSalB, CV_32F, 3, 1, 0, BORDER_DEFAULT);

	convertScaleAbs(localSal, localSal);
	convertScaleAbs(localSalA, localSalA);
	convertScaleAbs(localSalB, localSalB);
	uchar *pSal, *pSalA, *pSalB;
	for (int i = 0; i < localSal.rows; i++)
	{
		pSal = localSal.ptr<uchar>(i);
		pSalA = localSalA.ptr<uchar>(i);
		pSalB = localSalB.ptr<uchar>(i);
		for (int j = 0; j < localSal.cols; j++)
		{
			pSal[j] = (pSal[j] + pSalA[j] + pSalB[j]) / 3;
		}
	}

	localSal.convertTo(localSal, CV_32FC1);

	return defaultGaussianBlur(localSal);
}





