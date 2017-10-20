// BLS_saliency.cpp : Define o ponto de entrada para a aplicação de console.
//

#include "stdafx.h"
#include "BLS.h"


using namespace cv;


int main()
{
	cv::Mat img = imread("C:\\Pesquisa\\busca visual\\bases de dados\\ponce\\imagens teste\\test17.jpg", IMREAD_COLOR);
	
	namedWindow("InputImage", CV_WINDOW_FREERATIO);
	imshow("InputImage", img);

	BLS bls = BLS();

	cv::Mat sal = bls.backgroundLaplacianSaliency(img);
	
	namedWindow("SaliencyMap", CV_WINDOW_FREERATIO);
	imshow("SaliencyMap", sal);
	cvWaitKey();
	normalize(sal, sal, 0, 255, NORM_MINMAX);
	imwrite("saliencymap.png", sal);
    return 0;
}

