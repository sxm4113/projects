
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

#define RANGE ((1<<16)-1)

using namespace cv;
using namespace std;
using namespace std::chrono;

//Define size of image
#define WIDTH 1280
#define HEIGHT 1024
#define PI 3.14159265

const int total_pixels = WIDTH * HEIGHT;

ushort frame[HEIGHT][WIDTH]; 
 
int Round(double number)
{
	return (number > 0.0) ? (number + 0.5) : (number - 0.5);
}

/*Clipped histogram equalization
* input : input image
* output : CHE image
*/
Mat HistEqualization(Mat img) {

int max = 0;
int min = 65536;

int* px = new int[65536];
int* px_copy = new int[65536];
float* CDF = new float[65536];
float* normalize = new float[65536];

memset(px, 0, sizeof(int) * 65536);
memset(px_copy, 0, sizeof(int) * 65536);
memset(CDF, 0, sizeof(int) * 65536);
memset(normalize, 0, sizeof(int) * 65536);

int clip = 0;
int idx = 0;

//find PDF
for (int i = 0; i < HEIGHT; i++) {
	for (int j = 0; j < WIDTH; j++) {
		idx = img.at<ushort>(i, j);
		px[idx] ++;
	}
}

memcpy(px_copy, px, sizeof(int) * 65536);

int exceed_count = 0;

//find clip value that makes the exceeded number of pixel same as not exceeded number of pixel. 
while (exceed_count < total_pixels / 2) {
	for (int i = 0; i < 65536; i++) {
		if (px_copy[i] > 0) {
			px_copy[i] -= 1;
			exceed_count += 1;
		}
		if (exceed_count == total_pixels / 2) {

			break;
		}
	}
	clip += 1;
}

//cout << "clip = " << clip << endl;

//find total number of exceeded pixels 
int exceed = 0;
for (int i = 0; i < 65536; i++) {
	if (px[i] > clip) {

		exceed += px[i] - clip;
		px[i] = clip;
	}
}
//distribute exceeded pixels over the bins
while (exceed > 0) {
	for (int j = 0; j < 65536; j++) {
		if (px[j] > 0) {

			px[j] += 1;
			exceed -= 1;

		}
		if (exceed == 0) { break; }
	}
}

// calculate CDF corresponding to px 
int accumulation = 0;
for (int i = 0; i < 65536; i++) {
	accumulation += px[i];
	CDF[i] = accumulation;
}

// using general histogram equalization formula 
for (int i = 0; i < 65536; i++) {
	normalize[i] = ((CDF[i] - CDF[0]) / (img.rows * img.cols - CDF[0])) * 65535;
	normalize[i] = Round(normalize[i]);
}

Mat output(img.rows, img.cols, CV_16U);
Mat_<ushort>::iterator it_out = output.begin<ushort>();
Mat_<ushort>::iterator it_ori = img.begin<ushort>();
Mat_<ushort>::iterator itend_ori = img.end<ushort>();

for (; it_ori != itend_ori; it_ori++) {

	ushort pixel_value = static_cast<ushort>(*it_ori);
	*it_out = normalize[pixel_value];
	it_out++;
}

delete[] px;
delete[] px_copy;
delete[] normalize;
delete[] CDF;

return output;
}

/*add extra rows and cols (padding)*/
//input : full size image, size of filter
//output : padded image. 
Mat symmetric_boundary(Mat image, int scale) {

	int extra{ (scale - 1) / 2 };
	cv::Mat image_symmetric = cv::Mat::zeros(HEIGHT + 2 * extra, WIDTH + 2 * extra, CV_16U);

	image.copyTo(image_symmetric(cv::Rect(extra, extra, WIDTH, HEIGHT)));

	for (int i = 0; i < extra; i++) {
		//first row
		image.rowRange(i, i + 1).copyTo(image_symmetric(Range(i, i + 1), Range(extra, WIDTH + extra)));
		//first column
		image.colRange(i, i + 1).copyTo(image_symmetric(Range(extra, HEIGHT + extra), Range(i, i + 1)));
		//last row
		image.rowRange(HEIGHT - (i + 1), HEIGHT - i).copyTo(image_symmetric(Range((HEIGHT + extra * 2) - (i + 1), (HEIGHT + extra * 2) - i), Range(extra, WIDTH + extra)));
		//last column
		image.colRange(WIDTH - (i + 1), WIDTH - i).copyTo(image_symmetric(Range(extra, HEIGHT + extra), Range((WIDTH + extra * 2) - (i + 1), (WIDTH + extra * 2) - i)));
		//top left corner
		image(Range(0, extra), Range(0, extra)).copyTo(image_symmetric(Range(0, extra), Range(0, extra)));
		//bottom left corner
		image(Range(HEIGHT - extra, HEIGHT), Range(0, extra)).copyTo(image_symmetric(Range(HEIGHT + extra, HEIGHT + extra * 2), Range(0, extra)));
		//top right corner
		image(Range(0, extra), Range(WIDTH - extra, WIDTH)).copyTo(image_symmetric(Range(0, extra), Range(WIDTH + extra, WIDTH + extra * 2)));
		//bottom right corner
		image(Range(HEIGHT - extra, HEIGHT), Range(WIDTH - extra, WIDTH)).copyTo(image_symmetric(Range(HEIGHT + extra, HEIGHT + extra * 2), Range(WIDTH + extra, WIDTH + extra * 2)));
	}
	return image_symmetric;
}

/*Dilation function.*/
//input : input image, size of kernel
//output : result of dilation.
Mat toDilation(Mat image, ushort scale) {

	double minVal;
	double maxVal;

	int kernel_center = (scale - 1) / 2;
	Mat output(HEIGHT, WIDTH, CV_16U);

	for (int j = kernel_center; j < HEIGHT + kernel_center; j++) {
		for (int i = kernel_center; i < WIDTH + kernel_center; i++) {

			Mat roiImage(image, Rect(i - kernel_center, j - kernel_center, scale, scale));

			minMaxLoc(roiImage, &minVal, &maxVal);

			output.at<ushort>(j - kernel_center, i - kernel_center) = maxVal;
		}
	}
	return output;
}

/*Erosion function. */
//input : input image, size of kernel
//output : result of erosion.
Mat toErosion(Mat image, ushort scale) {

	double minVal, maxVal;

	int kernel_center = (scale - 1) / 2;
	Mat output(HEIGHT, WIDTH, CV_16U);

	for (int j = kernel_center; j < HEIGHT + kernel_center; j++) {
		for (int i = kernel_center; i < WIDTH + kernel_center; i++) {

			Mat roiImage(image, Rect(i - kernel_center, j - kernel_center, scale, scale));

			//Find min value under the kernel
			minMaxLoc(roiImage, &minVal, &maxVal);

			output.at<ushort>(j - kernel_center, i - kernel_center) = minVal;

		}
	}
	return output;
}

//Subtract opening/closing image from the original image.  
Mat subtract(Mat imageA, Mat imageB) {

	Mat output(HEIGHT, WIDTH, CV_16U);
	
	for (int j = 0; j < HEIGHT; j++) {
		for (int i = 0; i < WIDTH; i++) {

			if (imageA.at<ushort>(j, i) > imageB.at<ushort>(j, i))
				output.at<ushort>(j, i) = imageA.at<ushort>(j, i) - imageB.at<ushort>(j, i);

			else output.at<ushort>(j, i) = 0;
		}
	}

	return output;
}

/*add two images*/
Mat add(Mat imageA, Mat imageB) {

	Mat output(HEIGHT, WIDTH, CV_16U);
	for (int j = 0; j < HEIGHT; j++) {
		for (int i = 0; i < WIDTH; i++) {
			//cout << "i = " << i << ", j = " << j << endl;

			if (imageA.at<ushort>(j, i) >= 65535 - imageB.at<ushort>(j, i) || imageB.at<ushort>(j, i) >= 65535 - imageA.at<ushort>(j, i))
				output.at<ushort>(j, i) = 65535;
			else output.at<ushort>(j, i) = imageA.at<ushort>(j, i) + imageB.at<ushort>(j, i);
		}
	}
	return output;
}

/*add first two images and subtract the last image*/
Mat add_final(Mat imageA, Mat imageB, Mat imageC) {

	Mat output(HEIGHT, WIDTH, CV_16U);
	int final_value = 0;
	for (int j = 0; j < HEIGHT; j++) {
		for (int i = 0; i < WIDTH; i++) {
			final_value = 0;
		 final_value = imageA.at<ushort>(j, i) + imageB.at<ushort>(j, i) - imageC.at<ushort>(j, i);
		 if (final_value > RANGE)
			 final_value = RANGE;
		 if (final_value < 0)
			 final_value = 0;
		 output.at<ushort>(j, i) = final_value;

		}
	}
 
	return output;
}
  
int main(int argc, char** argv)
{  
	int count = 0;
	ifstream fp; 
	Mat output;
 
	/*********** 24 bit **********/
	//fp.open("M:/Software/image_processing/Datasets/12-16-2020/raw data/target-narrow", ios::binary | ios::in);
    //char dummy;
    //for (int x = 0; x < HEIGHT; x++)
    //{
    //    for (int y = 0; y < WIDTH; y++)
    //    {
    //        fp.read((char*)&frame[x][y], 2);
    //
    //        fp.read((char*)&dummy, 1); // throw away the third byte
    //    }
    //}
	/*******************************/

	//**********  narcissus ring removal  **********
	//Mat lowpassed;
	//cv::blur(original_image, lowpassed, cv::Size(100, 100)); 
	//double minVal_lowp[1];
	//double maxVal_lowp;
	//minMaxLoc(lowpassed, minVal_lowp, &maxVal_lowp);
	//
	//original_image = original_image + maxVal_lowp - lowpassed;
	/********************************/

	/********** 16-bit **********/
	fp.open("C:/Users/SYunMoon/Desktop/raw_NU_corrected/second", ios::binary | ios::in);
	fp.read((char*)&frame, sizeof(ushort) * total_pixels);
	/****************************/

	Mat original_image = Mat(HEIGHT, WIDTH, CV_16U, frame);
	cout << "original_image loaded" << endl;
 
	//********** tiff file ***********/ 
	//original_image = imread("H:/SYunMoon/FLIR_temp/FLIR_04524.tiff", IMREAD_UNCHANGED);
	//original_image = imread("M:/Software/image_processing/FLIR_ADAS_1_3/FLIR_ADAS_1_3/val/thermal_16_bit/FLIR_09927.TIFF", IMREAD_UNCHANGED); 
	/*********************************/

	//openinging : erosion first and then dilation
	//tophat : subtract opening from input image
	Mat tophat3;
	tophat3 = subtract(original_image, toDilation(symmetric_boundary(toErosion(symmetric_boundary(original_image, 3), 3), 3), 3));
	cout << "tophat3 " << endl;
	
	Mat tophat5;
	tophat5 = subtract(original_image, toDilation(symmetric_boundary(toErosion(symmetric_boundary(original_image, 5), 5), 5), 5));
	cout << "tophat5 " << endl;
	
	Mat tophat7;
	tophat7 = subtract(original_image, toDilation(symmetric_boundary(toErosion(symmetric_boundary(original_image, 7), 7), 7), 7));
	cout << "tophat7 " << endl;

	Mat tophat9;
	tophat9 = subtract(original_image, toDilation(symmetric_boundary(toErosion(symmetric_boundary(original_image, 9), 9), 9), 9));
	cout << "tophat9 " << endl;
 
	//closing : dilation first and then erosion
	//blackhat : subtract closing from input image 
	Mat blackhat3;
	blackhat3 = subtract(toErosion(symmetric_boundary(toDilation(symmetric_boundary(original_image, 3), 3), 3), 3), original_image);
	cout << "blackhat3" << endl;
	
	Mat blackhat5;
	blackhat5 = subtract(toErosion(symmetric_boundary(toDilation(symmetric_boundary(original_image, 5), 5), 5), 5), original_image);
	cout << "blackhat5 " << endl;
	
	Mat blackhat7;
	blackhat7 = subtract(toErosion(symmetric_boundary(toDilation(symmetric_boundary(original_image, 7), 7), 7), 7), original_image);
	cout << "blackhat7 " << endl;
	
	Mat blackhat9;
	blackhat9 = subtract( toErosion(symmetric_boundary(toDilation(symmetric_boundary(original_image, 9), 9), 9), 9), original_image);
	cout << "blackhat9 " << endl; 

	//add all blackhats
	Mat sum_blackhat =  add(add(add(0.5*blackhat3, 0.5 *blackhat5), 0.5 *blackhat7), 0.5 * blackhat9);
	cout << "sum_blackhat" << endl;
	
	//add all tophats
	Mat sum_tophat =  add(add(add(0.5 * tophat3, 0.5 * tophat5), 0.5 *tophat7), 0.5 * tophat9);
	cout << "sum_tophat" << endl;
	
	//add tophat and subtract blackhat
	Mat final = add_final(original_image, sum_tophat, sum_blackhat);
	cout << "final" << endl;
	    
	original_image = HistEqualization(original_image); 
	final = HistEqualization(final);
  
	imshow("final", final);
	imshow("Input", original_image);

	/// save final image as .bmp file format. 
	final = (final / 65535) * 255;
	original_image = (original_image / 65535) * 255; 
	imwrite("final.bmp", final);
	imwrite("input.bmp", original_image);
	
	waitKey(0);  
	
	return 0;
}
