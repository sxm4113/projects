// image_pyramid_for_16bit_image.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include <vector> // for 2D vector

using namespace cv;
using namespace std;
using namespace std::chrono;

#define WIDTH 1280
#define HEIGHT 1024

const int total_pixels = WIDTH * HEIGHT;

// #define RANGE ((1<<16)-1) // left shift 1, so 1*2^(16)-1 = 65536-1 = 65535
constexpr auto RANGE = ((1 << 16) - 1);
ushort frame[HEIGHT][WIDTH];
ushort CEG256[HEIGHT][WIDTH];
 
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
    
    //find total exceeded pixels 
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

    cv::Mat output(img.rows, img.cols, CV_16U);
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

void adjustment(Mat original_image, Mat rolp_result)
{
    int final_value{ 0 };
     
    // this function takes the global variable CEG256 and alters it, so after this function has been called, CEG256 will have the correct calculated values in it
    ushort CE_EXP;
    int count = 0;
    for (int i = 0; i < WIDTH; i++)
    {
        for (int j = 0; j < HEIGHT; j++)
        { 
            if (i > 1 && i < WIDTH - 1 && j >1 && j < HEIGHT) {

                if (rolp_result.at<float>(j, i) < 1)
                {
                    // now I have to loop through the rows [j-1, j+1) and columns [i-1,i+1) of the original image and find the minimum
                    // maybe I should come back and separate these minimum and maximums into their own functions
                    ushort minimum = original_image.at<ushort>(j - 1, i - 1);
                    for (int original_j = j - 1; original_j < j + 1; original_j++)
                    {
                        for (int original_i = i - 1; original_i < i + 1; original_i++)
                        {
                            if (original_image.at<ushort>(original_j, original_i) < minimum)
                            {
                                minimum = original_image.at<ushort>(original_j, original_i);
                            }
                        }
                    }
                    // now I have found the minimum 
                    CE_EXP = minimum;
                }
                else if (rolp_result.at<float>(j, i) > 1)
                {
                    // now I have to loop through the rows [j-1, j+1) and columns [i-1,i+1) of the original image and find the maxmimum
                    ushort maximum = original_image.at<ushort>(j - 1, i - 1);

                    for (int original_j = j - 1; original_j < j + 1; original_j++)
                    {
                        for (int original_i = i - 1; original_i < i + 1; original_i++)
                        {
                            if (original_image.at<ushort>(original_j, original_i) > maximum)
                            {
                                maximum = original_image.at<ushort>(original_j, original_i);
                            }
                        }
                    }
                    // now I have found the maximum
                    CE_EXP = maximum;
                }
                else CE_EXP = original_image.at<ushort>(j, i);
            }
            else CE_EXP = original_image.at<ushort>(j, i);
               
         // now i have CE_EXP
         final_value = trunc(rolp_result.at<float>(j, i) * 2 * CE_EXP);
         
         if (rolp_result.at<float>(j, i) >= (65335/2)) count += 1;

         if (final_value > RANGE) 
             final_value = RANGE;  
         
         CEG256[j][i] = final_value;
            
        }
    }
 
}

/*find ROLP (ratio of lowpass)*/
Mat rolp(Mat original_image, Mat expanded)
{
    Mat output = Mat::zeros(original_image.rows, original_image.cols, CV_32FC1);
    Mat float_expanded;
    Mat float_original_image;

    expanded.convertTo(float_expanded, CV_32FC1);
    original_image.convertTo(float_original_image, CV_32FC1);

    // First, I must loop through and multiply each element in the original_image by 4.0
    // but stackoverflow says that opencv supports scalar and matrix multiplication
     //expanded_times4 = 4.0 * expanded;
    // Now, divide each element of the original image by its respective element in output and assign the result to output
    //cv::divide(original_image, expanded_times4, output); // for some reason this just results in a black image
    // so maybe I'll write the element-wise division from scratch instead
    // iterate through each pixel in original_image, divide it by its respective pixel in expanded_times4, and store the result in the respective pixel of output
    for (int j = 0; j < original_image.rows; j++) {
        for (int i = 0; i < original_image.cols; i++) {

            output.at<float>(j, i) = float_original_image.at<float>(j, i) / (4.0 * float_expanded.at<float>(j, i));

            // this had division by 0, but it's probably because we haven't added the boundary_sym function yet, so there are zeroes at the boundaries
            // update: we've added the symmetric boundary function but it's still claiming error: unhandled excpetion 0xC0000094: Integer deivision, which is division by zero
            // so there must be some pixel in expanded_times4 that is zero
        }
    }

    return output;
}

/*expand the size of image*/
Mat upsample(Mat original_image)
{
    Mat upsampled_image = Mat::zeros(HEIGHT, WIDTH, CV_16U);
    for (int i = 0; i < WIDTH; i += 2)
    {
        for (int j = 0; j < HEIGHT; j += 2)
        {
            upsampled_image.at<ushort>(j, i) = original_image.at<ushort>(j / 2, i / 2);
        }
    }
    return upsampled_image;
}

/*shrink the size of image*/
 Mat downsample(Mat original_image)
{
    cv::Mat downsampled_image = cv::Mat::zeros(HEIGHT / 2, WIDTH / 2, CV_16U);

    for (int i = 0; i < WIDTH; i += 2)
    {
        // I'll create t as a vector here
        std::vector<ushort> t(HEIGHT, 0);
        // now copy over the i'th column of image into t
        int t_idx = 0;
        for (std::vector<ushort>::iterator it = t.begin(); it != t.end(); it++)
        {
            // now, index original_image at row t_idx, col i, and copy that into the vector t at position it
            *it = original_image.at<ushort>(t_idx, i);
            t_idx++;
        }
        // now t has the i'th column of original_image
        for (int j = 0; j < HEIGHT; j += 2)
        {
            // I need to access the j'th element of vector t
            downsampled_image.at<ushort>(j / 2, i / 2) = t[j];
        }
    }
    return downsampled_image;
}

 /*calculate convolution*/
//input : full size image, filter
//output : result of convolution (filtered image)

double convolution(Mat sub_image, std::vector<std::vector <double>> kernel)
{
    double convolution_value{ 0 };
    for (int j = 0; j < sub_image.rows; j++) {
        for (int i = 0; i < sub_image.cols; i++) {

            convolution_value = convolution_value + sub_image.at<ushort>(j, i) * kernel.at(j).at(i);
        }
    }
    if (convolution_value < 1)
        return 1;

    else return convolution_value;
}

/*calculate convolution*/
//input : 5 by 5 image, filter
//output : pixel value of the center of 5 by 5 image. 

Mat convolution_image(Mat image, std::vector<std::vector <double>> kernel)
{
    Mat output = Mat::zeros(HEIGHT, WIDTH, CV_16UC1);

    auto duration_roiImage_sum = 0;
    for (int i = kernel.size() / 2; i < image.cols - kernel.size() / 2; i++) {
        for (int j = kernel.size() / 2; j < image.rows - kernel.size() / 2; j++) {

            Mat roiImage(image, Rect(i - kernel.size() / 2, j - kernel.size() / 2, kernel.size(), kernel.size()));
            output.at<ushort>(j - kernel.size() / 2, i - kernel.size() / 2) = convolution(roiImage, kernel);
        }
    }
    return output;
}

/*add extra rows and cols (padding)*/
//input : full size image, size of filter
//output : padded image.  
Mat symmetric_boundary(Mat image, int scale) {

    int height_symm = image.rows;
    int width_symm = image.cols;

    int extra{ (scale - 1) / 2 };
    cv::Mat image_symmetric = cv::Mat::zeros(height_symm + 2 * extra, width_symm + 2 * extra, CV_16U);

    image.copyTo(image_symmetric(cv::Rect(extra, extra, width_symm, height_symm)));

    for (int i = 0; i < extra; i++) {
        //first row
        image.rowRange(i, i + 1).copyTo(image_symmetric(Range(i, i + 1), Range(extra, width_symm + extra)));
        //first column
        image.colRange(i, i + 1).copyTo(image_symmetric(Range(extra, height_symm + extra), Range(i, i + 1)));
        //last row
        image.rowRange(height_symm - (i + 1), height_symm - i).copyTo(image_symmetric(Range((height_symm + extra * 2) - (i + 1), (height_symm + extra * 2) - i), Range(extra, width_symm + extra)));
        //last column
        image.colRange(width_symm - (i + 1), width_symm - i).copyTo(image_symmetric(Range(extra, height_symm + extra), Range((width_symm + extra * 2) - (i + 1), (width_symm + extra * 2) - i)));
        //top left corner
        image(Range(0, extra), Range(0, extra)).copyTo(image_symmetric(Range(0, extra), Range(0, extra)));
        //bottom left corner
        image(Range(height_symm - extra, height_symm), Range(0, extra)).copyTo(image_symmetric(Range(height_symm + extra, height_symm + extra * 2), Range(0, extra)));
        //top right corner
        image(Range(0, extra), Range(width_symm - extra, width_symm)).copyTo(image_symmetric(Range(0, extra), Range(width_symm + extra, width_symm + extra * 2)));
        //bottom right corner
        image(Range(height_symm - extra, height_symm), Range(width_symm - extra, width_symm)).copyTo(image_symmetric(Range(height_symm + extra, height_symm + extra * 2), Range(width_symm + extra, width_symm + extra * 2)));
    }
    return image_symmetric;
}
int main(int argc, char** argv)
{
    ifstream fp;
    Mat output;
 
    /********** 24 bit **********/
    //fp.open("M:/Software/image_processing/Datasets/12-16-2020/raw data/towers-mid", ios::binary | ios::in);
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
    
    /********** 16-bit **********/
    fp.open("C:/Users/SYunMoon/Desktop/raw_NU_corrected/third", ios::binary | ios::in);
  
    fp.read((char*)&frame, sizeof(ushort) * total_pixels);
    Mat image = Mat(HEIGHT, WIDTH, CV_16U, frame);
    /*****************************/

    /**********Narcisus ring removal**********/
    //Mat lowpassed;
    //cv::blur(image, lowpassed, cv::Size(100, 100));
    //double minVal_lowp[1];
    //double maxVal_lowp;
    //minMaxLoc(lowpassed, minVal_lowp, &maxVal_lowp);
    //
    //image = image + maxVal_lowp - lowpassed;
    /******************************************/

    /*********** .tif image **********/
    //image = imread("M:/Software/image_processing/FLIR_ADAS_1_3/FLIR_ADAS_1_3/train/thermal_16_bit/FLIR_06962.tiff", IMREAD_UNCHANGED);
    /**********************************/
    
    std::vector<std::vector <double>> kernel = { {1.0 / 256, 4.0 / 256, 6.0 / 256, 4.0 / 256, 1.0 / 256} ,
                                        {4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256},
                                        {6.0 / 256, 24.0 / 256, 36.0 / 256, 24.0 / 256, 6.0 / 256},
                                        {4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256},
                                        {1.0 / 256, 4.0 / 256, 6.0 / 256, 4.0 / 256, 1.0 / 256 } };
      
    std::vector<std::vector <double>> kernel3 = { {1.0 / 16, 2.0 / 16, 1.0 / 16} ,
                                                  {2.0 / 16, 4.0 / 16, 2.0 / 16},
                                                  {1.0 / 16, 2.0 / 16, 1.0 / 16 } };
 

    std::cout << "Original Image Loaded" << std::endl;
        
    output = downsample(image);
    std::cout << "Downsample Done" << std::endl;

    output = symmetric_boundary(output,5);
    std::cout << "Symmetric boundary Done" << std::endl;
    
    output = convolution_image(output, kernel);
    std::cout << "Convolution Done" << std::endl;

    output = upsample(output);
    std::cout << "Upsample Done" << std::endl;
    
    output = symmetric_boundary(output, 5);
    std::cout << "Upsampled, Symmetric Done" << std::endl;
    
    output = convolution_image(output, kernel);
    std::cout << "Upsampled, Convolution done" << std::endl;

    Mat temp = Mat(output, Rect(100, 100, 8, 8));
    cout << "temp "<< temp << endl;
    Mat temp2 = Mat(image, Rect(100, 100, 8, 8));
    cout << "input" << temp2 << endl;
   // output = HistEqualization(output);
   // output = output / 65535 * 255;
   // output.convertTo(output, CV_8U);
   // imshow("Final Image", output);
   // waitKey();

    output = rolp(image, output);
    std::cout << "ROLP Done" << std::endl;
    
    adjustment(image, output); 
    output = Mat(HEIGHT, WIDTH, CV_16U, CEG256);
     
    // global contrast enhancement after image pyramid
    output = HistEqualization(output);
    image = HistEqualization(image);
  
    /// convert final image to 8-bit for .bmp file format.  
    output = output / 65535 * 255;
    image = image / 65535 * 255;

    output.convertTo(output, CV_8U);
    image.convertTo(image, CV_8U);
    
    imwrite("final.bmp", output);
    imwrite("input.bmp", image);
  
     
    imshow("Input Image", image);
    waitKey(0);

    fp.close();
    return 0;
}