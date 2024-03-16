#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <math.h> 

using namespace std;
using namespace cv;

#define WIDTH 1280
#define HEIGHT 1024
ushort frame[HEIGHT][WIDTH];

int Round(double number)
{
    return (number > 0.0) ? (number + 0.5) : (number - 0.5);
}

Mat HistEqualization(Mat img) {

    int px[65536] = { 0 };

    int max = { 0 };
    int min = { 65536 };

    // pdf
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int idx;
            idx = img.at<ushort>(i, j);

            px[idx]++;
        }
    }
    int exceed = 0;
    int temp_exceed = 0;
    int clip = 1;

    //find clip limit that makes exceeded pixels and receded pixels equal. 
    while (temp_exceed < 1024 * 1280 / 2) {

        for (int i = 0; i < 65536; i++) {
            if (px[i] - clip >= 0) {
                temp_exceed += 1;
                if (temp_exceed > 1024 * 1280 / 2)
                    break;
            }
        }
        clip += 1;
    }
    cout << "clip =" << clip << endl;

    for (int i = 0; i < 65536; i++) {
        if (px[i] > clip) {

            exceed += px[i] - clip;
            px[i] = clip;
        }
    }

    while (exceed > 0) {
        for (int j = 0; j < 65536; j++) {
            if (px[j] > 0) {

                px[j] += 1;
                exceed -= 1;
            }
            if (exceed == 0) { break; }
        }
    }

    // calculate CDF
    float CDF[65536] = { 0 };
    int accumulation = 0;
    for (int i = 0; i < 65536; i++) {
        accumulation += px[i];
        CDF[i] = accumulation;
    }

    // using general histogram equalization formula
    float normalize[65536] = { 0 };
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
    return output;
}

//Gaussian function with u and sigma for frequency filtering.
void createGaussian(Size size, Mat& output, int uX, int uY, float sigmaX, float sigmaY, float amplitude = 1.0f) {
    Mat temp = Mat(size, CV_32F);
    for (int r = 0; r < size.height; r++) {
        for (int c = 0; c < size.width; c++) {
            float x = ((c - uX) * ((float)c - uX)) / (2.0f * sigmaX * sigmaX);
            float y = ((r - uY) * ((float)r - uY)) / (2.0f * sigmaY * sigmaY);
            float value = amplitude * exp(-(x + y));
            temp.at<float>(r, c) = value;
        }
    }
    normalize(temp, temp, 0.0f, 1.0f, NORM_MINMAX);
    output = temp;
}

Mat fft_cal(Mat original) {

    //conver to floating point
    Mat original_float;
    original.convertTo(original_float, CV_32F);
    
    Mat original_complex[2] = { Mat::zeros(original.size() , CV_32FC1), Mat::zeros(original.size() , CV_32FC1) };
    Mat sub_complex[2] = { Mat::zeros(original.size() , CV_32FC1), Mat::zeros(original.size() , CV_32FC1) };
    original_float.copyTo(original_complex[0]);
    
    Mat complexIm;
    merge(original_complex, 2, complexIm);//Merge channel
    dft(complexIm, complexIm);//Fourier transform
    
    ////////// recentering ////////// 
    split(complexIm, original_complex);//separation channel
    
    // real part//
    int cx = original_complex[0].cols / 2; int cy = original_complex[0].rows / 2;
    Mat part1_r(original_complex[0], Rect(0, 0, cx, cy)); // 
    Mat part2_r(original_complex[0], Rect(cx, 0, cx, cy));
    Mat part3_r(original_complex[0], Rect(0, cy, cx, cy));
    Mat part4_r(original_complex[0], Rect(cx, cy, cx, cy));
    
    Mat temp;
    part1_r.copyTo(temp);
    part4_r.copyTo(part1_r);
    temp.copyTo(part4_r);
    
    part2_r.copyTo(temp);
    part3_r.copyTo(part2_r);
    temp.copyTo(part3_r);
    
    // imaginary part
    Mat part1_i(original_complex[1], Rect(0, 0, cx, cy));
    Mat part2_i(original_complex[1], Rect(cx, 0, cx, cy));
    Mat part3_i(original_complex[1], Rect(0, cy, cx, cy));
    Mat part4_i(original_complex[1], Rect(cx, cy, cx, cy));
    
    part1_i.copyTo(temp);
    part4_i.copyTo(part1_i);
    temp.copyTo(part4_i);
    
    part2_i.copyTo(temp);
    part3_i.copyTo(part2_i);
    temp.copyTo(part3_i);
    
    Mat blur_r, blur_i, BLUR;
    Mat filter;
    createGaussian(Size(1280, 1024), filter, 1280 / 2, 1024 / 2, 10, 10, 1);
    
    multiply(original_complex[0], filter, blur_r); //filter (real part)
    multiply(original_complex[1], filter, blur_i); // filter (imaginary part)
    
    // display FFT
    magnitude(original_complex[0], original_complex[1], original_complex[0]);
    original_complex[0] += Scalar::all(1);
    log(original_complex[0], original_complex[0]);
    
    Mat temp_merge[] = { blur_r, blur_i };
    merge(temp_merge, 2, BLUR);
    
    // inverse FFT of filtered image 
    dft(BLUR, BLUR, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    
    //inverse FFT of original image
    dft(complexIm, complexIm, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    Mat inverse_complex[2] = { Mat::zeros(original.size() , CV_32FC1), Mat::zeros(original.size() , CV_32FC1) };
    
    split(BLUR, inverse_complex);
    magnitude(inverse_complex[0], inverse_complex[1], inverse_complex[0]);
    
    split(complexIm, sub_complex);
    magnitude(sub_complex[0], sub_complex[1], sub_complex[0]);
    
    //subtract filtered image from original image
    inverse_complex[0] = sub_complex[0] - inverse_complex[0];
    normalize(inverse_complex[0], inverse_complex[0], 65335, 0, NORM_MINMAX);
    
    inverse_complex[0].convertTo(inverse_complex[0], CV_16U);
    
    //clipped histogramequalization
    Mat result =  inverse_complex[0];
 
    return result;
}

int main() {
    ifstream fp;
    
    fp.open("M:/Software/image_processing/Datasets/12-16-2020/raw data/target-mid", ios::binary | ios::in);

    ////////// 24 bit //////////
    char dummy;
    for (int x = 0; x < HEIGHT; x++)
    {
        for (int y = 0; y < WIDTH; y++)
        {
            fp.read((char*)&frame[x][y], 2);

            fp.read((char*)&dummy, 1); // throw away the third byte
        }
    }
    ////////// 16-bit //////////
    //fp.read((char*)&frame, sizeof(ushort) * HEIGHT*WIDTH);
    
    Mat original = Mat(HEIGHT, WIDTH, CV_16U, frame); 
    Mat fft_image = fft_cal(original);

    // clipped histogram equalization
    Mat result = HistEqualization(fft_image);

    // save result as .bmp
    double minVal_guassian[1];
    double maxVal_guassian;

    minMaxLoc(result, minVal_guassian, &maxVal_guassian);
    result = (result / maxVal_guassian) * 255;
    imwrite("output.bmp", result);
 
    return 0;

}
