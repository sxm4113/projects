#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

 
using namespace cv;
using namespace std;
using namespace std::chrono;

//Define size of image

#define WIDTH 1280
#define HEIGHT 1024

const int total_pixels = WIDTH * HEIGHT;

#define COL (blockIdx.x * blockDim.x + threadIdx.x)
#define ROW  (blockIdx.y * blockDim.y + threadIdx.y)
#define INDEX (COL + ROW * (WIDTH))

unsigned short frame[HEIGHT][WIDTH];
constexpr auto RANGE = ((1 << 16) - 1);

__constant__ float dev_filter[25];

texture<unsigned short, cudaTextureType2D> tex16u;
texture<unsigned short, cudaTextureType2D> tex16u_down;
texture<unsigned short, cudaTextureType2D> tex16u_up;


int Round(double number)
{
    return (number > 0.0) ? (number + 0.5) : (number - 0.5);
}

Mat HistEqualization(Mat img) {

    int px[65536] = { 0 };
    int* px_copy = new int[sizeof(int) * 65535];
    int max = { 0 };
    int min = { 65536 };

    int clip = { 0 };

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int idx;
            idx = img.at<ushort>(i, j);

            px[idx]++;
        }
    }
    memcpy(px_copy, px, 65536 * sizeof(int));

    int exceed_count = { 0 };

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

    cout << "clip = " << clip << endl;
    int exceed = 0;
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

    // calculate CDF corresponding to px
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
    delete[] px_copy;
    return output;
}

//Box Filter Kernel For Gray scale image with 8bit depth
__global__ void dev_downsampling(unsigned short* output, const size_t pitch)
{
    int index_down = ROW * pitch / 2 + COL;

    if ((COL < WIDTH / 2) && (ROW < HEIGHT / 2))
        output[index_down] = tex2D(tex16u, 2 * COL, 2 * ROW);

}

__global__ void dev_upsampling(unsigned short* input, const size_t input_pitch, unsigned short* output, const size_t output_pitch)
 
{
    int output_index = ROW * output_pitch / 2 + COL;
     
    int temp = 0;

    if ((COL < WIDTH) && (ROW < HEIGHT)) {
        if (ROW % 2 == 0 && COL % 2 == 0) {
            output[ROW * output_pitch / 2 + COL] = input[ROW / 2 * input_pitch/2 + (COL / 2)];
        }

        else output[ROW * output_pitch / 2 + COL] = 0;
    }
}

__global__ void dev_convolution(unsigned short* output, const size_t pitch, const int fWidth, const int fHeight)
{
    const int filter_offset_x = fWidth / 2;
    const int filter_offset_y = fHeight / 2;

    //Make sure the current thread is inside the image bounds
    if (COL < WIDTH && ROW < HEIGHT)
    {
        float output_value = 0;

        //Sum the window pixels
        for (int i = -filter_offset_x; i <= filter_offset_x; i++)
        {
            for (int j = -filter_offset_y; j <= filter_offset_y; j++)
            {
                //No need to worry about Out-Of-Range access. tex2D automatically handles it.
                int filter_index = (i + 2) * 5 + (j + 2);
                output_value += tex2D(tex16u_down, COL + i, ROW + j) * dev_filter[filter_index];
               
            }
        }

        //Write the averaged value to the output.
        //Transform 2D index to 1D index, because image is actually in linear memory
        int index = ROW * pitch / 2 + COL;

        output[index] = static_cast<unsigned short>(output_value);

    }
}

__global__ void dev_convolution_up(unsigned short* output, const size_t pitch, const int fWidth, const int fHeight)
{
    const int filter_offset_x = fWidth / 2;
    const int filter_offset_y = fHeight / 2;
     

    //Make sure the current thread is inside the image bounds
    if (COL < WIDTH && ROW < HEIGHT)
    {
        float output_value = 0;
        //Sum the window pixels
        for (int i = -filter_offset_x; i <= filter_offset_x; i++)
        {
            for (int j = -filter_offset_y; j <= filter_offset_y; j++)
            {
                //No need to worry about Out-Of-Range access. tex2D automatically handles it.
                int filter_index = (i + 2) * 5 + (j + 2);
                output_value += tex2D(tex16u_up, COL + i, ROW + j) * dev_filter[filter_index];

            }
        }

        //Write the averaged value to the output.
        //Transform 2D index to 1D index, because image is actually in linear memory
        int index = ROW * pitch / 2 + COL;

        output[index] = 4* static_cast<unsigned short>(output_value);

    }
}
__global__ void dev_ROLP(unsigned short* expanded, const size_t pitch, float* ROLP, unsigned short* final_image) {
    
    //if (COL >= 1 && ROW >= 1 && COL < WIDTH - 1 && ROW < HEIGHT - 1) {
        //if (COL < 3 && ROW < 3) {
        float rolp_value = 0;

        ROLP[INDEX] = tex2D(tex16u, COL, ROW) / static_cast<float>(expanded[ROW * pitch / 2 + COL]);

        int original_value = 0;
        int final_value = 0;

        if (ROLP[INDEX] > 1) {
            int max = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {

                    original_value = tex2D(tex16u, COL + i, ROW + j);
                    if (max <= original_value)
                        max = original_value;

                }
            }
            if ((max * ROLP[INDEX] < 65335 / 2))
                final_value = 2 * max * ROLP[INDEX];
            else final_value = 65335;
        }

        else if (ROLP[INDEX] < 1) {
            int min = 65535;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {

                    original_value = tex2D(tex16u, COL + i, ROW + j);
                    if (min >= original_value)
                        min = original_value;
                }
            }
            if ((min * ROLP[INDEX]) < 65335 / 2)
                final_value = 2 * min * ROLP[INDEX];
            else final_value = 65335;
        }

        else {
            if (tex2D(tex16u, COL, ROW) < 65335 / 2)
                final_value = 2 * tex2D(tex16u, COL, ROW);
            else final_value = 65335;
        }

        final_image[INDEX] = final_value;
    //}
    //else
    //    if (tex2D(tex16u, COL, ROW) < 65335 / 2)
    //        final_image[INDEX] = 2 * tex2D(tex16u, COL, ROW);
    //    else final_image[INDEX] = 65535;
    
    
}

__global__ void minmax(unsigned short* image, unsigned short* d_out, int numRows, int numCols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int c_t = threadIdx.x;
    int r_t = threadIdx.y;
    int pos_1D = row * numCols + col;
    int pos_1D_t = r_t * blockDim.x + c_t;

    extern __shared__ unsigned short sh_mem[];

    sh_mem[pos_1D_t] = (pos_1D >= numCols * numRows) ? -999999.0f : image[pos_1D];
    __syncthreads();

    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (pos_1D_t < s)
            if (sh_mem[pos_1D_t] < sh_mem[pos_1D_t + s])
                sh_mem[pos_1D_t] = sh_mem[pos_1D_t + s];
        __syncthreads();
    }

    if (r_t == 0 && c_t == 0)
        d_out[blockIdx.y * gridDim.x + blockIdx.x] = sh_mem[0];
}


void box_filter_8u_c1(Mat CPUinput, unsigned short* CPUoutput, const int widthStep, const int filterWidth, const int filterHeight)
{
    /*
    * 2D memory is allocated as strided linear memory on GPU.
    * The terminologies "Pitch", "WidthStep", and "Stride" are exactly the same thing.
    * It is the size of a row in bytes.
    * It is not necessary that width = widthStep.
    * Total bytes occupied by the image = widthStep x height.
    */

    /*downsampling - convolution - updampling - convolution - ROLP - adjustment*/

    //Declare GPU pointer
    float* ROLP;
    unsigned short* GPU_input, *GPU_output, *upsampling, *expanded, *final_image;
    unsigned short* dev_max = NULL;
    unsigned short* d_out = NULL;
    unsigned short max_cpu = 0;
    unsigned short* dev_image = NULL;
    int THREADS_PER_BLOCK = 16;
    //float* dev_filter;

    unsigned short* ROLP_host = new unsigned short[HEIGHT * WIDTH];

    unsigned short* h_expanded = new unsigned short[HEIGHT * WIDTH];
    unsigned short* temp = new unsigned short[HEIGHT * WIDTH];

    float* filter = new float[5 * 5]
    { 1.0 / 256, 4.0 / 256,  6.0 / 256,  4.0 / 256,  1.0 / 256,
      4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,
      6.0 / 256, 24.0 / 256, 36.0 / 256, 24.0 / 256, 6.0 / 256,
      4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,
      1.0 / 256, 4.0 / 256,  6.0 / 256,  4.0 / 256,  1.0 / 256 };

    /*
    * Specify the grid size for the GPU.
    */
    dim3 block_size_down(16, 16);
    dim3 grid_size_down(40, 32);

    dim3 block_size(16, 16);
    dim3 grid_size(80, 64);
     
    //Allocate 2D memory on GPU. Also known as Pitch Linear Memory
    size_t gpu_image_pitch;
    size_t gpu_image_pitch2;
     
    cudaMalloc((void**)&ROLP, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&final_image, WIDTH * HEIGHT * sizeof(unsigned short));
    
    cudaMallocPitch((void**)&GPU_input, &gpu_image_pitch, WIDTH * sizeof(unsigned short), HEIGHT);
    cudaMallocPitch((void**)&expanded, &gpu_image_pitch, WIDTH * sizeof(unsigned short), HEIGHT);
    cudaMallocPitch((void**)&upsampling, &gpu_image_pitch, WIDTH * sizeof(unsigned short), HEIGHT);
    cudaMallocPitch((void**)&GPU_output, &gpu_image_pitch2, WIDTH / 2 * sizeof(unsigned short), HEIGHT / 2);
    //cudaMallocPitch((void**)&final_image, &gpu_image_pitch, WIDTH * sizeof(unsigned short), HEIGHT);
    cudaMalloc(&dev_max, sizeof(unsigned short) * grid_size.y * grid_size.x);
    cudaMalloc(&d_out, sizeof(unsigned short));
    
    //Use constant memory for filter
    cudaMemcpyToSymbol(dev_filter, filter, 5 * 5 * sizeof(float));
    
    //Copy data from host to device.
    cudaMemcpy2D(GPU_input, gpu_image_pitch, CPUinput.data, WIDTH * sizeof(unsigned short), WIDTH * sizeof(unsigned short), HEIGHT, cudaMemcpyHostToDevice);

    // Mat GPU_input_m = Mat(HEIGHT, WIDTH, CV_16U, CPUinput);
   // Mat GPU_input_mm(CPUinput, Rect(0, 0, 10, 10));
   //
   // cout << "CPU_input: " << endl;
   // cout << GPU_input_mm << endl;

    //Bind the image to the texture. Now the kernel will read the input image through the texture cache.
    //Use tex2D function to read the image
    cudaBindTexture2D(NULL, tex16u, GPU_input, WIDTH, HEIGHT, gpu_image_pitch);

    /*
    * Set the behavior of tex2D for out-of-range image reads.
    * cudaAddressModeBorder = Read Zero
    * cudaAddressModeClamp  = Read the nearest border pixel
    * We can skip this step. The default mode is Clamp.
    */
    tex16u.addressMode[0] = tex16u.addressMode[1] = cudaAddressModeWrap;
    //texture_clamp.addressMode[0] = cudaAddressModeClamp; 

    /*
    * Specify a block size. 256 threads per block are sufficient.
    * It can be increased, but keep in mind the limitations of the GPU.
    * Older GPUs allow maximum 512 threads per block.
    * Current GPUs allow maximum 1024 threads per block
    */

    //Launch the kernel
    dev_downsampling << <grid_size_down, block_size_down >> > (GPU_output, gpu_image_pitch2);

    cudaBindTexture2D(NULL, tex16u_down, GPU_output, WIDTH / 2, HEIGHT / 2, gpu_image_pitch2);
    //cudaAddressModeWrap ,cudaAddressModeClamp, cudaAddressModeMirror  ,cudaAddressModeBorder 
    tex16u_down.addressMode[0] = tex16u_down.addressMode[1] = cudaAddressModeWrap;

    dev_convolution << <grid_size_down, block_size_down >> > (GPU_output,  gpu_image_pitch2, filterWidth, filterHeight);

    dev_upsampling << <grid_size, block_size >> > (GPU_output, gpu_image_pitch2, upsampling, gpu_image_pitch);

    cudaBindTexture2D(NULL, tex16u_up, upsampling, WIDTH, HEIGHT, gpu_image_pitch);
    tex16u_up.addressMode[0] = tex16u_up.addressMode[1] = cudaAddressModeWrap;

    dev_convolution_up << <grid_size, block_size >> > (expanded,  gpu_image_pitch, filterWidth, filterHeight);
    
    dev_ROLP << <grid_size, block_size >> > (expanded,  gpu_image_pitch , ROLP, final_image);

    //Copy the results back to CPU
    
    //cudaMemcpy2D(CPUoutput, widthStep * sizeof(unsigned short), expanded, gpu_image_pitch, WIDTH * sizeof(unsigned short), HEIGHT, cudaMemcpyDeviceToHost);
    
    cudaMemcpy(CPUoutput, final_image, WIDTH * HEIGHT * sizeof(unsigned short), cudaMemcpyDeviceToHost);
  
    // cudaMemcpy2D(h_expanded, widthStep * sizeof(unsigned short), expanded, gpu_image_pitch, WIDTH * sizeof(unsigned short), HEIGHT, cudaMemcpyDeviceToHost);
   // 
   // Mat h_exp = Mat(HEIGHT, WIDTH, CV_16U, h_expanded);
   // Mat h_expanded_mat(h_exp, Rect(0, 0, 10, 10));
   //
   // cout << "h_expanded: " << endl;
   // cout << h_expanded_mat << endl;
   //
   // Mat final_func = Mat(HEIGHT, WIDTH, CV_16U, CPUoutput);
   // Mat final_mat(final_func, Rect(0, 0, 10, 10));
   // 
   // cout << "final_mat: " << endl;
   // cout << final_mat << endl;

    //cudaMemcpy(&max_cpu, d_out, sizeof(unsigned short), cudaMemcpyDeviceToHost);
    //cout << "max = " << max_cpu << endl;
    ////Release the texture
    cudaUnbindTexture(tex16u);

    //Free GPU memory
    cudaFree(GPU_input);
    cudaFree(GPU_output); 
    cudaFree(dev_filter);
}

int main()
{ 
    float elapsedTime;
    auto start_all = steady_clock::now();
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    //unsigned char* dev_original_image;
    //unsigned char* dev_filtered_image;
    int* dev_max;
    unsigned short* h_result = new unsigned short[HEIGHT * WIDTH];

    ifstream fp;
    // read the file
//M:/Software/image_processing/Datasets/old data/outing/raw_NU_corrected/sixth
    //fp.open("M:/Software/image_processing/Datasets/old data/outing/raw_NU_corrected/sixth", ios::binary | ios::in);
    fp.open("C:/Users/SYunMoon/Desktop/raw_NU_corrected/sixth", ios::binary | ios::in);
    ////////// 24 bit //////////
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
    ////////// 16-bit //////////
    fp.read((char*)&frame, sizeof(ushort) * WIDTH * HEIGHT);
    ///////////////////////////

    //Mat CPUinput = imread("C:/Users/SYunMoon/Pictures/1004.bmp", IMREAD_GRAYSCALE);

    Mat input_whole = Mat(HEIGHT, WIDTH, CV_16U, frame);
    //Mat input_part(input_whole, Rect(0, 0, 640, 512));

    box_filter_8u_c1(input_whole, h_result, WIDTH, 5, 5);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("GPU Elapsed time : %f ms\n", elapsedTime);
    Mat final = Mat(HEIGHT, WIDTH, CV_16U, h_result);
    final = HistEqualization(final);

    auto stop_all = steady_clock::now();

    auto duration = duration_cast<milliseconds>(stop_all - start_all);

    cout << "Time taken by function (CHE and Image Pyramid): "
        << duration.count() << " milliseconds" << endl;

    //double minVal[1];
    //double maxVal;
    //minMaxLoc(final, minVal, &maxVal);
    //cout << "final : min = " << minVal[0] << ", max = " << maxVal << endl;
    //final = (final - minVal[0]) / (maxVal - minVal[0]) * 255;
    //
    //Mat temp_final = Mat(final, Rect(1, 1, WIDTH - 1, HEIGHT - 1));
    final = (final / 65535) * 255;
    final.convertTo(final, CV_8U);
    imwrite("final.bmp", final);

    input_whole = HistEqualization (input_whole);
    input_whole = (input_whole / 65535) * 255;
    imwrite("input.bmp", input_whole);

    //imshow("final", final);
    //waitKey();

    return 0;
}