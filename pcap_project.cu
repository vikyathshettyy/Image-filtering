#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Brighten kernel
__global__ void brighten_kernel(uchar3* input, uchar3* output, int width, int height, int brightness) {
    // Calculate the x and y indices of the current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the current pixel is within the image bounds
    if (x < width && y < height) {
        int index = y * width + x;
        uchar3 pixel = input[index];
        uchar3 brightened_pixel = make_uchar3(
            min(static_cast<int>(pixel.x) + brightness, 255),
            min(static_cast<int>(pixel.y) + brightness, 255),
            min(static_cast<int>(pixel.z) + brightness, 255)
        );
        output[index] = brightened_pixel;
    }
}

// Grayscale kernel
__global__ void grayscale_kernel(uchar3* input, uchar1* output, int width, int height) {
    // Calculate the x and y indices of the current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the current pixel is within the image bounds
    if (x < width && y < height) {
        int index = y * width + x;
        uchar3 pixel = input[index];
        float luminosity = 0.21f * pixel.x + 0.72f * pixel.y + 0.07f * pixel.z;
        output[index].x = static_cast<unsigned char>(roundf(luminosity));
    }
}

// Blur kernel
__global__ void blur_kernel(uchar3* input, uchar3* output, int width, int height, int blur_radius) {
    // Calculate the x and y indices of the current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the current pixel is within the image bounds
    if (x < width && y < height) {
        int index = y * width + x;
        int pixel_count = 0;
        int sum_r = 0, sum_g = 0, sum_b = 0;

        // Iterate through the neighboring pixels within the blur radius
        for (int i = -blur_radius; i <= blur_radius; ++i) {
            for (int j = -blur_radius; j <= blur_radius; ++j) {
                int current_x = x + j;
                int current_y = y + i;
                // Check if the current neighboring pixel is within the image bounds
                if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height) {
                    uchar3 current_pixel = input[current_y * width + current_x];
                    sum_r += current_pixel.x;
                    sum_g += current_pixel.y;
                    sum_b += current_pixel.z;
                    pixel_count++;
                }
            }
        }

        // Calculate the average color of the neighboring pixels and assign it to the output pixel
        output[index] = make_uchar3(
            static_cast<unsigned char>(sum_r / pixel_count),
            static_cast<unsigned char>(sum_g / pixel_count),
            static_cast<unsigned char>(sum_b / pixel_count)
        );
    }
}

int main(int argc, char** argv) {
    // Read input image
    cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat output_image(input_image.size(), input_image.type());
    cv::Mat grayscale_image(input_image.size(), CV_8UC1);
    cv::Mat blurred_image(input_image.size(), input_image.type());

    // Allocate device memory
    uchar3* d_input_image;
    uchar3* d_brightened_image;
    uchar1* d_grayscale_image;
    uchar3* d_blurred_image;
    cudaMalloc(&d_input_image, input_image.rows * input_image.cols * sizeof(uchar3));
    cudaMalloc(&d_brightened_image, input_image.rows * input_image.cols * sizeof(uchar3));
    cudaMalloc(&d_grayscale_image, input_image.rows * input_image.cols * sizeof(uchar1));
    cudaMalloc(&d_blurred_image, input_image.rows * input_image.cols * sizeof(uchar3));

    // Copy input image to device memory
    cudaMemcpy(d_input_image, input_image.ptr(), input_image.rows * input_image.cols * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Set grid and block dimensions for the brighten kernel
    dim3 dim_grid((input_image.cols - 1) / 16 + 1, (input_image.rows - 1) / 16 + 1, 1);
    dim3 dim_block(16, 16, 1);

    // Launch the brighten kernel with a brightness value of 50
    brighten_kernel<<<dim_grid, dim_block>>>(d_input_image, d_brightened_image, input_image.cols, input_image.rows, 50);

    // Copy brightened image from device memory
    cudaMemcpy(output_image.ptr(), d_brightened_image, input_image.rows * input_image.cols * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Save the brightened image
    cv::imwrite("brightened_image.jpg", output_image);

    // Set grid and block dimensions for the grayscale kernel
    dim3 dim_grid2((input_image.cols - 1) / 16 + 1, (input_image.rows - 1) / 16 + 1, 1);
    dim3 dim_block2(16, 16, 1);

    // Launch the grayscale kernel
    grayscale_kernel<<<dim_grid2, dim_block2>>>(d_input_image, d_grayscale_image, input_image.cols, input_image.rows);

    // Copy grayscale image from device memory
    cudaMemcpy(grayscale_image.ptr(), d_grayscale_image, input_image.rows * input_image.cols * sizeof(uchar1), cudaMemcpyDeviceToHost);

    // Save the grayscale image
    cv::imwrite("grayscale_image.jpg", grayscale_image);

    // Set grid and block dimensions for the blur kernel
    dim3 dim_grid3((input_image.cols - 1) / 16 + 1, (input_image.rows - 1) / 16 + 1, 1);
    dim3 dim_block3(16, 16, 1);

    // Launch the blur kernel with a blur radius of 9
    blur_kernel<<<dim_grid3, dim_block3>>>(d_input_image, d_blurred_image, input_image.cols, input_image.rows, 9);

    // Copy blurred image from device memory
    cudaMemcpy(blurred_image.ptr(), d_blurred_image, input_image.rows * input_image.cols * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Save the blurred image
    cv::imwrite("blurred_image.jpg", blurred_image);

        // Free device memory
    cudaFree(d_input_image);
    cudaFree(d_brightened_image);
    cudaFree(d_grayscale_image);
    cudaFree(d_blurred_image);

    return 0;
}


