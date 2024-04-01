#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHANNELS 3

__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int h, int w) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    int r = blockIdx.y*blockDim.y + threadIdx.y;

    if (c<w && r<h) {
        int i = r*w + c;
		int offset = i*CHANNELS;
        out[i] = 0.2989*x[offset] + 0.5870*x[offset+1] + 0.1140*x[offset+2];
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./rgb2gray <image_path>" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    unsigned char *d_in, *d_out;
    int imgSize = img.rows * img.cols * img.channels();
    int outSize = img.rows * img.cols;

    clock_t start = clock();

    cudaMalloc((void**)&d_in, imgSize);
    cudaMalloc((void**)&d_out, outSize);

    cudaMemcpy(d_in, img.data, imgSize, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((img.cols + dimBlock.x - 1) / dimBlock.x, (img.rows + dimBlock.y - 1) / dimBlock.y);

    rgb_to_grayscale_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, img.rows, img.cols);

    unsigned char* h_out = new unsigned char[outSize];
    cudaMemcpy(h_out, d_out, outSize, cudaMemcpyDeviceToHost);

    clock_t end = clock();
	double time_taken = ((double)end - start) / CLOCKS_PER_SEC;
    printf("Time taken to convert image: %f seconds\n", time_taken);

    cv::Mat outImg(img.rows, img.cols, CV_8UC1, h_out);
    cv::imwrite("output.jpg", outImg);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_out;

    return 0;
}