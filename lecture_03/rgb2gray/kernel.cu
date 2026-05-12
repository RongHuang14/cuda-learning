#include "common.h"
#include "timer.h"

__global__ void rgb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < height && col < width) {
        unsigned int i = row*width + col;
        gray[i] = red[i]*3/10 + green[i]*6/10 + blue[i]*1/10;
    }
}

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    Timer timer;

    startTime(&timer);
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**) &red_d,   width*height*sizeof(unsigned char));
    cudaMalloc((void**) &green_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &blue_d,  width*height*sizeof(unsigned char));
    cudaMalloc((void**) &gray_d,  width*height*sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    startTime(&timer);
    cudaMemcpy(red_d,   red,   width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d,  blue,  width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    startTime(&timer);
    dim3 numThreadsPerBlock(32, 32, 1);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y, 1);
    rgb2gray_kernel<<< numBlocks, numThreadsPerBlock >>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    startTime(&timer);
    cudaMemcpy(gray, gray_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    startTime(&timer);
    cudaFree(red_d); cudaFree(green_d); cudaFree(blue_d); cudaFree(gray_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}
