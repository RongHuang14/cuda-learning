#include "common.h"
#include "timer.h"

#define BLUR_SIZE 1  // 窗口半径：1→3x3，2→5x5

// 每个线程负责一个输出像素的均值模糊
__global__ void blur_kernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;  // 线程 → 像素坐标
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;

    if (outRow < height && outCol < width) {
        unsigned int average = 0;

        for (int inRow = outRow - BLUR_SIZE; inRow < outRow + BLUR_SIZE + 1; ++inRow) {
            for (int inCol = outCol - BLUR_SIZE; inCol < outCol + BLUR_SIZE + 1; ++inCol) {
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)  // 边界截断
                    average += image[inRow*width + inCol];
            }
        }

        // 除以完整窗口大小（边缘像素有效采样数不足，略偏暗）
        blurred[outRow*width + outCol] = (unsigned char)(average / ((2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)));
    }
}

void blur_gpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
    Timer timer;
    unsigned char *image_d, *blurred_d;

    startTime(&timer);
    cudaMalloc((void**) &image_d,   width*height*sizeof(unsigned char));
    cudaMalloc((void**) &blurred_d, width*height*sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    startTime(&timer);
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);  // H→D
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    startTime(&timer);
    dim3 numThreadsPerBlock(16, 16);  // 每 block 256 线程
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,
                   (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    blur_kernel<<< numBlocks, numThreadsPerBlock >>>(image_d, blurred_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    startTime(&timer);
    cudaMemcpy(blurred, blurred_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);  // D→H
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    startTime(&timer);
    cudaFree(image_d);
    cudaFree(blurred_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}
