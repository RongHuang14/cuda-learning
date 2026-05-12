#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "timer.h"

void blur_gpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height);

void blur_cpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
    int BLUR_SIZE = 1;
    for (unsigned int row = 0; row < height; row++) {
        for (unsigned int col = 0; col < width; col++) {
            unsigned int average = 0;
            for (int inRow = row - BLUR_SIZE; inRow < (int)row + BLUR_SIZE + 1; ++inRow) {
                for (int inCol = col - BLUR_SIZE; inCol < (int)col + BLUR_SIZE + 1; ++inCol) {
                    if (inRow >= 0 && inRow < (int)height && inCol >= 0 && inCol < (int)width)
                        average += image[inRow*width + inCol];
                }
            }
            blurred[row*width + col] = (unsigned char)(average / ((2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)));
        }
    }
}

int main() {
    unsigned int width = 1024, height = 1024;
    unsigned int size = width * height;
    Timer timer;

    unsigned char *image   = (unsigned char*)malloc(size);
    unsigned char *blurred = (unsigned char*)malloc(size);

    for (unsigned int i = 0; i < size; i++)
        image[i] = rand() % 256;

    unsigned char* dummy;
    cudaMalloc((void**)&dummy, 1);
    cudaFree(dummy);

    startTime(&timer);
    blur_cpu(image, blurred, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", "\033[0;36m");

    startTime(&timer);
    blur_gpu(image, blurred, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", GREEN);

    free(image); free(blurred);
    return 0;
}
