#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "timer.h"

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height);

void rgb2gray_cpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    for (unsigned int row = 0; row < height; row++) {
        for (unsigned int col = 0; col < width; col++) {
            unsigned int i = row*width + col;
            gray[i] = red[i]*3/10 + green[i]*6/10 + blue[i]*1/10;
        }
    }
}

int main() {
    unsigned int width = 1024, height = 1024;
    unsigned int size = width * height;
    Timer timer;

    unsigned char *red   = (unsigned char*)malloc(size);
    unsigned char *green = (unsigned char*)malloc(size);
    unsigned char *blue  = (unsigned char*)malloc(size);
    unsigned char *gray  = (unsigned char*)malloc(size);

    for (unsigned int i = 0; i < size; i++) {
        red[i] = rand() % 256;
        green[i] = rand() % 256;
        blue[i] = rand() % 256;
    }

    // CUDA warmup
    unsigned char* dummy;
    cudaMalloc((void**)&dummy, 1);
    cudaFree(dummy);

    startTime(&timer);
    rgb2gray_cpu(red, green, blue, gray, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", "\033[0;36m");

    startTime(&timer);
    rgb2gray_gpu(red, green, blue, gray, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", GREEN);

    free(red); free(green); free(blue); free(gray);
    return 0;
}
