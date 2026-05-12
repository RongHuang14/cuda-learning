#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "timer.h"

void mm_gpu(float* A, float* B, float* C, unsigned int N);

void mm_cpu(float* A, float* B, float* C, unsigned int N) {
    for (unsigned int row = 0; row < N; ++row)
        for (unsigned int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (unsigned int i = 0; i < N; ++i)
                sum += A[row*N + i] * B[i*N + col];
            C[row*N + col] = sum;
        }
}

int main() {
    unsigned int N = 512;
    Timer timer;

    float *A = (float*)malloc(N*N*sizeof(float));
    float *B = (float*)malloc(N*N*sizeof(float));
    float *C = (float*)malloc(N*N*sizeof(float));

    for (unsigned int i = 0; i < N*N; ++i) {
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }

    // CUDA warmup
    float* dummy;
    cudaMalloc((void**)&dummy, 1);
    cudaFree(dummy);

    startTime(&timer);
    mm_cpu(A, B, C, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", "\033[0;36m");

    startTime(&timer);
    mm_gpu(A, B, C, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", GREEN);

    free(A); free(B); free(C);
    return 0;
}
