#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

__host__ __device__ float f(float a, float b) {
    return a + b;
}

void vecadd_cpu(float* x, float* y, float* z, int N) {
    for (unsigned int i = 0; i < N; i++) {
        z[i] = f(x[i], y[i]);
    }
}

__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        z[i] = f(x[i], y[i]);
    }
}

void vecadd_gpu(float* x, float* y, float* z, int N, float* gpu_time_ms) {
    // Allocate GPU memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N * sizeof(float));
    cudaMalloc((void**)&y_d, N * sizeof(float));
    cudaMalloc((void**)&z_d, N * sizeof(float));

    // Copy to the GPU
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run the GPU Code
    // Call a GPU kernel function (launch a grid of threads)
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(gpu_time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy from the GPU
    cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main(int argc, char** argv) {
    // Allocate memory and initialize data
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);

    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));
    float* z_cpu = (float*)malloc(N * sizeof(float));
    float* z_gpu = (float*)malloc(N * sizeof(float));

    for (unsigned int i = 0; i < N; ++i) {
        x[i] = rand() / (float)RAND_MAX;
        y[i] = rand() / (float)RAND_MAX;
    }

    // Vector addition on CPU
    clock_t cpu_start = clock();
    vecadd_cpu(x, y, z_cpu, N);
    clock_t cpu_stop = clock();

    double cpu_time_ms = 1000.0 * (cpu_stop - cpu_start) / CLOCKS_PER_SEC;

    // Vector addition on GPU
    float gpu_time_ms = 0.0f;
    vecadd_gpu(x, y, z_gpu, N, &gpu_time_ms);

    // Print elapsed time
    printf("CPU time: %.4f ms\n", cpu_time_ms);
    printf("GPU kernel time: %.4f ms\n", gpu_time_ms);

    // Print some results
    printf("\nFirst 10 results:\n");
    for (unsigned int i = 0; i < 10 && i < N; i++) {
        printf("x[%u] = %f, y[%u] = %f, z_gpu[%u] = %f\n",
               i, x[i], i, y[i], i, z_gpu[i]);
    }

    // Check result
    int errors = 0;
    for (unsigned int i = 0; i < N; i++) {
        if (fabs(z_cpu[i] - z_gpu[i]) > 1e-5) {
            errors++;
            if (errors < 10) {
                printf("Mismatch at %u: CPU=%f GPU=%f\n", i, z_cpu[i], z_gpu[i]);
            }
        }
    }

    if (errors == 0) {
        printf("\nVector addition completed successfully!\n");
    } else {
        printf("\nFound %d errors.\n", errors);
    }

    // Free memory
    free(x);
    free(y);
    free(z_cpu);
    free(z_gpu);

    return 0;
}
