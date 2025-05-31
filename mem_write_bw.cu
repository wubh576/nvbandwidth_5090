#include <cuda_runtime.h>
#include <iostream>

__global__ void mem_write_kernel(float* __restrict__ data, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < N; i += blockDim.x * gridDim.x) {
        data[i] = 3.14f;
    }
}

int main() {
    size_t N = (size_t)1 << 28;  // 1 GiB
    size_t bytes = N * sizeof(float);

    float* d_data;
    cudaMalloc(&d_data, bytes);

    dim3 block(256);
    dim3 grid(1024);  // 共 256K 线程

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    mem_write_kernel<<<grid, block>>>(d_data, N);
    cudaDeviceSynchronize();

    // timed run
    cudaEventRecord(start);
    mem_write_kernel<<<grid, block>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float gb = bytes / (1024.0f * 1024.0f * 1024.0f);
    float bandwidth = gb / (ms / 1000.0f);

    std::cout << "Memory WRITE bandwidth: " << bandwidth << " GB/s" << std::endl;

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
