#include <cuda_runtime.h>
#include <iostream>

__global__ void mem_read_kernel(const float* __restrict__ data, float* __restrict__ sink, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (size_t i = tid; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }
    sink[tid] = sum;  // 防止编译器优化
}

int main() {
    size_t N = (size_t)1 << 28;  // 1 GiB，共 268,435,456 个 float
    size_t bytes = N * sizeof(float);

    float* d_data;
    float* d_sink;

    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_sink, sizeof(float) * 1024);  // dummy 输出防优化

    cudaMemset(d_data, 1, bytes);

    dim3 block(256);
    dim3 grid(1024);  // 共 256K 个线程

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    mem_read_kernel<<<grid, block>>>(d_data, d_sink, N);
    cudaDeviceSynchronize();

    // timed run
    cudaEventRecord(start);
    mem_read_kernel<<<grid, block>>>(d_data, d_sink, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float gb = bytes / (1024.0f * 1024.0f * 1024.0f);
    float bandwidth = gb / (ms / 1000.0f);

    std::cout << "Memory READ bandwidth: " << bandwidth << " GB/s" << std::endl;

    cudaFree(d_data);
    cudaFree(d_sink);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
