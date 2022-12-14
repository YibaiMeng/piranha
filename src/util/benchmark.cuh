#pragma once

#include <cmath>
#include <chrono>
#include <thrust/device_vector.h>
#include <loguru.hpp>
#include "../gpu/gemm.cuh"
#include "../gpu/DeviceData.h"

template <typename T>
__global__ void initializeMatrixKernel(T *matrix, int rows, int columns, int seed = 0) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < rows && j < columns) {
        int offset = i + j * rows;
        int const k = 16807;
        int const m = 16;
        T value = T(((offset + seed) * k % m) - m / 2);
        matrix[offset] = value;
    }
}

template<typename T>
cudaError_t initializeMatrix(T *matrix, int rows, int columns, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);
  initializeMatrixKernel<<< grid, block >>>(matrix, rows, columns, seed);
  return cudaGetLastError();
}

void heatup() {
    for (int t = 0; t < 10; t++) {
        CUDA_CHECK(cudaSetDevice(0));
        auto M = 4096;
        auto N = 4096;
        auto K = 4096;
        DeviceData<int> A(M * K);
        DeviceData<int> B(K * N);
        DeviceData<int> C(M * N);
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&A.begin()[0]), M, K, 238));
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&B.begin()[0]), K, N, 456));
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&C.begin()[0]), M, N, 831));
        gpu::gemm(M, N, K, &A, true, &B, true, &C, true);
    }
}

void benchmark_gemm(int is, int js, int ks) {
    vector<float> times(is * js * ks);
    heatup();

    for (int i = 0; i < is; i++) {
        for (int j = 0; j < js; j++) {
            LOG_F(INFO, "Running multi-GPU GeMM for %d, %d", i, j);
            for (int k = 0; k < ks; k++) {
                CUDA_CHECK(cudaSetDevice(0));
                auto M = pow(4, i);
                auto N = pow(4, j);
                auto K = pow(4, k * 2);
                DeviceData<int> A(M * K);
                DeviceData<int> B(K * N);
                DeviceData<int> C(M * N);
                CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&A.begin()[0]), M, K, 238));
                CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&B.begin()[0]), K, N, 456));
                CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&C.begin()[0]), M, N, 831));
                float total = 0;
                for (int t = 0; t < 3; t++) {
                    auto start = std::chrono::system_clock::now();
                    gpu::gemm(M, N, K, &A, true, &B, true, &C, true);
                    auto stop = std::chrono::system_clock::now();
                    total += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                }
                times[i * js * ks + j * ks + k] = total / 3;
            }
        }
    }

    for (int i = 0; i < is; i++) {
        for (int j = 0; j < js; j++) {
            LOG_F(INFO, "Running single-GPU GeMM for %d, %d", i, j);
            for (int k = 0; k < ks; k ++) {
                CUDA_CHECK(cudaSetDevice(0));
                auto M = pow(4, i);
                auto N = pow(4, j);
                auto K = pow(4, k * 2);
                DeviceData<int> A(M * K);
                DeviceData<int> B(K * N);
                DeviceData<int> C(M * N);
                CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&A.begin()[0]), M, K, 859));
                CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&B.begin()[0]), K, N, 129));
                CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&C.begin()[0]), M, N, 596));
                float total = 0;
                for (int t = 0; t < 3; t++) {
                    auto start = std::chrono::system_clock::now();
                    gpu::gemm(M, N, K, &A, false, &B, false, &C, false, {0});
                    auto stop = std::chrono::system_clock::now();
                    total += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                }
                times[i * js * ks + j * ks + k] /= total / 3;
            }
        }
    }

    for (int k = 0; k < ks; k ++) {
        for (int i = 0; i < is; i++) {
            for (int j = 0; j < js; j++) {
                printf("%f\t", times[i * js * ks + j * ks + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void benchmark_gemm(int ks) {
    vector<float> multi_times(ks);
    vector<float> single_times(ks);
    heatup();

    for (int t = 0; t < 10; t++) {
        CUDA_CHECK(cudaSetDevice(0));
        auto M = 4096;
        auto N = 4096;
        auto K = 4096;
        DeviceData<int> A(M * K);
        DeviceData<int> B(K * N);
        DeviceData<int> C(M * N);
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&A.begin()[0]), M, K, 238));
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&B.begin()[0]), K, N, 456));
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&C.begin()[0]), M, N, 831));
        gpu::gemm(M, N, K, &A, true, &B, true, &C, true);
    }

    for (int k = 0; k < ks; k++) {
        CUDA_CHECK(cudaSetDevice(0));
        auto M = 4096;
        auto N = 4096;
        int K = pow(4, k);
        LOG_F(INFO, "Running multi-GPU GeMM for K=%d", K);
        DeviceData<int> A(M * K);
        DeviceData<int> B(K * N);
        DeviceData<int> C(M * N);
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&A.begin()[0]), M, K, 238));
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&B.begin()[0]), K, N, 456));
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&C.begin()[0]), M, N, 831));
        float total = 0;
        for (int t = 0; t < 5; t++) {
            auto start = std::chrono::system_clock::now();
            gpu::gemm(M, N, K, &A, true, &B, true, &C, true);
            auto stop = std::chrono::system_clock::now();
            total += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        }
        multi_times[k] = total / 5;
    }

    for (int k = 0; k < ks; k++) {
        CUDA_CHECK(cudaSetDevice(0));
        auto M = 4096;
        auto N = 4096;
        int K = pow(4, k);
        LOG_F(INFO, "Running single-GPU GeMM for K=%d", K);
        DeviceData<int> A(M * K);
        DeviceData<int> B(K * N);
        DeviceData<int> C(M * N);
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&A.begin()[0]), M, K, 859));
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&B.begin()[0]), K, N, 129));
        CUDA_CHECK(initializeMatrix(thrust::raw_pointer_cast(&C.begin()[0]), M, N, 596));
        float total = 0;
        for (int t = 0; t < 5; t++) {
            auto start = std::chrono::system_clock::now();
            gpu::gemm(M, N, K, &A, false, &B, false, &C, false, {0});
            auto stop = std::chrono::system_clock::now();
            total += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        }
        single_times[k] = total / 5;
    }

    for (int k = 0; k < ks; k++) {
        printf("%f\t", single_times[k]);
    }
    printf("\n");
    for (int k = 0; k < ks; k++) {
        printf("%f\t", multi_times[k]);
    }
    printf("\n");
}
