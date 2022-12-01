
#pragma once

#include "gpu.h"

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/device/gemm.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

#include "DeviceData.h"
#include "../globals.h"
#include "../util/util.cuh"

#include <loguru.hpp>

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

template<typename T>
using CutlassGemmNNN = cutlass::gemm::device::Gemm<T, ColumnMajor, T, ColumnMajor, T, ColumnMajor>;
template<typename T>
using CutlassGemmTNN = cutlass::gemm::device::Gemm<T, RowMajor, T, ColumnMajor, T, ColumnMajor>;
template<typename T>
using CutlassGemmNTN = cutlass::gemm::device::Gemm<T, ColumnMajor, T, RowMajor, T, ColumnMajor>;
template<typename T>
using CutlassGemmTTN = cutlass::gemm::device::Gemm<T, RowMajor, T, RowMajor, T, ColumnMajor>;

template<typename T>
using CutlassGemmNNT = cutlass::gemm::device::Gemm<T, ColumnMajor, T, ColumnMajor, T, RowMajor>;
template<typename T>
using CutlassGemmTNT = cutlass::gemm::device::Gemm<T, RowMajor, T, ColumnMajor, T, RowMajor>;
template<typename T>
using CutlassGemmNTT = cutlass::gemm::device::Gemm<T, ColumnMajor, T, RowMajor, T, RowMajor>;
template<typename T>
using CutlassGemmTTT = cutlass::gemm::device::Gemm<T, RowMajor, T, RowMajor, T, RowMajor>;

template<typename T, template<typename> typename CutlassGemm>
void CutlassGemmOp(
        int M, int N, int K,
        T alpha,
        T const *A, int lda,
        T const *B, int ldb,
        T beta,
        T *C, int ldc,
        cudaStream_t stream = nullptr) {

    CutlassGemm<T> gemm_operator;

    typename CutlassGemm<T>::Arguments args(
            {M, N, K},
            {A, lda}, {B, ldb}, {C, ldc}, {C, ldc},
            {alpha, beta});
    
    cudaPointerAttributes attra, attrb, attrc;
    cudaPointerGetAttributes(&attra, A);
    cudaPointerGetAttributes(&attrb, B);
    cudaPointerGetAttributes(&attrc, C);
    LOG_F(INFO, "Pointers from: %i, %i, %i", attra.device, attrb.device, attrc.device);

    // https://github.com/NVIDIA/cutlass/blob/e773429f7e31db051a67daeb58ed35d20095de9b/examples/24_gemm_grouped/gemm_grouped.cu#L1072-L1081
    CUTLASS_CHECK(gemm_operator.initialize(args));
    CUTLASS_CHECK(gemm_operator(stream));
}

namespace gpu {

template<typename T>
void gemmAsync(int M, int N, int K,
        T const *A, bool transpose_a,
        T const *B, bool transpose_b,
        T *C, bool transpose_c,
        cudaStream_t stream = nullptr) {

    if (transpose_c) {
        if (transpose_a) {
            if (transpose_b) {
                CutlassGemmOp<T, CutlassGemmTTT>(M, N, K, (T)1, A, K, B, N, (T)0, C, N, stream);
            } else {
                CutlassGemmOp<T, CutlassGemmTNT>(M, N, K, (T)1, A, K, B, K, (T)0, C, N, stream);
            }
        } else {
            if (transpose_b) {
                CutlassGemmOp<T, CutlassGemmNTT>(M, N, K, (T)1, A, M, B, N, (T)0, C, N, stream);
            } else {
                CutlassGemmOp<T, CutlassGemmNNT>(M, N, K, (T)1, A, M, B, K, (T)0, C, N, stream);
            }
        }
    } else {
        if (transpose_a) {
            if (transpose_b) {
                CutlassGemmOp<T, CutlassGemmTTN>(M, N, K, (T)1, A, K, B, N, (T)0, C, M, stream);
            } else {
                CutlassGemmOp<T, CutlassGemmTNN>(M, N, K, (T)1, A, K, B, K, (T)0, C, M, stream);
            }
        } else {
            if (transpose_b) {
                CutlassGemmOp<T, CutlassGemmNTN>(M, N, K, (T)1, A, M, B, N, (T)0, C, M, stream);
            } else {
                CutlassGemmOp<T, CutlassGemmNNN>(M, N, K, (T)1, A, M, B, K, (T)0, C, M, stream);
            }
        }
    }
}

// C = alpha A * B + beta C
// A : M by K
// B : K by N
// C : M by N
// Possibly parallel by splitting A or B
template<typename T>
void gemm(int M, int N, int K,
    const DeviceData<T>* A, bool ta,
    const DeviceData<T>* B, bool tb,
    const DeviceData<T>* C, bool tc,
    const std::vector<int>* devs = nullptr) {
    if (devs == nullptr) {
        int deviceCount; cudaGetDeviceCount(&deviceCount);
        std::vector<int> devices(deviceCount);
        std::iota(devices.begin(), devices.end(), 0);
        devs = &devices;
    }
    if (ta != tc && tb != tc || devs->size() == 1) {
        CUDA_CHECK(cudaSetDevice(devs->at(0)));
        LOG_F(INFO, "Cannot split work");
        gemmAsync<T>(M, N, K,
            thrust::raw_pointer_cast(&A->begin()[0]), ta,
            thrust::raw_pointer_cast(&B->begin()[0]), tb,
            thrust::raw_pointer_cast(&C->begin()[0]), tc);
        cudaStreamSynchronize(0);
        return;
    }
    int len = devs->size();
    std::vector<cudaStream_t> streams(len);
    bool split_a = ta == tc;
    auto S = split_a ? A : B;
    auto U = split_a ? B : A;
    auto S_dim = split_a ? M : N;
    auto strip = S_dim / len;
    auto U_dim = split_a ? N : M;
    for (int i = 0; i < len; i++) {
        auto id = devs->at(i);
        CUDA_CHECK(cudaSetDevice(id));
        cudaStreamCreate(&streams[i]);
        auto st = streams[i];
        struct slice {
            T* ptr;
            int range;
            bool owned;
        } a, b, c;
        auto splitted = split_a ? &a : &b;
        auto unsplitted = split_a ? &b : &a;
        LOG_F(INFO, "Copying stripped part");
        auto sp = thrust::raw_pointer_cast(&S->begin()[i * strip * K]);
        auto srange = (i == len - 1) ? strip + S_dim % len : strip;
        if (id == S->cudaDeviceID()) {
            *splitted = { sp, srange, false };
        } else {
            T* sptr; CUDA_CHECK(cudaMallocAsync(&sptr, srange * K * sizeof(T), st));
            cudaMemcpyPeerAsync(sptr, id, sp, S->cudaDeviceID(), srange * K * sizeof(T), st);
            *splitted = { sptr, srange, true };
        }
        LOG_F(INFO, "Copying unstripped part");
        auto up = thrust::raw_pointer_cast(&U->begin()[0]);
        if (id == U->cudaDeviceID()) {
            *unsplitted = { up, U_dim, false };
        } else {
            T* uptr; CUDA_CHECK(cudaMallocAsync(&uptr, U_dim * K * sizeof(T), st));
            cudaMemcpyPeerAsync(uptr, id, up, U->cudaDeviceID(), U_dim * K * sizeof(T), st);
            *unsplitted = { uptr, U_dim, true };
        }
        LOG_F(INFO, "Allocating result buffer");
        auto cp = thrust::raw_pointer_cast(&C->begin()[i * strip * U_dim]);
        if (id == C->cudaDeviceID()) {
            c = { cp, srange, false };
        } else {
            T* cptr; CUDA_CHECK(cudaMallocAsync(&cptr, srange * U_dim * sizeof(T), st));
            c = { cptr, srange, true };
        }
        LOG_F(INFO, "Executing GeMM");
        gemmAsync<T>(a.range, b.range, K, a.ptr, ta, b.ptr, tb, c.ptr, tc, st);
        LOG_F(INFO, "Collecting results");
        cudaMemcpyPeerAsync(cp, C->cudaDeviceID(), c.ptr, id, c.range * U_dim * sizeof(T));
        // Cleanup
        if (a.owned) cudaFreeAsync(a.ptr, st);
        if (b.owned) cudaFreeAsync(b.ptr, st);
        if (c.owned) cudaFreeAsync(c.ptr, st);
    }
    // Synchronize
    for (auto stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}

} // namespace gpu

