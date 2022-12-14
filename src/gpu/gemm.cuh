
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
        cudaStream_t stream = 0) {

    CutlassGemm<T> gemm_operator;

    typename CutlassGemm<T>::Arguments args(
            {M, N, K},
            {A, lda}, {B, ldb}, {C, ldc}, {C, ldc},
            {alpha, beta});
    
    cudaPointerAttributes attra, attrb, attrc;
    CUDA_CHECK(cudaPointerGetAttributes(&attra, A));
    CUDA_CHECK(cudaPointerGetAttributes(&attrb, B));
    CUDA_CHECK(cudaPointerGetAttributes(&attrc, C));
    LOG_F(1, "Pointers from: %i, %i, %i", attra.device, attrb.device, attrc.device);

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
        cudaStream_t stream = 0) {

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
    std::vector<int> devs = {}) {
    int old_device; CUDA_CHECK(cudaGetDevice(&old_device));
    if (devs.size() == 0) {
        int deviceCount; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        std::vector<int> devices(deviceCount);
        std::iota(devices.begin(), devices.end(), 0);
        devs = devices;
    }
    LOG_F(1, "%d devices available for GeMM, with M, N, K = %d, %d, %d and ta, tb, tc = %d, %d, %d", devs.size(), M, N, K, ta, tb, tc);
    if ((!ta && tc) || (tb && !tc)) {
        CUDA_CHECK(cudaSetDevice(A->cudaDeviceID()));
        LOG_F(1, "Cannot split work");
        gemmAsync<T>(M, N, K,
            thrust::raw_pointer_cast(&A->begin()[0]), ta,
            thrust::raw_pointer_cast(&B->begin()[0]), tb,
            thrust::raw_pointer_cast(&C->begin()[0]), tc, 0);
        CUDA_CHECK(cudaStreamSynchronize(0));
        CUDA_CHECK(cudaSetDevice(old_device));
        return;
    }
    // cudaEvent_t start_event; CUDA_CHECK(cudaEventCreate(&start_event));
    // CUDA_CHECK(cudaEventRecord(start_event));
    std::vector<cudaStream_t> streams;
    struct timer {
        int device;
        cudaEvent_t pre_start, pre_stop;
        cudaEvent_t gemm_start, gemm_stop;
        cudaEvent_t post_start, post_stop;
    };
    std::vector<timer> timers;
    int len = devs.size();
    bool split_a = ta == tc;
    auto S = split_a ? A : B;
    auto U = split_a ? B : A;
    auto S_dim = split_a ? M : N;
    auto strip = S_dim / len;
    auto U_dim = split_a ? N : M;
    LOG_F(1, "Stripping %s with strip %d", split_a ? "A" : "B", strip);
    for (int i = 0; i < len; i++) {
        auto srange = (i == len - 1) ? strip + (S_dim % len) : strip;
        if (srange == 0) continue;
        auto id = devs[i];
        LOG_F(1, "Preparing GeMM on device %d", id);
        CUDA_CHECK(cudaSetDevice(id));
        // cudaStream_t st; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
        streams.push_back(st);
        timer timer;
        timer.device = id;

        CUDA_CHECK(cudaEventCreate(&timer.pre_start));
        CUDA_CHECK(cudaEventCreate(&timer.pre_stop));
        CUDA_CHECK(cudaEventRecord(timer.pre_start, st));
        struct slice {
            T* ptr;
            int range;
            bool owned;
        } a, b, c;
        auto splitted = split_a ? &a : &b;
        auto unsplitted = split_a ? &b : &a;
        auto sp = thrust::raw_pointer_cast(&S->begin()[i * strip * K]);
        if (id == S->cudaDeviceID()) {
            *splitted = { sp, srange, false };
        } else {
            T* sptr; CUDA_CHECK(cudaMallocAsync(&sptr, srange * K * sizeof(T), st));
            CUDA_CHECK(cudaMemcpyPeerAsync(sptr, id, sp, S->cudaDeviceID(), srange * K * sizeof(T), st));
            *splitted = { sptr, srange, true };
        }
        auto up = thrust::raw_pointer_cast(&U->begin()[0]);
        if (id == U->cudaDeviceID()) {
            *unsplitted = { up, U_dim, false };
        } else {
            T* uptr; CUDA_CHECK(cudaMallocAsync(&uptr, U_dim * K * sizeof(T), st));
            CUDA_CHECK(cudaMemcpyPeerAsync(uptr, id, up, U->cudaDeviceID(), U_dim * K * sizeof(T), st));
            *unsplitted = { uptr, U_dim, true };
        }
        auto cp = thrust::raw_pointer_cast(&C->begin()[i * strip * U_dim]);
        if (id == C->cudaDeviceID()) {
            c = { cp, srange, false };
        } else {
            T* cptr; CUDA_CHECK(cudaMallocAsync(&cptr, srange * U_dim * sizeof(T), st));
            c = { cptr, srange, true };
        }
        CUDA_CHECK(cudaEventRecord(timer.pre_stop, st));

        CUDA_CHECK(cudaEventCreate(&timer.gemm_start));
        CUDA_CHECK(cudaEventCreate(&timer.gemm_stop));
        CUDA_CHECK(cudaEventRecord(timer.gemm_start, st));
        gemmAsync<T>(a.range, b.range, K, a.ptr, ta, b.ptr, tb, c.ptr, tc, st);
        CUDA_CHECK(cudaEventRecord(timer.gemm_stop, st));
        
        CUDA_CHECK(cudaEventCreate(&timer.post_start));
        CUDA_CHECK(cudaEventCreate(&timer.post_stop));
        CUDA_CHECK(cudaEventRecord(timer.post_start, st));
        if (a.owned) CUDA_CHECK(cudaFreeAsync(a.ptr, st));
        if (b.owned) CUDA_CHECK(cudaFreeAsync(b.ptr, st));
        if (c.owned) {
            CUDA_CHECK(cudaMemcpyPeerAsync(cp, C->cudaDeviceID(), c.ptr, id, c.range * U_dim * sizeof(T), st));
            CUDA_CHECK(cudaFreeAsync(c.ptr, st));
        }
        CUDA_CHECK(cudaEventRecord(timer.post_stop, st));
        timers.push_back(timer);
    }
    // Synchronize and report
    // CUDA_CHECK(cudaEventSynchronize(start_event));
    for (int i = 0; i < streams.size(); i++) {
        auto stream = streams[i];
        auto timer = timers[i];
        // CUDA_CHECK(cudaSetDevice(timer.device));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float pre, pre_start, gemm, gemm_start, post, post_start;
        CUDA_CHECK(cudaEventElapsedTime(&pre, timer.pre_start, timer.pre_stop));
        // CUDA_CHECK(cudaEventElapsedTime(&pre_start, start_event, timer.pre_start));
        CUDA_CHECK(cudaEventElapsedTime(&gemm, timer.gemm_start, timer.gemm_stop));
        // CUDA_CHECK(cudaEventElapsedTime(&gemm_start, start_event, timer.gemm_start));
        CUDA_CHECK(cudaEventElapsedTime(&post, timer.post_start, timer.post_stop));
        // CUDA_CHECK(cudaEventElapsedTime(&post_start, start_event, timer.post_start));
        LOG_F(1, "Time on device %d:", timer.device);
        // LOG_F(1, "  Preprocessing: %f, starting at %f", pre, pre_start);
        // LOG_F(1, "  GeMM: %f, starting at %f", gemm, gemm_start);
        // LOG_F(1, "  Postprocessing: %f, starting at %f", post, post_start);
        LOG_F(1, "  Preprocessing: %f", pre);
        LOG_F(1, "  GeMM: %f", gemm);
        LOG_F(1, "  Postprocessing: %f", post);
        CUDA_CHECK(cudaEventDestroy(timer.pre_start));
        CUDA_CHECK(cudaEventDestroy(timer.pre_stop));
        CUDA_CHECK(cudaEventDestroy(timer.gemm_start));
        CUDA_CHECK(cudaEventDestroy(timer.gemm_stop));
        CUDA_CHECK(cudaEventDestroy(timer.post_start));
        CUDA_CHECK(cudaEventDestroy(timer.post_stop));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    // CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaSetDevice(old_device));
}

} // namespace gpu

