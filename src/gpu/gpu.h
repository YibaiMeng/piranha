
#pragma once
#include <loguru.hpp>
#define CUTLASS_CHECK(status)                                                                      \
    do {                                                                                           \
        cutlass::Status err_ = status;                                                             \
        if (err_ != cutlass::Status::kSuccess) {                                                   \
            LOG_F(ERROR, "CUTLASS error %d at %s:%d\n", err_, __FILE__, __LINE__);                 \
            throw std::runtime_error("CUTLASS error");                                                \
        }                                                                                          \
    } while (0)                                                                                    \
 

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            LOG_F(ERROR, "CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

#define THRUST_CHECK(err)                                            \
    do { \
    try {                                                             \
       (err);                                                        \
    }                                                                                         \
    catch(thrust::system_error e)  { \
        LOG_S(ERROR) << "Thrust error" << e.what() << std::endl; \
    } \
    } while (0)