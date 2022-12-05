
#pragma once
#include <loguru.hpp>
#define CUTLASS_CHECK(status)                                                                      \
{                                                                                                  \
    cutlass::Status error = status;                                                                \
    if (error != cutlass::Status::kSuccess) {                                                      \
        LOG_S(ERROR) << "Cutlass error " << cutlassGetStatusString(error) << " at " << __FILE__ << ":" << __LINE__;\
    }                                                                                              \
}

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