
#pragma once

#include <thrust/device_vector.h>

#include "SecretShare.h"

namespace kernel {

template<typename T>
__global__ void matrixMultiplication(
        thrust::device_vector<T> &a, thrust::device_vector<T> &b,
        thrust::device_vector<T> &c, bool transpose_a, bool transpose_b,
        int rows, int shared, int cols);

}

namespace gpu {

template<typename T>
void matrixMultiplication(
        SecretShare<T> &a, SecretShare<T> &b, SecretShare<T> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols);

}
