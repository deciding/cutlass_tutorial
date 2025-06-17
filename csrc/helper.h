/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cuda_runtime.h"
#include <iostream>

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer
{
    cudaStream_t _stream_id;
    cudaEvent_t _start;
    cudaEvent_t _stop;

    /// Constructor
    GpuTimer() : _stream_id(0)
    {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        CUDA_CHECK(cudaEventDestroy(_start));
        CUDA_CHECK(cudaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        CUDA_CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        CUDA_CHECK(cudaEventSynchronize(_stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

/// Naive reference GEMM computation.
template <class TA, class TB, class TC,
          class TI>
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  TI alpha,
  TA const *A,
  int lda,
  TB const *B,
  int ldb,
  TI beta,
  TC *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      //accumulator += A[i + k * lda] * B[k + j * ldb];
      // TN
      accumulator += A[k + i * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
template <typename TA, typename TB, typename TC, typename TI>
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  TI alpha,
  TA const *A,
  int lda,
  TB const *B,
  int ldb,
  TI beta,
  TC *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

// Function to check if two host_vectors are approximately equal within atol and rtol
template <typename T>
bool is_close(const thrust::host_vector<T>& a, 
              const thrust::host_vector<T>& b, 
              T atol = static_cast<T>(1e-8f),
              T rtol = static_cast<T>(1e-5f)) {
    if (a.size() != b.size()) {
        return false;  // Different sizes, cannot be equal
    }

    for (size_t i = 0; i < a.size(); ++i) {
        T diff = static_cast<T>(std::abs(static_cast<float>(a[i] - b[i])));
        T tol  = atol + rtol * static_cast<T>(std::abs(static_cast<float>(b[i])));
        
        if (diff > tol) {
            std::cout << "Mismatch at index " << i << ": "
                      << a[i] << " vs " << b[i] 
                      << " (diff = " << diff << ", tol = " << tol << ")\n";
            return false;
        }
    }
    return true;
}

template <typename T>
void print_host_vector_pretty(const thrust::host_vector<T>& vec, int width = 6, bool scientific = false, size_t max_elements = 6) {
    std::cout << "[";
    size_t n = vec.size();

    if (n <= max_elements) {
        // Print all elements if vector is small
        for (size_t i = 0; i < n; ++i) {
            if (scientific) {
                std::cout << std::scientific << std::setprecision(4) << vec[i];
            } else {
                std::cout << std::setw(width) << vec[i];
            }
            if (i != n - 1) std::cout << ", ";
        }
    } else {
        // Print first 3 and last 3 elements
        for (size_t i = 0; i < 3; ++i) {
            if (scientific) {
                std::cout << std::scientific << std::setprecision(4) << vec[i];
            } else {
                std::cout << std::setw(width) << vec[i];
            }
            std::cout << ", ";
        }
        std::cout << "..., ";
        for (size_t i = n - 3; i < n; ++i) {
            if (scientific) {
                std::cout << std::scientific << std::setprecision(4) << vec[i];
            } else {
                std::cout << std::setw(width) << vec[i];
            }
            if (i != n - 1) std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
