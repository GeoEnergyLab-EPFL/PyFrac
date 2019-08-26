//==============================================================================
//
// Copyright 2018 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

// Trying to benchmark the difference in between Blaze and
#include <chrono>
#include <cstdio>

#include <blaze/Math.h>
#include <mkl_cblas.h>

#include <mm_malloc.h>

int main() {
  std::size_t n = 16;
  std::size_t nb_loop = 10000000;

  for (int r = 0; r < 3; ++r) {
    {
      blaze::DynamicMatrix<double> A(n, n);
      blaze::DynamicMatrix<double> B(n, n);
      blaze::DynamicMatrix<double> C(n, n);
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          A(i, j) = 0.0;
          B(i, j) = 0.0;
          C(i, j) = 0.0;
        }
      }

      auto start = std::chrono::high_resolution_clock::now();
      for (std::size_t i = 0; i < nb_loop; ++i) {
        C = A * B;
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time =
          1.0e-9 *
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      double gflops = 1.0e-9 * (2 * n * n * n * nb_loop) / time;

      std::printf("Performance for Blaze: %7.2f Gflops\n", gflops);
    }

    double* A = (double*)_mm_malloc(n * n * sizeof(double), 32);
    double* B = (double*)_mm_malloc(n * n * sizeof(double), 32);
    double* C = (double*)_mm_malloc(n * n * sizeof(double), 32);
    {
      for (std::size_t k = 0; k < n * n; ++k) {
        A[k] = 0.0;
        B[k] = 0.0;
        C[k] = 0.0;
      }

      auto start = std::chrono::high_resolution_clock::now();
      for (std::size_t i = 0; i < nb_loop; ++i) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(n), static_cast<int>(n),
                    static_cast<int>(n), 1.0, A, static_cast<int>(n), B,
                    static_cast<int>(n), 0.0, C, static_cast<int>(n));
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time =
          1.0e-9 *
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      double gflops = 1.0e-9 * (2 * n * n * n * nb_loop) / time;

      std::printf("Performance for MKL:   %7.2f Gflops\n", gflops);
    }

    {
      for (std::size_t k = 0; k < n * n; ++k) {
        A[k] = 0.0;
        B[k] = 0.0;
        C[k] = 0.0;
      }

      int n_int = static_cast<int>(n);
      auto start = std::chrono::high_resolution_clock::now();
      for (std::size_t j = 0; j < nb_loop; ++j) {
        for (int i = 0; i < n_int; i++) {
          for (int k = 0; k < n_int; k++) {
            double* A_loc = A + i * n;
            double* C_loc = C + k * n;
            double coeff = B[i * n + k];
            for (int j = 0; j < n_int; j++) {
              // A[i * n + j] += B[i * n + k] * C[k * n + j];
              A_loc[j] += coeff * C_loc[j];
            }
          }
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time =
          1.0e-9 *
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      double gflops = 1.0e-9 * (2 * n * n * n * nb_loop) / time;

      std::printf("Performance for Man:   %7.2f Gflops\n", gflops);
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
  }

  return 0;
}