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

#include <chrono>
#include <cstdio>

//#include <blaze/math/DynamicMatrix.h>
#include <il/linearAlgebra/dense/blas/blas.h>

//#include "mkl.h"

int main() {
  il::StaticArray3D<double, 6, 9, 3> A{};
  il::StaticArray<double, 9> B{};

  auto C = il::dot_1_0(A, B);

  //  for (int i = 0; i < 15; ++i) {
  //    std::size_t n{3000};
  //    blaze::DynamicMatrix<double> A(n, n, 1.0);
  //    blaze::DynamicMatrix<double> B(n, n, 2.0);
  //    blaze::DynamicMatrix<double> C(n, n, 0.0);
  //
  //    auto begin = std::chrono::high_resolution_clock::now();
  //    C = A * B;
  //    auto end = std::chrono::high_resolution_clock::now();
  //    double time{
  //        1.0e-9 *
  //            std::chrono::duration_cast<std::chrono::nanoseconds>(end -
  //            begin)
  //                .count()};
  //    double gflops{1.0e-9 * 2 * n * n * n / time};
  //
  //    std::printf("Performance of Blaze: %7.2f Gflops\n", gflops);
  //  }

  // for (int i = 0; i < 15; ++i) {
  // int n = 3000;
  // double* A = new double[n * n];
  // double* B = new double[n * n];
  // double* C = new double[n * n];
  // for (int k = 0; k < n * n; ++k) {
  // A[k] = 0.0;
  // B[k] = 0.0;
  // C[k] = 0.0;
  //}

  // auto begin = std::chrono::high_resolution_clock::now();
  // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n,
  // B, n, 0.0, C, n);
  // auto end = std::chrono::high_resolution_clock::now();
  // double time{
  // 1.0e-9 *
  // std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
  //.count()};
  // double gflops{1.0e-9 * 2 * n * n * n / time};

  // std::printf("Performance of MKL  : %7.2f Gflops\n", gflops);

  // delete[] A;
  // delete[] B;
  // delete[] C;
  //}

  return 0;
}
