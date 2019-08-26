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

#ifndef IL_CUDA_BLAS_H
#define IL_CUDA_BLAS_H

#include <il/container/cuda/1d/CudaArray.h>
#include <il/container/cuda/2d/CudaArray2D.h>
#include <il/linearAlgebra/cuda/dense/blas/CublasHandle.h>

namespace il {

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 1
////////////////////////////////////////////////////////////////////////////////

inline void blas(float alpha, const il::CudaArray<float>& x, float beta,
                 il::io_t, il::CudaArray<float>& y, il::CublasHandle& handle) {
  IL_EXPECT_FAST(x.size() == y.size());

  const int n = static_cast<int>(x.size());
  const int incx = 1;
  const int incy = 1;
  cublasStatus_t status;
  if (beta == 1.0f) {
    status =
        cublasSaxpy(handle.handle(), n, &alpha, x.data(), incx, y.data(), incy);
    IL_EXPECT_FAST(status == CUBLAS_STATUS_SUCCESS);
  } else {
    const float beta_minus_one = beta - 1.0f;
    status = cublasSaxpy(handle.handle(), n, &beta_minus_one, y.data(), incx,
                         y.data(), incy);

    IL_EXPECT_FAST(status == CUBLAS_STATUS_SUCCESS);
    status =
        cublasSaxpy(handle.handle(), n, &alpha, x.data(), incx, y.data(), incy);
    IL_EXPECT_FAST(status == CUBLAS_STATUS_SUCCESS);
  }
}

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 3
////////////////////////////////////////////////////////////////////////////////

inline void blas(float alpha, const il::CudaArray2D<float>& A,
                 const il::CudaArray2D<float>& B, float beta, il::io_t,
                 il::CudaArray2D<float>& C, il::CublasHandle& handle) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(A.size(0) == C.size(0));
  IL_EXPECT_FAST(B.size(1) == C.size(1));

  IL_EXPECT_FAST(A.size(1) == A.size(0));
  IL_EXPECT_FAST(B.size(1) == A.size(0));

  const int n = A.size(0);
  cublasSgemm(handle.handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
              A.data(), n, B.data(), n, &beta, C.data(), n);
}

inline void blas(double alpha, const il::CudaArray2D<double>& A,
                 const il::CudaArray2D<double>& B, double beta, il::io_t,
                 il::CudaArray2D<double>& C, il::CublasHandle& handle) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(A.size(0) == C.size(0));
  IL_EXPECT_FAST(B.size(1) == C.size(1));

  IL_EXPECT_FAST(A.size(1) == A.size(0));
  IL_EXPECT_FAST(B.size(1) == A.size(0));

  const int n = A.size(0);
  cublasDgemm(handle.handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
              A.data(), n, B.data(), n, &beta, C.data(), n);
}
}  // namespace il

#endif  // IL_CUDA_BLAS_H
