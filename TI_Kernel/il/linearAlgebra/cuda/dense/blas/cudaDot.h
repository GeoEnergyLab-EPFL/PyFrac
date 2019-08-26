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

#ifndef IL_CUDA_DOT_H
#define IL_CUDA_DOT_H

#include <il/container/cuda/1d/CudaArray.h>
#include <il/container/cuda/2d/CudaArray2D.h>
#include <il/linearAlgebra/cuda/dense/blas/CublasHandle.h>

#include <cublas_v2.h>

namespace il {

inline float dot(const il::CudaArray<float>& x, const il::CudaArray<float>& y,
                 il::io_t, il::CublasHandle& handle) {
  IL_EXPECT_FAST(x.size() == y.size());

  const int n = x.size();
  const int incx = 1;
  const int incy = 1;
  float ans;
  cublasStatus_t status =
      cublasSdot(handle.handle(), n, x.data(), incx, y.data(), incy, &ans);

  return ans;
}

}  // namespace il

#endif  // IL_CUDA_DOT_H
