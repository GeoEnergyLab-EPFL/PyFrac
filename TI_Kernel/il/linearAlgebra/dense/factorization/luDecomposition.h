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

#ifndef IL_LU_DECOMPOSITION_H
#define IL_LU_DECOMPOSITION_H

#include <complex>

#include <il/Array2CView.h>
#include <il/Array2DView.h>
#include <il/ArrayView.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif

namespace il {

void luDecomposition(il::io_t, il::ArrayEdit<int> pivot,
                     il::Array2DEdit<double> A) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(A.size(0) == pivot.size());

  const int layout = LAPACK_COL_MAJOR;
  const lapack_int m = static_cast<lapack_int>(A.size(0));
  const lapack_int n = static_cast<lapack_int>(A.size(1));
  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
  const lapack_int lapack_error =
      LAPACKE_dgetrf(layout, m, n, A.Data(), lda, pivot.Data());

  IL_EXPECT_FAST(lapack_error == 0);
}

void luDecomposition(il::io_t, il::ArrayEdit<int> pivot,
                     il::Array2DEdit<std::complex<double>> A) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(A.size(0) == pivot.size());

  const int layout = LAPACK_COL_MAJOR;
  const lapack_int m = static_cast<lapack_int>(A.size(0));
  const lapack_int n = static_cast<lapack_int>(A.size(1));
  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
  const lapack_int lapack_error = LAPACKE_zgetrf(
      layout, m, n, reinterpret_cast<lapack_complex_double*>(A.Data()), lda,
      pivot.Data());

  IL_EXPECT_FAST(lapack_error == 0);
}

}  // namespace il

#endif  // IL_LU_DECOMPOSITION_H
