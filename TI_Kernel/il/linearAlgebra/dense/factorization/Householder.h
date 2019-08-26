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

#ifndef IL_HOUSEHOLDER_H
#define IL_HOUSEHOLDER_H

#include <il/Status.h>
#include <il/container/1d/Array.h>
#include <il/container/2d/Array2D.h>
#include <il/linearAlgebra/dense/factorization/norm.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif

namespace il {

template <typename MatrixType>
class Householder {};

template <>
class Householder<il::Array2D<double>> {
 private:
  il::Array<double> reflexion_;
  il::Array2D<double> house_holder_;

 public:
  // Computes a QR factorization of a general n0 x n1 matrix A
  //
  //  A = Q.R
  //
  Householder(il::Array2D<double> A);

  // Size of the matrix
  //  il::int_t size(il::int_t d) const;

  // Solve the system of equation with one second member
  //  il::Array<double> solve(il::Array<double> y) const;
};

Householder<il::Array2D<double>>::Householder(il::Array2D<double> A)
    : house_holder_{} {
  IL_EXPECT_FAST(A.size(0) > 0);
  IL_EXPECT_FAST(A.size(1) > 0);

  const int layout = LAPACK_COL_MAJOR;
  const char trans = 'N';
  const lapack_int m = static_cast<lapack_int>(A.size(0));
  const lapack_int n = static_cast<lapack_int>(A.size(1));
  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
  il::Array<double> reflexion{m < n ? m : n};
  const lapack_int lapack_error =
      LAPACKE_dgeqrf(layout, m, n, A.data(), lda, reflexion.data());
  IL_EXPECT_FAST(lapack_error >= 0);
}

}  // namespace il

#endif  // IL_HOUSEHOLDER_H
