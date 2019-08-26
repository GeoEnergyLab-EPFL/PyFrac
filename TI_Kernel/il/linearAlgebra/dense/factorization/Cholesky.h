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

#ifndef IL_CHOLESKY_H
#define IL_CHOLESKY_H

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/LowerArray2D.h>
#include <il/Status.h>
#include <il/linearAlgebra/dense/norm.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif

namespace il {

template <typename MatrixType>
class Cholesky {};

template <>
class Cholesky<il::Array2D<double>> {
 private:
  il::Array2D<double> l_;

 public:
  // Computes a Cholesky factorization of a real symmetric positive-definite
  // matrix. The factorization has the form
  //
  //   A = L.L^t
  //
  // where L is a lower triangular matrix.
  Cholesky(il::Array2D<double> A, il::io_t, il::Status& status);

  // Size of the matrix
  il::int_t size(il::int_t d) const;

  // Solve the system of equation with one second member
  il::Array<double> solve(il::Array<double> y) const;

  // Compute the inverse of the matrix
  il::Array2D<double> inverse() const;

  // Compute an approximation of the condition number
  double conditionNumber(il::Norm norm_type, double norm_a) const;
};

Cholesky<il::Array2D<double>>::Cholesky(il::Array2D<double> A, il::io_t,
                                        il::Status& status)
    : l_{} {
  IL_EXPECT_FAST(A.size(0) == A.size(1));

  const int layout = LAPACK_COL_MAJOR;
  const char uplo = 'L';
  const lapack_int n = static_cast<lapack_int>(A.size(0));
  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
  const lapack_int lapack_error =
      LAPACKE_dpotrf(layout, uplo, n, A.data(), lda);
  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
    l_ = std::move(A);
  } else {
    status.SetError(il::Error::FloatingPointNegative);
  }
}

il::int_t Cholesky<il::Array2D<double>>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));

  return l_.size(d);
}

il::Array<double> Cholesky<il::Array2D<double>>::solve(
    il::Array<double> y) const {
  IL_EXPECT_FAST(l_.size(0) == y.size());

  const int layout = LAPACK_COL_MAJOR;
  const char uplo = 'L';
  const lapack_int n = static_cast<lapack_int>(l_.size(0));
  const lapack_int nrhs = 1;
  const lapack_int lda = static_cast<lapack_int>(l_.stride(1));
  const lapack_int ldy = n;
  const lapack_int lapack_error =
      LAPACKE_dpotrs(layout, uplo, n, nrhs, l_.data(), lda, y.data(), ldy);
  IL_EXPECT_FAST(lapack_error == 0);

  return y;
}

il::Array2D<double> Cholesky<il::Array2D<double>>::inverse() const {
  il::Array2D<double> inverse{l_};
  const int layout = LAPACK_COL_MAJOR;
  const char uplo = 'L';
  const lapack_int n = static_cast<lapack_int>(inverse.size(0));
  const lapack_int lda = static_cast<lapack_int>(inverse.stride(1));
  const lapack_int lapack_error =
      LAPACKE_dpotri(layout, uplo, n, inverse.data(), lda);
  IL_EXPECT_FAST(lapack_error == 0);

  return inverse;
}

double Cholesky<il::Array2D<double>>::conditionNumber(il::Norm norm_type,
                                                      double norm_a) const {
  IL_EXPECT_FAST(norm_type == il::Norm::L1 || norm_type == il::Norm::Linf);

  const int layout = LAPACK_COL_MAJOR;
  const char uplo = 'L';
  const char lapack_norm = (norm_type == il::Norm::L1) ? '1' : 'I';
  const lapack_int n = static_cast<lapack_int>(l_.size(0));
  const lapack_int lda = static_cast<lapack_int>(l_.stride(1));
  double rcond;
  const lapack_int lapack_error =
      LAPACKE_dpocon(layout, lapack_norm, n, l_.data(), lda, norm_a, &rcond);
  IL_EXPECT_FAST(lapack_error == 0);

  return 1.0 / rcond;
}

template <>
class Cholesky<LowerArray2D<double>> {
 private:
  il::LowerArray2D<double> l_;

 public:
  Cholesky(il::LowerArray2D<double> A, il::io_t, il::Status& status);
};

Cholesky<LowerArray2D<double>>::Cholesky(il::LowerArray2D<double> A, il::io_t,
                                         il::Status& status)
    : l_{} {
  const int layout = LAPACK_COL_MAJOR;
  const char uplo = 'L';
  const lapack_int n = static_cast<lapack_int>(A.size());
  const lapack_int lapack_error = LAPACKE_dpptrf(layout, uplo, n, A.Data());
  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
    l_ = std::move(A);
  } else {
    status.SetError(il::Error::FloatingPointNonPositive);
  }
}

}  // namespace il

#endif  // IL_CHOLESKY_H
