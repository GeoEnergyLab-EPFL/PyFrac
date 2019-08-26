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

#ifndef IL_PARTIALLU_H
#define IL_PARTIALLU_H

#include <il/Status.h>
#include <il/container/1d/Array.h>
#include <il/container/2d/Array2C.h>
#include <il/container/2d/Array2D.h>
#include <il/container/2d/LowerArray2D.h>
#include <il/container/2d/StaticArray2D.h>
#include <il/container/2d/UpperArray2D.h>
#include <il/linearAlgebra/dense/norm.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif

namespace il {

template <typename MatrixType>
class LU {};

template <il::int_t n>
class LU<il::StaticArray2D<double, n, n>> {
 private:
  il::StaticArray<lapack_int, n> ipiv_;
  il::StaticArray2D<double, n, n> lu_;

 public:
  // Computes a LU factorization of a general n0 x n1 matrix A using partial
  // pivoting with row interchanges. The factorization has the form
  //
  //  A = P.L.U
  //
  // where P is a permutation matrix, L is lower triangular with unit diagonal
  // elements, and U is upper triangular.
  LU(il::StaticArray2D<double, n, n> A, il::io_t, il::Status &status);

  // Compute the inverse of the matrix
  il::StaticArray2D<double, n, n> inverse() const;

  // Compute an approximation of the condition number
  //  double conditionNumber(il::Norm norm_type, double norm_a) const;
};

template <il::int_t n>
class LU<il::StaticArray2D<std::complex<double>, n, n>> {
 private:
  il::StaticArray<lapack_int, n> ipiv_;
  il::StaticArray2D<std::complex<double>, n, n> lu_;

 public:
  // Computes a LU factorization of a general n0 x n1 matrix A using partial
  // pivoting with row interchanges. The factorization has the form
  //
  //  A = P.L.U
  //
  // where P is a permutation matrix, L is lower triangular with unit diagonal
  // elements, and U is upper triangular.
  LU(il::StaticArray2D<std::complex<double>, n, n> A, il::io_t,
     il::Status &status);

  // Compute the inverse of the matrix
  il::StaticArray2D<std::complex<double>, n, n> inverse() const;

  // Compute an approximation of the condition number
  //  double conditionNumber(il::Norm norm_type, double norm_a) const;
};

template <il::int_t n>
LU<il::StaticArray2D<double, n, n>>::LU(il::StaticArray2D<double, n, n> A,
                                        il::io_t, il::Status &status)
    : ipiv_{}, lu_{} {
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int lapack_n = static_cast<lapack_int>(n);
  il::StaticArray<lapack_int, n> ipiv{};
  const lapack_int lapack_error = LAPACKE_dgetrf(
      layout, lapack_n, lapack_n, A.Data(), lapack_n, ipiv.Data());
  IL_EXPECT_FAST(lapack_error >= 0);

  if (lapack_error == 0) {
    status.SetOk();
    ipiv_ = ipiv;
    lu_ = A;
  } else {
    status.SetError(il::Error::MatrixSingular);
    IL_SET_SOURCE(status);
    status.SetInfo("rank", il::int_t{lapack_error - 1});
  }
}

template <il::int_t n>
il::StaticArray2D<double, n, n> LU<il::StaticArray2D<double, n, n>>::inverse()
    const {
  il::StaticArray2D<double, n, n> inverse = lu_;
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int lapack_n = static_cast<lapack_int>(n);
  const lapack_int lapack_error =
      LAPACKE_dgetri(layout, lapack_n, inverse.Data(), lapack_n, ipiv_.data());
  IL_EXPECT_FAST(lapack_error == 0);

  return inverse;
}

template <il::int_t n>
LU<il::StaticArray2D<std::complex<double>, n, n>>::LU(
    il::StaticArray2D<std::complex<double>, n, n> A, il::io_t,
    il::Status &status)
    : ipiv_{}, lu_{} {
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int lapack_n = static_cast<lapack_int>(n);
  il::StaticArray<lapack_int, n> ipiv{};
  const lapack_int lapack_error =
      LAPACKE_zgetrf(layout, lapack_n, lapack_n,
                     reinterpret_cast<lapack_complex_double *>(A.Data()),
                     lapack_n, ipiv.Data());
  IL_EXPECT_FAST(lapack_error >= 0);

  if (lapack_error == 0) {
    status.SetOk();
    ipiv_ = ipiv;
    lu_ = A;
  } else {
    status.SetError(il::Error::MatrixSingular);
    IL_SET_SOURCE(status);
    status.SetInfo("rank", il::int_t{lapack_error - 1});
  }
}

template <il::int_t n>
il::StaticArray2D<std::complex<double>, n, n>
LU<il::StaticArray2D<std::complex<double>, n, n>>::inverse() const {
  il::StaticArray2D<std::complex<double>, n, n> inverse = lu_;
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int lapack_n = static_cast<lapack_int>(n);
  const lapack_int lapack_error =
      LAPACKE_zgetri(layout, lapack_n,
                     reinterpret_cast<lapack_complex_double *>(inverse.Data()),
                     lapack_n, ipiv_.data());
  IL_EXPECT_FAST(lapack_error == 0);

  return inverse;
}

template <>
class LU<il::Array2D<double>> {
 private:
  il::Array<lapack_int> ipiv_;
  il::Array2D<double> lu_;

 public:
  // Computes a LU factorization of a general n0 x n1 matrix A using partial
  // pivoting with row interchanges. The factorization has the form
  //
  //  A = P.L.U
  //
  // where P is a permutation matrix, L is lower triangular with unit diagonal
  // elements, and U is upper triangular.
  LU(il::Array2D<double> A, il::io_t, il::Status &status);

  // Size of the matrix
  il::int_t size(il::int_t d) const;

  // Read access to the L part of the decomposition
  const double &L(il::int_t i, il::int_t j) const;

  // Read access to the U part of the decomposition
  const double &U(il::int_t i, il::int_t j) const;

  // Solve the system of equation with one second member
  il::Array<double> solve(il::Array<double> y) const;

  // Solve the system of equation with many second member
  il::Array2D<double> solve(il::Array2D<double> y) const;

  // Compute the inverse of the matrix
  il::Array2D<double> inverse() const;

  // Compute the determinant of the matrix
  double determinant() const;

  // Compute an approximation of the condition number
  double conditionNumber(il::Norm norm_type, double norm_a) const;

  // Get the L part of the matrix
  il::LowerArray2D<double> L() const;

  // Get the U part of the matrix
  il::UpperArray2D<double> U() const;
};

inline LU<il::Array2D<double>>::LU(il::Array2D<double> A, il::io_t,
                                   il::Status &status)
    : ipiv_{}, lu_{} {
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int m = static_cast<lapack_int>(A.size(0));
  const lapack_int n = static_cast<lapack_int>(A.size(1));
  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
  il::Array<lapack_int> ipiv{A.size(0) < A.size(1) ? A.size(0) : A.size(1)};
  const lapack_int lapack_error =
      LAPACKE_dgetrf(layout, m, n, A.Data(), lda, ipiv.Data());

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
    ipiv_ = std::move(ipiv);
    lu_ = std::move(A);
  } else {
    status.SetError(il::Error::MatrixSingular);
    IL_SET_SOURCE(status);
    status.SetInfo("rank", il::int_t{lapack_error - 1});
  }
}

inline il::int_t LU<il::Array2D<double>>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return lu_.size(d);
}

inline il::Array<double> LU<il::Array2D<double>>::solve(
    il::Array<double> y) const {
  IL_EXPECT_FAST(lu_.size(0) == lu_.size(1));

  const int layout = LAPACK_COL_MAJOR;
  const char trans = 'N';
  const lapack_int n = static_cast<lapack_int>(lu_.size(0));
  const lapack_int nrhs = 1;
  const lapack_int lda = static_cast<lapack_int>(lu_.size(0));
  const lapack_int ldy = n;
  const lapack_int lapack_error = LAPACKE_dgetrs(
      layout, trans, n, nrhs, lu_.data(), lda, ipiv_.data(), y.Data(), ldy);
  IL_EXPECT_FAST(lapack_error == 0);

  return y;
}

inline il::Array2D<double> LU<il::Array2D<double>>::solve(
    il::Array2D<double> y) const {
  IL_EXPECT_FAST(lu_.size(0) == lu_.size(1));
  IL_EXPECT_FAST(lu_.size(0) == y.size(0));

  const int layout = LAPACK_COL_MAJOR;
  const char trans = 'N';
  const lapack_int n = static_cast<lapack_int>(lu_.size(0));
  const lapack_int nrhs = static_cast<lapack_int>(y.size(1));
  const lapack_int lda = static_cast<lapack_int>(lu_.stride(1));
  const lapack_int ldy = n;
  const lapack_int lapack_error = LAPACKE_dgetrs(
      layout, trans, n, nrhs, lu_.data(), lda, ipiv_.data(), y.Data(), ldy);
  IL_EXPECT_FAST(lapack_error == 0);

  return y;
}

inline il::Array2D<double> LU<il::Array2D<double>>::inverse() const {
  IL_EXPECT_FAST(lu_.size(0) == lu_.size(1));

  il::Array2D<double> inverse{lu_};
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int n = static_cast<lapack_int>(inverse.size(0));
  const lapack_int lda = static_cast<lapack_int>(inverse.stride(1));
  const lapack_int lapack_error =
      LAPACKE_dgetri(layout, n, inverse.Data(), lda, ipiv_.data());
  IL_EXPECT_FAST(lapack_error == 0);

  return inverse;
}

inline double LU<il::Array2D<double>>::determinant() const {
  IL_EXPECT_FAST(lu_.size(0) == lu_.size(1));

  double det = 1.0;
  for (il::int_t i = 0; i < lu_.size(0); ++i) {
    det *= lu_(i, i);
  }

  return det;
}

inline double LU<il::Array2D<double>>::conditionNumber(il::Norm norm_type,
                                                       double norm_a) const {
  IL_EXPECT_FAST(lu_.size(0) == lu_.size(1));
  IL_EXPECT_FAST(norm_type == il::Norm::L1 || norm_type == il::Norm::Linf);

  const int layout = LAPACK_COL_MAJOR;
  const char lapack_norm = (norm_type == il::Norm::L1) ? '1' : 'I';
  const lapack_int n = static_cast<lapack_int>(lu_.size(0));
  const lapack_int lda = static_cast<lapack_int>(lu_.stride(1));
  double rcond;
  const lapack_int lapack_error =
      LAPACKE_dgecon(layout, lapack_norm, n, lu_.data(), lda, norm_a, &rcond);
  IL_EXPECT_FAST(lapack_error == 0);

  return 1.0 / rcond;
}

inline const double &LU<il::Array2D<double>>::L(il::int_t i,
                                                il::int_t j) const {
  IL_EXPECT_MEDIUM(j < i);
  return lu_(i, j);
}

inline const double &LU<il::Array2D<double>>::U(il::int_t i,
                                                il::int_t j) const {
  IL_EXPECT_MEDIUM(j >= i);
  return lu_(i, j);
}

/*
template <> class LU<il::Array2C<double>> {
private:
  il::Array<lapack_int> ipiv_;
  il::Array2C<double> lu_;

public:
  // Computes a LU factorization of a general n0 x n1 matrix A using partial
  // pivoting with row interchanges. The factorization has the form
  //
  //  A = P.L.U
  //
  // where P is a permutation matrix, L is lower triangular with unit diagonal
  // elements, and U is upper triangular.
  LU(il::Array2C<double> A, il::io_t, il::Status &status);
};

template <>
LU<il::Array2C<double>>::LU(il::Array2C<double> A, il::io_t, il::Status &status)
    : ipiv_{}, lu_{} {
  IL_EXPECT_FAST(A.size(0) == A.size(1));

  const int layout = LAPACK_ROW_MAJOR;
  const lapack_int m = static_cast<lapack_int>(A.size(0));
  const lapack_int n = static_cast<lapack_int>(A.size(1));
  const lapack_int lda = static_cast<lapack_int>(A.stride(0));
  il::Array<lapack_int> ipiv{A.size(0) < A.size(1) ? A.size(0) : A.size(1)};
  const lapack_int lapack_error =
      LAPACKE_dgetrf(layout, m, n, A.data(), lda, ipiv.data());

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
    ipiv_ = std::move(ipiv);
    lu_ = std::move(A);
  } else {
    status.SetError(il::Error::MatrixSingular);
    IL_SET_SOURCE(status);
    status.SetInfo("rank", il::int_t{lapack_error - 1});
  }
}

template <> il::LowerArray2D<double> LU<il::Array2D<double>>::L() const {
  IL_EXPECT_FAST(size(0) == size(1));

  const il::int_t n = size(0);
  il::LowerArray2D<double> L{n};
  for (il::int_t i1 = 0; i1 < n; ++i1) {
    L(i1, i1) = 1.0;
    for (il::int_t i0 = i1 + 1; i0 < n; ++i0) {
      L(i0, i1) = lu_(i0, i1);
    }
  }
  return L;
}

template <> il::UpperArray2D<double> LU<il::Array2D<double>>::U() const {
  IL_EXPECT_FAST(size(0) == size(1));

  const il::int_t n = size(0);
  il::UpperArray2D<double> U{n};
  for (il::int_t i1 = 0; i1 < n; ++i1) {
    for (il::int_t i0 = 0; i0 <= i1; ++i0) {
      U(i0, i1) = lu_(i0, i1);
    }
  }
  return U;
}

// template <>
// class LU<il::Array2C<double>> {
// private:
//  il::Array<lapack_int> ipiv_;
//  il::Array2C<double> lu_;
//
// public:
//  // Computes a LU factorization of a general n0 x n1 matrix A using partial
//  // pivoting with row interchanges. The factorization has the form
//  //
//  //  A = P.L.U
//  //
//  // where P is a permutation matrix, L is lower triangular with unit diagonal
//  // elements, and U is upper triangular.
//  LU(il::Array2C<double> A, il::io_t, il::Status& status);
//};
//
// template <>
// LU<il::Array2C<double>>::LU(il::Array2C<double> A, il::io_t, il::Status&
// status)
//    : ipiv_{}, lu_{} {
//  IL_EXPECT_FAST(A.size(0) == A.size(1));
//
//  const int layout = LAPACK_ROW_MAJOR;
//  const lapack_int m = static_cast<lapack_int>(A.size(0));
//  const lapack_int n = static_cast<lapack_int>(A.size(1));
//  const lapack_int lda = static_cast<lapack_int>(A.stride(0));
//  il::Array<lapack_int> ipiv{A.size(0) < A.size(1) ? A.size(0) : A.size(1)};
//  const lapack_int lapack_error =
//      LAPACKE_dgetrf(layout, m, n, A.data(), lda, ipiv.data());
//
//  IL_EXPECT_FAST(lapack_error >= 0);
//  if (lapack_error == 0) {
//    status.SetOk();
//    ipiv_ = std::move(ipiv);
//    lu_ = std::move(A);
//  } else {
//    status.SetError(il::Error::MatrixSingular);
//    IL_SET_SOURCE(status);
//    status.SetInfo("rank", il::int_t{lapack_error - 1});
//  }
//}
 */

}  // namespace il

#endif  // IL_PARTIALLU_H
