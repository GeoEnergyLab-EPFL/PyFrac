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

#ifndef IL_SINGULAR_H
#define IL_SINGULAR_H

#include <complex>

#include <il/Status.h>
#include <il/container/1d/Array.h>
#include <il/container/1d/StaticArray.h>
#include <il/container/2d/Array2D.h>
#include <il/container/2d/StaticArray2D.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif

namespace il {

template <il::int_t n>
il::StaticArray<double, n> singularValues(il::StaticArray2D<double, n, n> A,
                                          il::io_t, il::Status &status) {
  static_assert(n > 0, "il::singularValues<n>: n must be > 0");

  il::StaticArray<double, n> d{};
  il::StaticArray<double, (n == 1) ? 1 : (n - 1)> e{};
  il::StaticArray<double, n> tauq{};
  il::StaticArray<double, n> taup{};
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int lapack_n = static_cast<lapack_int>(n);
  const lapack_int lapack_error_0 =
      LAPACKE_dgebrd(layout, lapack_n, lapack_n, A.Data(), lapack_n, d.Data(),
                     e.Data(), tauq.Data(), taup.Data());
  IL_EXPECT_FAST(lapack_error_0 >= 0);

  const char uplo = 'U';
  const lapack_int ncvt = 0;
  const lapack_int ldvt = 1;
  const lapack_int nru = 0;
  const lapack_int ldu = 1;
  const lapack_int ncc = 0;
  const lapack_int ldc = 1;
  il::StaticArray<double, ldvt * ncvt> vt{};
  il::StaticArray<double, ldu * n> u{};
  il::StaticArray<double, 1> c{};
  const lapack_int lapack_error_1 =
      LAPACKE_dbdsqr(layout, uplo, lapack_n, ncvt, nru, ncc, d.Data(), e.Data(),
                     vt.Data(), ldvt, u.Data(), ldu, c.Data(), ldc);
  IL_EXPECT_FAST(lapack_error_1 >= 0);

  if (lapack_error_0 == 0 && lapack_error_1 == 0) {
    status.SetOk();
  } else {
    status.SetError(il::Error::Undefined);
  }
  return d;
}

template <il::int_t n>
il::StaticArray<double, n> singularValues(
    il::StaticArray2D<std::complex<double>, n, n> A, il::io_t,
    il::Status &status) {
  static_assert(n > 0, "il::singularValues<n>: n must be > 0");

  il::StaticArray<double, n> d{};
  il::StaticArray<double, (n == 1) ? 1 : (n - 1)> e{};
  il::StaticArray<std::complex<double>, n> tauq{};
  il::StaticArray<std::complex<double>, n> taup{};
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int lapack_n = static_cast<lapack_int>(n);
  const lapack_int lapack_error_0 = LAPACKE_zgebrd(
      layout, lapack_n, lapack_n,
      reinterpret_cast<lapack_complex_double *>(A.Data()), lapack_n, d.Data(),
      e.Data(), reinterpret_cast<lapack_complex_double *>(tauq.Data()),
      reinterpret_cast<lapack_complex_double *>(taup.Data()));
  IL_EXPECT_FAST(lapack_error_0 >= 0);

  const char uplo = 'U';
  const lapack_int ncvt = 0;
  const lapack_int ldvt = 1;
  const lapack_int nru = 0;
  const lapack_int ldu = 1;
  const lapack_int ncc = 0;
  const lapack_int ldc = 1;
  il::StaticArray<std::complex<double>, ldvt * ncvt> vt{};
  il::StaticArray<std::complex<double>, ldu * n> u{};
  il::StaticArray<std::complex<double>, 1> c{};
  const lapack_int lapack_error_1 =
      LAPACKE_zbdsqr(layout, uplo, lapack_n, ncvt, nru, ncc, d.Data(), e.Data(),
                     reinterpret_cast<lapack_complex_double *>(vt.Data()), ldvt,
                     reinterpret_cast<lapack_complex_double *>(u.Data()), ldu,
                     reinterpret_cast<lapack_complex_double *>(c.Data()), ldc);
  IL_EXPECT_FAST(lapack_error_1 >= 0);

  if (lapack_error_0 == 0 && lapack_error_1 == 0) {
    status.SetOk();
  } else {
    status.SetError(il::Error::Undefined);
  }
  return d;
}

template <typename MatrixType>
class Singular {};

template <il::int_t n>
class Singular<il::StaticArray2D<double, n, n>> {
 private:
  il::StaticArray<double, n> singular_value_;

 public:
  // Computes singular values of A
  Singular(const il::StaticArray2D<double, n, n> &A, il::io_t,
           il::Status &status);
};

template <il::int_t n>
Singular<il::StaticArray2D<double, n, n>>::Singular(
    const il::StaticArray2D<double, n, n> &A, il::io_t, il::Status &status)
    : singular_value_{} {
  static_assert(n > 0, "il::StaticArray<T, n>: n must be > 0");

  il::StaticArray2D<double, n, n> A_local = A;

  const int layout = LAPACK_COL_MAJOR;
  const lapack_int lapack_n = static_cast<lapack_int>(n);

  il::StaticArray<double, n> d{n};
  il::StaticArray<double, (n == 1) ? 1 : (n - 1)> e{};
  il::StaticArray<double, n> tauq{};
  il::StaticArray<double, n> taup{};
  lapack_int lapack_error =
      LAPACKE_dgebrd(layout, lapack_n, lapack_n, A_local.data(), lapack_n,
                     d.data(), e.data(), tauq.data(), taup.data());
  IL_EXPECT_FAST(lapack_error >= 0);

  const char uplo = 'U';
  const lapack_int ncvt = 0;
  const lapack_int ldvt = 1;
  const lapack_int nru = 0;
  const lapack_int ldu = 1;
  const lapack_int ncc = 0;
  const lapack_int ldc = 1;
  il::Array<double> vt{ldvt * ncvt};
  il::Array<double> u{ldu * n};
  il::Array<double> c{1};
  lapack_error =
      LAPACKE_dbdsqr(layout, uplo, lapack_n, ncvt, nru, ncc, d.data(), e.data(),
                     vt.data(), ldvt, u.data(), ldu, c.data(), ldc);

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
    singular_value_ = std::move(d);
  } else {
    status.SetError(il::Error::Undefined);
  }
}

// template <> class Singular<il::Array2D<double>> {
// private:
//  il::Array<double> singular_value_;
//
// public:
//  // Computes singular values of A
//  Singular(il::Array2D<double> A, il::io_t, il::Status &status);
//};
//
// template <>
// Singular<il::Array2D<double>>::Singular(il::Array2D<double> A, il::io_t,
//                                        il::Status &status)
//    : singular_value_{} {
//  IL_EXPECT_FAST(A.size(0) > 0);
//  IL_EXPECT_FAST(A.size(1) > 0);
//  IL_EXPECT_FAST(A.size(0) == A.size(1));
//
//  const int layout = LAPACK_COL_MAJOR;
//  const lapack_int m = static_cast<lapack_int>(A.size(0));
//  const lapack_int n = static_cast<lapack_int>(A.size(1));
//  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
//  const il::int_t min_mn = m < n ? m : n;
//  il::Array<double> d{min_mn};
//  il::Array<double> e{(min_mn == 1) ? 1 : (min_mn - 1)};
//  il::Array<double> tauq{min_mn};
//  il::Array<double> taup{min_mn};
//  lapack_int lapack_error =
//      LAPACKE_dgebrd(layout, m, n, A.Data(), lda, d.Data(), e.Data(),
//                     tauq.Data(), taup.Data());
//  IL_EXPECT_FAST(lapack_error >= 0);
//
//  const char uplo = (m >= n) ? 'U' : 'L';
//  const lapack_int ncvt = 0;
//  const lapack_int ldvt = 1;
//  const lapack_int nru = 0;
//  const lapack_int ldu = 1;
//  const lapack_int ncc = 0; // No matrix C is upplied
//  const lapack_int ldc = 1; // No matrix C is upplied
//  il::Array<double> vt{ldvt * ncvt};
//  il::Array<double> u{ldu * n};
//  il::Array<double> c{1}; // Should be useless
//  lapack_error =
//      LAPACKE_dbdsqr(layout, uplo, n, ncvt, nru, ncc, d.Data(), e.Data(),
//                     vt.Data(), ldvt, u.Data(), ldu, c.Data(), ldc);
//
//  IL_EXPECT_FAST(lapack_error >= 0);
//  if (lapack_error == 0) {
//    status.SetOk();
//    singular_value_ = std::move(d);
//  } else {
//    status.SetError(il::Error::Undefined);
//  }
//}
}  // namespace il

#endif  // IL_SINGULAR_H
