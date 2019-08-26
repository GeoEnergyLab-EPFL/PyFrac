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

#ifndef IL_LINEAR_SOLVE_H
#define IL_LINEAR_SOLVE_H

#include <il/Status.h>

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>
#include <il/BandArray2C.h>
#include <il/TriDiagonal.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <lapacke.h>
#endif

namespace il {

#ifdef IL_BLAS
inline il::Array<double> linearSolve(il::Array2D<double> A, il::Array<double> y,
                                     il::io_t, il::Status &status) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));
  IL_EXPECT_FAST(y.size() == A.size(0));

  il::Array<lapack_int> ipiv{A.size(0)};

  const int layout{LAPACK_COL_MAJOR};
  const lapack_int n{static_cast<lapack_int>(A.size(0))};
  const lapack_int nrhs = 1;
  const lapack_int lda{static_cast<lapack_int>(A.stride(1))};
  const lapack_int ldx{n};
  const lapack_int lapack_error{LAPACKE_dgesv(layout, n, nrhs, A.Data(), lda,
                                              ipiv.Data(), y.Data(), ldx)};

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
  } else {
    status.SetError(il::Error::MatrixSingular);
    IL_SET_SOURCE(status);
  }

  return y;
}

inline il::Array<double> linearSolve(il::Array2C<double> A, il::Array<double> y,
                                     il::io_t, il::Status &status) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));
  IL_EXPECT_FAST(y.size() == A.size(1));

  il::Array<lapack_int> ipiv{A.size(0)};

  const int layout{LAPACK_ROW_MAJOR};
  const lapack_int n{static_cast<lapack_int>(A.size(0))};
  const lapack_int nrhs = 1;
  const lapack_int lda{static_cast<lapack_int>(A.stride(0))};
  const lapack_int ldx = 1;
  const lapack_int lapack_error{LAPACKE_dgesv(layout, n, nrhs, A.Data(), lda,
                                              ipiv.Data(), y.Data(), ldx)};

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
  } else {
    status.SetError(il::Error::MatrixSingular);
    IL_SET_SOURCE(status);
  }

  return y;
}

inline il::Array<double> linearSolve(il::BandArray2C<double> A,
                                     il::Array<double> y, il::io_t,
                                     il::Status &status) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));
  IL_EXPECT_FAST(y.size() == A.size(1));
  IL_EXPECT_FAST(A.capacityRight() >= A.widthLeft() + A.widthRight());

  il::Array<lapack_int> ipiv{A.size(1)};

  const int layout{LAPACK_ROW_MAJOR};
  const lapack_int n{static_cast<lapack_int>(A.size(1))};
  const lapack_int kl{static_cast<lapack_int>(A.widthLeft())};
  const lapack_int ku{static_cast<lapack_int>(A.widthRight())};
  const lapack_int nrhs = 1;
  const lapack_int ldab{
      static_cast<lapack_int>(A.widthLeft() + 1 + A.capacityRight())};
  const lapack_int ldb = 1;
  const lapack_int lapack_error{LAPACKE_dgbsv(
      layout, n, kl, ku, nrhs, A.Data(), ldab, ipiv.Data(), y.Data(), ldb)};

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
  } else {
    status.SetError(il::Error::MatrixSingular);
    IL_SET_SOURCE(status);
    status.SetInfo("rank", il::int_t{lapack_error - 1});
  }

  return y;
}

inline il::Array<double> linearSolve(il::TriDiagonal<double> A,
                                     il::Array<double> y, il::io_t,
                                     il::Status &status) {
  IL_EXPECT_FAST(A.size() == y.size());

  const int layout{LAPACK_ROW_MAJOR};
  const lapack_int n{static_cast<lapack_int>(A.size())};
  const lapack_int nrhs = 1;
  const lapack_int ldb = 1;
  const lapack_int lapack_error{LAPACKE_dgtsv(layout, n, nrhs, A.LowerData(),
                                              A.DiagonalData(), A.UpperData(),
                                              y.Data(), ldb)};

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
  } else {
    status.SetError(il::Error::MatrixSingular);
    IL_SET_SOURCE(status);
  }

  return y;
}

#endif
}  // namespace il

#endif  // IL_LINEAR_SOLVE_H