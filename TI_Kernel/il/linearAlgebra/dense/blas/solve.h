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

#ifndef IL_SOLVE_H
#define IL_SOLVE_H

#include <complex>

#include <il/Array2CView.h>
#include <il/Array2DView.h>
#include <il/ArrayView.h>
#include <il/linearAlgebra/Matrix.h>

#include <il/linearAlgebra/dense/blas/blas_config.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif

namespace il {

inline void solve(il::ArrayView<lapack_int> P, il::io_t,
                  il::Array2DEdit<double> Y) {
  IL_EXPECT_MEDIUM(P.size() == Y.size(0));

  const int layout = LAPACK_COL_MAJOR;
  const lapack_int n = static_cast<lapack_int>(Y.size(1));
  const lapack_int lda = static_cast<lapack_int>(Y.stride(1));
  const lapack_int k1 = 1;
  const lapack_int k2 = P.size();
  const lapack_int incx = 1;
  const lapack_int lapack_error =
      LAPACKE_dlaswp(layout, n, Y.Data(), lda, k1, k2, P.data(), incx);
  IL_EXPECT_MEDIUM(lapack_error == 0);
}

inline void solve(il::Array2DView<double> A, il::MatrixType type, il::io_t,
                  il::Array2DEdit<double> Y) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(Y.size(0) == A.size(0));
  IL_EXPECT_FAST(type == il::MatrixType::LowerUnit ||
                 type == il::MatrixType::UpperNonUnit);

  if (type == il::MatrixType::LowerUnit) {
    const IL_CBLAS_LAYOUT layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasLower;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasUnit;
    const MKL_INT m = static_cast<MKL_INT>(Y.size(0));
    const MKL_INT n = static_cast<MKL_INT>(Y.size(1));
    const double alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(Y.stride(1));
    cblas_dtrsm(layout, side, uplo, transa, diag, m, n, alpha, A.data(), lda,
                Y.Data(), ldb);
  } else if (type == il::MatrixType::UpperNonUnit) {
    const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasUpper;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasNonUnit;
    const MKL_INT m = static_cast<MKL_INT>(Y.size(0));
    const MKL_INT n = static_cast<MKL_INT>(Y.size(1));
    const double alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(Y.stride(1));
    cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
                lda, Y.Data(), ldb);
  }
}

inline void solve(il::ArrayView<int> pivot, il::Array2DView<double> A,
                  il::MatrixType type, il::io_t, il::Array2DEdit<double> B) {
  IL_EXPECT_MEDIUM(pivot.size() == A.size(0));
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(A.size(0) == B.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::LowerUnit);

  {
    const int layout = LAPACK_COL_MAJOR;
    const lapack_int n = static_cast<lapack_int>(B.size(1));
    const lapack_int lda = static_cast<lapack_int>(B.stride(1));
    const lapack_int k1 = 1;
    const lapack_int k2 = pivot.size();
    const lapack_int incx = 1;
    const lapack_int lapack_error =
        LAPACKE_dlaswp(layout, n, B.Data(), lda, k1, k2, pivot.data(), incx);
    IL_EXPECT_MEDIUM(lapack_error == 0);
  }

  {
    const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasLower;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasUnit;
    const MKL_INT m = static_cast<MKL_INT>(B.size(0));
    const MKL_INT n = static_cast<MKL_INT>(B.size(1));
    const double alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(B.stride(1));
    cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
                lda, B.Data(), ldb);
  }
}

inline void solve(il::Array2DView<double> A, il::MatrixType type, il::io_t,
                  il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(x.size() == A.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::UpperNonUnit);

  const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
  const CBLAS_SIDE side = CblasLeft;
  const CBLAS_UPLO uplo = CblasUpper;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_DIAG diag = CblasNonUnit;
  const MKL_INT m = static_cast<MKL_INT>(x.size());
  const MKL_INT n = static_cast<MKL_INT>(1);
  const double alpha = 1.0;
  const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
  const MKL_INT ldb = static_cast<MKL_INT>(x.size());
  cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
              lda, x.Data(), ldb);
}

inline void solve(il::Array2DView<double> A, il::MatrixType type, il::Dot op,
                  il::io_t, il::Array2DEdit<double> B) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));
  IL_EXPECT_FAST(B.size(0) == A.size(0));
  IL_EXPECT_FAST(type == il::MatrixType::UpperNonUnit);
  IL_EXPECT_FAST(op == il::Dot::Transpose);

  const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
  const CBLAS_SIDE side = CblasLeft;
  const CBLAS_UPLO uplo = CblasUpper;
  const CBLAS_TRANSPOSE transa = CblasTrans;
  const CBLAS_DIAG diag = CblasNonUnit;
  const MKL_INT m = static_cast<MKL_INT>(B.size(0));
  const MKL_INT n = static_cast<MKL_INT>(B.size(1));
  const double alpha = 1.0;
  const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
  const MKL_INT ldb = static_cast<MKL_INT>(B.stride(1));
  cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
              lda, B.Data(), ldb);
}

inline void solve(il::ArrayView<int> pivot, il::Array2DView<double> A,
                  il::MatrixType type, il::io_t, il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(pivot.size() == A.size(0));
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(x.size() == A.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::LowerUnit);

  {
    const int layout = LAPACK_COL_MAJOR;
    const lapack_int n = 1;
    const lapack_int lda = x.size();
    const lapack_int k1 = 1;
    const lapack_int k2 = x.size();
    const lapack_int incx = 1;
    const lapack_int lapack_error =
        LAPACKE_dlaswp(layout, n, x.Data(), lda, k1, k2, pivot.data(), incx);
    IL_EXPECT_MEDIUM(lapack_error == 0);
  }

  {
    const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasLower;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasUnit;
    const MKL_INT m = static_cast<MKL_INT>(x.size());
    const MKL_INT n = static_cast<MKL_INT>(1);
    const double alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(x.size());
    cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
                lda, x.Data(), ldb);
  }
}

inline void solveRight(il::Array2DView<double> A, il::MatrixType type, il::io_t,
                       il::Array2DEdit<double> X) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(X.size(1) == A.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::UpperNonUnit);

  const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
  const CBLAS_SIDE side = CblasRight;
  const CBLAS_UPLO uplo = CblasUpper;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_DIAG diag = CblasNonUnit;
  const MKL_INT m = static_cast<MKL_INT>(X.size(0));
  const MKL_INT n = static_cast<MKL_INT>(X.size(1));
  const double alpha = 1.0;
  const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
  const MKL_INT ldb = static_cast<MKL_INT>(X.stride(1));
  cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
              lda, X.Data(), ldb);
}

// new with complex

inline void solve(il::ArrayView<lapack_int> P, il::io_t,
                  il::Array2DEdit<std::complex<double>> Y) {
  IL_EXPECT_MEDIUM(P.size() == Y.size(0));

  const int layout = LAPACK_COL_MAJOR;
  const lapack_int n = static_cast<lapack_int>(Y.size(1));
  const lapack_int lda = static_cast<lapack_int>(Y.stride(1));
  const lapack_int k1 = 1;
  const lapack_int k2 = P.size();
  const lapack_int incx = 1;
  const lapack_int lapack_error = LAPACKE_zlaswp(
      layout, n, reinterpret_cast<lapack_complex_double*>(Y.Data()), lda, k1,
      k2, P.data(), incx);
  IL_EXPECT_MEDIUM(lapack_error == 0);
}

inline void solve(il::Array2DView<std::complex<double>> A, il::MatrixType type,
                  il::io_t, il::Array2DEdit<std::complex<double>> Y) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(Y.size(0) == A.size(0));
  IL_EXPECT_FAST(type == il::MatrixType::LowerUnit ||
                 type == il::MatrixType::UpperNonUnit);

  if (type == il::MatrixType::LowerUnit) {
    const IL_CBLAS_LAYOUT layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasLower;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasUnit;
    const MKL_INT m = static_cast<MKL_INT>(Y.size(0));
    const MKL_INT n = static_cast<MKL_INT>(Y.size(1));
    const std::complex<double> alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(Y.stride(1));
    cblas_ztrsm(layout, side, uplo, transa, diag, m, n, &alpha, A.data(), lda,
                Y.Data(), ldb);
  } else if (type == il::MatrixType::UpperNonUnit) {
    const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasUpper;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasNonUnit;
    const MKL_INT m = static_cast<MKL_INT>(Y.size(0));
    const MKL_INT n = static_cast<MKL_INT>(Y.size(1));
    const std::complex<double> alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(Y.stride(1));
    cblas_ztrsm(cblas_layout, side, uplo, transa, diag, m, n, &alpha, A.data(),
                lda, Y.Data(), ldb);
  }
}

inline void solve(il::ArrayView<int> pivot,
                  il::Array2DView<std::complex<double>> A, il::MatrixType type,
                  il::io_t, il::Array2DEdit<std::complex<double>> B) {
  IL_EXPECT_MEDIUM(pivot.size() == A.size(0));
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(A.size(0) == B.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::LowerUnit);

  {
    const int layout = LAPACK_COL_MAJOR;
    const lapack_int n = static_cast<lapack_int>(B.size(1));
    const lapack_int lda = static_cast<lapack_int>(B.stride(1));
    const lapack_int k1 = 1;
    const lapack_int k2 = pivot.size();
    const lapack_int incx = 1;
    const lapack_int lapack_error =
        LAPACKE_zlaswp(layout, n, reinterpret_cast<lapack_complex_double*>(B.Data()),
                       lda, k1, k2, pivot.data(), incx);
    IL_EXPECT_MEDIUM(lapack_error == 0);
  }

  {
    const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasLower;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasUnit;
    const MKL_INT m = static_cast<MKL_INT>(B.size(0));
    const MKL_INT n = static_cast<MKL_INT>(B.size(1));
    const std::complex<double> alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(B.stride(1));
    cblas_ztrsm(cblas_layout, side, uplo, transa, diag, m, n, &alpha, A.data(),
                lda, B.Data(), ldb);
  }
}

inline void solve(il::Array2DView<std::complex<double>> A, il::MatrixType type,
                  il::io_t, il::ArrayEdit<std::complex<double>> x) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(x.size() == A.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::UpperNonUnit);

  const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
  const CBLAS_SIDE side = CblasLeft;
  const CBLAS_UPLO uplo = CblasUpper;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_DIAG diag = CblasNonUnit;
  const MKL_INT m = static_cast<MKL_INT>(x.size());
  const MKL_INT n = static_cast<MKL_INT>(1);
  const std::complex<double> alpha = 1.0;
  const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
  const MKL_INT ldb = static_cast<MKL_INT>(x.size());
  cblas_ztrsm(cblas_layout, side, uplo, transa, diag, m, n, &alpha, A.data(),
              lda, x.Data(), ldb);
}

inline void solve(il::Array2DView<std::complex<double>> A, il::MatrixType type,
                  il::Dot op, il::io_t,
                  il::Array2DEdit<std::complex<double>> B) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));
  IL_EXPECT_FAST(B.size(0) == A.size(0));
  IL_EXPECT_FAST(type == il::MatrixType::UpperNonUnit);
  IL_EXPECT_FAST(op == il::Dot::Transpose);

  const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
  const CBLAS_SIDE side = CblasLeft;
  const CBLAS_UPLO uplo = CblasUpper;
  const CBLAS_TRANSPOSE transa = CblasTrans;
  const CBLAS_DIAG diag = CblasNonUnit;
  const MKL_INT m = static_cast<MKL_INT>(B.size(0));
  const MKL_INT n = static_cast<MKL_INT>(B.size(1));
  const std::complex<double> alpha = 1.0;
  const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
  const MKL_INT ldb = static_cast<MKL_INT>(B.stride(1));
  cblas_ztrsm(cblas_layout, side, uplo, transa, diag, m, n, &alpha, A.data(),
              lda, B.Data(), ldb);
}

inline void solve(il::ArrayView<int> pivot,
                  il::Array2DView<std::complex<double>> A, il::MatrixType type,
                  il::io_t, il::ArrayEdit<std::complex<double>> x) {
  IL_EXPECT_MEDIUM(pivot.size() == A.size(0));
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(x.size() == A.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::LowerUnit);

  {
    const int layout = LAPACK_COL_MAJOR;
    const lapack_int n = 1;
    const lapack_int lda = x.size();
    const lapack_int k1 = 1;
    const lapack_int k2 = x.size();
    const lapack_int incx = 1;
    const lapack_int lapack_error = LAPACKE_zlaswp(
        layout, n, reinterpret_cast<lapack_complex_double*>(x.Data()), lda, k1,
        k2, pivot.data(), incx);
    IL_EXPECT_MEDIUM(lapack_error == 0);
  }

  {
    const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasLower;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasUnit;
    const MKL_INT m = static_cast<MKL_INT>(x.size());
    const MKL_INT n = static_cast<MKL_INT>(1);
    const std::complex<double> alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(x.size());
    cblas_ztrsm(cblas_layout, side, uplo, transa, diag, m, n, &alpha, A.data(),
                lda, x.Data(), ldb);
  }
}

inline void solveRight(il::Array2DView<std::complex<double>> A,
                       il::MatrixType type, il::io_t,
                       il::Array2DEdit<std::complex<double>> X) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(X.size(1) == A.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::UpperNonUnit);

  const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
  const CBLAS_SIDE side = CblasRight;
  const CBLAS_UPLO uplo = CblasUpper;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_DIAG diag = CblasNonUnit;
  const MKL_INT m = static_cast<MKL_INT>(X.size(0));
  const MKL_INT n = static_cast<MKL_INT>(X.size(1));
  const std::complex<double> alpha = 1.0;
  const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
  const MKL_INT ldb = static_cast<MKL_INT>(X.stride(1));
  cblas_ztrsm(cblas_layout, side, uplo, transa, diag, m, n, &alpha, A.data(),
              lda, X.Data(), ldb);
}

}  // namespace il

#endif  // IL_SOLVE_H
