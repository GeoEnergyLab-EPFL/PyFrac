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

#ifndef IL_BLAS_H
#define IL_BLAS_H

#include <complex>

#include <il/Array2CView.h>
#include <il/Array2DView.h>
#include <il/ArrayView.h>
#include <il/linearAlgebra/Matrix.h>

#include <il/linearAlgebra/dense/blas/blas_config.h>

// BLAS level 1
//   y = alpha * x + y
//   y = alpha * x + beta * y
//   Scalar product of x and y, with conjugate options for complex
//   A = alpha * B + A

namespace il {

inline CBLAS_TRANSPOSE blas_from_dot(il::Dot op) {
  switch (op) {
    case il::Dot::None:
      return CblasNoTrans;
    case il::Dot::Transpose:
      return CblasTrans;
    case il::Dot::Star:
      return CblasConjTrans;
    default:
      IL_UNREACHABLE;
  }
}

// y = alpha * x + y

inline void blas(float alpha, il::ArrayView<float> x, il::io_t,
                 il::ArrayEdit<float> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  cblas_saxpy(n, alpha, x.data(), incx, y.Data(), incy);
}

inline void blas(double alpha, il::ArrayView<double> x, il::io_t,
                 il::ArrayEdit<double> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  cblas_daxpy(n, alpha, x.data(), incx, y.Data(), incy);
}

inline void blas(std::complex<float> alpha,
                 il::ArrayView<std::complex<float>> x, il::io_t,
                 il::ArrayEdit<std::complex<float>> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  cblas_caxpy(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(x.data()), incx,
              reinterpret_cast<IL_CBLAS_PCOMPLEX64>(y.Data()), incy);
}

inline void blas(std::complex<double> alpha,
                 il::ArrayView<std::complex<double>> x, il::io_t,
                 il::ArrayEdit<std::complex<double>> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  cblas_zaxpy(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(x.data()), incx,
              reinterpret_cast<IL_CBLAS_PCOMPLEX128>(y.Data()), incy);
}

// y = alpha * x + beta * y

inline void blas(float alpha, il::ArrayView<float> x, float beta, il::io_t,
                 il::ArrayEdit<float> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  cblas_saxpby(n, alpha, x.data(), incx, beta, y.Data(), incy);
}

inline void blas(double alpha, il::ArrayView<double> x, double beta, il::io_t,
                 il::ArrayEdit<double> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  cblas_daxpby(n, alpha, x.data(), incx, beta, y.Data(), incy);
}

inline void blas(std::complex<float> alpha,
                 il::ArrayView<std::complex<float>> x, std::complex<float> beta,
                 il::io_t, il::ArrayEdit<std::complex<float>> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  cblas_caxpby(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&alpha),
               reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(x.data()), incx,
               reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&beta),
               reinterpret_cast<IL_CBLAS_PCOMPLEX64>(y.Data()), incy);
}

inline void blas(std::complex<double> alpha,
                 il::ArrayView<std::complex<double>> x,
                 std::complex<double> beta, il::io_t,
                 il::ArrayEdit<std::complex<double>> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  cblas_zaxpby(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&alpha),
               reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(x.data()), incx,
               reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&beta),
               reinterpret_cast<IL_CBLAS_PCOMPLEX128>(y.Data()), incy);
}

// Scalar product of x and y

inline float dot(il::ArrayView<float> x, il::ArrayView<float> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  const float ans = cblas_sdot(n, x.data(), incx, y.data(), incy);
  return ans;
}

inline double dot(il::ArrayView<double> x, il::ArrayView<double> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  const double ans = cblas_ddot(n, x.data(), incx, y.data(), incy);
  return ans;
}

inline std::complex<float> dot(il::ArrayView<std::complex<float>> x,
                               il::ArrayView<std::complex<float>> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  std::complex<float> ans;
  cblas_cdotu_sub(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(x.data()),
                  incx, reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(y.data()),
                  incy, reinterpret_cast<IL_CBLAS_PCOMPLEX64_ANS>(&ans));
  return ans;
}

inline std::complex<double> dot(il::ArrayView<std::complex<double>> x,
                                il::ArrayView<std::complex<double>> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;
  std::complex<double> ans;
  cblas_zdotu_sub(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(x.data()),
                  incx, reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(y.data()),
                  incy, reinterpret_cast<IL_CBLAS_PCOMPLEX128_ANS>(&ans));
  return ans;
}

inline std::complex<float> dot(il::ArrayView<std::complex<float>> x, il::Dot op,
                               il::ArrayView<std::complex<float>> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  switch (op) {
    case il::Dot::Star: {
      const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
      const IL_CBLAS_INT incx = 1;
      const IL_CBLAS_INT incy = 1;
      std::complex<float> ans;
      cblas_cdotc_sub(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(x.data()),
                      incx,
                      reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(y.data()),
                      incy, reinterpret_cast<IL_CBLAS_PCOMPLEX64_ANS>(&ans));
      return ans;
    }
    case il::Dot::None:
      return il::dot(x, y);
    default:
      IL_UNREACHABLE;
  }
  IL_UNREACHABLE;
  return 0.0;
}

inline std::complex<double> dot(il::ArrayView<std::complex<double>> x,
                                il::Dot op,
                                il::ArrayView<std::complex<double>> y) {
  IL_EXPECT_FAST(x.size() == y.size());

  switch (op) {
    case il::Dot::Star: {
      const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
      const IL_CBLAS_INT incx = 1;
      const IL_CBLAS_INT incy = 1;
      std::complex<double> ans;
      cblas_zdotc_sub(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(x.data()),
                      incx,
                      reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(y.data()),
                      incy, reinterpret_cast<IL_CBLAS_PCOMPLEX128_ANS>(&ans));
      return ans;
    }
    case il::Dot::None:
      return il::dot(x, y);
    default:
      IL_UNREACHABLE;
  }
  IL_UNREACHABLE;
  return 0.0;
}

inline std::complex<float> dot(il::ArrayView<std::complex<float>> x,
                               il::ArrayView<std::complex<float>> y,
                               il::Dot op) {
  IL_EXPECT_FAST(x.size() == y.size());

  switch (op) {
    case il::Dot::Star: {
      const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
      const IL_CBLAS_INT incx = 1;
      const IL_CBLAS_INT incy = 1;
      std::complex<float> ans;
      cblas_cdotc_sub(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(x.data()),
                      incx,
                      reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(y.data()),
                      incy, reinterpret_cast<IL_CBLAS_PCOMPLEX64_ANS>(&ans));
      return std::conj(ans);
    }
    case il::Dot::None:
      return il::dot(x, y);
    default:
      IL_UNREACHABLE;
  }
  IL_UNREACHABLE;
  return 0.0;
}

inline std::complex<double> dot(il::ArrayView<std::complex<double>> x,
                                il::ArrayView<std::complex<double>> y,
                                il::Dot op) {
  IL_EXPECT_FAST(x.size() == y.size());

  switch (op) {
    case il::Dot::Star: {
      const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(x.size());
      const IL_CBLAS_INT incx = 1;
      const IL_CBLAS_INT incy = 1;
      std::complex<double> ans;
      cblas_zdotc_sub(n, reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(x.data()),
                      incx,
                      reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(y.data()),
                      incy, reinterpret_cast<IL_CBLAS_PCOMPLEX128_ANS>(&ans));
      return std::conj(ans);
    }
    case il::Dot::None:
      return il::dot(x, y);
    default:
      IL_UNREACHABLE;
  }
  IL_UNREACHABLE;
  return 0.0;
}

inline void blas(double alpha, il::Array2DView<double> X, double beta, il::io_t,
                 il::Array2DEdit<double> Y) {
  IL_EXPECT_MEDIUM(X.size(0) == Y.size(0));
  IL_EXPECT_MEDIUM(X.size(1) == Y.size(1));

  if (beta == 0) {
    for (il::int_t i1 = 0; i1 < Y.size(1); ++i1) {
      for (il::int_t i0 = 0; i0 < Y.size(0); ++i0) {
        Y(i0, i1) = alpha * X(i0, i1);
      }
    }
  } else {
    for (il::int_t i1 = 0; i1 < Y.size(1); ++i1) {
      for (il::int_t i0 = 0; i0 < Y.size(0); ++i0) {
        Y(i0, i1) = alpha * X(i0, i1) + beta * Y(i0, i1);
      }
    }
  }
}

inline void blas(std::complex<double> alpha,
                 il::Array2DView<std::complex<double>> X,
                 std::complex<double> beta, il::io_t,
                 il::Array2DEdit<std::complex<double>> Y) {
  IL_EXPECT_MEDIUM(X.size(0) == Y.size(0));
  IL_EXPECT_MEDIUM(X.size(1) == Y.size(1));

  if (beta == std::complex<double>{0.0}) {
    for (il::int_t i1 = 0; i1 < Y.size(1); ++i1) {
      for (il::int_t i0 = 0; i0 < Y.size(0); ++i0) {
        Y(i0, i1) = alpha * X(i0, i1);
      }
    }
  } else {
    for (il::int_t i1 = 0; i1 < Y.size(1); ++i1) {
      for (il::int_t i0 = 0; i0 < Y.size(0); ++i0) {
        Y(i0, i1) = alpha * X(i0, i1) + beta * Y(i0, i1);
      }
    }
  }
}

// BLAS level 2
//   y = alpha * A.x + beta * y

inline void blas(float alpha, il::Array2DView<float> A, il::Dot op,
                 il::ArrayView<float> x, float beta, il::io_t,
                 il::ArrayEdit<float> y) {
  switch (op) {
    case il::Dot::None:
      IL_EXPECT_FAST(A.size(0) == y.size());
      IL_EXPECT_FAST(A.size(1) == x.size());
      break;
    case il::Dot::Transpose:
    case il::Dot::Star:
      IL_EXPECT_FAST(A.size(0) == x.size());
      IL_EXPECT_FAST(A.size(1) == y.size());
      break;
    default:
      IL_UNREACHABLE;
  }
  IL_EXPECT_FAST(x.data() != y.data());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(op);
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_sgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.Data(), incy);
}

inline void blas(float alpha, il::Array2DView<float> A, il::ArrayView<float> x,
                 float beta, il::io_t, il::ArrayEdit<float> y) {
  il::blas(alpha, A, il::Dot::None, x, beta, il::io, y);
}

inline void blas(double alpha, il::Array2DView<double> A, il::Dot op,
                 il::ArrayView<double> x, double beta, il::io_t,
                 il::ArrayEdit<double> y) {
  switch (op) {
    case il::Dot::None:
      IL_EXPECT_FAST(A.size(0) == y.size());
      IL_EXPECT_FAST(A.size(1) == x.size());
      break;
    case il::Dot::Transpose:
    case il::Dot::Star:
      IL_EXPECT_FAST(A.size(0) == x.size());
      IL_EXPECT_FAST(A.size(1) == y.size());
      break;
    default:
      IL_UNREACHABLE;
  }
  IL_EXPECT_FAST(x.data() != y.data());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(op);
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_dgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.Data(), incy);
}

inline void blas(double alpha, il::Array2DView<double> A,
                 il::ArrayView<double> x, double beta, il::io_t,
                 il::ArrayEdit<double> y) {
  il::blas(alpha, A, il::Dot::None, x, beta, il::io, y);
}

inline void blas(std::complex<float> alpha,
                 il::Array2DView<std::complex<float>> A, il::Dot dot,
                 il::ArrayView<std::complex<float>> x, std::complex<float> beta,
                 il::io_t, il::ArrayEdit<std::complex<float>> y) {
  switch (dot) {
    case il::Dot::None:
      IL_EXPECT_FAST(A.size(0) == y.size());
      IL_EXPECT_FAST(A.size(1) == x.size());
      break;
    case il::Dot::Transpose:
    case il::Dot::Star:
      IL_EXPECT_FAST(A.size(0) == x.size());
      IL_EXPECT_FAST(A.size(1) == y.size());
      break;
    default:
      IL_UNREACHABLE;
  }
  IL_EXPECT_FAST(x.data() != y.data());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(dot);
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_cgemv(layout, transa, m, n,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(A.data()), lda,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(x.data()), incx,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&beta),
              reinterpret_cast<IL_CBLAS_PCOMPLEX64>(y.Data()), incy);
}

inline void blas(std::complex<float> alpha,
                 il::Array2DView<std::complex<float>> A,
                 il::ArrayView<std::complex<float>> x, std::complex<float> beta,
                 il::io_t, il::ArrayEdit<std::complex<float>> y) {
  il::blas(alpha, A, il::Dot::None, x, beta, il::io, y);
}

inline void blas(std::complex<double> alpha,
                 il::Array2DView<std::complex<double>> A, il::Dot dot,
                 il::ArrayView<std::complex<double>> x,
                 std::complex<double> beta, il::io_t,
                 il::ArrayEdit<std::complex<double>> y) {
  switch (dot) {
    case il::Dot::None:
      IL_EXPECT_FAST(A.size(0) == y.size());
      IL_EXPECT_FAST(A.size(1) == x.size());
      break;
    case il::Dot::Transpose:
    case il::Dot::Star:
      IL_EXPECT_FAST(A.size(0) == x.size());
      IL_EXPECT_FAST(A.size(1) == y.size());
      break;
    default:
      IL_UNREACHABLE;
  }
  IL_EXPECT_FAST(x.data() != y.data());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(dot);
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_zgemv(layout, transa, m, n,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(A.data()), lda,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(x.data()), incx,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&beta),
              reinterpret_cast<IL_CBLAS_PCOMPLEX128>(y.Data()), incy);
}

inline void blas(std::complex<double> alpha,
                 il::Array2DView<std::complex<double>> A,
                 il::ArrayView<std::complex<double>> x,
                 std::complex<double> beta, il::io_t,
                 il::ArrayEdit<std::complex<double>> y) {
  il::blas(alpha, A, il::Dot::None, x, beta, il::io, y);
}

inline void blas(float alpha, il::Array2CView<float> A, il::Dot op,
                 il::ArrayView<float> x, float beta, il::io_t,
                 il::ArrayEdit<float> y) {
  switch (op) {
    case il::Dot::None:
      IL_EXPECT_FAST(A.size(0) == y.size());
      IL_EXPECT_FAST(A.size(1) == x.size());
      break;
    case il::Dot::Transpose:
    case il::Dot::Star:
      IL_EXPECT_FAST(A.size(0) == x.size());
      IL_EXPECT_FAST(A.size(1) == y.size());
      break;
    default:
      IL_UNREACHABLE;
  }
  IL_EXPECT_FAST(x.data() != y.data());

  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(op);
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(0));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_sgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.Data(), incy);
}

inline void blas(float alpha, il::Array2CView<float> A, il::ArrayView<float> x,
                 float beta, il::io_t, il::ArrayEdit<float> y) {
  il::blas(alpha, A, il::Dot::None, x, beta, il::io, y);
}

inline void blas(double alpha, il::Array2CView<double> A, il::Dot op,
                 il::ArrayView<double> x, double beta, il::io_t,
                 il::ArrayEdit<double> y) {
  switch (op) {
    case il::Dot::None:
      IL_EXPECT_FAST(A.size(0) == y.size());
      IL_EXPECT_FAST(A.size(1) == x.size());
      break;
    case il::Dot::Transpose:
      IL_EXPECT_FAST(A.size(0) == x.size());
      IL_EXPECT_FAST(A.size(1) == y.size());
      break;
    default:
      IL_UNREACHABLE;
  }
  IL_EXPECT_FAST(x.data() != y.data());

  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(op);
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(0));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_dgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.Data(), incy);
}

inline void blas(double alpha, il::Array2CView<double> A,
                 il::ArrayView<double> x, double beta, il::io_t,
                 il::ArrayEdit<double> y) {
  il::blas(alpha, A, il::Dot::None, x, beta, il::io, y);
}

inline void blas(std::complex<float> alpha,
                 il::Array2CView<std::complex<float>> A, il::Dot dot,
                 il::ArrayView<std::complex<float>> x, std::complex<float> beta,
                 il::io_t, il::ArrayEdit<std::complex<float>> y) {
  switch (dot) {
    case il::Dot::None:
      IL_EXPECT_FAST(A.size(0) == y.size());
      IL_EXPECT_FAST(A.size(1) == x.size());
      break;
    case il::Dot::Transpose:
    case il::Dot::Star:
      IL_EXPECT_FAST(A.size(0) == x.size());
      IL_EXPECT_FAST(A.size(1) == y.size());
      break;
    default:
      IL_UNREACHABLE;
  }
  IL_EXPECT_FAST(x.data() != y.data());

  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(dot);
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(0));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_cgemv(layout, transa, m, n,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(A.data()), lda,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(x.data()), incx,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&beta),
              reinterpret_cast<IL_CBLAS_PCOMPLEX64>(y.Data()), incy);
}

inline void blas(std::complex<float> alpha,
                 il::Array2CView<std::complex<float>> A,
                 il::ArrayView<std::complex<float>> x, std::complex<float> beta,
                 il::io_t, il::ArrayEdit<std::complex<float>> y) {
  il::blas(alpha, A, il::Dot::None, x, beta, il::io, y);
}

inline void blas(std::complex<double> alpha,
                 il::Array2CView<std::complex<double>> A, il::Dot dot,
                 il::ArrayView<std::complex<double>> x,
                 std::complex<double> beta, il::io_t,
                 il::ArrayEdit<std::complex<double>> y) {
  switch (dot) {
    case il::Dot::None:
      IL_EXPECT_FAST(A.size(0) == y.size());
      IL_EXPECT_FAST(A.size(1) == x.size());
      break;
    case il::Dot::Transpose:
    case il::Dot::Star:
      IL_EXPECT_FAST(A.size(0) == x.size());
      IL_EXPECT_FAST(A.size(1) == y.size());
      break;
    default:
      IL_UNREACHABLE;
  }
  IL_EXPECT_FAST(x.data() != y.data());

  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(dot);
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(0));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_zgemv(layout, transa, m, n,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(A.data()), lda,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(x.data()), incx,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&beta),
              reinterpret_cast<IL_CBLAS_PCOMPLEX128>(y.Data()), incy);
}

inline void blas(std::complex<double> alpha,
                 il::Array2CView<std::complex<double>> A,
                 il::ArrayView<std::complex<double>> x,
                 std::complex<double> beta, il::io_t,
                 il::ArrayEdit<std::complex<double>> y) {
  il::blas(alpha, A, il::Dot::None, x, beta, il::io, y);
}

// BLAS level 3
//   C = alpha * A.B + beta * C

inline void blas(float alpha, il::Array2DView<float> A, il::Dot opa,
                 il::Array2DView<float> B, il::Dot opb, float beta, il::io_t,
                 il::Array2DEdit<float> C) {
  if (opa == il::Dot::None && opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  } else if (opa == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(1));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(0));
  } else if (opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(0) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(1));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  }
  IL_EXPECT_FAST(A.data() != C.data());
  IL_EXPECT_FAST(B.data() != C.data());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(opa);
  const CBLAS_TRANSPOSE transb = il::blas_from_dot(opb);
  const IL_CBLAS_INT m =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(0) : A.size(1));
  const IL_CBLAS_INT n =
      static_cast<IL_CBLAS_INT>(opb == il::Dot::None ? B.size(1) : B.size(0));
  const IL_CBLAS_INT k =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(1) : A.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
  cblas_sgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.Data(), ldc);
}

inline void blas(float alpha, il::Array2DView<float> A,
                 il::Array2DView<float> B, float beta, il::io_t,
                 il::Array2DEdit<float> C) {
  il::blas(alpha, A, il::Dot::None, B, il::Dot::None, beta, il::io, C);
}

inline void blas(float alpha, il::Array2DView<float> A, il::Dot op,
                 il::Array2DView<float> B, float beta, il::io_t,
                 il::Array2DEdit<float> C) {
  il::blas(alpha, A, op, B, il::Dot::None, beta, il::io, C);
}

inline void blas(float alpha, il::Array2DView<float> A,
                 il::Array2DView<float> B, il::Dot op, float beta, il::io_t,
                 il::Array2DEdit<float> C) {
  il::blas(alpha, A, il::Dot::None, B, op, beta, il::io, C);
}

inline void blas(double alpha, il::Array2DView<double> A, il::Dot opa,
                 il::Array2DView<double> B, il::Dot opb, double beta, il::io_t,
                 il::Array2DEdit<double> C) {
  if (opa == il::Dot::None && opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  } else if (opa == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(1));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(0));
  } else if (opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(0) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(1));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  }
  IL_EXPECT_FAST(A.data() != C.data());
  IL_EXPECT_FAST(B.data() != C.data());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(opa);
  const CBLAS_TRANSPOSE transb = il::blas_from_dot(opb);
  const IL_CBLAS_INT m =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(0) : A.size(1));
  const IL_CBLAS_INT n =
      static_cast<IL_CBLAS_INT>(opb == il::Dot::None ? B.size(1) : B.size(0));
  const IL_CBLAS_INT k =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(1) : A.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
  cblas_dgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.Data(), ldc);
}

inline void blas(double alpha, il::Array2DView<double> A,
                 il::Array2DView<double> B, double beta, il::io_t,
                 il::Array2DEdit<double> C) {
  il::blas(alpha, A, il::Dot::None, B, il::Dot::None, beta, il::io, C);
}

inline void blas(double alpha, il::Array2DView<double> A, il::Dot op,
                 il::Array2DView<double> B, double beta, il::io_t,
                 il::Array2DEdit<double> C) {
  il::blas(alpha, A, op, B, il::Dot::None, beta, il::io, C);
}

inline void blas(double alpha, il::Array2DView<double> A,
                 il::Array2DView<double> B, il::Dot op, double beta, il::io_t,
                 il::Array2DEdit<double> C) {
  il::blas(alpha, A, il::Dot::None, B, op, beta, il::io, C);
}

inline void blas(std::complex<float> alpha,
                 il::Array2DView<std::complex<float>> A, il::Dot opa,
                 il::Array2DView<std::complex<float>> B, il::Dot opb,
                 std::complex<float> beta, il::io_t,
                 il::Array2DEdit<std::complex<float>> C) {
  if (opa == il::Dot::None && opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  } else if (opa == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(1));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(0));
  } else if (opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(0) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(1));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  }
  IL_EXPECT_FAST(A.data() != C.data());
  IL_EXPECT_FAST(B.data() != C.data());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(opa);
  const CBLAS_TRANSPOSE transb = il::blas_from_dot(opb);
  const IL_CBLAS_INT m =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(0) : A.size(1));
  const IL_CBLAS_INT n =
      static_cast<IL_CBLAS_INT>(opb == il::Dot::None ? B.size(1) : B.size(0));
  const IL_CBLAS_INT k =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(1) : A.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
#ifdef IL_MKL
  cblas_cgemm3m(layout, transa, transb, m, n, k, &alpha, A.data(), lda,
                B.data(), ldb, &beta, C.Data(), ldc);
#else
  cblas_cgemm(layout, transa, transb, m, n, k,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(A.data()), lda,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(B.data()), ldb,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&beta),
              reinterpret_cast<IL_CBLAS_PCOMPLEX64>(C.Data()), ldc);
#endif
}

inline void blas(std::complex<float> alpha,
                 il::Array2DView<std::complex<float>> A,
                 il::Array2DView<std::complex<float>> B,
                 std::complex<float> beta, il::io_t,
                 il::Array2DEdit<std::complex<float>> C) {
  il::blas(alpha, A, il::Dot::None, B, il::Dot::None, beta, il::io, C);
}

inline void blas(std::complex<float> alpha,
                 il::Array2DView<std::complex<float>> A, il::Dot op,
                 il::Array2DView<std::complex<float>> B,
                 std::complex<float> beta, il::io_t,
                 il::Array2DEdit<std::complex<float>> C) {
  il::blas(alpha, A, op, B, il::Dot::None, beta, il::io, C);
}

inline void blas(std::complex<float> alpha,
                 il::Array2DView<std::complex<float>> A,
                 il::Array2DView<std::complex<float>> B, il::Dot op,
                 std::complex<float> beta, il::io_t,
                 il::Array2DEdit<std::complex<float>> C) {
  il::blas(alpha, A, il::Dot::None, B, op, beta, il::io, C);
}

inline void blas(std::complex<double> alpha,
                 il::Array2DView<std::complex<double>> A, il::Dot opa,
                 il::Array2DView<std::complex<double>> B, il::Dot opb,
                 std::complex<double> beta, il::io_t,
                 il::Array2DEdit<std::complex<double>> C) {
  if (opa == il::Dot::None && opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  } else if (opa == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(1));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(0));
  } else if (opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(0) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(1));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  }
  IL_EXPECT_FAST(A.data() != C.data());
  IL_EXPECT_FAST(B.data() != C.data());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(opa);
  const CBLAS_TRANSPOSE transb = il::blas_from_dot(opb);
  const IL_CBLAS_INT m =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(0) : A.size(1));
  const IL_CBLAS_INT n =
      static_cast<IL_CBLAS_INT>(opb == il::Dot::None ? B.size(1) : B.size(0));
  const IL_CBLAS_INT k =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(1) : A.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
#ifdef IL_MKL
  cblas_zgemm3m(layout, transa, transb, m, n, k, &alpha, A.data(), lda,
                B.data(), ldb, &beta, C.Data(), ldc);
#else
  cblas_zgemm(layout, transa, transb, m, n, k,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(A.data()), lda,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(B.data()), ldb,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&beta),
              reinterpret_cast<IL_CBLAS_PCOMPLEX128>(C.Data()), ldc);
#endif
}

inline void blas(std::complex<double> alpha,
                 il::Array2DView<std::complex<double>> A,
                 il::Array2DView<std::complex<double>> B,
                 std::complex<double> beta, il::io_t,
                 il::Array2DEdit<std::complex<double>> C) {
  il::blas(alpha, A, il::Dot::None, B, il::Dot::None, beta, il::io, C);
}

inline void blas(std::complex<double> alpha,
                 il::Array2DView<std::complex<double>> A, il::Dot op,
                 il::Array2DView<std::complex<double>> B,
                 std::complex<double> beta, il::io_t,
                 il::Array2DEdit<std::complex<double>> C) {
  il::blas(alpha, A, op, B, il::Dot::None, beta, il::io, C);
}

inline void blas(std::complex<double> alpha,
                 il::Array2DView<std::complex<double>> A,
                 il::Array2DView<std::complex<double>> B, il::Dot op,
                 std::complex<double> beta, il::io_t,
                 il::Array2DEdit<std::complex<double>> C) {
  il::blas(alpha, A, il::Dot::None, B, op, beta, il::io, C);
}

inline void blas(float alpha, il::Array2CView<float> A, il::Dot opa,
                 il::Array2CView<float> B, il::Dot opb, float beta, il::io_t,
                 il::Array2CEdit<float> C) {
  if (opa == il::Dot::None && opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  } else if (opa == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(1));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(0));
  } else if (opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(0) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(1));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  }
  IL_EXPECT_FAST(A.data() != C.data());
  IL_EXPECT_FAST(B.data() != C.data());

  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(opa);
  const CBLAS_TRANSPOSE transb = il::blas_from_dot(opb);
  const IL_CBLAS_INT m =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(0) : A.size(1));
  const IL_CBLAS_INT n =
      static_cast<IL_CBLAS_INT>(opb == il::Dot::None ? B.size(1) : B.size(0));
  const IL_CBLAS_INT k =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(1) : A.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(0));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(0));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(0));
  cblas_sgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.Data(), ldc);
}

inline void blas(float alpha, il::Array2CView<float> A,
                 il::Array2CView<float> B, float beta, il::io_t,
                 il::Array2CEdit<float> C) {
  il::blas(alpha, A, il::Dot::None, B, il::Dot::None, beta, il::io, C);
}

inline void blas(float alpha, il::Array2CView<float> A, il::Dot op,
                 il::Array2CView<float> B, float beta, il::io_t,
                 il::Array2CEdit<float> C) {
  il::blas(alpha, A, op, B, il::Dot::None, beta, il::io, C);
}

inline void blas(float alpha, il::Array2CView<float> A,
                 il::Array2CView<float> B, il::Dot op, float beta, il::io_t,
                 il::Array2CEdit<float> C) {
  il::blas(alpha, A, il::Dot::None, B, op, beta, il::io, C);
}

inline void blas(double alpha, il::Array2CView<double> A, il::Dot opa,
                 il::Array2CView<double> B, il::Dot opb, double beta, il::io_t,
                 il::Array2CEdit<double> C) {
  if (opa == il::Dot::None && opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  } else if (opa == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(1));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(0));
  } else if (opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(0) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(1));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  }
  IL_EXPECT_FAST(A.data() != C.data());
  IL_EXPECT_FAST(B.data() != C.data());

  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(opa);
  const CBLAS_TRANSPOSE transb = il::blas_from_dot(opb);
  const IL_CBLAS_INT m =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(0) : A.size(1));
  const IL_CBLAS_INT n =
      static_cast<IL_CBLAS_INT>(opb == il::Dot::None ? B.size(1) : B.size(0));
  const IL_CBLAS_INT k =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(1) : A.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(0));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(0));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(0));
  cblas_dgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.Data(), ldc);
}

inline void blas(double alpha, il::Array2CView<double> A,
                 il::Array2CView<double> B, double beta, il::io_t,
                 il::Array2CEdit<double> C) {
  il::blas(alpha, A, il::Dot::None, B, il::Dot::None, beta, il::io, C);
}

inline void blas(double alpha, il::Array2CView<double> A, il::Dot op,
                 il::Array2CView<double> B, double beta, il::io_t,
                 il::Array2CEdit<double> C) {
  il::blas(alpha, A, op, B, il::Dot::None, beta, il::io, C);
}

inline void blas(double alpha, il::Array2CView<double> A,
                 il::Array2CView<double> B, il::Dot op, double beta, il::io_t,
                 il::Array2CEdit<double> C) {
  il::blas(alpha, A, il::Dot::None, B, op, beta, il::io, C);
}

inline void blas(std::complex<float> alpha,
                 il::Array2CView<std::complex<float>> A, il::Dot opa,
                 il::Array2CView<std::complex<float>> B, il::Dot opb,
                 std::complex<float> beta, il::io_t,
                 il::Array2CEdit<std::complex<float>> C) {
  if (opa == il::Dot::None && opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  } else if (opa == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(1));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(0));
  } else if (opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(0) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(1));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  }
  IL_EXPECT_FAST(A.data() != C.data());
  IL_EXPECT_FAST(B.data() != C.data());

  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(opa);
  const CBLAS_TRANSPOSE transb = il::blas_from_dot(opb);
  const IL_CBLAS_INT m =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(0) : A.size(1));
  const IL_CBLAS_INT n =
      static_cast<IL_CBLAS_INT>(opb == il::Dot::None ? B.size(1) : B.size(0));
  const IL_CBLAS_INT k =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(1) : A.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(0));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(0));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(0));
#ifdef IL_MKL
  cblas_cgemm3m(layout, transa, transb, m, n, k, &alpha, A.data(), lda,
                B.data(), ldb, &beta, C.Data(), ldc);
#else
  cblas_cgemm(layout, transa, transb, m, n, k,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(A.data()), lda,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(B.data()), ldb,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX64>(&beta),
              reinterpret_cast<IL_CBLAS_PCOMPLEX64>(C.Data()), ldc);
#endif
}

inline void blas(std::complex<float> alpha,
                 il::Array2CView<std::complex<float>> A,
                 il::Array2CView<std::complex<float>> B,
                 std::complex<float> beta, il::io_t,
                 il::Array2CEdit<std::complex<float>> C) {
  il::blas(alpha, A, il::Dot::None, B, il::Dot::None, beta, il::io, C);
}

inline void blas(std::complex<float> alpha,
                 il::Array2CView<std::complex<float>> A, il::Dot op,
                 il::Array2CView<std::complex<float>> B,
                 std::complex<float> beta, il::io_t,
                 il::Array2CEdit<std::complex<float>> C) {
  il::blas(alpha, A, op, B, il::Dot::None, beta, il::io, C);
}

inline void blas(std::complex<float> alpha,
                 il::Array2CView<std::complex<float>> A,
                 il::Array2CView<std::complex<float>> B, il::Dot op,
                 std::complex<float> beta, il::io_t,
                 il::Array2CEdit<std::complex<float>> C) {
  il::blas(alpha, A, il::Dot::None, B, op, beta, il::io, C);
}

inline void blas(std::complex<double> alpha,
                 il::Array2CView<std::complex<double>> A, il::Dot opa,
                 il::Array2CView<std::complex<double>> B, il::Dot opb,
                 std::complex<double> beta, il::io_t,
                 il::Array2CEdit<std::complex<double>> C) {
  if (opa == il::Dot::None && opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  } else if (opa == il::Dot::None) {
    IL_EXPECT_FAST(A.size(1) == B.size(1));
    IL_EXPECT_FAST(C.size(0) == A.size(0));
    IL_EXPECT_FAST(C.size(1) == B.size(0));
  } else if (opb == il::Dot::None) {
    IL_EXPECT_FAST(A.size(0) == B.size(0));
    IL_EXPECT_FAST(C.size(0) == A.size(1));
    IL_EXPECT_FAST(C.size(1) == B.size(1));
  }
  IL_EXPECT_FAST(A.data() != C.data());
  IL_EXPECT_FAST(B.data() != C.data());

  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transa = il::blas_from_dot(opa);
  const CBLAS_TRANSPOSE transb = il::blas_from_dot(opb);
  const IL_CBLAS_INT m =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(0) : A.size(1));
  const IL_CBLAS_INT n =
      static_cast<IL_CBLAS_INT>(opb == il::Dot::None ? B.size(1) : B.size(0));
  const IL_CBLAS_INT k =
      static_cast<IL_CBLAS_INT>(opa == il::Dot::None ? A.size(1) : A.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(0));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(0));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(0));
#ifdef IL_MKL
  cblas_zgemm3m(layout, transa, transb, m, n, k, &alpha, A.data(), lda,
                B.data(), ldb, &beta, C.Data(), ldc);
#else
  cblas_zgemm(layout, transa, transb, m, n, k,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&alpha),
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(A.data()), lda,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(B.data()), ldb,
              reinterpret_cast<const IL_CBLAS_PCOMPLEX128>(&beta),
              reinterpret_cast<IL_CBLAS_PCOMPLEX128>(C.Data()), ldc);
#endif
}

inline void blas(std::complex<double> alpha,
                 il::Array2CView<std::complex<double>> A,
                 il::Array2CView<std::complex<double>> B,
                 std::complex<double> beta, il::io_t,
                 il::Array2CEdit<std::complex<double>> C) {
  il::blas(alpha, A, il::Dot::None, B, il::Dot::None, beta, il::io, C);
}

inline void blas(std::complex<double> alpha,
                 il::Array2CView<std::complex<double>> A, il::Dot op,
                 il::Array2CView<std::complex<double>> B,
                 std::complex<double> beta, il::io_t,
                 il::Array2CEdit<std::complex<double>> C) {
  il::blas(alpha, A, op, B, il::Dot::None, beta, il::io, C);
}

inline void blas(std::complex<double> alpha,
                 il::Array2CView<std::complex<double>> A,
                 il::Array2CView<std::complex<double>> B, il::Dot op,
                 std::complex<double> beta, il::io_t,
                 il::Array2CEdit<std::complex<double>> C) {
  il::blas(alpha, A, il::Dot::None, B, op, beta, il::io, C);
}

}  // namespace il

#endif  // IL_BLAS_H
