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

#ifndef IL_BLAS_STATIC_H
#define IL_BLAS_STATIC_H

#include <il/StaticArray.h>
#include <il/StaticArray2D.h>
#include <il/StaticArray3D.h>
#include <il/StaticArray4D.h>
#include <il/linearAlgebra/Matrix.h>

namespace il {

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
void blas(T alpha, const il::StaticArray3D<T, n0, n1, n2> &A, T beta, il::io_t,
          il::StaticArray3D<T, n0, n1, n2> &B) {
  for (il::int_t i2 = 0; i2 < n2; ++i2) {
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        B(i0, i1, i2) = alpha * A(i0, i1, i2) + beta * B(i0, i1, i2);
      }
    }
  }
}

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
//template <il::int_t n>
//void blas(double alpha, const il::StaticArray2D<double, n, n> &A,
//          const il::StaticArray2D<double, n, n> &B, double beta, il::io_t,
//          il::StaticArray2D<double, n, n> &C) {
//  IL_EXPECT_FAST(&A != &C);
//  IL_EXPECT_FAST(&B != &C);
//
//  const IL_CBLAS_LAYOUT layout = CblasColMajor;
//  const CBLAS_TRANSPOSE transa = CblasNoTrans;
//  const CBLAS_TRANSPOSE transb = CblasNoTrans;
//  const IL_CBLAS_INT lapack_n = static_cast<IL_CBLAS_INT>(n);
//
//  cblas_dgemm(layout, transa, transb, lapack_n, lapack_n, lapack_n, alpha,
//              A.data(), lapack_n, B.data(), lapack_n, beta, C.Data(), lapack_n);
//}

template <typename T, il::int_t n0, il::int_t n>
void blas(double alpha, const il::StaticArray2D<T, n0, n> &A,
          const il::StaticArray<T, n> &B, double beta, il::io_t,
          il::StaticArray<T, n0> &C) {
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    C[i0] *= beta;
    for (il::int_t i = 0; i < n; ++i) {
      C[i0] += alpha * A(i0, i) * B[i];
    }
  }
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n>
void blas(double alpha, const il::StaticArray3D<T, n0, n1, n> &A,
          const il::StaticArray<T, n> &B, double beta, il::io_t,
          il::StaticArray2D<T, n0, n1> &C) {
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      C(i0, i1) *= beta;
      for (il::int_t i = 0; i < n; ++i) {
        C(i0, i1) += alpha * A(i0, i1, i) * B[i];
      }
    }
  }
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n>
void blas(double alpha, const il::StaticArray4D<T, n0, n1, n2, n> &A,
          const il::StaticArray<T, n> &B, double beta, il::io_t,
          il::StaticArray3D<T, n0, n1, n2> &C) {
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        C(i0, i1, i2) *= beta;
        for (il::int_t i = 0; i < n; ++i) {
          C(i0, i1, i2) += alpha * A(i0, i1, i2, i) * B[i];
        }
      }
    }
  }
}

template <typename T, il::int_t n>
T dot(const il::StaticArray<T, n>& x, const il::StaticArray<T, n>& y) {
  T ans{0};
  for (il::int_t i = 0; i < n; ++i) {
    ans += x[i] * y[i];
  }
  return ans;
}


template <typename T, il::int_t n0, il::int_t n>
il::StaticArray<T, n0> dot(const il::StaticArray2D<T, n0, n>& A,
                           const il::StaticArray<T, n>& B) {
  il::StaticArray<T, n0> C{0};
  for (il::int_t i = 0; i < n; ++i) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      C[i0] += A(i0, i) * B[i];
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n>
il::StaticArray<T, n0> dot(const il::StaticArray2D<T, n, n0>& A, il::Dot A_info,
                           const il::StaticArray<T, n>& B) {
  IL_EXPECT_FAST(A_info == il::Dot::Transpose);
  IL_UNUSED(A_info);

  il::StaticArray<T, n0> C{};
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    C[i0] = 0;
    for (il::int_t i = 0; i < n; ++i) {
      C[i0] += A(i, i0) * B[i];
    }
  }
  return C;
}

template <typename T, il::int_t n, il::int_t n1>
il::StaticArray<T, n1> dot(const il::StaticArray<T, n>& A,
                           const il::StaticArray2D<T, n, n1>& B) {
  il::StaticArray<T, n1> C{0};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i = 0; i < n; ++i) {
      C[i1] += A[i] * B(i, i1);
    }
  }

  return C;
}

template <typename T, il::int_t n0, il::int_t n, il::int_t n1>
il::StaticArray2D<T, n0, n1> dot(const il::StaticArray2D<T, n0, n>& A,
                                 const il::StaticArray2D<T, n, n1>& B) {
  il::StaticArray2D<T, n0, n1> C{0};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i = 0; i < n; ++i) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        C(i0, i1) += A(i0, i) * B(i, i1);
      }
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n, il::int_t n1>
il::StaticArray2D<T, n0, n1> dot(const il::StaticArray2D<T, n, n0>& A,
                                 il::Dot A_info,
                                 const il::StaticArray2D<T, n, n1>& B) {
  IL_EXPECT_FAST(A_info == il::Dot::Transpose);
  IL_UNUSED(A_info);

  il::StaticArray2D<T, n0, n1> C{};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      C(i0, i1) = 0;
      for (il::int_t i = 0; i < n; ++i) {
        C(i0, i1) += A(i, i0) * B(i, i1);
      }
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n, il::int_t n1, il::int_t n2>
il::StaticArray3D<T, n0, n1, n2> dot(const il::StaticArray2D<T, n0, n>& A,
                                     const il::StaticArray3D<T, n, n1, n2>& B) {
  il::StaticArray3D<T, n0, n1, n2> C{0};
  for (il::int_t i = 0; i < n; ++i) {
    for (il::int_t i2 = 0; i2 < n2; ++i2) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          C(i0, i1, i2) += A(i0, i) * B(i, i1, i2);
        }
      }
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n>
il::StaticArray2D<T, n0, n1> dot(const il::StaticArray3D<T, n0, n1, n>& A,
                                 const il::StaticArray<T, n>& B) {
  il::StaticArray2D<T, n0, n1> C{};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      C(i0, i1) = 0;
      for (il::int_t i = 0; i < n; ++i) {
        C(i0, i1) += A(i0, i1, i) * B[i];
      }
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n>
il::StaticArray3D<T, n0, n1, n2> dot(
    const il::StaticArray4D<T, n0, n1, n2, n>& A,
    const il::StaticArray<T, n>& B) {
  il::StaticArray3D<T, n0, n1, n2> C{};
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        C(i0, i1, i2) = 0.0;
        for (il::int_t i = 0; i < n; ++i) {
          C(i0, i1, i2) += A(i0, i1, i2, i) * B[i];
        }
      }
    }
  }
  return C;
}

}  // namespace il

#endif  // IL_BLAS_STATIC_H
