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

#ifndef IL_DOT_H
#define IL_DOT_H

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>
#include <il/Array3D.h>
#include <il/Array4D.h>

#include <il/linearAlgebra/dense/blas/blas.h>

namespace il {

inline float dot(const il::Array<float>& x, const il::Array<float>& y) {
  return il::dot(x.view(), y.view());
}

inline double dot(const il::Array<double>& x, const il::Array<double>& y) {
  return il::dot(x.view(), y.view());
}

inline std::complex<float> dot(const il::Array<std::complex<float>>& x,
                               const il::Array<std::complex<float>>& y) {
  return il::dot(x.view(), y.view());
}

inline std::complex<float> dot(const il::Array<std::complex<float>>& x,
                               il::Dot op,
                               const il::Array<std::complex<float>>& y) {
  return il::dot(x.view(), op, y.view());
}

inline std::complex<float> dot(const il::Array<std::complex<float>>& x,
                               const il::Array<std::complex<float>>& y,
                               il::Dot op) {
  return il::dot(x.view(), y.view(), op);
}

inline std::complex<double> dot(const il::Array<std::complex<double>>& x,
                                const il::Array<std::complex<double>>& y) {
  return il::dot(x.view(), y.view());
}

inline std::complex<double> dot(const il::Array<std::complex<double>>& x,
                                il::Dot op,
                                const il::Array<std::complex<double>>& y) {
  return il::dot(x.view(), op, y.view());
}

inline std::complex<double> dot(const il::Array<std::complex<double>>& x,
                                const il::Array<std::complex<double>>& y,
                                il::Dot op) {
  return il::dot(x.view(), y.view(), op);
}

inline il::Array<double> dot(const il::Array2D<double>& A,
                             const il::Array<double>& x) {
  IL_EXPECT_FAST(A.size(1) == x.size());

  il::Array<double> y{A.size(0)};
  il::blas(1.0, A.view(), x.view(), 0.0, il::io, y.Edit());

  return y;
}

inline il::Array<double> dot(const il::Array2C<double>& A,
                             const il::Array<double>& x) {
  IL_EXPECT_FAST(A.size(1) == x.size());

  il::Array<double> y{A.size(0)};
  il::blas(1.0, A.view(), x.view(), 0.0, il::io, y.Edit());

  return y;
}

inline il::Array2D<float> dot(const il::Array2D<float>& A,
                              const il::Array2D<float>& B) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  il::Array2D<float> C{A.size(0), B.size(1)};
  il::blas(1.0f, A.view(), B.view(), 0.0f, il::io, C.Edit());
  return C;
}

inline il::Array2D<double> dot(const il::Array2D<double>& A,
                               const il::Array2D<double>& B) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  il::Array2D<double> C{A.size(0), B.size(1)};
  il::blas(1.0, A.view(), B.view(), 0.0, il::io, C.Edit());

  return C;
}

inline il::Array2C<double> dot(const il::Array2C<double>& A,
                               const il::Array2C<double>& B) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  il::Array2C<double> C{A.size(0), B.size(1)};
  il::blas(1.0, A.view(), B.view(), 0.0, il::io, C.Edit());

  return C;
}

}  // namespace il

#endif  // IL_DOT_H
