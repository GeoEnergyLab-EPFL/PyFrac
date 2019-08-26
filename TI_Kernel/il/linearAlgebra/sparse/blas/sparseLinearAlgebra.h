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

#include <il/SparseMatrixCSR.h>

template <typename T>
double norm(const il::SparseArray2D<T> &A, Norm norm_type,
            const il::Array<T> &beta, const il::Array<T> &alpha) {
  IL_EXPECT_FAST(alpha.size() == A.size(1));
  IL_EXPECT_FAST(beta.size() == A.size(0));

  auto norm = T{0.0};
  switch (norm_type) {
    case Norm::infinity:
      for (il::int_t i = 0; i < A.size(0); ++i) {
        double sum = 0.0;
        for (il::int_t k = A.row(i); k < A.row(i + 1); ++k) {
          sum += il::abs(A[k] * alpha[A.column(k)] / beta[i]);
        }
        norm = il::max(norm, sum);
      }
      break;
    default:
      IL_EXPECT_FAST(false);
  }

  return norm;
}