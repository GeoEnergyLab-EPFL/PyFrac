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

#ifndef IL_NORM_H
#define IL_NORM_H

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/StaticArray.h>
#include <il/TriDiagonal.h>
#include <il/math.h>

namespace il {

enum class Norm { L1, L2, Linf };

template <il::int_t n>
double norm(const il::StaticArray<double, n>& v, il::Norm norm_type) {
  double ans = 0.0;
  switch (norm_type) {
    case Norm::L1:
      for (il::int_t i = 0; i < n; ++i) {
        ans += il::abs(v[i]);
      }
      break;
    case Norm::L2:
      for (il::int_t i = 0; i < n; ++i) {
        ans += il::ipow<2>(v[i]);
      }
      ans = std::sqrt(ans);
      break;
    case Norm::Linf:
      for (il::int_t i = 0; i < n; ++i) {
        ans = il::max(ans, il::abs(v[i]));
      }
      break;
    default:
      IL_UNREACHABLE;
  }
  return ans;
}

inline double norm(const il::Array<double>& v, Norm norm_type) {
  double ans = 0.0;
  switch (norm_type) {
    case Norm::L1:
      for (il::int_t i = 0; i < v.size(); ++i) {
        ans += il::abs(v[i]);
      }
      break;
    case Norm::L2:
      for (il::int_t i = 0; i < v.size(); ++i) {
        ans += il::ipow<2>(v[i]);
      }
      ans = std::sqrt(ans);
      break;
    case Norm::Linf:
      for (il::int_t i = 0; i < v.size(); ++i) {
        ans = il::max(ans, il::abs(v[i]));
      }
      break;
    default:
      IL_UNREACHABLE;
  }
  return ans;
}

inline double norm(const il::Array<double>& x, Norm norm_type,
                   const il::Array<double>& alpha) {
  IL_EXPECT_FAST(alpha.size() == x.size());
  IL_EXPECT_AXIOM("All alpha elements must be positive");

  double ans = 0.0;
  switch (norm_type) {
    case Norm::L1:
      for (il::int_t i = 0; i < x.size(); ++i) {
        ans += il::abs(x[i] / alpha[i]);
      }
      break;
    case Norm::L2:
      for (il::int_t i = 0; i < x.size(); ++i) {
        ans += il::ipow<2>(x[i] / alpha[i]);
      }
      ans = std::sqrt(ans);
      break;
    case Norm::Linf:
      for (il::int_t i = 0; i < x.size(); ++i) {
        ans = il::max(ans, il::abs(x[i] / alpha[i]));
      }
      break;
    default:
      IL_UNREACHABLE;
  }
  return ans;
}

template <typename T>
double norm(const il::Array2D<T>& A, Norm norm_type) {
  T ans = 0;

  switch (norm_type) {
    case Norm::L1: {
      for (il::int_t j = 0; j < A.size(1); ++j) {
        T sum_column = 0;
        for (il::int_t i = 0; i < A.size(0); ++i) {
          sum_column += il::abs(A(i, j));
        }
        ans = il::max(ans, sum_column);
      }
    } break;
    case Norm::Linf: {
      il::Array<T> sum_row{A.size(0), 0};
      for (il::int_t j = 0; j < A.size(1); ++j) {
        for (il::int_t i = 0; i < A.size(0); ++i) {
          sum_row[i] += il::abs(A(i, j));
        }
      }
      for (il::int_t i = 0; i < sum_row.size(); ++i) {
        ans = il::max(ans, sum_row[i]);
      }
    } break;
    default:
      IL_EXPECT_FAST(false);
  }

  return ans;
}

template <typename T>
double norm(const il::TriDiagonal<T>& A, Norm norm_type) {
  il::int_t n{A.size()};
  T ans = 0;

  switch (norm_type) {
    case Norm::Linf: {
      il::Array<T> sum_row{n};
      sum_row[0] = il::abs(A(0, 0)) + il::abs(A(0, 1));
      for (il::int_t i = 1; i < n - 1; ++i) {
        sum_row[i] = il::abs(A(i, -1)) + il::abs(A(i, 0)) + il::abs(A(i, 1));
      }
      sum_row[n - 1] = il::abs(A(n - 1, -1)) + il::abs(A(n - 1, 0));
      for (il::int_t i = 0; i < sum_row.size(); ++i) {
        ans = il::max(ans, sum_row[i]);
      }
    } break;
    default:
      IL_EXPECT_FAST(false);
  }

  return ans;
}

}  // namespace il

#endif  // IL_NORM_H
