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

#ifndef IL_HEAT_H
#define IL_HEAT_H

#include <cstdio>

#include <il/Timer.h>

#include <il/SparseMatrixCSR.h>

namespace il {

template <typename Index>
il::SparseMatrixCSR<Index, double> heat_1d(Index n) {
  il::Array<il::StaticArray<Index, 2>> position{};
  position.Append(il::StaticArray<Index, 2>{il::value, {0, 0}});
  position.Append(il::StaticArray<Index, 2>{il::value, {0, 1}});
  for (Index i = 1; i < n - 1; ++i) {
    position.Append(il::StaticArray<Index, 2>{il::value, {i, i - 1}});
    position.Append(il::StaticArray<Index, 2>{il::value, {i, i}});
    position.Append(il::StaticArray<Index, 2>{il::value, {i, i + 1}});
  }
  position.Append(il::StaticArray<Index, 2>{il::value, {n - 1, n - 2}});
  position.Append(il::StaticArray<Index, 2>{il::value, {n - 1, n - 1}});

  il::Array<Index> index{};
  il::SparseMatrixCSR<Index, double> A{n, position, il::io, index};
  const double a = 1.0;
  const double b = 2.0;
  Index k = 0;
  A[index[k++]] = b;
  A[index[k++]] = a;
  for (Index i = 1; i < n - 1; ++i) {
    A[index[k++]] = a;
    A[index[k++]] = b;
    A[index[k++]] = a;
  }
  A[index[k++]] = a;
  A[index[k++]] = b;

  return A;
}

template <typename Index>
il::SparseMatrixCSR<Index, double> heat_2d(Index n) {
  il::Array<il::StaticArray<Index, 2>> position{};
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      const Index idx = i * n + j;
      if (i >= 1) {
        position.Append(
            il::StaticArray<Index, 2>{il::value, {idx, (i - 1) * n + j}});
      }
      if (j >= 1) {
        position.Append(
            il::StaticArray<Index, 2>{il::value, {idx, i * n + (j - 1)}});
      }
      position.Append(il::StaticArray<Index, 2>{il::value, {idx, idx}});
      if (j < n - 1) {
        position.Append(
            il::StaticArray<Index, 2>{il::value, {idx, i * n + (j + 1)}});
      }
      if (i < n - 1) {
        position.Append(
            il::StaticArray<Index, 2>{il::value, {idx, (i + 1) * n + j}});
      }
    }
  }

  il::Array<Index> index{};
  il::SparseMatrixCSR<Index, double> A{n * n, position, il::io, index};
  const double a = -1.0;
  const double b = 5.0;
  Index k = 0;
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      if (i >= 1) {
        A[k++] = a;
      }
      if (j >= 1) {
        A[k++] = a;
      }
      A[k++] = b;
      if (j < n - 1) {
        A[k++] = a;
      }
      if (i < n - 1) {
        A[k++] = a;
      }
    }
  }

  return A;
}

template <typename Index, typename T>
il::SparseMatrixCSR<Index, T> heat3d(Index n) {
  il::Array<il::StaticArray<Index, 2>> position{};
  position.Resize(7 * n * n * n);
  Index ii = 0;
  for (Index k = 0; k < n; ++k) {
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        const Index idx = (k * n + j) * n + i;
        if (k >= 1) {
          position[ii][0] = idx;
          position[ii][1] = ((k - 1) * n + j) * n + i;
          ++ii;
        }
        if (j >= 1) {
          position[ii][0] = idx;
          position[ii][1] = (k * n + (j - 1)) * n + i;
          ++ii;
        }
        if (i >= 1) {
          position[ii][0] = idx;
          position[ii][1] = (k * n + j) * n + (i - 1);
          ++ii;
        }
        position[ii][0] = idx;
        position[ii][1] = idx;
        ++ii;
        if (i < n - 1) {
          position[ii][0] = idx;
          position[ii][1] = (k * n + j) * n + (i + 1);
          ++ii;
        }
        if (j < n - 1) {
          position[ii][0] = idx;
          position[ii][1] = (k * n + (j + 1)) * n + i;
          ++ii;
        }
        if (k < n - 1) {
          position[ii][0] = idx;
          position[ii][1] = ((k + 1) * n + j) * n + i;
          ++ii;
        }
      }
    }
  }
  position.Resize(ii);

  il::Array<Index> index{};
  il::SparseMatrixCSR<Index, T> A{n * n * n, position, il::io, index};

  const T a = -1.0;
  const T b = 7.0;
  Index idx = 0;
  for (Index k = 0; k < n; ++k) {
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        if (k >= 1) {
          A[idx++] = a;
        }
        if (j >= 1) {
          A[idx++] = a;
        }
        if (i >= 1) {
          A[idx++] = a;
        }
        A[idx++] = b;
        if (i < n - 1) {
          A[idx++] = a;
        }
        if (j < n - 1) {
          A[idx++] = a;
        }
        if (k < n - 1) {
          A[idx++] = a;
        }
      }
    }
  }

  return A;
}
}  // namespace il

#endif  // IL_HEAT_H
