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

#ifndef IL_SPARSE_DOT_H
#define IL_SPARSE_DOT_H

#include <il/SparseMatrixCSR.h>

#ifdef IL_MKL
#include <mkl_spblas.h>

namespace il {

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 2
////////////////////////////////////////////////////////////////////////////////

// A is a matrix, x, y are vectors
// C <- A.B
inline il::SparseMatrixCSR<int, double> dot(
    il::io_t, il::SparseMatrixCSR<int, double> &A,
    il::SparseMatrixCSR<int, double> &B) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  const char trans = 'n';
  const MKL_INT sort = 0;
  const MKL_INT m = A.size(0);
  const MKL_INT n = A.size(1);
  const MKL_INT k = B.size(1);
  const MKL_INT nzmax = 0;
  MKL_INT info;

  il::Array<double> element{};
  il::Array<int> column{};
  il::Array<int> row{m + 1};
  MKL_INT request = 1;

  int *A_row_data = A.rowData();
  int *A_column_data = A.columnData();
  for (int i = 0; i <= A.size(0); ++i) {
    ++A_row_data[i];
  }
  for (int i = 0; i <= A.nbNonZeros(); ++i) {
    ++A_column_data[i];
  }
  int *B_row_data = B.rowData();
  int *B_column_data = B.columnData();
  if (&A != &B) {
    for (int i = 0; i <= B.size(0); ++i) {
      ++B_row_data[i];
    }
    for (int i = 0; i <= B.nbNonZeros(); ++i) {
      ++B_column_data[i];
    }
  }

  mkl_dcsrmultcsr(&trans, &request, &sort, &m, &n, &k, A.elementData(),
                  A.columnData(), A.rowData(), B.elementData(), B.columnData(),
                  B.rowData(), element.data(), column.data(), row.data(),
                  &nzmax, &info);
  IL_EXPECT_FAST(info == 0);

  element.Resize(row[m] - 1);
  column.Resize(row[m] - 1);
  request = 2;
  mkl_dcsrmultcsr(&trans, &request, &sort, &m, &n, &k, A.elementData(),
                  A.columnData(), A.rowData(), B.elementData(), B.columnData(),
                  B.rowData(), element.data(), column.data(), row.data(),
                  &nzmax, &info);
  IL_EXPECT_FAST(info == 0);

  for (int i = 0; i <= A.size(0); ++i) {
    --A_row_data[i];
  }
  for (int i = 0; i <= A.nbNonZeros(); ++i) {
    --A_column_data[i];
  }
  if (&A != &B) {
    for (int i = 0; i <= B.size(0); ++i) {
      --B_row_data[i];
    }
    for (int i = 0; i <= B.nbNonZeros(); ++i) {
      --B_column_data[i];
    }
  }
  for (int i = 0; i < row.size(); ++i) {
    --row[i];
  }
  for (int i = 0; i < column.size(); ++i) {
    --column[i];
  }

  return il::SparseMatrixCSR<int, double>{m, k, std::move(column),
                                          std::move(row), std::move(element)};
}

}  // namespace il
#endif  // IL_MKL

#endif  // IL_SPARSE_DOT_H
