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

#ifndef IL_SPARSE_BLAS_H
#define IL_SPARSE_BLAS_H

#include <il/linearAlgebra/sparse/blas/SparseMatrixBlas.h>

#ifdef IL_MKL

namespace il {

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 2
////////////////////////////////////////////////////////////////////////////////

inline void blas(double alpha, const il::SparseMatrixCSR<int, double>& A,
                 const il::Array<double>& x, double beta, il::io_t,
                 il::Array<double>& y) {
  //#pragma omp parallel for
  for (int i = 0; i < A.size(0); ++i) {
    double sum = 0.0;
    for (int k = A.row(i); k < A.row(i + 1); ++k) {
      sum += A.element(k) * x[A.column(k)];
    }
    y[i] = alpha * sum + beta * y[i];
  }
}

inline void blas(double alpha, const il::SparseMatrixCSR<il::int_t, double>& A,
                 const il::Array<double>& x, double beta, il::io_t,
                 il::Array<double>& y) {
  //#pragma omp parallel for
  for (il::int_t i = 0; i < A.size(0); ++i) {
    double sum = 0.0;
    for (il::int_t k = A.row(i); k < A.row(i + 1); ++k) {
      sum += A.element(k) * x[A.column(k)];
    }
    y[i] = alpha * sum + beta * y[i];
  }
}

// inline void blas(const il::Array<double>& x, il::io_t,
//                 il::SparseMatrixCSR<int, double>& A, il::Array<double>& y) {
//  IL_EXPECT_FAST(y.size() == A.size(0));
//  IL_EXPECT_FAST(x.size() == A.size(1));
//  IL_EXPECT_FAST(A.size(0) == A.size(1));
//
//  const char transa = 'n';
//  const MKL_INT m = A.size(0);
//  MKL_INT info;
//
//  int* A_row_data = A.rowData();
//  int* A_column_data = A.columnData();
//  for (int i = 0; i <= A.size(0); ++i) {
//    ++A_row_data[i];
//  }
//  for (int i = 0; i <= A.nbNonZeros(); ++i) {
//    ++A_column_data[i];
//  }
//
//  mkl_dcsrgemv(&transa, &m, A.elementData(), A.rowData(), A.columnData(),
//               x.data(), y.data());
//
//  for (int i = 0; i <= A.size(0); ++i) {
//    --A_row_data[i];
//  }
//  for (int i = 0; i <= A.nbNonZeros(); ++i) {
//    --A_column_data[i];
//  }
//}

inline void blas(float alpha, il::SparseMatrixBlas<int, float>& A_optimized,
                 const il::Array<float>& x, float beta, il::io_t,
                 il::Array<float>& y) {
  const sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
  const sparse_matrix_t A = A_optimized.handle();
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;

  sparse_status_t status =
      mkl_sparse_s_mv(operation, alpha, A, descr, x.data(), beta, y.Data());
  IL_EXPECT_FAST(status == SPARSE_STATUS_SUCCESS);
}

inline void blas(double alpha, il::SparseMatrixBlas<int, double>& A_optimized,
                 const il::Array<double>& x, double beta, il::io_t,
                 il::Array<double>& y) {
  const sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
  const sparse_matrix_t A = A_optimized.handle();
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;

  sparse_status_t status =
      mkl_sparse_d_mv(operation, alpha, A, descr, x.data(), beta, y.Data());
  IL_EXPECT_FAST(status == SPARSE_STATUS_SUCCESS);
}

}  // namespace il

#endif  // IL_MKL

#endif  // IL_SPARSE_BLAS_H
