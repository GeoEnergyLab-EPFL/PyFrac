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

#ifndef IL_CUDA_SPARSE_BLAS_H
#define IL_CUDA_SPARSE_BLAS_H

#include <il/linearAlgebra/cuda/sparse/blas/CusparseHandle.h>

namespace il {

inline void blas(float alpha, const il::CudaSparseMatrixCSR<float> &A,
                 const il::CudaArray<float> &x, float beta, il::io_t,
                 il::CudaArray<float> &y, il::CusparseHandle &handle) {
  IL_EXPECT_FAST(A.size(1) == x.size());
  IL_EXPECT_FAST(A.size(0) == y.size());

  const cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const int m = static_cast<int>(A.size(0));
  const int n = static_cast<int>(A.size(1));
  const int nnz = static_cast<int>(A.nbNonZeros());
  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

  const cusparseStatus_t status = cusparseScsrmv(
      handle.handle(), transA, m, n, nnz, &alpha, descrA, A.elementData(),
      A.rowData(), A.columnData(), x.data(), &beta, y.data());
  IL_EXPECT_FAST(status == CUSPARSE_STATUS_SUCCESS);
}

}  // namespace il

#endif  // IL_CUDA_SPARSE_BLAS_H
