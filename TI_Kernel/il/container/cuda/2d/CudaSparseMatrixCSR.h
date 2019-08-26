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

#ifndef IL_CUDASPARSEMATRIXCSR_H
#define IL_CUDASPARSEMATRIXCSR_H

#include <il/container/cuda/1d/CudaArray.h>

namespace il {

template <typename T>
class CudaSparseMatrixCSR {
 private:
  il::int_t n0_;
  il::int_t n1_;
  il::CudaArray<int> row_;
  il::CudaArray<int> column_;
  il::CudaArray<T> element_;

 public:
  CudaSparseMatrixCSR(il::int_t n0, il::int_t n1, il::CudaArray<int> row,
                      il::CudaArray<int> column, il::CudaArray<T> element);
  il::int_t size(il::int_t d) const;
  il::int_t nbNonZeros() const;
  const int* rowData() const;
  const int* columnData() const;
  const T* elementData() const;
};

template <typename T>
CudaSparseMatrixCSR<T>::CudaSparseMatrixCSR(il::int_t n0, il::int_t n1,
                                            il::CudaArray<int> row,
                                            il::CudaArray<int> column,
                                            il::CudaArray<T> element)
    : row_{row}, column_{column}, element_{element} {
  n0_ = n0;
  n1_ = n1;
}

template <typename T>
il::int_t CudaSparseMatrixCSR<T>::size(il::int_t d) const {
  IL_EXPECT_FAST(d >= 0 && d < 2);

  return d == 0 ? n0_ : n1_;
}

template <typename T>
il::int_t CudaSparseMatrixCSR<T>::nbNonZeros() const {
  return element_.size();
}

template <typename T>
const int* CudaSparseMatrixCSR<T>::rowData() const {
  return row_.data();
}

template <typename T>
const int* CudaSparseMatrixCSR<T>::columnData() const {
  return column_.data();
}

template <typename T>
const T* CudaSparseMatrixCSR<T>::elementData() const {
  return element_.data();
}

}  // namespace il

#endif  // IL_CUDASPARSEMATRIXCSR_H
