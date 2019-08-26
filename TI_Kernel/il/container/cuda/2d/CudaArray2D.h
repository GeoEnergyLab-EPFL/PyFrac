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

#ifndef IL_CUDAARRAY2D_H
#define IL_CUDAARRAY2D_H

#include <cuda.h>
#include <cuda_runtime_api.h>

// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>

#include <il/core.h>

namespace il {

template <typename T>
class CudaArray2D {
 private:
  T *data_;
  il::int_t size_[2];
  il::int_t capacity_[2];

 public:
  /* \brief Construct an array of n elements
   */
  explicit CudaArray2D(il::int_t n0, il::int_t n1);

  /* \brief The destructor
   */
  ~CudaArray2D();

  /* \brief Get the size of the il::CudaArray2D<T>
   */
  il::int_t size(il::int_t d) const;

  /* \brief Get a pointer to the first element of the array
   */
  const T *data() const;

  /* \brief Get a pointer to the first element of the array
   */
  T *Data();
};

template <typename T>
CudaArray2D<T>::CudaArray2D(il::int_t n0, il::int_t n1) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);

  cudaMalloc(&data_, n0 * n1 * sizeof(T));
  size_[0] = n0;
  size_[1] = n1;
  capacity_[0] = n0;
  capacity_[1] = n1;
}

template <typename T>
CudaArray2D<T>::~CudaArray2D() {
  cudaFree(data_);
}

template <typename T>
il::int_t CudaArray2D<T>::size(il::int_t d) const {
  return size_[d];
}

template <typename T>
const T *CudaArray2D<T>::data() const {
  return data_;
}

template <typename T>
T *CudaArray2D<T>::Data() {
  return data_;
}

}  // namespace il
#endif  // IL_CUDAARRAY2D_H
