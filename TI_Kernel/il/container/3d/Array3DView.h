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

#ifndef IL_ARRAY3DVIEW_H
#define IL_ARRAY3DVIEW_H

#include <il/core.h>

namespace il {

template <typename T>
class Array3DView {
 protected:
#ifndef NDEBUG
  il::int_t debug_size_0_;
  il::int_t debug_size_1_;
  il::int_t debug_size_2_;
  il::int_t debug_stride_0_;
  il::int_t debug_stride_1_;
#endif
  T* data_;
  T* size_[3];
  T* stride_[2];

 public:
  /* \brief Default constructor
  // \details It creates a Array3DView of 0 rows, 0 columns and 0 slices
  */
  Array3DView();

  /* \brief Construct an il::Array3DView<T> from a C-array (a pointer) and
  // its size and the stride
  //
  // il::Array3DView<double> A{data, n, p, q, stride_0, stride_1};
  */
  Array3DView(const T* data, il::int_t n, il::int_t p, il::int_t q,
              il::int_t stride_0, il::int_t stride_1);

  /* \brief Accessor
  // \details Access (read only) the (i, j, k)-th element of the array view.
  // Bound  checking is done in debug mode but not in release mode.
  //
  // il::Array3DView<double> A{data, n, p, q, p, q};
  // std::cout << v(0, 0, 0) << std::endl;
  */
  const T& operator()(il::int_t i, il::int_t j, il::int_t k) const;

  /* \brief Get the size of the array view
  //
  // il::Array3DView<double> v{data, n, p, q, p, q};
  // for (il::int_t i = 0; i < v.size(0); ++i) {
  //   for (il::int_t j = 0; j < v.size(1); ++j) {
  //     for (il::int_t k = 0; k < v.size(2); ++k) {
  //       A(i, j, k) = 1.0 / (i + j + k + 3);
  //     }
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Memory distance (in sizeof(T)) in between A(i, j) and A(i + 1, j)
   */
  il::int_t stride(il::int_t d) const;
};

template <typename T>
Array3DView<T>::Array3DView() {
#ifndef NDEBUG
  debug_size_0_ = 0;
  debug_size_1_ = 0;
  debug_size_2_ = 0;
  debug_stride_0_ = 0;
  debug_stride_1_ = 0;
#endif
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  size_[2] = nullptr;
  stride_[0] = nullptr;
  stride_[1] = nullptr;
}

template <typename T>
Array3DView<T>::Array3DView(const T* data, il::int_t n, il::int_t p,
                            il::int_t q, il::int_t stride_0,
                            il::int_t stride_1) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(p >= 0);
  IL_EXPECT_FAST(q >= 0);
  IL_EXPECT_FAST(stride_0 >= 0);
  IL_EXPECT_FAST(stride_1 >= 0);
  data_ = const_cast<T*>(data);
#ifndef NDEBUG
  debug_size_0_ = n;
  debug_size_1_ = p;
  debug_size_2_ = q;
  debug_stride_0_ = stride_0;
  debug_stride_1_ = stride_1;
#endif
  size_[0] = data_ + n;
  size_[1] = data_ + p;
  size_[2] = data_ + q;
  stride_[0] = data_ + stride_0;
  stride_[1] = data_ + stride_1;
}

template <typename T>
const T& Array3DView<T>::operator()(il::int_t i, il::int_t j,
                                    il::int_t k) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(i) <
                 static_cast<std::size_t>(size(0)));
  IL_EXPECT_FAST(static_cast<std::size_t>(j) <
                 static_cast<std::size_t>(size(1)));
  IL_EXPECT_FAST(static_cast<std::size_t>(k) <
                 static_cast<std::size_t>(size(2)));
  return data_[(k * stride(1) + j) * stride(0) + i];
}

template <typename T>
il::int_t Array3DView<T>::size(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(3));
  return static_cast<il::int_t>(size_[d] - data_);
}

template <typename T>
const T* Array3DView<T>::data() const {
  return data_;
}

template <typename T>
il::int_t Array3DView<T>::stride(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return static_cast<il::int_t>(stride_[d] - data_);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
class Array3DEdit : public Array3DView<T> {
 public:
  /* \brief Default constructor
  // \details It creates a Array3DView of 0 rows, 0 columns and 0 slices
  */
  Array3DEdit();

  /* \brief Construct an il::Array3DView<T> from a C-array (a pointer) and
  // its size and the stride
  //
  // il::Array3DView<double> A{data, n, p, q, stride_0, stride_1};
  */
  Array3DEdit(T* data, il::int_t n, il::int_t p, il::int_t q,
              il::int_t stride_0, il::int_t stride_1);

  /* \brief Accessor
  // \details Access (read and write) the (i, j, k)-th element of the array
  // view. Bound  checking is done in debug mode but not in release mode.
  //
  // il::Array3DView<double> A{data, n, p, q, p, q};
  // std::cout << v(0, 0, 0) << std::endl;
  */
  T& operator()(il::int_t i, il::int_t j, il::int_t k);

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* data();
};

template <typename T>
Array3DEdit<T>::Array3DEdit() : Array3DView<T>{} {}

template <typename T>
Array3DEdit<T>::Array3DEdit(T* data, il::int_t n, il::int_t p, il::int_t q,
                            il::int_t stride_0, il::int_t stride_1)
    : Array3DView<T>{data, n, p, q, stride_0, stride_1} {}

template <typename T>
T& Array3DEdit<T>::operator()(il::int_t i, il::int_t j, il::int_t k) {
  IL_EXPECT_FAST(static_cast<std::size_t>(i) <
                 static_cast<std::size_t>(this->size(0)));
  IL_EXPECT_FAST(static_cast<std::size_t>(j) <
                 static_cast<std::size_t>(this->size(1)));
  IL_EXPECT_FAST(static_cast<std::size_t>(k) <
                 static_cast<std::size_t>(this->size(2)));
  return this->data_[(k * this->stride(1) + j) * this->stride(0) + i];
}

template <typename T>
T* Array3DEdit<T>::data() {
  return this->data_;
}
}  // namespace il

#endif  // IL_ARRAY3DVIEW_H
