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

#ifndef IL_ARRAY2DVIEW_H
#define IL_ARRAY2DVIEW_H

#include <il/core.h>

namespace il {

template <typename T>
class Array2DView {
 protected:
#ifdef IL_DEBUGGER_HELPERS
  il::int_t debug_size_0_;
  il::int_t debug_size_1_;
  il::int_t debug_stride_;
#endif
  T* data_;
  T* size_[2];
  T* stride_;
  short align_mod_;
  short align_r_;

 public:
  /* \brief Default constructor
  // \details It creates a Array2DView of 0 rows and 0 columns.
  */
  Array2DView();

  Array2DView(const T* data, il::int_t n0, il::int_t n1, il::int_t stride);

  /* \brief Construct an il::Array2DView<T> from a
  C-array (a pointer) and
  // its size and the stride
  //
  // void f(const double* data, il::int_t n, il::int_t p, il::int_t stride) {
  //   il::Array2DView<double> A{data, n, p, stride};
  //   ...
  // }
  */
  Array2DView(const T* data, il::int_t n0, il::int_t n1, il::int_t stride,
              short align_mod, short align_r);

  /* \brief Accessor
  // \details Access (read only) the (i, j)-th element of the array view. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::Array2DView<double> A{data, n, p};
  // std::cout << v(0, 0) << std::endl;
  */
  const T& operator()(il::int_t i0, il::int_t i1) const;

  /* \brief Get the size of the array view
  //
  // il::ConstArrayView<double> v{data, n, p, p};
  // for (il::int_t i = 0; i < v.size(0); ++i) {
  //   for (il::int_t j = 0; j < v.size(1); ++j) {
  //     A(i, j) = 1.0 / (i + j + 2);
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  Array2DView<T> view(il::Range r0, il::Range r1) const;

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Memory distance (in sizeof(T)) in between A(i, j) and A(i + 1, j)
   */
  il::int_t stride(il::int_t d) const;
};

template <typename T>
class Array2DEdit : public Array2DView<T> {
 public:
  /* \brief Default constructor
  // \details It creates a Array2DEdit of 0 rows and 0 columns.
  */
  Array2DEdit();

  /* \brief Construct an il::Array2DEdit<T> from a C-array (a
  pointer) and
  // its size and the stride
  //
  // void f(const double* data, il::int_t n, il::int_t, il::int_t stride) {
  //   il::Array2DEdit<double> A{data, n, p, stride};
  //   ...
  // }
  */
  Array2DEdit(T* data, il::int_t n0, il::int_t n1, il::int_t stride,
              short align_mod, short align_r);

  /* \brief Accessor
  // \details Access (read and write) the (i, j)-th element of the array view.
  // Bound checking is done in debug mode but not in release mode.
  */
  T& operator()(il::int_t i0, il::int_t i1);

  Array2DEdit<T> Edit(il::Range r0, il::Range r1);

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* Data();
};

template <typename T>
Array2DView<T>::Array2DView() {
#ifdef IL_DEBUGGER_HELPERS
  debug_size_0_ = 0;
  debug_size_1_ = 0;
  debug_stride_ = 0;
#endif
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  stride_ = nullptr;
  align_mod_ = 0;
  align_r_ = 0;
}

template <typename T>
Array2DView<T>::Array2DView(const T* data, il::int_t n0, il::int_t n1,
                            il::int_t stride) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(stride > 0);
  data_ = const_cast<T*>(data);
#ifdef IL_DEBUGGER_HELPERS
  debug_size_0_ = n0;
  debug_size_1_ = n1;
  debug_stride_ = stride;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  stride_ = data_ + stride;
  align_mod_ = 0;
  align_r_ = 0;
}

template <typename T>
Array2DView<T>::Array2DView(const T* data, il::int_t n0, il::int_t n1,
                            il::int_t stride, short align_mod, short align_r) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(stride > 0);
  data_ = const_cast<T*>(data);
#ifdef IL_DEBUGGER_HELPERS
  debug_size_0_ = n0;
  debug_size_1_ = n1;
  debug_stride_ = stride;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  stride_ = data_ + stride;
  align_mod_ = align_mod;
  align_r_ = align_r;
}

template <typename T>
const T& Array2DView<T>::operator()(il::int_t i0, il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));
  return data_[i1 * (stride_ - data_) + i0];
}

template <typename T>
il::int_t Array2DView<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return static_cast<il::int_t>(size_[d] - data_);
}

template <typename T>
Array2DView<T> Array2DView<T>::view(il::Range r0, il::Range r1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(r0.begin) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(r0.end) <=
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(r1.begin) <
                   static_cast<std::size_t>(size(1)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(r1.end) <=
                   static_cast<std::size_t>(size(1)));

  const il::int_t my_stride = stride(1);
  return il::Array2DView<T>{data() + r0.begin + r1.begin * my_stride,
                            r0.end - r0.begin,
                            r1.end - r1.begin,
                            my_stride,
                            0,
                            0};
}

template <typename T>
const T* Array2DView<T>::data() const {
  return data_;
}

template <typename T>
il::int_t Array2DView<T>::stride(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return (d == 0) ? 1 : static_cast<il::int_t>(stride_ - data_);
}

template <typename T>
Array2DEdit<T>::Array2DEdit() : Array2DView<T>{} {}

template <typename T>
Array2DEdit<T>::Array2DEdit(T* data, il::int_t n0, il::int_t n1,
                            il::int_t stride, short align_mod, short align_r)
    : Array2DView<T>{data, n0, n1, stride, align_mod, align_r} {}

template <typename T>
T& Array2DEdit<T>::operator()(il::int_t i0, il::int_t i1) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(this->size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(this->size(1)));
  return this->data_[i1 * (this->stride_ - this->data_) + i0];
}

template <typename T>
Array2DEdit<T> Array2DEdit<T>::Edit(il::Range r0, il::Range r1) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(r0.begin) <
                   static_cast<std::size_t>(this->size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(r0.end) <=
                   static_cast<std::size_t>(this->size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(r1.begin) <
                   static_cast<std::size_t>(this->size(1)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(r1.end) <=
                   static_cast<std::size_t>(this->size(1)));

  const il::int_t my_stride = this->stride(1);
  return il::Array2DEdit<T>{Data() + r0.begin + r1.begin * my_stride,
                            r0.end - r0.begin,
                            r1.end - r1.begin,
                            my_stride,
                            0,
                            0};
}

template <typename T>
T* Array2DEdit<T>::Data() {
  return this->data_;
}

template <typename T>
void copy(il::Array2DView<T> A, il::io_t, il::Array2DEdit<T> B) {
  IL_EXPECT_FAST(A.size(0) == B.size(0));
  IL_EXPECT_FAST(A.size(1) == B.size(1));

  for (il::int_t i1 = 0; i1 < A.size(1); ++i1) {
    for (il::int_t i0 = 0; i0 < A.size(0); ++i0) {
      B(i0, i1) = A(i0, i1);
    }
  }
}

}  // namespace il

#endif  // IL_ARRAY2DVIEW_H
