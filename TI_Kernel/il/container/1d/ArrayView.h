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

#ifndef IL_ARRAYVIEW_H
#define IL_ARRAYVIEW_H

#include <il/core.h>

namespace il {

template <typename T>
class ArrayView {
 protected:
  T* data_;
  T* size_;
  short align_r_;
  short alignment_;

 public:
  /* \brief Default constructor
  // \details It creates a ArrayView of size 0.
  */
  ArrayView();

  /* \brief Construct an il::ArrayView<T> from a C-array (a pointer) and
  // its size
  //
  // void f(const double* p, il::int_t n) {
  //   il::ArrayView<double> v{p, n};
  //   ...
  // }
  */
  explicit ArrayView(const T* data, il::int_t n);

  explicit ArrayView(const T* data, il::int_t n, il::int_t align_mod,
                     il::int_t align_r);

  /* \brief Accessor
  // \details Access (read only) the i-th element of the array view. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::ArrayView<double> v{p, n};
  // std::cout << v[0] << std::endl;
  */
  const T& operator[](il::int_t i) const;

  /* \brief Accessor on the last element
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  const T& back() const;

  /* \brief Get the size of the array view
  //
  // il::ArrayView<double> v{p, n};
  // for (il::int_t k = 0; k < v.size(); ++k) {
  //   std::cout << v[k] << std::endl;
  // }
  */
  il::int_t size() const;

  il::ArrayView<T> view() const;

  il::ArrayView<T> view(il::Range range) const;

  /* \brief Get a pointer to const to the first element of the array view
  // \details One should use this method only when using C-style API
  */
  const T* data() const;
};

template <typename T>
class ArrayEdit : public ArrayView<T> {
 public:
  /* \brief Default constructor
  // \details It creates a ArrayView of size 0.
  */
  ArrayEdit();

  /* \brief Construct an il::ArrayEdit<T> from a C-array (a pointer) and
  // its size
  //
  // void f(double* p, int n) {
  //   il::ArrayEdit<double> v{p, n};
  //   ...
  // }
  */
  explicit ArrayEdit(T* data, il::int_t n);

  explicit ArrayEdit(T* data, il::int_t n, il::int_t align_mod,
                     il::int_t align_r);

  /* \brief Accessor
  // \details Access (read or write) the i-th element of the array view. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::ArrayEdit<double> v{p, n};
  // v[0] = 0.0;
  // v[n] = 0.0; // Program is aborted in debug mode and has undefined
  //             // behavior in release mode
  */
  T& operator[](il::int_t i);

  /* \brief Accessor on the last element
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  T& Back();

  il::ArrayView<T> view() const;

  il::ArrayView<T> view(il::Range range) const;

  il::ArrayEdit<T> Edit();

  il::ArrayEdit<T> Edit(il::Range range);

  /* \brief Get a pointer to the first element of the array view
  // \details One should use this method only when using C-style API
  */
  T* Data();
};

template <typename T>
ArrayView<T>::ArrayView() {
  data_ = nullptr;
  size_ = nullptr;
  alignment_ = 0;
  align_r_ = 0;
}

template <typename T>
ArrayView<T>::ArrayView(const T* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  data_ = const_cast<T*>(data);
  size_ = const_cast<T*>(data) + n;
  alignment_ = 0;
  align_r_ = 0;
}

template <typename T>
ArrayView<T>::ArrayView(const T* data, il::int_t n, il::int_t align_mod,
                        il::int_t align_r) {
  IL_EXPECT_FAST(il::isTrivial<T>::value);
  IL_EXPECT_FAST(sizeof(T) % alignof(T) == 0);
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_mod <= SHRT_MAX);
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);
  IL_EXPECT_FAST(align_r <= SHRT_MAX);

  data_ = const_cast<T*>(data);
  size_ = const_cast<T*>(data) + n;
  alignment_ = 0;
  align_r_ = 0;
}

template <typename T>
const T& ArrayView<T>::operator[](il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));
  return data_[i];
}

template <typename T>
const T& ArrayView<T>::back() const {
  IL_EXPECT_MEDIUM(size() > 0);
  return size_[-1];
}

template <typename T>
il::int_t ArrayView<T>::size() const {
  return size_ - data_;
}

template <typename T>
il::ArrayView<T> ArrayView<T>::view() const {
  return il::ArrayView<T>{data_, size()};
};

template <typename T>
il::ArrayView<T> ArrayView<T>::view(il::Range range) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.begin) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.end) <=
                   static_cast<std::size_t>(size()));

  return il::ArrayView<T>{data_ + range.begin, range.end - range.begin};
};

template <typename T>
const T* ArrayView<T>::data() const {
  return data_;
}

template <typename T>
ArrayEdit<T>::ArrayEdit() : ArrayView<T>{} {}

template <typename T>
ArrayEdit<T>::ArrayEdit(T* data, il::int_t n) : ArrayView<T>{data, n} {}

template <typename T>
ArrayEdit<T>::ArrayEdit(T* data, il::int_t n, il::int_t align_mod,
                        il::int_t align_r)
    : ArrayView<T>{data, n, align_mod, align_r} {}

template <typename T>
T& ArrayEdit<T>::operator[](il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(this->size()));
  return this->data_[i];
}

template <typename T>
T& ArrayEdit<T>::Back() {
  IL_EXPECT_FAST(this->size() > 0);
  return this->size_[-1];
}

template <typename T>
il::ArrayView<T> ArrayEdit<T>::view() const {
  return il::ArrayView<T>{this->data_, this->size()};
};

template <typename T>
il::ArrayView<T> ArrayEdit<T>::view(il::Range range) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.begin) <
      static_cast<std::size_t>(this->size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.end) <=
      static_cast<std::size_t>(this->size()));

  return il::ArrayView<T>{this->data_ + range.begin, range.end - range.begin};
};

template <typename T>
il::ArrayEdit<T> ArrayEdit<T>::Edit() {
  return il::ArrayEdit<T>{this->data_, this->size()};
};

template <typename T>
il::ArrayEdit<T> ArrayEdit<T>::Edit(il::Range range) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.begin) <
                   static_cast<std::size_t>(this->size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.end) <=
                   static_cast<std::size_t>(this->size()));

  return il::ArrayEdit<T>{this->data_ + range.begin, range.end - range.begin};
};

template <typename T>
T* ArrayEdit<T>::Data() {
  return this->data_;
}

}  // namespace il

#endif  // IL_ARRAYVIEW_H
