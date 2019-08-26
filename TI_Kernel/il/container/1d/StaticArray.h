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

#ifndef IL_STATICARRAY_H
#define IL_STATICARRAY_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>

#include <il/container/1d/ArrayView.h>

namespace il {

template <typename T, il::int_t n>
class StaticArray {
  static_assert(n >= 0, "il::StaticArray<T, n>: n must be nonnegative");

 private:
  T data_[n > 0 ? n : 1];
  il::int_t size_ = n;

 public:
  /* \brief The default constructor
  // \details If T is a numeric value, the memory is
  // - (Debug mode) initialized to il::defaultValue<T>(). It is usually NaN
  //   if T is a floating point number or 666..666 if T is an integer.
  // - (Release mode) left uninitialized. This behavior is different from
  //   std::vector from the standard library which initializes all numeric
  //   values to 0.
  */
  StaticArray();

  /* \brief Construct a il::StaticArray<T, n> elements with a value
  /
  // // Construct a vector of double of length 5, initialized with 0.0
  // il::StaticArray<double, 5> v{0.0};
  */
  explicit StaticArray(const T& value);

  /* \brief Construct a il::StaticArray<T, n> from a brace-initialized list
  // \details In order to allow brace initialization in all cases, this
  // constructor has different syntax from the one found in std::array. You
  // must use the il::value value as a first argument. For instance:
  //
  // il::StaticArray<double, 4> v{il::value, {2.0, 3.14, 5.0, 7.0}};
  //
  // The length of the initializer list is checked against the vector length
  // in debug mode. In release mode, if the length do not match, the result
  // is undefined behavior.
  */
  StaticArray(il::value_t, std::initializer_list<T> list);

  /* \brief Accessor for a const il::StaticArray<T, n>
  // \details Access (read only) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::StaticArray<il::int_t, 4> v = 0;
  // std::cout << v[0] << std::endl;
  */
  const T& operator[](il::int_t i) const;

  /* \brief Accessor for a il::StaticArray<T, n>
  // \details Access (read only) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::StaticArray<double, 4> v{};
  // v[0] = 0.0;
  // v[4] = 0.0; // Program is aborted in debug mode and has undefined
  //             // behavior in release mode
  */
  T& operator[](il::int_t i);

  /* \brief Accessor on the last element for a const il::SaticArray<T, n>
  // \details This method does not compile for empty vectors
  */
  const T& back() const;

  /* \brief Accessor on the last element
  // \details This method does not compile for empty vectors
  */
  T& Back();

  /* \brief Get the size of the il::StaticArray<T, n>
  //
  // il::StaticArray<double, 4> v{};
  // for (il::int_t i = 0; i < v.size(); ++i) {
  //     v[i] = 1 / static_cast<double>(i);
  // }
  */
  il::int_t size() const;

  il::ArrayView<T> view() const;

  il::ArrayView<T> view(il::Range range) const;

  il::ArrayEdit<T> Edit();

  il::ArrayEdit<T> Edit(il::Range range);

  /* \brief Get a pointer to the first element of the array for a const
  // object
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Get a pointer to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* Data();
};

template <typename T, il::int_t n>
StaticArray<T, n>::StaticArray() {
  if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
    for (il::int_t i = 0; i < n; ++i) {
      data_[i] = il::defaultValue<T>();
    }
#endif
  }
}

template <typename T, il::int_t n>
StaticArray<T, n>::StaticArray(const T& value) {
  for (il::int_t i = 0; i < n; ++i) {
    data_[i] = value;
  }
}

template <typename T, il::int_t n>
StaticArray<T, n>::StaticArray(il::value_t, std::initializer_list<T> list) {
  IL_EXPECT_FAST(static_cast<std::size_t>(n) == list.size());

  if (il::isTrivial<T>::value) {
    memcpy(data_, list.begin(), n * sizeof(T));
  } else {
    for (il::int_t i = 0; i < n; ++i) {
      data_[i] = *(list.begin() + i);
    }
  }
}

template <typename T, il::int_t n>
const T& StaticArray<T, n>::operator[](il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) < static_cast<std::size_t>(n));
  return data_[i];
}

template <typename T, il::int_t n>
T& StaticArray<T, n>::operator[](il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) < static_cast<std::size_t>(n));
  return data_[i];
}

template <typename T, il::int_t n>
const T& StaticArray<T, n>::back() const {
  static_assert(n > 0,
                "il::StaticArray<T, n>: n must be positive to call back()");
  return data_[n - 1];
}

template <typename T, il::int_t n>
T& StaticArray<T, n>::Back() {
  static_assert(n > 0,
                "il::StaticArray<T, n>: n must be positive to call back()");
  return data_[n - 1];
}

template <typename T, il::int_t n>
il::int_t StaticArray<T, n>::size() const {
  return n;
}

template <typename T, il::int_t n>
il::ArrayView<T> StaticArray<T, n>::view() const {
  return il::ArrayView<T>{data_, n};
};

template <typename T, il::int_t n>
il::ArrayView<T> StaticArray<T, n>::view(il::Range range) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.begin) <
                   static_cast<std::size_t>(n));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.end) <=
                   static_cast<std::size_t>(n));

  return il::ArrayView<T>{data_ + range.begin, range.end - range.begin};
};

template <typename T, il::int_t n>
il::ArrayEdit<T> StaticArray<T, n>::Edit() {
  return il::ArrayEdit<T>{data_, n};
};

template <typename T, il::int_t n>
il::ArrayEdit<T> StaticArray<T, n>::Edit(il::Range range) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.begin) <
                   static_cast<std::size_t>(n));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.end) <=
                   static_cast<std::size_t>(n));

  return il::ArrayEdit<T>{data_ + range.begin, range.end - range.begin};
};

template <typename T, il::int_t n>
const T* StaticArray<T, n>::data() const {
  return data_;
}

template <typename T, il::int_t n>
T* StaticArray<T, n>::Data() {
  return data_;
}

}  // namespace il

#endif  // IL_STATICARRAY_H
