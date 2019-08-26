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

#ifndef IL_STATICARRAY3D_H
#define IL_STATICARRAY3D_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <type_traits> is needed for std::is_pod
#include <type_traits>

#include <il/core.h>

namespace il {

template <class T, il::int_t n0, il::int_t n1, il::int_t n2>
class StaticArray3D {
  static_assert(n0 >= 0,
                "il::StaticArray3D<T, n0, n1, n2>: n0 must be nonnegative");
  static_assert(n1 >= 0,
                "il::StaticArray3D<T, n0, n1, n2>: n1 must be nonnegative");
  static_assert(n2 >= 0,
                "il::StaticArray3D<T, n0, n1, n2>: n2 must be nonnegative");

 private:
  T data_[n0 * n1 * n2 > 0 ? (n0* n1* n2) : 1];
  il::int_t size_0_ = n0;
  il::int_t size_1_ = n1;
  il::int_t size_2_ = n2;

 public:
  /* \brief The default constructor
  // \details If T is a numeric value, the memory is
  // - (Debug mode) initialized to il::defaultValue<T>(). It is usually NaN
  //   if T is a floating point number or 666..666 if T is an integer.
  // - (Release mode) left uninitialized. This behavior is different from
  //   std::vector from the standard library which initializes all numeric
  //   values to 0.
  */
  StaticArray3D();

  /* \brief Construct a il::StaticArray3D<T, n0, n1, n2> elements with a value
  /
  // // Construct a static array of 3 rows, 5 columns and 7 slices, initialized
  // // with 0.0.
  // il::StaticArray3D<double, 3, 5, 7> A{0.0};
  */
  StaticArray3D(const T& value);

  /* \brief Construct a il::StaticArray3D<T, n0, n1, n2> from a
  brace-initialized
  // list
  // \details In order to allow brace initialization in all cases, this
  // constructor has different syntax from the one found in std::array. You
  // must use the il::value value as a first argument. For instance:
  //
  // // Construct an array of double with 2 rows, 3 columns and 2 slices from a
  // // list
  // il::StaticArray3D<double, 2, 3, 2> v{il::value,
  //                                      {2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
  //                                       2.5, 3.5, 4.5, 5.5, 6.5, 7.5}};
  */
  StaticArray3D(
      il::value_t,
      std::initializer_list<std::initializer_list<std::initializer_list<T>>>
          list);

  /* \brief Accessor for a const il::StaticArray3D<T, n0, n1, n2>
  // \details Access (read only) the (i, j, k)-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::StaticArray3D<double, 2, 5, 10> A{0.0};
  // std::cout << A(0, 0, 0) << std::endl;
  */
  const T& operator()(il::int_t i0, il::int_t i1, il::int_t i2) const;

  /* \brief Accessor for a il::StaticArray3D<T, n0, n1, n2>
  // \details Access (read and write) the (i, j, k)-th element of the array.
  // Bound checking is done in debug mode but not in release mode.
  //
  // il::StaticArray3D<double, 2, 5, 10> A{0.0};
  // A(0, 0, 0) = 0.0;
  */
  T& operator()(il::int_t i0, il::int_t i1, il::int_t i2);

  /* \brief Get the size of the il::StaticArray3D<T, n0, n1, n2>
  //
  // for (il::int_t i = 0; i < v.size(0); ++i) {
  //   for (il::int_t j = 0; j < v.size(1); ++j) {
  //     A(i, j) = 1.0 / (i + j + 2);
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Get an const array view to the container
   */
  //  ConstArray3DView<T> const_view() const;

  /* \brief Get an array view to the container
   */
  //  Array3DView<T> view();

  /* \brief Get an const array view to a subpart of the container with
  // the elements indexed by i with i_begin <= i < i_end, j with
  // j_begin <= j < j_end and k with k_begin <= k < k_end.
  */
  //  ConstArray3DView<T> const_view(il::int_t i_begin, il::int_t i_end,
  //                                 il::int_t j_begin, il::int_t j_end,
  //                                 il::int_t k_begin, il::int_t k_end) const;

  /* \brief Get an array view to a subpart of the container with
  // the elements indexed by i with i_begin <= i < i_end, j with
  // j_begin <= j < j_end and k with k_begin <= k < k_end.
  */
  //  Array3DView<T> view(il::int_t begin, il::int_t i_end, il::int_t j_begin,
  //                      il::int_t j_end, il::int_t k_begin, il::int_t k_end);

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

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
StaticArray3D<T, n0, n1, n2>::StaticArray3D() {
  if (il::isTrivial<T>::value) {
#ifndef NDEBUG
    for (il::int_t l = 0; l < n0 * n1 * n2; ++l) {
      data_[l] = il::defaultValue<T>();
    }
#endif
  }
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
StaticArray3D<T, n0, n1, n2>::StaticArray3D(const T& value) {
  for (il::int_t l = 0; l < n0 * n1 * n2; ++l) {
    data_[l] = value;
  }
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
StaticArray3D<T, n0, n1, n2>::StaticArray3D(
    il::value_t,
    std::initializer_list<std::initializer_list<std::initializer_list<T>>>
        list) {
  IL_EXPECT_FAST(n2 == static_cast<il::int_t>(list.size()));
  for (il::int_t i2 = 0; i2 < n2; ++i2) {
    IL_EXPECT_FAST(n1 == (list.begin() + i2)->size());
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      IL_EXPECT_FAST(n0 == ((list.begin() + i2)->begin() + i1)->size());
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        data_[(i2 * n1 + i1) * n0 + i0] =
            *(((list.begin() + i2)->begin() + i1)->begin() + i0);
      }
    }
  }
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
const T& StaticArray3D<T, n0, n1, n2>::operator()(il::int_t i0, il::int_t i1,
                                                  il::int_t i2) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(i0) < static_cast<std::size_t>(n0));
  IL_EXPECT_FAST(static_cast<std::size_t>(i1) < static_cast<std::size_t>(n1));
  IL_EXPECT_FAST(static_cast<std::size_t>(i2) < static_cast<std::size_t>(n2));
  return data_[(i2 * n1 + i1) * n0 + i0];
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
T& StaticArray3D<T, n0, n1, n2>::operator()(il::int_t i0, il::int_t i1,
                                            il::int_t i2) {
  IL_EXPECT_FAST(static_cast<std::size_t>(i0) < static_cast<std::size_t>(n0));
  IL_EXPECT_FAST(static_cast<std::size_t>(i1) < static_cast<std::size_t>(n1));
  IL_EXPECT_FAST(static_cast<std::size_t>(i2) < static_cast<std::size_t>(n2));
  return data_[(i2 * n1 + i1) * n0 + i0];
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
il::int_t StaticArray3D<T, n0, n1, n2>::size(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(3));
  return d == 0 ? n0 : (d == 1 ? n1 : n2);
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
const T* StaticArray3D<T, n0, n1, n2>::data() const {
  return data_;
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
T* StaticArray3D<T, n0, n1, n2>::Data() {
  return data_;
}

}  // namespace il

#endif  // IL_STATICARRAY3D_H
