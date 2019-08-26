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

#ifndef IL_STATICARRAY4D_H
#define IL_STATICARRAY4D_H

// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>

#include <il/core.h>

namespace il {

template <class T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3>
class StaticArray4D {
  static_assert(n0 >= 0,
                "il::StaticArray4D<T, n0, n1, n2, n3>: n0 must be nonnegative");
  static_assert(n1 >= 0,
                "il::StaticArray4D<T, n0, n1, n2, n3>: n1 must be nonnegative");
  static_assert(n2 >= 0,
                "il::StaticArray4D<T, n0, n1, n2, n3>: n2 must be nonnegative");

 private:
  T data_[n0 * n1 * n2 * n3 > 0 ? (n0* n1* n2* n3) : 1];
  il::int_t size_0_ = n0;
  il::int_t size_1_ = n1;
  il::int_t size_2_ = n2;
  il::int_t size_3_ = n3;

 public:
  StaticArray4D();
  StaticArray4D(const T& value);
  const T& operator()(il::int_t i0, il::int_t i1, il::int_t i2,
                      il::int_t i3) const;
  T& operator()(il::int_t i0, il::int_t i1, il::int_t i2, il::int_t i3);
  il::int_t size(il::int_t d) const;
  const T* data() const;
  T* Data();
};

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3>
StaticArray4D<T, n0, n1, n2, n3>::StaticArray4D() {
  if (il::isTrivial<T>::value) {
#ifndef NDEBUG
    for (il::int_t l = 0; l < n0 * n1 * n2 * n3; ++l) {
      data_[l] = il::defaultValue<T>();
    }
#endif
  }
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3>
StaticArray4D<T, n0, n1, n2, n3>::StaticArray4D(const T& value) {
  for (il::int_t l = 0; l < n0 * n1 * n2 * n3; ++l) {
    data_[l] = value;
  }
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3>
const T& StaticArray4D<T, n0, n1, n2, n3>::operator()(il::int_t i0,
                                                      il::int_t i1,
                                                      il::int_t i2,
                                                      il::int_t i3) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) < static_cast<std::size_t>(n0));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) < static_cast<std::size_t>(n1));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i2) < static_cast<std::size_t>(n2));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i3) < static_cast<std::size_t>(n3));

  return data_[((i3 * n2 + i2) * n1 + i1) * n0 + i0];
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3>
T& StaticArray4D<T, n0, n1, n2, n3>::operator()(il::int_t i0, il::int_t i1,
                                                il::int_t i2, il::int_t i3) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) < static_cast<std::size_t>(n0));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) < static_cast<std::size_t>(n1));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i2) < static_cast<std::size_t>(n2));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i3) < static_cast<std::size_t>(n3));

  return data_[((i3 * n2 + i2) * n1 + i1) * n0 + i0];
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3>
il::int_t StaticArray4D<T, n0, n1, n2, n3>::size(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(4));

  return d == 0 ? n0 : (d == 1 ? n1 : (d == 2 ? n2 : n3));
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3>
const T* StaticArray4D<T, n0, n1, n2, n3>::data() const {
  return data_;
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3>
T* StaticArray4D<T, n0, n1, n2, n3>::Data() {
  return data_;
}
}  // namespace il

#endif  // IL_STATICARRAY4D_H
