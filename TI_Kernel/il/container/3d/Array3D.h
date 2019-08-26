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

#ifndef IL_ARRAY3D_H
#define IL_ARRAY3D_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <new> is needed for placement new
#include <new>
// <utility> is needed for std::move
#include <utility>

#include <il/core.h>
#include <il/core/memory/allocate.h>

namespace il {

template <typename T>
class Array3D {
 private:
  T* data_;
  T* size_[3];
  T* capacity_[3];
  short alignment_;
  short align_r_;
  short align_mod_;
  short shift_;

 public:
  /* \brief Default constructor
  // \details The size and the capacity of the array are set to 0 and no memory
  // allocation is done during the process.
  */
  Array3D();

  /* \brief Construct an il::Array3D<T> of n0 rows, n1 columns and n2 slices
  // \details The row size and the row capacity of the array are set to n0. The
  // column size and the column capacity of the array are set to n1. The slice
  // size and the slice capacity of the array are set to n2.
  // - If T is a numeric value, the memory is
  //   - (Debug mode) initialized to il::defaultValue<T>(). It is usually NaN
  //     if T is a floating point number or 666..666 if T is an integer.
  //   - (Release mode) left uninitialized. This behavior is different from
  //     std::vector from the standard library which initializes all numeric
  //     values to 0.
  // - If T is an object with a default constructor, all objects are default
  //   constructed. A compile-time error is raised if T has no default
  //   constructor.
  //
  // // Construct an array of double with 3 rows, 5 columns and 7 slices.
  // il::Array3D<double> A{3, 5, 7};
  */
  explicit Array3D(il::int_t n0, il::int_t n1, il::int_t n2);

  /* \brief Construct an aligned array
  // \details The pointer data, when considered as an integer, satisfies
  // data_ = 0 (Modulo align_mod)
  */
  explicit Array3D(il::int_t n0, il::int_t n1, il::int_t n2, il::align_t,
                   il::int_t alignment);

  /* \brief Construct an aligned array
  // \details The pointer data, when considered as an integer, satisfies
  // data_ = align_r (Modulo align_mod)
  */
  explicit Array3D(il::int_t n0, il::int_t n1, il::int_t n2, il::align_t,
                   il::int_t alignment, il::int_t align_r, il::int_t align_mod);

  /* \brief Construct an array of n0 rows and n1 columns and n2 slices with a
  // value
  /
  // // Construct an array of double with 3 rows, 5 columns and 7 slices,
  // // initialized with 3.14.
  // il::Array2D<double> A{3, 5, 7, 3.14};
  */
  explicit Array3D(il::int_t n0, il::int_t n1, il::int_t n2, const T& x);

  explicit Array3D(il::int_t n0, il::int_t n1, il::int_t n2, const T& x,
                   il::align_t, il::int_t alignment);

  explicit Array3D(il::int_t n0, il::int_t n1, il::int_t n2, const T& x,
                   il::align_t, il::int_t alignment, il::int_t align_r,
                   il::int_t align_mod);

  /* \brief Construct an array from a brace-initialized list
  //
  // // Construct an array of double with 2 rows, 3 columns and 2 slices from a
  // // list
  // il::Array3D<double> v{il::value,
  //                       {{2.0, 3.0}, {4.0, 5.0}, {6.0, 7.0}},
  //                       {{2.5, 3.5}, {4.5, 5.5}, {6.5, 7.5}}};
  */
  explicit Array3D(
      il::value_t,
      std::initializer_list<std::initializer_list<std::initializer_list<T>>>
          list);

  /* \brief The copy constructor
  // \details The different size and capacity of the constructed il::Array3D<T>
  // are equal to the size of the source array.
  */
  Array3D(const Array3D<T>& A);

  /* \brief The move constructor
   */
  Array3D(Array3D<T>&& A);

  /* \brief The copy assignment
  // \details The size is the same as the one for the source array. The
  // capacity is not changed if it is enough for the copy to happen and is
  // set to the size of the source array if the initial capacity is too low.
  */
  Array3D& operator=(const Array3D<T>& A);

  /* \brief The move assignment
   */
  Array3D& operator=(Array3D<T>&& A);

  /* \brief The destructor
   */
  ~Array3D();

  /* \brief Accessor for a const il::3DArray<T>
  // \details Access (read only) the (i, j, k)-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::Array3D<double> A{4, 6, 7};
  // std::cout << A(3, 5, 6) << std::endl;
  */
  const T& operator()(il::int_t i0, il::int_t i1, il::int_t i2) const;

  /* \brief Accessor for a il::3DArray<T>
  // \details Access (read and write) the (i, j, k)-th element of the array.
  // Bound checking is done in debug mode but not in release mode.
  //
  // il::Array3D<double> A{4, 6, 7};
  // A(0, 0, 0) = 3.14;
  */
  T& operator()(il::int_t i0, il::int_t i1, il::int_t i2);

  /* \brief Get the size of the il::Array3D<T>
  // \details size(0) returns the number of rows of the array and size(1)
  // returns the number of columns of the same array and size(2) return the
  // number of slices of the array. The library has been designed in a way that
  // any compiler can prove that modifying A(i, j, k) can't
  // change the result of A.size(0), A.size(1) or A.size(2). As a consequence
  // a call to A.size(0), A.size(1) or A.size(2) are made just once at the very
  // beginning of the loop in the following example. It allows many
  // optimizations from the compiler, including automatic vectorization.
  //
  // il::Array3D<double> A{n, p, q};
  // for (il::int_t i = 0; i < v.size(0); ++i) {
  //   for (il::int_t j = 0; j < v.size(1); ++j) {
  //     for (il::int_t k = 0; k < v.size(2); ++k) {
  //       A(i, j, k) = 1.0 / (i + j + k + 3);
  //     }
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Resizing an il::Array3D<T>
  // \details No reallocation is performed if the new size is <= to the
  // capacity for both rows, columns and slices. In this case, the capacity is
  // unchanged. When one of the sizes is > than the the capacity, reallocation
  // is done and the array gets the same capacity as its size.
  */
  void Resize(il::int_t n0, il::int_t n1, il::int_t n2);

  /* \brief Get the capacity of the il::Array3D<T>
  // \details capacity(0) gives the capacity in terms of rows and capacity(1)
  // gives the capacity in terms of columns and capacity(2) gives the capacity
  // in terms of slices.
  */
  il::int_t capacity(il::int_t d) const;

  /* \brief Change the capacity of the array to at least r, s columns and t
  // slices
  // \details If the row capacity is >= to r, the column capacity is >= to s,
  // and the slice capacity is >= to t, nothing is done. Otherwise, reallocation
  // is done and the new capacity is set to r, s and t
  */
  void Reserve(il::int_t r0, il::int_t r1, il::int_t r2);

  /* \brief Get the alignment of the pointer returned by data()
   */
  il::int_t alignment() const;

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Get a pointer to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* Data();

  /* \brief Get a pointer to the first element of the column
  // \details One should use this method only when using C-style API
  */
  //  const T* data(il::int_t i1, il::int_t i2) const;

  /* \brief Get a pointer to the first element of the column
  // \details One should use this method only when using C-style API
  */
  //  T* Data(il::int_t i1, il::int_t i2);

  /* \brief The memory position of A(i0, i1, i2) is
  // data() + i2 * stride(2) + i1 * stride(1) + i0
  */
  il::int_t stride(il::int_t d) const;

 private:
  /* \brief Used internally in debug mode to check the invariance of the object
   */
  bool invariance() const;
};

template <typename T>
Array3D<T>::Array3D() {
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  size_[2] = nullptr;
  capacity_[0] = nullptr;
  capacity_[1] = nullptr;
  capacity_[2] = nullptr;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array3D<T>::Array3D(il::int_t n0, il::int_t n1, il::int_t n2) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(n2 >= 0);

  const il::int_t r0 = n0 > 0 ? n0 : ((n1 == 0 && n2 == 0) ? 0 : 1);
  const il::int_t r1 = n1 > 0 ? n1 : ((n0 == 0 && n2 == 0) ? 0 : 1);
  const il::int_t r2 = n2 > 0 ? n2 : ((n0 == 0 && n1 == 0) ? 0 : 1);
  bool error;
  const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    data_ = il::allocateArray<T>(r);
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = 0; i0 < n0; ++i0) {
            data_[(i2 * r1 + i1) * r0 + i0] = il::defaultValue<T>();
          }
        }
      }
#endif
    } else {
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = 0; i0 < n0; ++i0) {
            new (data_ + (i2 * r1 + i1) * r0 + i0) T{};
          }
        }
      }
    }
  } else {
    data_ = nullptr;
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array3D<T>::Array3D(il::int_t n0, il::int_t n1, il::int_t n2, il::align_t,
                    il::int_t alignment, il::int_t align_r,
                    il::int_t align_mod) {
  IL_EXPECT_FAST(il::isTrivial<T>::value);
  IL_EXPECT_FAST(sizeof(T) % alignof(T) == 0);
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(n2 >= 0);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_mod <= SHRT_MAX);
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);
  IL_EXPECT_FAST(align_r <= SHRT_MAX);
  IL_EXPECT_FAST(alignment > 0);
  IL_EXPECT_FAST(alignment % alignof(T) == 0);
  IL_EXPECT_FAST(alignment <= SHRT_MAX);
  IL_EXPECT_FAST(align_r % alignment == 0);
  IL_EXPECT_FAST(align_mod % alignment == 0);

  il::int_t r0;
  il::int_t r1;
  il::int_t r2;
  if (n0 > 0 && n1 > 0 && n2 > 0) {
    if (il::isTrivial<T>::value && align_mod != 0) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(alignment) / alignof(T));
      bool error = false;
      r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
      r1 = n1;
      r2 = n2;
    } else {
      r0 = n0;
      r1 = n1;
      r2 = n2;
    }
  } else if (n0 == 0 && n1 == 0 && n2 == 0) {
    r0 = 0;
    r1 = 0;
    r2 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
    r2 = (n2 == 0) ? 1 : n2;
  }
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    il::int_t shift;
    data_ = il::allocateArray<T>(r, align_r, align_mod, il::io, shift);
    shift_ = static_cast<short>(shift);
#ifdef IL_DEFAULT_VALUE
    for (il::int_t i2 = 0; i2 < n2; ++i2) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          data_[(i2 * r1 + i1) * r0 + i0] = il::defaultValue<T>();
        }
      }
    }
#endif
  } else {
    data_ = nullptr;
    shift_ = 0;
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  alignment_ = static_cast<short>(alignment);
  align_r_ = static_cast<short>(align_r);
  align_mod_ = static_cast<short>(align_mod);
}

template <typename T>
Array3D<T>::Array3D(il::int_t n0, il::int_t n1, il::int_t n2, il::align_t,
                    il::int_t alignment)
    : Array3D{n0, n1, n2, il::align, alignment, 0, alignment} {}

template <typename T>
Array3D<T>::Array3D(il::int_t n0, il::int_t n1, il::int_t n2, const T& x) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(n2 >= 0);

  const il::int_t r0 = n0 > 0 ? n0 : ((n1 == 0 && n2 == 0) ? 0 : 1);
  const il::int_t r1 = n1 > 0 ? n1 : ((n0 == 0 && n2 == 0) ? 0 : 1);
  const il::int_t r2 = n2 > 0 ? n2 : ((n0 == 0 && n1 == 0) ? 0 : 1);
  bool error;
  const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    data_ = il::allocateArray<T>(r);
    for (il::int_t i2 = 0; i2 < n2; ++i2) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          new (data_ + (i2 * r1 + i1) * r0 + i0) T(x);
        }
      }
    }
  } else {
    data_ = nullptr;
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array3D<T>::Array3D(il::int_t n0, il::int_t n1, il::int_t n2, const T& x,
                    il::align_t, il::int_t alignment, il::int_t align_r,
                    il::int_t align_mod) {
  IL_EXPECT_FAST(il::isTrivial<T>::value);
  IL_EXPECT_FAST(sizeof(T) % alignof(T) == 0);
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(n2 >= 0);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_mod <= SHRT_MAX);
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);
  IL_EXPECT_FAST(align_r <= SHRT_MAX);
  IL_EXPECT_FAST(alignment > 0);
  IL_EXPECT_FAST(alignment % alignof(T) == 0);
  IL_EXPECT_FAST(alignment <= SHRT_MAX);
  IL_EXPECT_FAST(align_r % alignment == 0);
  IL_EXPECT_FAST(align_mod % alignment == 0);

  il::int_t r0;
  il::int_t r1;
  il::int_t r2;
  if (n0 > 0 && n1 > 0 && n2 > 0) {
    if (il::isTrivial<T>::value && align_mod != 0) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(alignment) / alignof(T));
      bool error = false;
      r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
      r1 = n1;
      r2 = n2;
    } else {
      r0 = n0;
      r1 = n1;
      r2 = n2;
    }
  } else if (n0 == 0 && n1 == 0 && n2 == 0) {
    r0 = 0;
    r1 = 0;
    r2 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
    r2 = (n2 == 0) ? 1 : n2;
  }
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    il::int_t shift;
    data_ = il::allocateArray<T>(r, align_r, align_mod, il::io, shift);
    shift_ = static_cast<short>(shift);
#ifdef IL_DEFAULT_VALUE
    for (il::int_t i2 = 0; i2 < n2; ++i2) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          data_[(i2 * r1 + i1) * r0 + i0] = x;
        }
      }
    }
#endif
  } else {
    data_ = nullptr;
    shift_ = 0;
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  alignment_ = static_cast<short>(alignment);
  align_r_ = static_cast<short>(align_r);
  align_mod_ = static_cast<short>(align_mod);
}

template <typename T>
Array3D<T>::Array3D(il::int_t n0, il::int_t n1, il::int_t n2, const T& x,
                    il::align_t, il::int_t alignment)
    : Array3D{n0, n1, n2, x, il::align, alignment, 0, alignment} {}

template <typename T>
Array3D<T>::Array3D(
    il::value_t,
    std::initializer_list<std::initializer_list<std::initializer_list<T>>>
        list) {
  bool error;
  const il::int_t n2 = il::safeConvert<il::int_t>(list.size(), il::io, error);
  if (error) {
    il::abort();
  }
  const il::int_t n1 =
      n2 > 0 ? il::safeConvert<il::int_t>(list.begin()->size(), il::io, error)
             : 0;
  if (error) {
    il::abort();
  }
  const il::int_t n0 = n1 > 0
                           ? il::safeConvert<il::int_t>(
                                 list.begin()->begin()->size(), il::io, error)
                           : 0;
  if (error) {
    il::abort();
  }

  il::int_t r0;
  il::int_t r1;
  il::int_t r2;
  if (n0 > 0 && n1 > 0 && n2 > 0) {
    r0 = n0;
    r1 = n1;
    r2 = n2;
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
    if (error) {
      il::abort();
    }
    data_ = il::allocateArray<T>(r);
    if (il::isTrivial<T>::value) {
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        IL_EXPECT_FAST(static_cast<il::int_t>((list.begin() + i2)->size()) ==
                       n1);
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          IL_EXPECT_FAST(static_cast<il::int_t>(
                             ((list.begin() + i2)->begin() + i1)->size()) ==
                         n0);
          memcpy(data_ + (i2 * r1 + i1) * r0,
                 ((list.begin() + i2)->begin() + i1)->begin(), n0 * sizeof(T));
        }
      }
    } else {
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        IL_EXPECT_FAST(static_cast<il::int_t>((list.begin() + i2)->size()) ==
                       n1);
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          IL_EXPECT_FAST(static_cast<il::int_t>(
                             ((list.begin() + i2)->begin() + i1)->size()) ==
                         n0);
          for (il::int_t i0 = 0; i0 < n0; ++i0) {
            new (data_ + (i2 * r1 + i1) * r0 + i0)
                T(*(((list.begin() + i2)->begin() + i1)->begin() + i0));
          }
        }
      }
    }
  } else if (n0 == 0 && n1 == 0 && n2 == 0) {
    r0 = 0;
    r1 = 0;
    r2 = 0;
    data_ = nullptr;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
    r2 = (n2 == 0) ? 1 : n2;
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
    if (error) {
      il::abort();
    }
    data_ = il::allocateArray<T>(r);
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array3D<T>::Array3D(const Array3D<T>& A) {
  const il::int_t n0 = A.size(0);
  const il::int_t n1 = A.size(1);
  const il::int_t n2 = A.size(2);
  il::int_t r0;
  il::int_t r1;
  il::int_t r2;
  if (n0 > 0 && n1 > 0 && n2 > 0) {
    if (il::isTrivial<T>::value && A.alignment_ != 0) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(A.alignment_) / alignof(T));
      bool error = false;
      r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
      r1 = n1;
      r2 = n2;
    } else {
      r0 = n0;
      r1 = n1;
      r2 = n2;
    }
  } else if (n0 == 0 && n1 == 0 && n2 == 0) {
    r0 = 0;
    r1 = 0;
    r2 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
    r2 = (n2 == 0) ? 1 : n2;
  }
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
  if (error) {
    il::abort();
  }
  if (il::isTrivial<T>::value) {
    if (A.alignment_ == 0) {
      data_ = il::allocateArray<T>(r);
      shift_ = 0;
    } else {
      il::int_t shift;
      data_ = il::allocateArray<T>(r, A.align_r_, A.align_mod_, il::io, shift);
      shift_ = static_cast<short>(shift);
    }
    for (il::int_t i2 = 0; i2 < n2; ++i2) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        memcpy(data_ + (i2 * r1 + i1) * r0,
               A.data_ + (i2 * A.capacity(1) + i1) * A.capacity(0),
               n0 * sizeof(T));
      }
    }
  } else {
    data_ = il::allocateArray<T>(r);
    shift_ = 0;
    for (il::int_t i2 = 0; i2 < n2; ++i2) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          new (data_ + (i2 * r1 + i1) * r0 + i0) T(A(i0, i1, i2));
        }
      }
    }
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  alignment_ = A.alignment_;
  align_r_ = A.align_r_;
  align_mod_ = A.align_mod_;
}

template <typename T>
Array3D<T>::Array3D(Array3D<T>&& A) {
  data_ = A.data_;
  size_[0] = A.size_[0];
  size_[1] = A.size_[1];
  size_[2] = A.size_[2];
  capacity_[0] = A.capacity_[0];
  capacity_[1] = A.capacity_[1];
  capacity_[2] = A.capacity_[2];
  alignment_ = A.alignment_;
  align_r_ = A.align_r_;
  align_mod_ = A.align_mod_;
  shift_ = A.shift_;
  A.data_ = nullptr;
  A.size_[0] = nullptr;
  A.size_[1] = nullptr;
  A.size_[2] = nullptr;
  A.capacity_[0] = nullptr;
  A.capacity_[1] = nullptr;
  A.capacity_[2] = nullptr;
  A.alignment_ = 0;
  A.align_r_ = 0;
  A.align_mod_ = 0;
  A.shift_ = 0;
}

template <typename T>
Array3D<T>& Array3D<T>::operator=(const Array3D<T>& A) {
  if (this != &A) {
    const il::int_t n0 = A.size(0);
    const il::int_t n1 = A.size(1);
    const il::int_t n2 = A.size(2);
    const il::int_t alignment = A.alignment_;
    const il::int_t align_r = A.align_r_;
    const il::int_t align_mod = A.align_mod_;
    const bool need_memory = capacity(0) < n0 || capacity(1) < n1 ||
                             capacity(2) < n2 || align_mod_ != align_mod ||
                             align_r_ != align_r || alignment_ != alignment;
    if (need_memory) {
      il::int_t r0;
      il::int_t r1;
      il::int_t r2;
      if (n0 > 0 && n1 > 0 && n2 > 0) {
        if (il::isTrivial<T>::value && alignment != 0) {
          const il::int_t nb_lanes = static_cast<il::int_t>(
              static_cast<std::size_t>(alignment) / alignof(T));
          bool error = false;
          r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
          if (error) {
            il::abort();
          }
          r1 = n1;
          r2 = n2;
        } else {
          r0 = n0;
          r1 = n1;
          r2 = n2;
        }
      } else {
        r0 = (n0 == 0) ? 1 : n0;
        r1 = (n1 == 0) ? 1 : n1;
        r2 = (n2 == 0) ? 1 : n2;
      }
      bool error = false;
      const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
      if (error) {
        il::abort();
      }
      if (il::isTrivial<T>::value) {
        if (data_) {
          il::deallocate(data_ - shift_);
        }
        if (alignment == 0) {
          data_ = il::allocateArray<T>(r);
          shift_ = 0;
        } else {
          il::int_t shift;
          data_ = il::allocateArray<T>(r, align_r, align_mod, il::io, shift);
          shift_ = static_cast<short>(shift);
        }
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            memcpy(data_ + (i2 * r1 + i1) * r0,
                   A.data_ + (i2 * A.capacity(1) + i1) * A.capacity(0),
                   n0 * sizeof(T));
          }
        }
      } else {
        if (data_) {
          for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
            for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
              for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
                (data_ + (i2 * capacity(1) + i1) * capacity(0) + i0)->~T();
              }
            }
          }
          il::deallocate(data_);
        }
        data_ = il::allocateArray<T>(r);
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i0 = 0; i0 < n0; ++i0) {
              new (data_ + (i2 * r1 + i1) * r0 + i0) T(A(i0, i1, i2));
            }
          }
        }
      }
      size_[0] = data_ + n0;
      size_[1] = data_ + n1;
      size_[2] = data_ + n2;
      capacity_[0] = data_ + r0;
      capacity_[1] = data_ + r1;
      capacity_[2] = data_ + r2;
      alignment_ = static_cast<short>(alignment);
      align_r_ = static_cast<short>(align_r);
      align_mod_ = static_cast<short>(align_mod);
    } else {
      if (il::isTrivial<T>::value) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            memcpy(data_ + (i2 * capacity(1) + i1) * capacity(0),
                   A.data_ + (i2 * A.capacity(1) + i1) * A.capacity(0),
                   n0 * sizeof(T));
          }
        }
      } else {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i0 = 0; i0 < n0; ++i0) {
              data_[(i2 * capacity(1) + i1) * capacity(0) + i0] = A(i0, i1, i2);
            }
          }
        }
        for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
          for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
            for (il::int_t i0 = size(0) - 1;
                 i0 >= (i2 < n2 && i1 < n1 ? n0 : 0); --i0) {
              (data_ + (i2 * capacity(1) + i1) * capacity(0) + i0)->~T();
            }
          }
        }
      }
      size_[0] = data_ + n0;
      size_[1] = data_ + n1;
      size_[2] = data_ + n2;
    }
  }
  return *this;
}

template <typename T>
Array3D<T>& Array3D<T>::operator=(Array3D<T>&& A) {
  if (this != &A) {
    if (data_) {
      if (!il::isTrivial<T>::value) {
        for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
          for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
            for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
              (data_ + (i2 * capacity(1) + i1) * capacity(0) + i0)->~T();
            }
          }
        }
      }
      il::deallocate(data_ - shift_);
    }
    data_ = A.data_;
    size_[0] = A.size_[0];
    size_[1] = A.size_[1];
    size_[2] = A.size_[2];
    capacity_[0] = A.capacity_[0];
    capacity_[1] = A.capacity_[1];
    capacity_[2] = A.capacity_[2];
    alignment_ = A.alignment_;
    align_r_ = A.align_r_;
    align_mod_ = A.align_mod_;
    shift_ = A.shift_;
    A.data_ = nullptr;
    A.size_[0] = nullptr;
    A.size_[1] = nullptr;
    A.size_[2] = nullptr;
    A.capacity_[0] = nullptr;
    A.capacity_[1] = nullptr;
    A.capacity_[2] = nullptr;
    A.alignment_ = 0;
    A.align_r_ = 0;
    A.align_mod_ = 0;
    A.shift_ = 0;
  }
  return *this;
}

template <typename T>
Array3D<T>::~Array3D() {
  IL_EXPECT_FAST_NOTHROW(invariance());

  if (data_) {
    if (!il::isTrivial<T>::value) {
      for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
        for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
            (data_ + (i2 * capacity(1) + i1) * capacity(0) + i0)->~T();
          }
        }
      }
    }
    il::deallocate(data_ - shift_);
  }
}

template <typename T>
const T& Array3D<T>::operator()(il::int_t i0, il::int_t i1,
                                il::int_t i2) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i2) <
                   static_cast<std::size_t>(size(2)));

  return data_[(i2 * (capacity_[1] - data_) + i1) * (capacity_[0] - data_) +
               i0];
}

template <typename T>
T& Array3D<T>::operator()(il::int_t i0, il::int_t i1, il::int_t i2) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i2) <
                   static_cast<std::size_t>(size(2)));

  return data_[(i2 * (capacity_[1] - data_) + i1) * (capacity_[0] - data_) +
               i0];
}

template <typename T>
il::int_t Array3D<T>::size(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(3));

  return size_[d] - data_;
}

template <typename T>
void Array3D<T>::Resize(il::int_t n0, il::int_t n1, il::int_t n2) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(n2 >= 0);

  const il::int_t n0_old = size(0);
  const il::int_t n1_old = size(1);
  const il::int_t n2_old = size(2);
  const bool need_memory =
      (n0 > capacity(0)) || (n1 > capacity(1)) || (n2 > capacity(2));
  if (need_memory) {
    il::int_t r0;
    il::int_t r1;
    il::int_t r2;
    if (n0 > 0 && n1 > 0 && n2 > 0) {
      if (il::isTrivial<T>::value && alignment_ != 0) {
        const il::int_t nb_lanes = static_cast<il::int_t>(
            static_cast<std::size_t>(alignment_) / alignof(T));
        bool error = false;
        r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
        if (error) {
          il::abort();
        }
        r1 = n1;
        r2 = n2;
      } else {
        r0 = n0;
        r1 = n1;
        r2 = n2;
      }
    } else {
      r0 = (n0 == 0) ? 1 : n0;
      r1 = (n1 == 0) ? 1 : n1;
      r2 = (n2 == 0) ? 1 : n2;
    }
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
    if (error) {
      il::abort();
    }
    T* new_data;
    il::int_t new_shift;
    if (il::isTrivial<T>::value) {
      if (alignment_ == 0) {
        new_data = il::allocateArray<T>(r);
        new_shift = 0;
      } else {
        new_data =
            il::allocateArray<T>(r, align_r_, align_mod_, il::io, new_shift);
      }
      if (data_) {
        for (il::int_t i2 = 0; i2 < (n2 < n2_old ? n2 : n2_old); ++i2) {
          for (il::int_t i1 = 0; i1 < (n1 < n1_old ? n1 : n1_old); ++i1) {
            memcpy(new_data + (i2 * r1 + i1) * r0,
                   data_ + (i2 * capacity(1) + i1) * capacity(0),
                   (n0 < n0_old ? n0 : n0_old) * sizeof(T));
          }
        }
        il::deallocate(data_ - shift_);
      }
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = i2 < n2_old && i1 < n1_old ? n0_old : 0; i0 < n0;
               ++i0) {
            new_data[(i2 * r1 + i1) * r0 + i0] = il::defaultValue<T>();
          }
        }
      }
#endif
    } else {
      new_data = il::allocateArray<T>(r);
      new_shift = 0;
      if (data_) {
        for (il::int_t i2 = n2_old - 1; i2 >= 0; --i2) {
          for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
            for (il::int_t i0 = n0_old - 1; i0 >= (i2 < n2 && i1 < n1 ? n0 : 0);
                 --i0) {
              (data_ + (i2 * capacity(1) + i1) * capacity(0) + i0)->~T();
            }
          }
        }
        for (il::int_t i2 = (n2 < n2_old ? n2 : n2_old) - 1; i2 >= 0; --i2) {
          for (il::int_t i1 = (n1 < n1_old ? n1 : n1_old) - 1; i1 >= 0; --i1) {
            for (il::int_t i0 = (n0 < n0_old ? n0 : n0_old) - 1; i0 >= 0;
                 --i0) {
              new (new_data + (i2 * r1 + i1) * r0 + i0) T(
                  std::move(data_[(i2 * capacity(1) + i1) * capacity(0) + i0]));
              (data_ + (i2 * capacity(1) + i1) * capacity(0) + i0)->~T();
            }
          }
        }
        il::deallocate(data_);
      }
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = i2 < n2_old && i1 < n1_old ? n0_old : 0; i0 < n0;
               ++i0) {
            new (new_data + (i2 * r1 + i1) * r0 + i0) T{};
          }
        }
      }
    }
    data_ = new_data;
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    capacity_[2] = data_ + r2;
    shift_ = static_cast<short>(new_shift);
  } else {
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = (i2 < n2_old && i1 < n1_old) ? n0_old : 0;
               i0 < n0; ++i0) {
            data_[(i2 * capacity(1) + i1) * capacity(0) + i0] =
                il::defaultValue<T>();
          }
        }
      }
#endif
    } else {
      for (il::int_t i2 = n2_old - 1; i2 >= 0; --i2) {
        for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = n0_old - 1; i0 >= ((i2 < n2 && i1 < n1) ? n0 : 0);
               --i0) {
            (data_ + (i2 * capacity(1) + i1) * capacity(0) + i0)->~T();
          }
        }
      }
    }
    for (il::int_t i2 = 0; i2 < n2; ++i2) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = (i2 < n2_old && i1 < n1_old) ? n0_old : 0; i0 < n0;
             ++i0) {
          new (data_ + (i2 * capacity(1) + i1) * capacity(0) + i0) T{};
        }
      }
    }
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
}

template <typename T>
il::int_t Array3D<T>::capacity(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(3));

  return capacity_[d] - data_;
}

template <typename T>
void Array3D<T>::Reserve(il::int_t r0, il::int_t r1, il::int_t r2) {
  IL_EXPECT_FAST(r0 >= 0);
  IL_EXPECT_FAST(r1 >= 0);
  IL_EXPECT_FAST(r2 >= 0);

  r0 = (r0 > capacity(0)) ? r0 : capacity(0);
  r1 = (r1 > capacity(1)) ? r1 : capacity(1);
  r2 = (r2 > capacity(2)) ? r2 : capacity(2);
  const bool need_memory =
      r0 > capacity(0) || r1 > capacity(1) || r2 > capacity(2);
  if (need_memory) {
    const il::int_t n0_old = size(0);
    const il::int_t n1_old = size(1);
    const il::int_t n2_old = size(2);
    if (il::isTrivial<T>::value && alignment_ != 0) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(alignment_) / alignof(T));
      bool error = false;
      r0 = il::safeUpperRound(r0, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
    }
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, r2, il::io, error);
    if (error) {
      il::abort();
    }
    T* new_data;
    il::int_t new_shift;
    if (il::isTrivial<T>::value) {
      if (alignment_ == 0) {
        new_data = il::allocateArray<T>(r);
        new_shift = 0;
      } else {
        new_data =
            il::allocateArray<T>(r, align_r_, align_mod_, il::io, new_shift);
      }
      if (data_) {
        for (il::int_t i2 = 0; i2 < n2_old; ++i2) {
          for (il::int_t i1 = 0; i1 < n1_old; ++i1) {
            memcpy(new_data + (i2 * r1 + i1) * r0,
                   data_ + (i2 * capacity(1) + i1) * capacity(0),
                   n0_old * sizeof(T));
          }
        }
        il::deallocate(data_ - shift_);
      }
    } else {
      new_data = il::allocateArray<T>(r);
      new_shift = 0;
      for (il::int_t i2 = n2_old - 1; i2 >= 0; --i2) {
        for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = n0_old - 1; i0 >= 0; --i0) {
            new (new_data + (i2 * r1 + i1) * r0 + i0)
                T(std::move(data_[(i2 * capacity(1) + i1) * capacity(0) + i0]));
            (data_ + (i2 * capacity(1) + i1) * capacity(0) + i0)->~T();
          }
        }
      }
      il::deallocate(data_);
    }
    data_ = new_data;
    size_[0] = data_ + n0_old;
    size_[1] = data_ + n1_old;
    size_[2] = data_ + n2_old;
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    capacity_[2] = data_ + r2;
    shift_ = static_cast<short>(new_shift);
  }
}

template <typename T>
il::int_t Array3D<T>::alignment() const {
  return alignment_;
}

template <typename T>
const T* Array3D<T>::data() const {
  return data_;
}

template <typename T>
T* Array3D<T>::Data() {
  return data_;
}

template <typename T>
il::int_t Array3D<T>::stride(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(3));

  return (d == 0)
             ? 1
             : (capacity_[0] - data_) * ((d == 1) ? 1 : (capacity_[1] - data_));
}

template <typename T>
bool Array3D<T>::invariance() const {
  bool ans = true;

  if (data_ == nullptr) {
    ans = ans && (size_[0] == nullptr);
    ans = ans && (size_[1] == nullptr);
    ans = ans && (size_[2] == nullptr);
    ans = ans && (capacity_[0] == nullptr);
    ans = ans && (capacity_[1] == nullptr);
    ans = ans && (capacity_[2] == nullptr);
    ans = ans && (alignment_ == 0);
    ans = ans && (align_r_ == 0);
    ans = ans && (align_mod_ == 0);
    ans = ans && (shift_ == 0);
  } else {
    ans = ans && (size_[0] != nullptr);
    ans = ans && (size_[1] != nullptr);
    ans = ans && (size_[2] != nullptr);
    ans = ans && (capacity_[0] != nullptr);
    ans = ans && (capacity_[1] != nullptr);
    ans = ans && (capacity_[2] != nullptr);
    ans = ans && ((size_[0] - data_) <= (capacity_[0] - data_));
    ans = ans && ((size_[1] - data_) <= (capacity_[1] - data_));
    ans = ans && ((size_[2] - data_) <= (capacity_[2] - data_));
    if (il::isTrivial<T>::value) {
      ans = ans && (alignment_ >= 0);
      ans = ans && (align_r_ >= 0);
      ans = ans && (align_mod_ >= 0);
      ans = ans && (alignment_ % alignof(T) == 0);
      ans = ans && (align_r_ % alignof(T) == 0);
      ans = ans && (align_mod_ % alignof(T) == 0);
      if (alignment_ > 0) {
        ans = ans && (align_r_ % alignment_ == 0);
        ans = ans && (align_mod_ > 0);
        ans = ans && (align_mod_ % alignment_ == 0);
        ans = ans && (align_r_ < align_mod_);
        ans = ans && (reinterpret_cast<std::size_t>(data_) %
                          static_cast<std::size_t>(align_mod_) ==
                      static_cast<std::size_t>(align_r_));
      } else {
        ans = ans && (align_r_ == 0);
        ans = ans && (align_mod_ == 0);
      }
    } else {
      ans = ans && (align_r_ == 0);
      ans = ans && (align_mod_ == 0);
      ans = ans && (alignment_ == 0);
    }
  }
  return ans;
}
}  // namespace il

#endif  // IL_ARRAY3D_H