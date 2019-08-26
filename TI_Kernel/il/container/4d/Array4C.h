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

#ifndef IL_ARRAY4C_H
#define IL_ARRAY4C_H

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
class Array4C {
 private:
  T* data_;
  T* size_[4];
  T* capacity_[4];
  short alignment_;
  short align_r_;
  short align_mod_;
  short shift_;

 public:
  /* \brief Default constructor
  // \details The size and the capacity of the array are set to 0 and no memory
  // allocation is done during the process.
  */
  Array4C();

  /* \brief Construct an il::Array4C<T> of n rows and p columns and q slices
  // \details The row size and the row capacity of the array are set to n. The
  // column size and the column capacity of the array are set to p. The slice
  // size and the slice capacity of the array are set to q.
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
  // il::Array4C<double> A{3, 5, 7};
  */
  explicit Array4C(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3);

  /* \brief Construct an array of n rows and p columns with a value
  /
  // // Construct an array of double with 3 rows and 5 columns, initialized with
  // // 3.14.
  // il::Array2D<double> A{3, 5, 3.14};
  */
  explicit Array4C(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3,
                   const T& x);

  /* \brief The copy constructor
  // \details The different size and capacity of the constructed il::Array4C<T>
  // are equal to the size of the source array.
  */
  Array4C(const Array4C<T>& A);

  /* \brief The move constructor
   */
  Array4C(Array4C<T>&& A);

  /* \brief The copy assignment
  // \details The size is the same as the one for the source array. The
  // capacity is not changed if it is enough for the copy to happen and is
  // set to the size of the source array if the initial capacity is too low.
  */
  Array4C& operator=(const Array4C<T>& A);

  /* \brief The move assignment
   */
  Array4C& operator=(Array4C<T>&& A);

  /* \brief The destructor
   */
  ~Array4C();

  /* \brief Accessor for a const il::4DArray<T>
  // \details Access (read only) the (i0, i1, i2, i3)-th element of the array.
  // Bound checking is done in debug mode but not in release mode.
  //
  // il::Array4C<double> A{4, 6, 7, 8};
  // std::cout << A(3, 5, 6, 4) << std::endl;
  */
  const T& operator()(il::int_t i0, il::int_t i1, il::int_t i2,
                      il::int_t i3) const;

  /* \brief Accessor for a il::4DArray<T>
  // \details Access (read and write) the (i0, i1, i2, i3)-th element of the
  // array. Bound checking is done in debug mode but not in release mode.
  //
  // il::Array4C<double> A{4, 6, 7, 8};
  // A(0, 0, 0, 0) = 3.14;
  */
  T& operator()(il::int_t i0, il::int_t i1, il::int_t i2, il::int_t i3);

  /* \brief Get the size of the il::Array4C<T>
  // \details size(0) returns the number of rows of the array and size(1)
  // returns the number of columns of the same array and size(2) return the
  // number of slices of the array. The library has been designed in a way that
  // any compiler can prove that modifying A(i, j, k) can't
  // change the result of A.size(0), A.size(1) or A.size(2). As a consequence
  // a call to A.size(0), A.size(1) or A.size(2) are made just once at the very
  // beginning of the loop in the following example. It allows many
  // optimizations from the compiler, including automatic vectorization.
  //
  // il::Array4C<double> A{n0, n1, n2, n3};
  // for (il::int_t i3 = 0; i3 < v.size(3); ++i3) {
  //   for (il::int_t i2 = 0; i2 < v.size(2); ++i2) {
  //     for (il::int_t i1 = 0; i1 < v.size(1); ++i1) {
  //       for (il::int_t i0 = 0; i0 < v.size(0); ++i0) {
  //         A(i0, i1, i2, i3) = 1.0 / (1 + i0 + i1 + i2 + i3);
  //       }
  //     }
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Resizing an il::Array4C<T>
  // \details No reallocation is performed if the new size is <= to the
  // capacity for both rows, columns and slices. In this case, the capacity is
  // unchanged. When one of the sizes is > than the the capacity, reallocation
  // is done and the array gets the same capacity as its size.
  */
  void Resize(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3);

  /* \brief Get the capacity of the il::Array4C<T>
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
  void Reserve(il::int_t r0, il::int_t r1, il::int_t r2, il::int_t r3);

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Get a pointer to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* Data();

  /* \brief The memory position of A(i, j, k) is
  // data() + (i * stride(0) + j) * stride(1) + k
  */
  //  il::int_t stride(il::int_t d) const;

 private:
  /* \brief Used internally in debug mode to check the invariance of the object
   */
  bool invariance() const;
};

template <typename T>
Array4C<T>::Array4C() {
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  size_[2] = nullptr;
  size_[3] = nullptr;
  capacity_[0] = nullptr;
  capacity_[1] = nullptr;
  capacity_[2] = nullptr;
  capacity_[3] = nullptr;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array4C<T>::Array4C(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(n2 >= 0);
  IL_EXPECT_FAST(n3 >= 0);

  const il::int_t r0 = n0 > 0 ? n0 : ((n1 == 0 && n2 == 0 && n3 == 0) ? 0 : 1);
  const il::int_t r1 = n1 > 0 ? n1 : ((n0 == 0 && n2 == 0 && n3 == 0) ? 0 : 1);
  const il::int_t r2 = n2 > 0 ? n2 : ((n0 == 0 && n1 == 0 && n3 == 0) ? 0 : 1);
  const il::int_t r3 = n3 > 0 ? n3 : ((n0 == 0 && n1 == 0 && n2 == 0) ? 0 : 1);
  bool error;
  const il::int_t r = il::safeProduct(r0, r1, r2, r3, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    data_ = il::allocateArray<T>(r);
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i3 = 0; i3 < n3; ++i3) {
              data_[((i0 * r1 + i1) * r2 + i2) * r3 + i3] =
                  il::defaultValue<T>();
            }
          }
        }
      }
#endif
    } else {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i3 = 0; i3 < n3; ++i3) {
              new (data_ + ((i0 * r1 + i1) * r2 + i2) * r3 + i3) T{};
            }
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
  size_[3] = data_ + n3;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  capacity_[3] = data_ + r3;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array4C<T>::Array4C(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3,
                    const T& x) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(n2 >= 0);
  IL_EXPECT_FAST(n3 >= 0);

  const il::int_t r0 = n0 > 0 ? n0 : ((n1 == 0 && n2 == 0 && n3 == 0) ? 0 : 1);
  const il::int_t r1 = n1 > 0 ? n1 : ((n0 == 0 && n2 == 0 && n3 == 0) ? 0 : 1);
  const il::int_t r2 = n2 > 0 ? n2 : ((n0 == 0 && n1 == 0 && n3 == 0) ? 0 : 1);
  const il::int_t r3 = n3 > 0 ? n3 : ((n0 == 0 && n1 == 0 && n2 == 0) ? 0 : 1);
  bool error;
  const il::int_t r = il::safeProduct(r0, r1, r2, r3, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    data_ = il::allocateArray<T>(r);
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i3 = 0; i3 < n3; ++i3) {
            new (data_ + ((i0 * r1 + i1) * r2 + i2) * r3 + i3) T(x);
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
  size_[3] = data_ + n3;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  capacity_[3] = data_ + r3;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array4C<T>::Array4C(const Array4C<T>& A) {
  const il::int_t n0 = A.size(0);
  const il::int_t n1 = A.size(1);
  const il::int_t n2 = A.size(2);
  const il::int_t n3 = A.size(3);
  il::int_t r0;
  il::int_t r1;
  il::int_t r2;
  il::int_t r3;
  if (n0 > 0 && n1 > 0 && n2 > 0 && n3 > 0) {
    r0 = n0;
    r1 = n1;
    r2 = n2;
    if (il::isTrivial<T>::value && A.alignment_ != 0) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(A.alignment_) / alignof(T));
      bool error = false;
      r3 = il::safeUpperRound(n3, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
    } else {
      r3 = n3;
    }
  } else if (n0 == 0 && n1 == 0 && n2 == 0 && n3 == 0) {
    r0 = 0;
    r1 = 0;
    r2 = 0;
    r3 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
    r2 = (n2 == 0) ? 1 : n2;
    r3 = (n3 == 0) ? 1 : n3;
  }
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, r2, r3, il::io, error);
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
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          memcpy(data_ + ((i0 * r1 + i1) * r2 + i2) * r3,
                 A.data_ + ((i0 * A.capacity(1) + i1) * A.capacity(2) + i2) *
                               A.capacity(3),
                 n3 * sizeof(T));
        }
      }
    }
  } else {
    data_ = il::allocateArray<T>(r);
    shift_ = 0;
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i3 = 0; i3 < n3; ++i3) {
            new (data_ + ((i0 * r1 + i1) * r2 + i2) * r3 + i3)
                T(A(i0, i1, i2, i3));
          }
        }
      }
    }
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  size_[3] = data_ + n3;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  capacity_[3] = data_ + r3;
  alignment_ = A.alignment_;
  align_r_ = A.align_r_;
  align_mod_ = A.align_mod_;
}

template <typename T>
Array4C<T>::Array4C(Array4C<T>&& A) {
  data_ = A.data_;
  size_[0] = A.size_[0];
  size_[1] = A.size_[1];
  size_[2] = A.size_[2];
  size_[3] = A.size_[3];
  capacity_[0] = A.capacity_[0];
  capacity_[1] = A.capacity_[1];
  capacity_[2] = A.capacity_[2];
  capacity_[3] = A.capacity_[3];
  alignment_ = A.alignment_;
  align_r_ = A.align_r_;
  align_mod_ = A.align_mod_;
  shift_ = A.shift_;
  A.data_ = nullptr;
  A.size_[0] = nullptr;
  A.size_[1] = nullptr;
  A.size_[2] = nullptr;
  A.size_[3] = nullptr;
  A.capacity_[0] = nullptr;
  A.capacity_[1] = nullptr;
  A.capacity_[2] = nullptr;
  A.capacity_[3] = nullptr;
  A.alignment_ = 0;
  A.align_r_ = 0;
  A.align_mod_ = 0;
  A.shift_ = 0;
}

template <typename T>
Array4C<T>& Array4C<T>::operator=(const Array4C<T>& A) {
  if (this != &A) {
    const il::int_t n0 = A.size(0);
    const il::int_t n1 = A.size(1);
    const il::int_t n2 = A.size(2);
    const il::int_t n3 = A.size(3);
    const il::int_t alignment = A.alignment_;
    const il::int_t align_r = A.align_r_;
    const il::int_t align_mod = A.align_mod_;
    const bool need_memory = capacity(0) < n0 || capacity(1) < n1 ||
                             capacity(2) < n2 || capacity(3) < n3 ||
                             align_mod_ != align_mod || align_r_ != align_r ||
                             alignment_ != alignment;
    if (need_memory) {
      il::int_t r0;
      il::int_t r1;
      il::int_t r2;
      il::int_t r3;
      if (n0 > 0 && n1 > 0 && n2 > 0) {
        r0 = n0;
        r1 = n1;
        r2 = n2;
        if (il::isTrivial<T>::value && alignment != 0) {
          const il::int_t nb_lanes = static_cast<il::int_t>(
              static_cast<std::size_t>(alignment) / alignof(T));
          bool error = false;
          r3 = il::safeUpperRound(n3, nb_lanes, il::io, error);
          if (error) {
            il::abort();
          }
        } else {
          r3 = n3;
        }
      } else {
        r0 = (n0 == 0) ? 1 : n0;
        r1 = (n1 == 0) ? 1 : n1;
        r2 = (n2 == 0) ? 1 : n2;
        r3 = (n3 == 0) ? 1 : n3;
      }
      bool error = false;
      const il::int_t r = il::safeProduct(r0, r1, r2, r3, il::io, error);
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
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i2 = 0; i2 < n2; ++i2) {
              memcpy(
                  data_ + ((i0 * r1 + i1) * r2 + i2) * r3,
                  A.data_ + ((i0 * A.capacity(1) + i1) * A.capacity(2) + i2) *
                                A.capacity(3),
                  n3 * sizeof(T));
            }
          }
        }
      } else {
        if (data_) {
          for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
            for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
              for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
                for (il::int_t i3 = size(3) - 1; i3 >= 0; --i3) {
                  (data_ +
                   ((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) +
                   i3)
                      ->~T();
                }
              }
            }
          }
          il::deallocate(data_);
        }
        data_ = il::allocateArray<T>(r);
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i2 = 0; i2 < n2; ++i2) {
              for (il::int_t i3 = 0; i3 < n3; ++i3) {
                new (data_ + ((i0 * r1 + i1) * r2 + i2) * r3 + i3)
                    T(A(i0, i1, i2, i3));
              }
            }
          }
        }
      }
      size_[0] = data_ + n0;
      size_[1] = data_ + n1;
      size_[2] = data_ + n2;
      size_[3] = data_ + n3;
      capacity_[0] = data_ + r0;
      capacity_[1] = data_ + r1;
      capacity_[2] = data_ + r2;
      capacity_[3] = data_ + r3;
      alignment_ = static_cast<short>(alignment);
      align_r_ = static_cast<short>(align_r);
      align_mod_ = static_cast<short>(align_mod);
    } else {
      if (il::isTrivial<T>::value) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i2 = 0; i2 < n2; ++i2) {
              memcpy(
                  data_ + ((i0 * capacity(1) + i1) * capacity(2) + i2) *
                              capacity(3),
                  A.data_ + ((i0 * A.capacity(1) + i1) * A.capacity(2) + i2) *
                                A.capacity(3),
                  n3 * sizeof(T));
            }
          }
        }
      } else {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i2 = 0; i2 < n2; ++i2) {
              for (il::int_t i3 = 0; i3 < n3; ++i3) {
                data_[((i0 * capacity(1) + i1) * capacity(2) + i2) *
                          capacity(3) +
                      i0] = A(i0, i1, i2, i3);
              }
            }
          }
        }
        for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
          for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
            for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
              for (il::int_t i3 = size(3) - 1;
                   i3 >= (((i0 < n0) && (i1 < n1) && (i2 < n2)) ? n3 : 0);
                   --i3) {
                (data_ +
                 ((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) +
                 i3)
                    ->~T();
              }
            }
          }
        }
      }
      size_[0] = data_ + n0;
      size_[1] = data_ + n1;
      size_[2] = data_ + n2;
      size_[3] = data_ + n3;
    }
  }
  return *this;
}

template <typename T>
Array4C<T>& Array4C<T>::operator=(Array4C<T>&& A) {
  if (this != &A) {
    if (data_) {
      if (!il::isTrivial<T>::value) {
        for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
          for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
            for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
              for (il::int_t i3 = size(3) - 1; i3 >= 0; --i3) {
                (data_ +
                 ((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) +
                 i3)
                    ->~T();
              }
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
    size_[3] = A.size_[3];
    capacity_[0] = A.capacity_[0];
    capacity_[1] = A.capacity_[1];
    capacity_[2] = A.capacity_[2];
    capacity_[3] = A.capacity_[3];
    alignment_ = A.alignment_;
    align_r_ = A.align_r_;
    align_mod_ = A.align_mod_;
    shift_ = A.shift_;
    A.data_ = nullptr;
    A.size_[0] = nullptr;
    A.size_[1] = nullptr;
    A.size_[2] = nullptr;
    A.size_[3] = nullptr;
    A.capacity_[0] = nullptr;
    A.capacity_[1] = nullptr;
    A.capacity_[2] = nullptr;
    A.capacity_[3] = nullptr;
    A.alignment_ = 0;
    A.align_r_ = 0;
    A.align_mod_ = 0;
    A.shift_ = 0;
  }
  return *this;
}

template <typename T>
Array4C<T>::~Array4C() {
  IL_EXPECT_FAST_NOTHROW(invariance());

  if (data_) {
    if (!il::isTrivial<T>::value) {
      for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
        for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
          for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
            for (il::int_t i3 = size(3) - 1; i3 >= 0; --i3) {
              (data_ +
               ((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) + i3)
                  ->~T();
            }
          }
        }
      }
    }
    il::deallocate(data_ - shift_);
  }
}

template <typename T>
const T& Array4C<T>::operator()(il::int_t i0, il::int_t i1, il::int_t i2,
                                il::int_t i3) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i2) <
                   static_cast<std::size_t>(size(2)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i3) <
                   static_cast<std::size_t>(size(3)));

  return data_[((i0 * (capacity_[1] - data_) + i1) * (capacity_[2] - data_) +
                i2) *
                   (capacity_[3] - data_) +
               i3];
}

template <typename T>
T& Array4C<T>::operator()(il::int_t i0, il::int_t i1, il::int_t i2,
                          il::int_t i3) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i2) <
                   static_cast<std::size_t>(size(2)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i3) <
                   static_cast<std::size_t>(size(3)));

  return data_[((i0 * (capacity_[1] - data_) + i1) * (capacity_[2] - data_) +
                i2) *
                   (capacity_[3] - data_) +
               i3];
}

template <typename T>
il::int_t Array4C<T>::size(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(4));

  return size_[d] - data_;
}

template <typename T>
void Array4C<T>::Resize(il::int_t n0, il::int_t n1, il::int_t n2,
                        il::int_t n3) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(n2 >= 0);
  IL_EXPECT_FAST(n3 >= 0);

  const il::int_t n0_old = size(0);
  const il::int_t n1_old = size(1);
  const il::int_t n2_old = size(2);
  const il::int_t n3_old = size(3);
  const bool need_memory = n0 > capacity(0) || n1 > capacity(1) ||
                           n2 > capacity(2) || n3 > capacity(3);
  if (need_memory) {
    il::int_t r0;
    il::int_t r1;
    il::int_t r2;
    il::int_t r3;
    if (n0 > 0 && n1 > 0 && n2 > 0 && n3 > 0) {
      r0 = n0;
      r1 = n1;
      r2 = n2;
      if (il::isTrivial<T>::value && alignment_ != 0) {
        const il::int_t nb_lanes = static_cast<il::int_t>(
            static_cast<std::size_t>(alignment_) / alignof(T));
        bool error = false;
        r3 = il::safeUpperRound(n3, nb_lanes, il::io, error);
        if (error) {
          il::abort();
        }
      } else {
        r3 = n3;
      }
    } else {
      r0 = (n0 == 0) ? 1 : n0;
      r1 = (n1 == 0) ? 1 : n1;
      r2 = (n2 == 0) ? 1 : n2;
      r3 = (n3 == 0) ? 1 : n3;
    }
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, r2, r3, il::io, error);
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
        for (il::int_t i0 = 0; i0 < (n0 < n0_old ? n0 : n0_old); ++i0) {
          for (il::int_t i1 = 0; i1 < (n1 < n1_old ? n1 : n1_old); ++i1) {
            for (il::int_t i2 = 0; i2 < (n2 < n2_old ? n2 : n2_old); ++i2) {
              memcpy(new_data + ((i0 * r1 + i1) * r2 + i2) * r3,
                     data_ + ((i0 * capacity(1) + i1) * capacity(2) + i2) *
                                 capacity(3),
                     (n3 < n3_old ? n3 : n3_old) * sizeof(T));
            }
          }
        }
        il::deallocate(data_ - shift_);
      }
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i3 =
                     (i0 < n0_old && i1 < n1_old && i2 < n2_old) ? n3_old : 0;
                 i3 < n3; ++i3) {
              new_data[((i0 * r1 + i1) * r2 + i2) * r3 + i3] =
                  il::defaultValue<T>();
            }
          }
        }
      }
#endif
    } else {
      new_data = il::allocateArray<T>(r);
      new_shift = 0;
      if (data_) {
        for (il::int_t i0 = n0_old - 1; i0 >= 0; --i0) {
          for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
            for (il::int_t i2 = n2_old - 1; i2 >= 0; --i2) {
              for (il::int_t i3 = n3_old - 1;
                   i3 >= (i0 < n0 && i1 < n1 && i2 < n2 ? n3 : 0); --i3) {
                (data_ +
                 ((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) +
                 i3)
                    ->~T();
              }
            }
          }
        }
        for (il::int_t i0 = (n0 < n0_old ? n0 : n0_old) - 1; i0 >= 0; --i0) {
          for (il::int_t i1 = (n1 < n1_old ? n1 : n1_old) - 1; i1 >= 0; --i1) {
            for (il::int_t i2 = (n2 < n2_old ? n2 : n2_old) - 1; i2 >= 0;
                 --i2) {
              for (il::int_t i3 = (n3 < n3_old ? n3 : n3_old) - 1; i3 >= 0;
                   --i3) {
                new (new_data + ((i0 * r1 + i1) * r2 + i2) * r3 + i3)
                    T(std::move(
                        data_[((i0 * capacity(1) + i1) * capacity(2) + i2) *
                                  capacity(3) +
                              i3]));
                (data_ +
                 ((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) +
                 i3)
                    ->~T();
              }
            }
          }
        }
        il::deallocate(data_);
      }
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i3 =
                     (i0 < n0_old && i1 < n1_old && i2 < n2_old) ? n3_old : 0;
                 i3 < n3; ++i3) {
              new (new_data + ((i0 * r1 + i1) * r2 + i2) * r3 + i3) T{};
            }
          }
        }
      }
    }
    data_ = new_data;
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    capacity_[2] = data_ + r2;
    capacity_[3] = data_ + r3;
    shift_ = static_cast<short>(new_shift);
  } else {
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i3 =
                     (i0 < n0_old && i1 < n1_old && i2 < n2_old) ? n3_old : 0;
                 i3 < n3; ++i3) {
              data_[((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) +
                    i3] = il::defaultValue<T>();
            }
          }
        }
      }
#endif
    } else {
      for (il::int_t i0 = n0_old - 1; i0 >= 0; --i0) {
        for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
          for (il::int_t i2 = n2_old - 1; i2 >= 0; --i2) {
            for (il::int_t i3 = n3_old - 1;
                 i3 >= ((i0 < n0 && i1 < n1 && i2 < n2) ? n3 : 0); --i3) {
              (data_ +
               ((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) + i3)
                  ->~T();
            }
          }
        }
      }
    }
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i3 =
                   (i0 < n0_old && i1 < n1_old && i2 < n2_old) ? n3_old : 0;
               i3 < n3; ++i3) {
            new (data_ +
                 ((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) +
                 i3) T{};
          }
        }
      }
    }
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  size_[3] = data_ + n3;
}

template <typename T>
il::int_t Array4C<T>::capacity(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(4));

  return capacity_[d] - data_;
}

template <typename T>
void Array4C<T>::Reserve(il::int_t r0, il::int_t r1, il::int_t r2,
                         il::int_t r3) {
  IL_EXPECT_FAST(r0 >= 0);
  IL_EXPECT_FAST(r1 >= 0);
  IL_EXPECT_FAST(r2 >= 0);
  IL_EXPECT_FAST(r3 >= 0);

  r0 = (r0 > capacity(0)) ? r0 : capacity(0);
  r1 = (r1 > capacity(1)) ? r1 : capacity(1);
  r2 = (r2 > capacity(2)) ? r2 : capacity(2);
  r3 = (r3 > capacity(3)) ? r3 : capacity(3);
  const bool need_memory = r0 > capacity(0) || r1 > capacity(1) ||
                           r2 > capacity(2) || r3 > capacity(3);
  if (need_memory) {
    const il::int_t n0_old = size(0);
    const il::int_t n1_old = size(1);
    const il::int_t n2_old = size(2);
    const il::int_t n3_old = size(3);
    if (il::isTrivial<T>::value && alignment_ != 0) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(alignment_) / alignof(T));
      bool error = false;
      r3 = il::safeUpperRound(r3, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
    }
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, r2, r3, il::io, error);
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
        for (il::int_t i0 = 0; i0 < n0_old; ++i0) {
          for (il::int_t i1 = 0; i1 < n1_old; ++i1) {
            for (il::int_t i2 = 0; i2 < n2_old; ++i2) {
              memcpy(new_data + ((i0 * r1 + i1) * r2 + i2) * r3,
                     data_ + ((i0 * capacity(1) + i1) * capacity(2) + i2) *
                                 capacity(3),
                     n3_old * sizeof(T));
            }
          }
        }
        il::deallocate(data_ - shift_);
      }
    } else {
      new_data = il::allocateArray<T>(r);
      new_shift = 0;
      for (il::int_t i0 = n0_old - 1; i0 >= 0; --i0) {
        for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
          for (il::int_t i2 = n2_old - 1; i2 >= 0; --i2) {
            for (il::int_t i3 = n3_old - 1; i3 >= 0; --i3) {
              new (new_data + ((i0 * r1 + i1) * r2 + i2) * r3 + i3) T(
                  std::move(data_[((i0 * capacity(1) + i1) * capacity(2) + i2) *
                                      capacity(3) +
                                  i3]));
              (data_ +
               ((i0 * capacity(1) + i1) * capacity(2) + i2) * capacity(3) + i3)
                  ->~T();
            }
          }
        }
      }
      il::deallocate(data_);
    }
    data_ = new_data;
    size_[0] = data_ + n0_old;
    size_[1] = data_ + n1_old;
    size_[2] = data_ + n2_old;
    size_[3] = data_ + n3_old;
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    capacity_[2] = data_ + r2;
    capacity_[3] = data_ + r3;
    shift_ = static_cast<short>(new_shift);
  }
}

template <typename T>
const T* Array4C<T>::data() const {
  return data_;
}

template <typename T>
T* Array4C<T>::Data() {
  return data_;
}

template <typename T>
bool Array4C<T>::invariance() const {
  bool ans = true;

  if (data_ == nullptr) {
    ans = ans && (size_[0] == nullptr);
    ans = ans && (size_[1] == nullptr);
    ans = ans && (size_[2] == nullptr);
    ans = ans && (size_[3] == nullptr);
    ans = ans && (capacity_[0] == nullptr);
    ans = ans && (capacity_[1] == nullptr);
    ans = ans && (capacity_[2] == nullptr);
    ans = ans && (capacity_[3] == nullptr);
    ans = ans && (alignment_ == 0);
    ans = ans && (align_r_ == 0);
    ans = ans && (align_mod_ == 0);
    ans = ans && (shift_ == 0);
  } else {
    ans = ans && (size_[0] != nullptr);
    ans = ans && (size_[1] != nullptr);
    ans = ans && (size_[2] != nullptr);
    ans = ans && (size_[3] != nullptr);
    ans = ans && (capacity_[0] != nullptr);
    ans = ans && (capacity_[1] != nullptr);
    ans = ans && (capacity_[2] != nullptr);
    ans = ans && (capacity_[3] != nullptr);
    ans = ans && ((size_[0] - data_) <= (capacity_[0] - data_));
    ans = ans && ((size_[1] - data_) <= (capacity_[1] - data_));
    ans = ans && ((size_[2] - data_) <= (capacity_[2] - data_));
    ans = ans && ((size_[3] - data_) <= (capacity_[3] - data_));
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

#endif  // IL_ARRAY4C_H
