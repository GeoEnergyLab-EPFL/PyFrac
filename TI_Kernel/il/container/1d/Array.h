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

#ifndef IL_ARRAY_H
#define IL_ARRAY_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <new> is needed for placement new
#include <new>
// <utility> is needed for std::move
#include <utility>

#include <il/container/1d/ArrayView.h>
#include <il/core/memory/allocate.h>

namespace il {

template <typename T>
class Array {
 private:
  T* data_;
  T* size_;
  T* capacity_;
  short alignment_;
  short align_r_;
  short align_mod_;
  short shift_;

 public:
  /* \brief Default constructor
  // \details The size and the capacity of the array are set to 0. The alignment
  // is undefined. No memory allocation is done during the process.
  */
  Array();

  /* \brief Construct an array of n elements
  // \details The size and the capacity of the array are set to n.
  // - If T is a numeric value, the memory is
  //   - (Debug mode) initialized to il::defaultValue<T>(). It is usually
  NaN
  //     if T is a floating point number or 666..666 if T is an integer.
  //   - (Release mode) left uninitialized. This behavior is different from
  //     std::vector from the standard library which initializes all numeric
  //     values to 0.
  // - If T is an object with a default constructor, all objects are default
  //   constructed. A compile-time error is raised if T has no default
  //   constructor.
  //
  // // Construct an array of double of size 5
  // il::Array<double> v{5};
  */
  explicit Array(il::int_t n);

  /* \brief Construct an aligned array of n elements
  // \details The pointer data, when considered as an integer, satisfies
  // data_ = 0 (Modulo align_mod)
  */
  explicit Array(il::int_t n, il::align_t, il::int_t alignment);

  /* \brief Construct an aligned array of n elements
  // \details The pointer data, when considered as an integer, satisfies
  // data_ = align_r (Modulo align_mod)
  */
  explicit Array(il::int_t n, il::align_t, il::int_t alignment,
                 il::int_t align_r, il::int_t align_mod);

  /* \brief Construct an array of n elements with a value
  //
  // // Construct an array of double of length 5, initialized with 3.14
  // il::Array<double> v{5, 3.14};
  */
  explicit Array(il::int_t n, const T& x);

  template <typename... Args>
  explicit Array(il::int_t n, il::emplace_t, Args&&... args);

  /* \brief Construct an array of n elements with a value
  //
  // // Construct an array of double of length 5, initialized with 3.14
  // il::Array<double> v{5, 3.14};
  */
  explicit Array(il::int_t n, const T& x, il::align_t, il::int_t alignment,
                 il::int_t align_r, il::int_t align_mod);

  explicit Array(il::int_t n, const T& x, il::align_t, il::int_t alignment);

  explicit Array(il::ArrayView<T> v);

  /* \brief Construct an array from a brace-initialized list
  // \details The size and the capacity of the il::Array<T> is adjusted
  to
  // the size of the initializer list. The tag il::value is used to allow brace
  // initialization of il::Array<T> everywhere.
  //
  // // Construct an array of double from a list
  // il::Array<double> v{il::value, {2.0, 3.14, 5.0, 7.0}};
  */
  Array(il::value_t, std::initializer_list<T> list);

  /* \brief The copy constructor
  // \details The size and the capacity of the constructed il::Array<T>
  are
  // equal to the size of the source array.
  */
  Array(const Array<T>& v);

  /* \brief The move constructor
   */
  Array(Array<T>&& v);

  /* \brief The copy assignment
  // \details The size is the same as the one for the source array. The
  // capacity is not changed if it is enough for the copy to happen and is
  // set to the size of the source array if the initial capacity is too low.
  */
  Array& operator=(const Array<T>& v);

  /* \brief The move assignment
   */
  Array& operator=(Array<T>&& v);

  /* \brief The destructor
   */
  ~Array();

  /* \brief Accessor for a const il::Array<T>
  // \details Access (read only) the i-th element of the array. Bound checking
  // is done in debug mode but not in release mode.
  //
  // il::Array<double> v{4};
  // std::cout << v[0] << std::endl;
  */
  const T& operator[](il::int_t i) const;

  /* \brief Accessor for an il::Array<T>
  // \details Access (read or write) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::Array<double> v{4};
  // v[0] = 0.0;
  // v[4] = 0.0; // Program is aborted in debug mode and has undefined
  //             // behavior in release mode
  */
  T& operator[](il::int_t i);

  void Set(const T& x);

  /* \brief Accessor to the last element of a const il::Array<T>
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  const T& back() const;

  /* \brief Accessor to the last element of a il::Array<T>
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  T& Back();

  /* \brief Get the size of the il::Array<T>
  // \details The library has been designed in a way that any compiler can prove
  // that modifying v[k] can't change the result of v.size(). As a consequence
  // a call to v.size() is made just once at the very beginning of the loop
  // in the following example. It allows many optimizations from the compiler,
  // including automatic vectorization.
  //
  // il::Array<double> v{n};
  // for (il::int_t k = 0; k < v.size(); ++k) {
  //     v[k] = 1.0 / (k + 1);
  // }
  */
  il::int_t size() const;

  /* \brief Resizing an il::Array<T>
  // \details No reallocation is performed if the new size is <= to the
  // capacity. In this case, the capacity is unchanged. When the size is > than
  // the current capacity, reallocation is done and the array gets the same
  // capacity as its size.
  */
  void Resize(il::int_t n);

  void Resize(il::int_t n, const T& x);

  template <typename... Args>
  void Resize(il::int_t n, il::emplace_t, Args&&... args);

  /* \brief Get the capacity of the il::Array<T>
   */
  il::int_t capacity() const;

  /* \brief Change the capacity of the array to at least p
  // \details If the capacity is >= to p, nothing is done. Otherwise,
  // reallocation is done and the new capacity is set to p.
  */
  void Reserve(il::int_t r);

  /* \brief Add an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  void Append(const T& x);

  /* \brief Add an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  void Append(T&& x);

  /* \brief Construct an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  template <typename... Args>
  void Append(il::emplace_t, Args&&... args);

  /* \brief Get the alignment of the pointer returned by data()
   */
  il::int_t alignment() const;

  il::ArrayView<T> view() const;

  il::ArrayView<T> view(il::Range range) const;

  il::ArrayEdit<T> Edit();

  il::ArrayEdit<T> Edit(il::Range range);

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Get a pointer to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* Data();

  const T* begin() const;
  const T* cbegin() const;
  T* begin();
  const T* end() const;
  const T* cend() const;
  T* end();

 private:
  /* \brief Used internally to increase the capacity of the array
   */
  void IncreaseCapacity(il::int_t r);

  /* \brief Used internally in debug mode to check the invariance of the object
   */
  bool invariance() const;
};

template <typename T>
Array<T>::Array() {
  data_ = nullptr;
  size_ = nullptr;
  capacity_ = nullptr;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array<T>::Array(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n > 0) {
    data_ = il::allocateArray<T>(n);
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i = 0; i < n; ++i) {
        data_[i] = il::defaultValue<T>();
      }
#endif
    } else {
      for (il::int_t i = 0; i < n; ++i) {
        new (data_ + i) T{};
      }
    }
  } else {
    data_ = nullptr;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array<T>::Array(il::int_t n, il::align_t, il::int_t alignment,
                il::int_t align_r, il::int_t align_mod) {
  IL_EXPECT_FAST(il::isTrivial<T>::value);
  IL_EXPECT_FAST(sizeof(T) % alignof(T) == 0);
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(alignment > 0);
  IL_EXPECT_FAST(alignment % alignof(T) == 0);
  IL_EXPECT_FAST(alignment <= SHRT_MAX);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_mod % alignment == 0);
  IL_EXPECT_FAST(align_mod <= SHRT_MAX);
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);
  IL_EXPECT_FAST(align_r % alignment == 0);
  IL_EXPECT_FAST(align_r <= SHRT_MAX);

  if (n > 0) {
    il::int_t shift;
    data_ = il::allocateArray<T>(n, align_r, align_mod, il::io, shift);
    alignment_ = static_cast<short>(alignment);
    align_r_ = static_cast<short>(align_r);
    align_mod_ = static_cast<short>(align_mod);
    shift_ = static_cast<short>(shift);
#ifdef IL_DEFAULT_VALUE
    for (il::int_t i = 0; i < n; ++i) {
      data_[i] = il::defaultValue<T>();
    }
#endif
  } else {
    data_ = nullptr;
    alignment_ = 0;
    align_r_ = 0;
    align_mod_ = 0;
    shift_ = 0;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
Array<T>::Array(il::int_t n, il::align_t, il::int_t alignment)
    : Array{n, il::align, alignment, 0, alignment} {}

template <typename T>
Array<T>::Array(il::int_t n, const T& x) {
  IL_EXPECT_FAST(n >= 0);

  if (n > 0) {
    data_ = il::allocateArray<T>(n);
    for (il::int_t i = 0; i < n; ++i) {
      new (data_ + i) T(x);
    }
  } else {
    data_ = nullptr;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
template <typename... Args>
Array<T>::Array(il::int_t n, il::emplace_t, Args&&... args) {
  IL_EXPECT_FAST(n >= 0);

  if (n > 0) {
    data_ = il::allocateArray<T>(n);
    for (il::int_t i = 0; i < n; ++i) {
      new (data_ + i) T(std::forward<Args>(args)...);
    }
  } else {
    data_ = nullptr;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array<T>::Array(il::int_t n, const T& x, il::align_t, il::int_t alignment,
                il::int_t align_r, il::int_t align_mod) {
  IL_EXPECT_FAST(il::isTrivial<T>::value);
  IL_EXPECT_FAST(sizeof(T) % alignof(T) == 0);
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(alignment > 0);
  IL_EXPECT_FAST(alignment % alignof(T) == 0);
  IL_EXPECT_FAST(alignment <= SHRT_MAX);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_mod % alignment == 0);
  IL_EXPECT_FAST(align_mod <= SHRT_MAX);
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);
  IL_EXPECT_FAST(align_r % alignment == 0);
  IL_EXPECT_FAST(align_r <= SHRT_MAX);

  if (n > 0) {
    il::int_t shift;
    data_ = il::allocateArray<T>(n, align_r, align_mod, il::io, shift);
    alignment_ = static_cast<short>(alignment);
    align_r_ = static_cast<short>(align_r);
    align_mod_ = static_cast<short>(align_mod);
    shift_ = static_cast<short>(shift);
    for (il::int_t i = 0; i < n; ++i) {
      data_[i] = x;
    }
  } else {
    data_ = nullptr;
    alignment_ = 0;
    align_r_ = 0;
    align_mod_ = 0;
    shift_ = 0;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
Array<T>::Array(il::int_t n, const T& x, il::align_t, il::int_t alignment)
    : Array{n, x, il::align, alignment, 0, alignment} {}

template <typename T>
Array<T>::Array(il::ArrayView<T> v) {
  const il::int_t n = v.size();

  if (n > 0) {
    data_ = il::allocateArray<T>(n);
    for (il::int_t i = 0; i < n; ++i) {
      new (data_ + i) T{v[i]};
    }
  } else {
    data_ = nullptr;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array<T>::Array(il::value_t, std::initializer_list<T> list) {
  bool error = false;
  const il::int_t n = il::safeConvert<il::int_t>(list.size(), il::io, error);
  if (error) {
    il::abort();
  }

  if (n > 0) {
    data_ = il::allocateArray<T>(n);
    if (il::isTrivial<T>::value) {
      memcpy(data_, list.begin(), n * sizeof(T));
    } else {
      for (il::int_t i = 0; i < n; ++i) {
        new (data_ + i) T(*(list.begin() + i));
      }
    }
  } else {
    data_ = nullptr;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array<T>::Array(const Array<T>& v) {
  const il::int_t n = v.size();
  const il::int_t alignment = v.alignment_;
  const il::int_t align_r = v.align_r_;
  const il::int_t align_mod = v.align_mod_;
  if (alignment == 0) {
    data_ = il::allocateArray<T>(n);
    alignment_ = 0;
    align_r_ = 0;
    align_mod_ = 0;
    shift_ = 0;
  } else {
    il::int_t shift;
    data_ = il::allocateArray<T>(n, align_r, align_mod, il::io, shift);
    alignment_ = static_cast<short>(alignment);
    align_r_ = static_cast<short>(align_r);
    align_mod_ = static_cast<short>(align_mod);
    shift_ = static_cast<short>(shift);
  }
  if (il::isTrivial<T>::value) {
    memcpy(data_, v.data_, n * sizeof(T));
  } else {
    for (il::int_t i = 0; i < n; ++i) {
      new (data_ + i) T(v.data_[i]);
    }
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
Array<T>::Array(Array<T>&& v) {
  data_ = v.data_;
  size_ = v.size_;
  capacity_ = v.capacity_;
  alignment_ = v.alignment_;
  align_r_ = v.align_r_;
  align_mod_ = v.align_mod_;
  shift_ = v.shift_;
  v.data_ = nullptr;
  v.size_ = nullptr;
  v.capacity_ = nullptr;
  v.alignment_ = 0;
  v.align_r_ = 0;
  v.align_mod_ = 0;
  v.shift_ = 0;
}

template <typename T>
Array<T>& Array<T>::operator=(const Array<T>& v) {
  if (this == &v) {
    return *this;
  }

  const il::int_t n = v.size();
  const il::int_t alignment = v.alignment_;
  const il::int_t align_r = v.align_r_;
  const il::int_t align_mod = v.align_mod_;
  const bool needs_memory = n > capacity() || alignment_ != alignment ||
                            align_r_ != align_r || align_mod_ != align_mod;
  if (needs_memory) {
    if (data_) {
      if (!il::isTrivial<T>::value) {
        for (il::int_t i = size() - 1; i >= 0; --i) {
          (data_ + i)->~T();
        }
      }
      il::deallocate(data_ - shift_);
    }
    if (alignment == 0) {
      data_ = il::allocateArray<T>(n);
      alignment_ = 0;
      align_r_ = 0;
      align_mod_ = 0;
      shift_ = 0;
    } else {
      il::int_t shift;
      data_ = il::allocateArray<T>(n, align_r, align_mod, il::io, shift);
      alignment_ = static_cast<short>(alignment);
      align_r_ = static_cast<short>(align_r);
      align_mod_ = static_cast<short>(align_mod);
      shift_ = static_cast<short>(shift);
    }
    if (il::isTrivial<T>::value) {
      memcpy(data_, v.data_, n * sizeof(T));
    } else {
      for (il::int_t i = 0; i < n; ++i) {
        new (data_ + i) T(v.data_[i]);
      }
    }
    size_ = data_ + n;
    capacity_ = data_ + n;
  } else {
    if (il::isTrivial<T>::value) {
      memcpy(data_, v.data_, n * sizeof(T));
    } else {
      for (il::int_t i = 0; i < n; ++i) {
        data_[i] = v.data_[i];
      }
      for (il::int_t i = size() - 1; i >= n; --i) {
        (data_ + i)->~T();
      }
    }
    size_ = data_ + n;
  }
  return *this;
}

template <typename T>
Array<T>& Array<T>::operator=(Array<T>&& v) {
  if (this == &v) {
    return *this;
  }

  if (data_) {
    if (!il::isTrivial<T>::value) {
      for (il::int_t i = size() - 1; i >= 0; --i) {
        (data_ + i)->~T();
      }
    }
    il::deallocate(data_ - shift_);
  }
  data_ = v.data_;
  size_ = v.size_;
  capacity_ = v.capacity_;
  alignment_ = v.alignment_;
  align_r_ = v.align_r_;
  align_mod_ = v.align_mod_;
  shift_ = v.shift_;
  v.data_ = nullptr;
  v.size_ = nullptr;
  v.capacity_ = nullptr;
  v.alignment_ = 0;
  v.align_r_ = 0;
  v.align_mod_ = 0;
  v.shift_ = 0;
  return *this;
}

template <typename T>
Array<T>::~Array() {
  IL_EXPECT_FAST_NOTHROW(invariance());

  if (data_) {
    if (!il::isTrivial<T>::value) {
      for (il::int_t i = size() - 1; i >= 0; --i) {
        (data_ + i)->~T();
      }
    }
    il::deallocate(data_ - shift_);
  }
}

template <typename T>
const T& Array<T>::operator[](il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i];
}

template <typename T>
T& Array<T>::operator[](il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i];
}

template <typename T>
void Array<T>::Set(const T& x) {
  for (il::int_t i = 0; i < capacity_ - data_; ++i) {
    data_[i] = x;
  }
}

template <typename T>
const T& Array<T>::back() const {
  IL_EXPECT_MEDIUM(size() > 0);

  return size_[-1];
}

template <typename T>
T& Array<T>::Back() {
  IL_EXPECT_MEDIUM(size() > 0);

  return size_[-1];
}

template <typename T>
il::int_t Array<T>::size() const {
  return size_ - data_;
}

template <typename T>
void Array<T>::Resize(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= capacity()) {
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i = size(); i < n; ++i) {
        data_[i] = il::defaultValue<T>();
      }
#endif
    } else {
      for (il::int_t i = size() - 1; i >= n; --i) {
        (data_ + i)->~T();
      }
      for (il::int_t i = size(); i < n; ++i) {
        new (data_ + i) T();
      }
    }
  } else {
    const il::int_t n_old = size();
    IncreaseCapacity(n);
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i = n_old; i < n; ++i) {
        data_[i] = il::defaultValue<T>();
      }
#endif
    } else {
      for (il::int_t i = n_old; i < n; ++i) {
        new (data_ + i) T{};
      }
    }
  }
  size_ = data_ + n;
}

template <typename T>
void Array<T>::Resize(il::int_t n, const T& x) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= capacity()) {
    if (il::isTrivial<T>::value) {
      for (il::int_t i = size(); i < n; ++i) {
        data_[i] = x;
      }
    } else {
      for (il::int_t i = size() - 1; i >= n; --i) {
        (data_ + i)->~T();
      }
      for (il::int_t i = size(); i < n; ++i) {
        new (data_ + i) T{x};
      }
    }
  } else {
    const il::int_t n_old = size();
    IncreaseCapacity(n);
    if (il::isTrivial<T>::value) {
      for (il::int_t i = n_old; i < n; ++i) {
        data_[i] = x;
      }
    } else {
      for (il::int_t i = n_old; i < n; ++i) {
        new (data_ + i) T{x};
      }
    }
  }
  size_ = data_ + n;
}

template <typename T>
template <typename... Args>
void Array<T>::Resize(il::int_t n, il::emplace_t, Args&&... args) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= capacity()) {
    if (il::isTrivial<T>::value) {
      for (il::int_t i = size(); i < n; ++i) {
        data_[i] = T(std::forward<Args>(args)...);
      }
    } else {
      for (il::int_t i = size() - 1; i >= n; --i) {
        (data_ + i)->~T();
      }
      for (il::int_t i = size(); i < n; ++i) {
        new (data_ + i) T(std::forward<Args>(args)...);
      }
    }
  } else {
    const il::int_t n_old = size();
    IncreaseCapacity(n);
    if (il::isTrivial<T>::value) {
      for (il::int_t i = n_old; i < n; ++i) {
        data_[i] = T(std::forward<Args>(args)...);
      }
    } else {
      for (il::int_t i = n_old; i < n; ++i) {
        new (data_ + i) T(std::forward<Args>(args)...);
      }
    }
  }
  size_ = data_ + n;
};

template <typename T>
il::int_t Array<T>::capacity() const {
  return capacity_ - data_;
}

template <typename T>
void Array<T>::Reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  if (r > capacity()) {
    IncreaseCapacity(r);
  }
}

template <typename T>
void Array<T>::Append(const T& x) {
  if (size_ == capacity_) {
    const il::int_t n = size();
    bool error = false;
    il::int_t new_capacity =
        n > 1 ? il::safeProduct(static_cast<il::int_t>(2), n, il::io, error)
              : il::safeSum(n, static_cast<il::int_t>(1), il::io, error);
    if (error) {
      il::abort();
    }
    T x_copy = x;
    IncreaseCapacity(new_capacity);
    new (size_) T(std::move(x_copy));
  } else {
    new (size_) T(x);
  }
  ++size_;
}

template <typename T>
void Array<T>::Append(T&& x) {
  if (size_ == capacity_) {
    const il::int_t n = size();
    bool error = false;
    il::int_t new_capacity =
        n > 1 ? il::safeProduct(static_cast<il::int_t>(2), n, il::io, error)
              : il::safeSum(n, static_cast<il::int_t>(1), il::io, error);
    if (error) {
      il::abort();
    }
    IncreaseCapacity(new_capacity);
  }
  //  if (il::isTrivial<T>::value) {
  //    *size_ = std::move(x);
  //  } else {
  new (size_) T(std::move(x));
  //  }
  ++size_;
}

template <typename T>
template <typename... Args>
void Array<T>::Append(il::emplace_t, Args&&... args) {
  if (size_ == capacity_) {
    const il::int_t n = size();
    bool error = false;
    il::int_t new_capacity =
        n > 1 ? il::safeProduct(static_cast<il::int_t>(2), n, il::io, error)
              : il::safeSum(n, static_cast<il::int_t>(1), il::io, error);
    if (error) {
      il::abort();
    }
    IncreaseCapacity(new_capacity);
  };
  new (size_) T(std::forward<Args>(args)...);
  ++size_;
}

template <typename T>
il::int_t Array<T>::alignment() const {
  return alignment_;
}

template <typename T>
il::ArrayView<T> Array<T>::view() const {
  return il::ArrayView<T>{data_, size()};
};

template <typename T>
il::ArrayView<T> Array<T>::view(il::Range range) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.begin) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.end) <=
                   static_cast<std::size_t>(size()));

  return il::ArrayView<T>{data_ + range.begin, range.end - range.begin};
};

template <typename T>
il::ArrayEdit<T> Array<T>::Edit() {
  return il::ArrayEdit<T>{data_, size()};
};

template <typename T>
il::ArrayEdit<T> Array<T>::Edit(il::Range range) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.begin) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.end) <=
                   static_cast<std::size_t>(size()));

  return il::ArrayEdit<T>{data_ + range.begin, range.end - range.begin};
};

template <typename T>
const T* Array<T>::data() const {
  return data_;
}

template <typename T>
T* Array<T>::Data() {
  return data_;
}

template <typename T>
const T* Array<T>::begin() const {
  return data_;
}

template <typename T>
const T* Array<T>::cbegin() const {
  return data_;
}

template <typename T>
T* Array<T>::begin() {
  return data_;
}

template <typename T>
const T* Array<T>::end() const {
  return size_;
}

template <typename T>
const T* Array<T>::cend() const {
  return size_;
}

template <typename T>
T* Array<T>::end() {
  return size_;
}

template <typename T>
void Array<T>::IncreaseCapacity(il::int_t r) {
  IL_EXPECT_FAST(capacity() < r);

  const il::int_t n = size();
  T* new_data;
  il::int_t new_shift;
  if (alignment_ == 0) {
    new_data = il::allocateArray<T>(r);
    new_shift = 0;
  } else {
    new_data = il::allocateArray<T>(n, align_r_, align_mod_, il::io, new_shift);
  }
  if (data_) {
    if (il::isTrivial<T>::value) {
      memcpy(new_data, data_, n * sizeof(T));
    } else {
      for (il::int_t i = n - 1; i >= 0; --i) {
        new (new_data + i) T(std::move(data_[i]));
        (data_ + i)->~T();
      }
    }
    il::deallocate(data_ - shift_);
  }
  data_ = new_data;
  size_ = data_ + n;
  capacity_ = data_ + r;
  shift_ = static_cast<short>(new_shift);
}

template <typename T>
bool Array<T>::invariance() const {
  bool ans = true;

  if (data_ == nullptr) {
    ans = ans && (size_ == nullptr);
    ans = ans && (capacity_ == nullptr);
  } else {
    ans = ans && (size_ != nullptr);
    ans = ans && (capacity_ != nullptr);
    ans = ans && ((size_ - data_) <= (capacity_ - data_));
  }
  if (!il::isTrivial<T>::value) {
    ans = ans && (align_mod_ == 0);
  }
  if (align_mod_ == 0) {
    ans = ans && (align_r_ == 0);
    ans = ans && (shift_ == 0);
  } else {
    ans = ans && (align_r_ < align_mod_);
    ans = ans && (reinterpret_cast<std::size_t>(data_) %
                      static_cast<std::size_t>(align_mod_) ==
                  static_cast<std::size_t>(align_r_));
  }
  return ans;
}

}  // namespace il

#endif  // IL_ARRAY_H
