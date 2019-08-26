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

#ifndef IL_SMALLARRAY_H
#define IL_SMALLARRAY_H

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

template <typename T, il::int_t small_size>
class SmallArray {
  static_assert(small_size >= 1,
                "il::SmallArray<T, small_size>: small_size must be positive");

 private:
  T* data_;
  T* size_;
  T* capacity_;
  alignas(T) char small_data_[small_size * sizeof(T)];

 public:
  /* \brief Default constructor
  // \details The size is set to 0 and the capacity is set to small_size.
  */
  SmallArray();

  /* \brief Construct a small array of n elements
  // \details The size and the capacity of the array are set to n.
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
  // // Construct a small array of double of size 5 with a stack memory of
  // // size 10.
  // il::SmallArray<double, 10> v{5};
  */
  explicit SmallArray(il::int_t n);

  /* \brief Construct a small array of n elements from a value
  // \details Initialize the array of length n with a given value.
  /
  // // Construct an array of double of length 5, initialized with 0.0
  // il::SmallArray<double, 10> v{5, 0.0};
  */
  explicit SmallArray(il::int_t n, const T& x);

  /* \brief Construct a il::SmallArray<T, small_size> from a brace-initialized
  // list
  // \details The size and the capacity of the il::SmallArray<T, small_size> is
  // adjusted to the size of the initializer list. The tag il::value is used to
  // allow brace initialization of il::Array<T> everywhere.
  //
  // // Construct an array of double from a list
  // il::SmallArray<double, 4> v{il::value, {2.0, 3.14, 5.0, 7.0}};
  */
  explicit SmallArray(il::value_t, std::initializer_list<T> list);

  /* \brief The copy constructor
  // \details The size of the constructed array is the same as the one for the
  // source array. However, its capacity is the same as its size even though
  // the source array had a larger capacity.
  */
  SmallArray(const SmallArray<T, small_size>& A);

  /* \brief The move constructor
   */
  SmallArray(SmallArray<T, small_size>&& A);

  /* \brief The copy assignment
  // \details The size is the same as the one for the source array. The
  // capacity is not changed if it is enough for the copy to happen and is
  // set to the size of the source array if the initial capacity is too low.
  */
  SmallArray& operator=(const SmallArray<T, small_size>& A);

  /* \brief The move assignment
   */
  SmallArray& operator=(SmallArray<T, small_size>&& A);

  /* \brief The destructor
  // \details If T is an object, they are destructed from the last one to the
  // first one. Then, the allocated memory is released.
  */
  ~SmallArray();

  /* \brief Accessor for a const object
  // \details Access (read only) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::SmallArray<double, 10> v{4};
  // std::cout << v[0] << std::endl;
  */
  const T& operator[](il::int_t i) const;

  /* \brief Accessor
  // \details Access (read or write) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::SmallArray<double, 10> v{4};
  // v[0] = 0.0;
  // v[4] = 0.0; // Program is aborted in debug mode and has undefined
  //             // behavior in release mode
  */
  T& operator[](il::int_t i);

  /* \brief Accessor to the last element of a const
  // il::SmallArray<T, small_size>
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  const T& back() const;

  /* \brief Accessor to the last element of a il::SmallArray<T, small_size>
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  T& Back();

  /* \brief Get the size of the array
  //
  // il::SmallArray<double, 10> v{4};
  // for (il::int_t k = 0; k < v.size(); ++k) {
  //     v[k] = 1.0 / (k + 1);
  // }
  */
  il::int_t size() const;

  /* \brief Change the size of the array
  // \details No reallocation is performed if the new size is <= to the
  // capacity. In this case, the capacity is unchanged. When the size is > at
  // the current capacity, reallocation is done and the array gets the same
  // capacity as its size.
  */
  void Resize(il::int_t n);

  /* \brief Get the capacity of the array
   */
  il::int_t capacity() const;

  /* \brief Change the capacity of the array to at least p
  // \details If the capacity is >= to p, nothing is done. Otherwise,
  // reallocation is done and the new capacity is set to p.
  */
  void Reserve(il::int_t p);

  /* \brief Add an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  void Append(const T& x);

  /* \brief Construct an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  template <typename... Args>
  void Append(Args&&... args);

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

 private:
  /* \brief Used internally to check if the stack array is used
   */
  bool smallDataUsed() const;

  /* \brief Used internally to increase the capacity of the array
   */
  void IncreaseCapacity(il::int_t r);

  /* \brief Used internally in debug mode to check the invariance of the object
   */
  bool invariance() const;
};

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray() {
  data_ = reinterpret_cast<T*>(small_data_);
  size_ = data_;
  capacity_ = data_ + small_size;
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= small_size) {
    data_ = reinterpret_cast<T*>(small_data_);
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
    data_ = il::allocateArray<T>(n);
    size_ = data_ + n;
    capacity_ = data_ + n;
  }
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
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(il::int_t n, const T& x) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= small_size) {
    data_ = reinterpret_cast<T*>(small_data_);
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
    data_ = il::allocateArray<T>(n);
    size_ = data_ + n;
    capacity_ = data_ + n;
  }
  for (il::int_t i = 0; i < n; ++i) {
    new (data_ + i) T(x);
  }
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(il::value_t,
                                      std::initializer_list<T> list) {
  const il::int_t n = static_cast<il::int_t>(list.size());
  if (n <= small_size) {
    data_ = reinterpret_cast<T*>(small_data_);
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
    data_ = il::allocateArray<T>(n);
    size_ = data_ + n;
    capacity_ = data_ + n;
  }
  if (il::isTrivial<T>::value) {
    memcpy(data_, list.begin(), n * sizeof(T));
  } else {
    for (il::int_t i = 0; i < n; ++i) {
      new (data_ + i) T(*(list.begin() + i));
    }
  }
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(const SmallArray<T, small_size>& A) {
  const il::int_t n = A.size();
  if (n <= small_size) {
    data_ = reinterpret_cast<T*>(small_data_);
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
    data_ = il::allocateArray<T>(n);
    size_ = data_ + n;
    capacity_ = data_ + n;
  }
  if (il::isTrivial<T>::value) {
    memcpy(data_, A.data_, n * sizeof(T));
  } else {
    for (il::int_t i = 0; i < n; ++i) {
      new (data_ + i) T(A.data_[i]);
    }
  }
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(SmallArray<T, small_size>&& A) {
  const il::int_t n = A.size();
  if (A.smallDataUsed()) {
    data_ = reinterpret_cast<T*>(small_data_);
    if (il::isTrivial<T>::value) {
      memcpy(data_, A.data_, n * sizeof(T));
    } else {
      for (il::int_t i = n - 1; i >= 0; --i) {
        new (data_ + i) T(std::move(A.data_[i]));
        (A.data_ + i)->~T();
      }
    }
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
    data_ = A.data_;
    size_ = A.size_;
    capacity_ = A.capacity_;
  }
  A.data_ = reinterpret_cast<T*>(A.small_data_);
  A.size_ = A.data_ + 0;
  A.capacity_ = A.data_ + small_size;
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>& SmallArray<T, small_size>::operator=(
    const SmallArray<T, small_size>& A) {
  if (this != &A) {
    const il::int_t n = A.size();
    const bool needs_memory = capacity() < n;
    if (needs_memory) {
      if (il::isTrivial<T>::value) {
        if (!smallDataUsed()) {
          il::deallocate(data_);
        }
        data_ = il::allocateArray<T>(n);
        memcpy(data_, A.data_, n * sizeof(T));
      } else {
        for (il::int_t i = size() - 1; i >= 0; --i) {
          (data_ + i)->~T();
        }
        if (!smallDataUsed()) {
          il::deallocate(data_);
        }
        data_ = il::allocateArray<T>(n);
        for (il::int_t i = 0; i < n; ++i) {
          new (data_ + i) T(A.data_[i]);
        }
      }
      size_ = data_ + n;
      capacity_ = data_ + n;
    } else {
      if (!il::isTrivial<T>::value) {
        for (il::int_t i = size() - 1; i >= n; --i) {
          (data_ + i)->~T();
        }
      }
      for (il::int_t i = 0; i < n; ++i) {
        data_[i] = A.data_[i];
      }
      size_ = data_ + n;
    }
  }
  return *this;
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>& SmallArray<T, small_size>::operator=(
    SmallArray<T, small_size>&& A) {
  if (this != &A) {
    if (il::isTrivial<T>::value) {
      if (!smallDataUsed()) {
        il::deallocate(data_);
      }
    } else {
      for (il::int_t i = size() - 1; i >= 0; --i) {
        (data_ + i)->~T();
      }
      if (!smallDataUsed()) {
        il::deallocate(data_);
      }
    }
    const il::int_t n = A.size();
    if (A.smallDataUsed()) {
      data_ = reinterpret_cast<T*>(small_data_);
      if (il::isTrivial<T>::value) {
        memcpy(data_, A.data_, n * sizeof(T));
      } else {
        for (il::int_t i = n - 1; i >= 0; --i) {
          new (data_ + i) T(std::move(A.data_[i]));
          (data_ + i)->~T();
        }
      }
      size_ = data_ + n;
      capacity_ = data_ + small_size;
    } else {
      data_ = A.data_;
      size_ = A.size_;
      capacity_ = A.capacity_;
    }
    data_ = reinterpret_cast<T*>(small_data_);
    A.size_ = A.data_ + 0;
    A.capacity_ = A.data_ + small_size;
  }
  return *this;
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::~SmallArray() {
  IL_EXPECT_FAST_NOTHROW(invariance());

  if (!il::isTrivial<T>::value) {
    for (il::int_t i = size() - 1; i >= 0; --i) {
      (data_ + i)->~T();
    }
  }
  if (!smallDataUsed()) {
    il::deallocate(data_);
  }
}

template <typename T, il::int_t small_size>
const T& SmallArray<T, small_size>::operator[](il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));
  return data_[i];
}

template <typename T, il::int_t small_size>
T& SmallArray<T, small_size>::operator[](il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));
  return data_[i];
}

template <typename T, il::int_t small_size>
const T& SmallArray<T, small_size>::back() const {
  IL_EXPECT_MEDIUM(size() > 0);
  return size_[-1];
}

template <typename T, il::int_t small_size>
T& SmallArray<T, small_size>::Back() {
  IL_EXPECT_MEDIUM(size() > 0);
  return size_[-1];
}

template <typename T, il::int_t small_size>
il::int_t SmallArray<T, small_size>::size() const {
  return size_ - data_;
}

template <typename T, il::int_t small_size>
void SmallArray<T, small_size>::Resize(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= capacity()) {
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i = size(); i < n; ++i) {
        data_[i] = il::defaultValue<T>();
      }
#endif
    } else {
      for (il::int_t i = size() - 1; i >= 0; --i) {
        (data_ + i)->~T();
      }
      for (il::int_t i = size(); i < n; ++i) {
        new (data_ + i) T{};
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

template <typename T, il::int_t small_size>
il::int_t SmallArray<T, small_size>::capacity() const {
  return capacity_ - data_;
}

template <typename T, il::int_t small_size>
void SmallArray<T, small_size>::Reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  if (r > capacity()) {
    IncreaseCapacity(r);
  }
}

template <typename T, il::int_t small_size>
void SmallArray<T, small_size>::Append(const T& x) {
  if (size_ == capacity_) {
    const il::int_t n = size();
    bool error = false;
    il::int_t new_capacity =
        n > 1 ? il::safeSum(n, n / 2, il::io, error)
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

template <typename T, il::int_t small_size>
template <typename... Args>
void SmallArray<T, small_size>::Append(Args&&... args) {
  if (size_ == capacity_) {
    const il::int_t n = size();
    bool error = false;
    il::int_t new_capacity =
        n > 1 ? il::safeSum(n, n / 2, il::io, error)
              : il::safeSum(n, static_cast<il::int_t>(1), il::io, error);
    if (error) {
      il::abort();
    }
    IncreaseCapacity(new_capacity);
  };
  new (size_) T(args...);
  ++size_;
}

template <typename T, il::int_t small_size>
T* SmallArray<T, small_size>::Data() {
  return data_;
}

template <typename T, il::int_t small_size>
il::ArrayView<T> SmallArray<T, small_size>::view() const {
  return il::ArrayView<T>{data_, size()};
};

template <typename T, il::int_t small_size>
il::ArrayView<T> SmallArray<T, small_size>::view(il::Range range) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.begin) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.end) <=
                   static_cast<std::size_t>(size()));

  return il::ArrayView<T>{data_ + range.begin, range.end - range.begin};
};

template <typename T, il::int_t small_size>
il::ArrayEdit<T> SmallArray<T, small_size>::Edit() {
  return il::ArrayEdit<T>{data_, size()};
};

template <typename T, il::int_t small_size>
il::ArrayEdit<T> SmallArray<T, small_size>::Edit(il::Range range) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.begin) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(range.end) <=
                   static_cast<std::size_t>(size()));

  return il::ArrayEdit<T>{data_ + range.begin, range.end - range.begin};
};

template <typename T, il::int_t small_size>
const T* SmallArray<T, small_size>::data() const {
  return data_;
}

template <typename T, il::int_t small_size>
bool SmallArray<T, small_size>::smallDataUsed() const {
  return data_ == reinterpret_cast<const T*>(small_data_);
}

template <typename T, il::int_t small_size>
void SmallArray<T, small_size>::IncreaseCapacity(il::int_t r) {
  IL_EXPECT_FAST(size() <= r);

  const il::int_t n = size();
  T* new_data;
  new_data = il::allocateArray<T>(r);
  if (il::isTrivial<T>::value) {
    memcpy(new_data, data_, n * sizeof(T));
    if (!smallDataUsed()) {
      il::deallocate(data_);
    }
  } else {
    for (il::int_t i = n - 1; i >= 0; --i) {
      new (new_data + i) T(std::move(data_[i]));
      (data_ + i)->~T();
    }
    if (!smallDataUsed()) {
      il::deallocate(data_);
    }
  }
  data_ = new_data;
  size_ = data_ + n;
  capacity_ = data_ + r;
}

template <typename T, il::int_t small_size>
bool SmallArray<T, small_size>::invariance() const {
  bool ans = true;

  if (data_ == reinterpret_cast<const T*>(small_data_)) {
    ans = ans && (size_ - data_ <= small_size);
    ans = ans && (capacity_ - data_ == small_size);
  } else {
    ans = ans && (size_ - data_ >= 0);
    ans = ans && (capacity_ - data_ >= 0);
    ans = ans && ((size_ - data_) <= (capacity_ - data_));
  }
  return ans;
}
}  // namespace il

#endif  // IL_SMALLARRAY_H
