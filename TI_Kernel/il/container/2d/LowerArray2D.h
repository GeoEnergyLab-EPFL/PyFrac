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

#ifndef IL_LOWERARRAY2D_H
#define IL_LOWERARRAY2D_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <new> is needed for ::operator new
#include <new>
// <type_traits> is needed for std::is_pod
#include <type_traits>
// <utility> is needed for std::move
#include <utility>

#include <il/core.h>

namespace il {

template <typename T>
class LowerArray2D {
 private:
#ifdef IL_DEBUG_VISUALIZER
  il::int_t debug_size_;
  il::int_t debug_capacity_;
#endif
  T* data_;
  T* size_;
  T* capacity_;

 public:
  LowerArray2D();
  LowerArray2D(il::int_t n);
  LowerArray2D(const LowerArray2D<T>& A);
  LowerArray2D(LowerArray2D<T>&& A);
  LowerArray2D<T>& operator=(const LowerArray2D<T>& A);
  LowerArray2D<T>& operator=(LowerArray2D<T>&& A);
  ~LowerArray2D();
  const T& operator()(il::int_t i0, il::int_t i1) const;
  T& operator()(il::int_t i0, il::int_t i1);
  il::int_t size() const;
  il::int_t capacity() const;
  T* Data();
};

template <typename T>
LowerArray2D<T>::LowerArray2D() {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = 0;
  debug_capacity_ = 0;
#endif
  data_ = nullptr;
  size_ = nullptr;
  capacity_ = nullptr;
}

template <typename T>
LowerArray2D<T>::LowerArray2D(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  if (n > 0) {
    const il::int_t nb_elements{(n * (n + 1)) / 2};
    if (il::isTrivial<T>::value) {
      data_ = new T[nb_elements];
#ifdef IL_DEFAULT_VALUE
      for (il::int_t k = 0; k < nb_elements; ++k) {
        data_[k] = il::defaultValue<T>();
      }
#endif
    } else {
      data_ = static_cast<T*>(::operator new(nb_elements * sizeof(T)));
      for (il::int_t k = 0; k < nb_elements; ++k) {
        new (data_ + k) T{};
      }
    }
  } else {
    data_ = nullptr;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = n;
  debug_capacity_ = n;
#endif
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
LowerArray2D<T>::LowerArray2D(const LowerArray2D<T>& A) {
  const il::int_t n{A.size()};
  if (n > 0) {
    const il::int_t nb_elements{(n * (n + 1)) / 2};
    if (il::isTrivial<T>::value) {
      data_ = new T[nb_elements];
      memcpy(data_, A.data_, nb_elements * sizeof(T));
    } else {
      data_ = static_cast<T*>(::operator new(nb_elements * sizeof(T)));
      for (il::int_t k = 0; k < nb_elements; ++k) {
        new (data_ + k) T{A.data_[k]};
      }
    }
  } else {
    data_ = nullptr;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = n;
  debug_capacity_ = n;
#endif
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
LowerArray2D<T>::LowerArray2D(LowerArray2D<T>&& A) {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = A.debug_size_;
  debug_capacity_ = A.debug_capacity_;
#endif
  data_ = A.data_;
  size_ = A.size_;
  capacity_ = A.capacity_;
#ifdef IL_DEBUG_VISUALIZER
  A.debug_size_ = 0;
  A.debug_capacity_ = 0;
#endif
  A.data_ = nullptr;
  A.size_ = nullptr;
  A.capacity_ = nullptr;
}

template <typename T>
LowerArray2D<T>& LowerArray2D<T>::operator=(const LowerArray2D<T>& A) {
  if (this != &A) {
    const il::int_t n{A.size()};
    const il::int_t nb_elements{(n * (n + 1)) / 2};
    const bool needs_memory{n > capacity()};
    if (needs_memory) {
      if (il::isTrivial<T>::value) {
        if (data_) {
          delete[] data_;
        }
        data_ = new T[nb_elements];
        memcpy(data_, A.data_, nb_elements * sizeof(T));
      } else {
        if (data_) {
          for (il::int_t k{nb_elements - 1}; k >= 0; --k) {
            (data_ + k)->~T();
          }
          ::operator delete(data_);
        }
        data_ = static_cast<T*>(::operator new(nb_elements * sizeof(T)));
        for (il::int_t k = 0; k < nb_elements; ++k) {
          new (data_ + k) T{A.data_[k]};
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_ = n;
      debug_capacity_ = n;
#endif
      size_ = data_ + n;
      capacity_ = data_ + n;
    } else {
      if (il::isTrivial<T>::value) {
        memcpy(data_, A.data_, n * sizeof(T));
      } else {
        for (il::int_t k = 0; k < nb_elements; ++k) {
          data_[k] = A.data_[k];
        }
        const il::int_t nb_elements_old{(size() * (size() + 1)) / 2};
        for (il::int_t k{nb_elements_old - 1}; k >= nb_elements; --k) {
          (data_ + k)->~T();
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_ = n;
#endif
      size_ = data_ + n;
    }
  }
  return *this;
}

template <typename T>
LowerArray2D<T>& LowerArray2D<T>::operator=(LowerArray2D<T>&& A) {
  if (this != &A) {
    if (data_) {
      if (il::isTrivial<T>::value) {
        delete[] data_;
      } else {
        const il::int_t nb_elements{(size() * (size() + 1)) / 2};
        for (il::int_t k{nb_elements - 1}; k >= 0; --k) {
          (data_ + k)->~T();
        }
        ::operator delete(data_);
      }
    }
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = A.debug_size_;
    debug_capacity_ = A.debug_capacity_;
#endif
    data_ = A.data_;
    size_ = A.size_;
    capacity_ = A.capacity_;
#ifdef IL_DEBUG_VISUALIZER
    A.debug_size_ = 0;
    A.debug_capacity_ = 0;
#endif
    A.data_ = nullptr;
    A.size_ = nullptr;
    A.capacity_ = nullptr;
  }
  return *this;
}

template <typename T>
LowerArray2D<T>::~LowerArray2D() {
  if (data_) {
    if (il::isTrivial<T>::value) {
      delete[] data_;
    } else {
      const il::int_t nb_elements{(size() * (size() + 1)) / 2};
      for (il::int_t k{nb_elements - 1}; k >= 0; --k) {
        (data_ + k)->~T();
      }
      ::operator delete(data_);
    }
  }
}

template <typename T>
const T& LowerArray2D<T>::operator()(il::int_t i0, il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(i1 <= i0);
  return data_[(i1 * (2 * (size_ - data_) - (1 + i1))) / 2 + i0];
}

template <typename T>
T& LowerArray2D<T>::operator()(il::int_t i0, il::int_t i1) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(i1 <= i0);
  return data_[(i1 * (2 * (size_ - data_) - (1 + i1))) / 2 + i0];
}

template <typename T>
il::int_t LowerArray2D<T>::size() const {
  return static_cast<il::int_t>(size_ - data_);
}

template <typename T>
il::int_t LowerArray2D<T>::capacity() const {
  return static_cast<il::int_t>(capacity_ - data_);
}

template <typename T>
T* LowerArray2D<T>::Data() {
  return data_;
}
}  // namespace il

#endif  // IL_LOWERARRAY2D_H
