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

#ifndef IL_TRIDIAGONAL_H
#define IL_TRIDIAGONAL_H

#include <il/Array.h>

namespace il {

template <typename T>
class TriDiagonal {
 private:
  il::int_t size_;
  il::Array<T> data_;

 public:
  TriDiagonal(il::int_t n) : data_{3 * n} { size_ = n; }
  const T& operator()(il::int_t i, il::int_t k) const {
    IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                     static_cast<std::size_t>(size_));
    IL_EXPECT_MEDIUM(k >= -1 && k <= 1);
    return data_[size_ * (k + 1) + i];
  }
  T& operator()(il::int_t i, il::int_t k) {
    IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                     static_cast<std::size_t>(size_));
    IL_EXPECT_MEDIUM(k >= -1 && k <= 1);
    return data_[size_ * (k + 1) + i];
  }
  il::int_t size() const { return size_; }
  T* LowerData() { return data_.Data() + 1; }
  T* DiagonalData() { return data_.Data() + size_; }
  T* UpperData() { return data_.Data() + 2 * size_; }
};

}  // namespace il

#endif  // IL_TRIDIAGONAL_H
