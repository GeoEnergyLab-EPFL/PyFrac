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

#ifndef IL_ALGORITHMARRAY2D_H
#define IL_ALGORITHMARRAY2D_H

#include <limits>

#include <il/Array2D.h>
#include <il/algorithmArray.h>

namespace il {

template <typename T>
MinMax<T> minMax(const il::Array2D<T>& A, il::Range i0_range, il::int_t i1) {
  IL_EXPECT_FAST(i0_range.begin < i0_range.end);

  MinMax<T> ans{};
  ans.min = std::numeric_limits<T>::max();
  ans.max = -std::numeric_limits<T>::max();
  for (il::int_t i0 = i0_range.begin; i0 < i0_range.end; ++i0) {
    if (A(i0, i1) < ans.min) {
      ans.min = A(i0, i1);
    }
    if (A(i0, i1) > ans.max) {
      ans.max = A(i0, i1);
    }
  }
  return ans;
}

}  // namespace il

#endif  // IL_ALGORITHMARRAY2D_H
