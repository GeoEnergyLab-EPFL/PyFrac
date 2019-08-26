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

#ifndef IL_CROSS_H
#define IL_CROSS_H

#include <il/StaticArray.h>

namespace il {

template <typename T>
il::StaticArray<T, 2> cross(const il::StaticArray<T, 2>& x) {
  il::StaticArray<T, 2> ans{};
  ans[0] = -x[1];
  ans[1] = x[0];
  return ans;
}

template <typename T>
il::StaticArray<T, 3> cross(const il::StaticArray<T, 3>& x,
                            const il::StaticArray<T, 3>& y) {
  il::StaticArray<T, 3> ans{};
  ans[0] = x[1] * y[2] - x[2] * y[1];
  ans[1] = x[2] * y[0] - x[0] * y[2];
  ans[2] = x[0] * y[1] - x[1] * y[0];
  return ans;
}

}  // namespace il

#endif  // IL_CROSS_H
