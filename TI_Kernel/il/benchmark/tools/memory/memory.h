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

#ifndef IL_MEMORY_H
#define IL_MEMORY_H

namespace il {

void escape(void* p) { asm volatile("" : : "g"(p) : "memory"); }

void clobber() { asm volatile("" : : : "memory"); }

// template <typename T>
// void commit_memory(il::io_t, il::Array2D<T>& A) {
//   il::int_t stride{static_cast<il::int_t>(il::page / sizeof(T))};
//   T* const data{A.data()};
//   for (il::int_t k = 0; k < A.capacity(0) * A.capacity(1); k += stride) {
//     data[k] = T{};
//   }
// }

// template <typename T>
// void warm_cache(il::io_t, il::Array2D<T>& A) {
//   T* const data{A.data()};
//   for (il::int_t k = 0; k < A.capacity(0) * A.capacity(1); ++k) {
//     data[k] = T{};
//   }
// }
}  // namespace il

#endif  // IL_MEMORY_H
