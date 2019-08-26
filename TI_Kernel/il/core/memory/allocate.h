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

#ifndef IL_ALLOCATE_H
#define IL_ALLOCATE_H

// <cstdlib> is used for std::malloc
#include <cstdlib>

#include <il/core/math/safe_arithmetic.h>
#include <il/math.h>


//#include <iostream>

namespace il {

template <typename T>
T* allocateArray(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  const std::size_t u_n = static_cast<std::size_t>(n);
  const std::size_t u_max_integer =
      static_cast<std::size_t>(1)
      << (sizeof(std::size_t) * 8 - (1 + il::nextLog2(sizeof(T))));
  if (u_n >= u_max_integer) {
    il::abort();
  }
  const std::size_t n_bytes = sizeof(T) * u_n;

  T* p = static_cast<T*>(std::malloc(n_bytes));
  if (!p && n_bytes > 0) {
    il::abort();
  }

//  std::cout << "Allocated: " << p << std::endl;
//
  return p;
}

template <typename T>
T* allocateArray(il::int_t n, il::int_t align_r, il::int_t align_mod, il::io_t,
                 il::int_t& shift) {
  IL_EXPECT_FAST(sizeof(T) % alignof(T) == 0);
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);

  const std::size_t n_unsigned = static_cast<std::size_t>(n);
  const std::size_t align_mod_unsigned = static_cast<std::size_t>(align_mod);

  bool product_error = false;
  bool sum_error = false;
  const std::size_t n_bytes =
      il::safeSum(il::safeProduct(n_unsigned, sizeof(T), il::io, product_error),
                  align_mod_unsigned - 1, il::io, sum_error);
  if (product_error || sum_error) {
    il::abort();
  }

  T* p = static_cast<T*>(std::malloc(n_bytes));
  if (!p && n_bytes > 0) {
    il::abort();
  }
  const std::size_t align_r_unsigned = static_cast<std::size_t>(align_r);
  const std::size_t p_int = reinterpret_cast<std::size_t>(p);
  const std::size_t r = p_int % align_mod_unsigned;
  const std::size_t local_shift =
      align_r_unsigned >= r
          ? (align_r_unsigned - r) / sizeof(T)
          : (align_mod_unsigned - (r - align_r_unsigned)) / sizeof(T);

  IL_ENSURE(local_shift < static_cast<std::size_t>(align_mod));

  shift = static_cast<il::int_t>(local_shift);
  T* aligned_p = p + local_shift;
  return aligned_p;
}

inline void deallocate(void* p) {
//  std::cout << "Deallocated: " << p << std::endl;
//
  std::free(p); }
}  // namespace il

#endif  // IL_ALLOCATE_H
