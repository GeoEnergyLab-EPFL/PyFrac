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

#ifndef IL_SAFE_ARITHMETIC_H
#define IL_SAFE_ARITHMETIC_H

#include <il/core.h>

namespace il {

////////////////////////////////////////////////////////////////////////////////
// int
////////////////////////////////////////////////////////////////////////////////

inline int safeSum(int a, int b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  int ans;
  error = __builtin_sadd_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > INT_MAX - b : a < INT_MIN - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline int safeDifference(int a, int b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  int ans;
  error = __builtin_ssub_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a < INT_MIN + b : a > INT_MAX + b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a - b;
  }
#endif
}

inline int safeProduct(int a, int b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  int ans;
  error = __builtin_smul_overflow(a, b, &ans);
  return ans;
#elif INT_MAX < LLONG_MAX
  const long long a_llong = a;
  const long long b_llong = b;
  const long long product_llong = a_llong * b_llong;
  if (product_llong > INT_MAX || product_llong < INT_MIN) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<int>(product_llong);
  }
#else
  if (b > 0 ? a > INT_MAX / b || a < INT_MIN / b
            : (b < -1 ? a > INT_MIN / b || a < INT_MAX / b
                      : b == -1 && a == INT_MIN)) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

inline int safeProduct(int a, int b, int c, il::io_t, bool& error) {
  bool error_first;
  const int ab = il::safeProduct(a, b, il::io, error_first);
  bool error_second;
  const int abc = il::safeProduct(ab, c, il::io, error_second);
  if (error_first || error_second) {
    error = true;
    return 0;
  } else {
    error = false;
    return abc;
  }
}

inline int safeProduct(int a, int b, int c, int d, il::io_t, bool& error) {
  bool error_first;
  const int ab = il::safeProduct(a, b, il::io, error_first);
  bool error_second;
  const int cd = il::safeProduct(c, d, il::io, error_second);
  bool error_third;
  const int abcd = il::safeProduct(ab, cd, il::io, error_third);
  if (error_first || error_second || error_third) {
    error = true;
    return 0;
  } else {
    error = false;
    return abcd;
  }
}

inline int safeDivision(int a, int b, il::io_t, bool& error) {
  if (b == 0 || (b == -1 && a == INT_MIN)) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

inline unsigned safeSum(unsigned a, unsigned b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned ans;
  error = __builtin_uadd_overflow(a, b, &ans);
  return ans;
#else
  if (a > UINT_MAX - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline unsigned safeProduct(unsigned a, unsigned b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned ans;
  error = __builtin_umul_overflow(a, b, &ans);
  return ans;
#elif INT_MAX < LONG_MAX
  const unsigned long a_long = a;
  const unsigned long b_long = b;
  const unsigned long product_long = a_long * b_long;
  if (product_long > UINT_MAX) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<unsigned>(product_long);
  }
#else
  if (b > 0 && a > INT_MAX / b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// long
////////////////////////////////////////////////////////////////////////////////

inline long safeSum(long a, long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long ans;
  error = __builtin_saddl_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > LONG_MAX - b : a < LONG_MIN - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline long safeDifference(long a, long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long ans;
  error = __builtin_ssubl_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a < LONG_MIN + b : a > LONG_MAX + b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a - b;
  }
#endif
}

inline long safeProduct2Positive(long a, il::io_t, bool& error) {
  const unsigned long ua = static_cast<unsigned long>(a);
  constexpr unsigned long max = static_cast<unsigned long>(2)
                                << (sizeof(unsigned long) * 8 - 2);
  if (ua < max) {
    error = false;
    return 2 * a;
  } else {
    error = true;
    return 0;
  }
}

inline long safeProduct(long a, long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long ans;
  error = __builtin_smull_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > LONG_MAX / b || a < LONG_MIN / b
            : (b < -1 ? a > LONG_MIN / b || a < LONG_MAX / b
                      : b == -1 && a == LONG_MIN)) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

inline long safeProduct(long a, long b, long c, il::io_t, bool& error) {
  bool error_first;
  const long ab = il::safeProduct(a, b, il::io, error_first);
  bool error_second;
  const long abc = il::safeProduct(ab, c, il::io, error_second);
  if (error_first || error_second) {
    error = true;
    return 0;
  } else {
    error = false;
    return abc;
  }
}

inline long safeProduct(long a, long b, long c, long d, il::io_t, bool& error) {
  bool error_first;
  const long ab = il::safeProduct(a, b, il::io, error_first);
  bool error_second;
  const long cd = il::safeProduct(c, d, il::io, error_second);
  bool error_third;
  const long abcd = il::safeProduct(ab, cd, il::io, error_third);
  if (error_first || error_second || error_third) {
    error = true;
    return 0;
  } else {
    error = false;
    return abcd;
  }
}

inline long safeDivision(long a, long b, il::io_t, bool& error) {
  if (b == 0 || (b == -1 && a == LONG_MIN)) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

inline unsigned long safeSum(unsigned long a, unsigned long b, il::io_t,
                             bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned long ans;
  error = __builtin_uaddl_overflow(a, b, &ans);
  return ans;
#else
  if (a > ULONG_MAX - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline unsigned long safeProduct(unsigned long a, unsigned long b, il::io_t,
                                 bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned long ans;
  error = __builtin_umull_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 && a > ULONG_MAX / b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// long long
////////////////////////////////////////////////////////////////////////////////

inline long long safeSum(long long a, long long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long long ans;
  error = __builtin_saddll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > LLONG_MAX - b : a < LLONG_MIN - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline long long safeDifference(long long a, long long b, il::io_t,
                                bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long long ans;
  error = __builtin_ssubll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a < LLONG_MIN + b : a > LLONG_MAX + b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a - b;
  }
#endif
}

inline long long safeProduct(long long a, long long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long long ans;
  error = __builtin_smulll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > LLONG_MAX / b || a < LLONG_MIN / b
            : (b < -1 ? a > LLONG_MIN / b || a < LLONG_MAX / b
                      : b == -1 && a == LLONG_MIN)) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

inline long long safeProduct(long long a, long long b, long long c, il::io_t,
                             bool& error) {
  bool error_first;
  const long long ab = il::safeProduct(a, b, il::io, error_first);
  bool error_second;
  const long long abc = il::safeProduct(ab, c, il::io, error_second);
  if (error_first || error_second) {
    error = true;
    return 0;
  } else {
    error = false;
    return abc;
  }
}

inline long long safeProduct(long long a, long long b, long long c, long long d,
                             il::io_t, bool& error) {
  bool error_first;
  const long long ab = il::safeProduct(a, b, il::io, error_first);
  bool error_second;
  const long long cd = il::safeProduct(c, d, il::io, error_second);
  bool error_third;
  const long long abcd = il::safeProduct(ab, cd, il::io, error_third);
  if (error_first || error_second || error_third) {
    error = true;
    return 0;
  } else {
    error = false;
    return abcd;
  }
}

inline long long safeDivision(long long a, long long b, il::io_t, bool& error) {
  if (b == 0 || (b == -1 && a == LLONG_MIN)) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

inline unsigned long long safeSum(unsigned long long a, unsigned long long b,
                                  il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned long long ans;
  error = __builtin_uaddll_overflow(a, b, &ans);
  return ans;
#else
  if (a > ULLONG_MAX - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline unsigned long long safeProduct(unsigned long long a,
                                      unsigned long long b, il::io_t,
                                      bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned long long ans;
  error = __builtin_umulll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 && a > ULLONG_MAX / b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// rounding
////////////////////////////////////////////////////////////////////////////////

inline int safeUpperRound(int a, int b, il::io_t, bool& error) {
  IL_EXPECT_FAST(a >= 0);
  IL_EXPECT_FAST(b > 0);

  const unsigned q = static_cast<unsigned>(a) / static_cast<unsigned>(b);
  const unsigned r = static_cast<unsigned>(a) % static_cast<unsigned>(b);
  if (r == 0) {
    error = false;
    return a;
  } else {
    bool error_sum = false;
    bool error_product = false;
    const int q_plus_one = il::safeSum(static_cast<int>(q), static_cast<int>(1),
                                       il::io, error_sum);
    const int ans = il::safeProduct(q_plus_one, b, il::io, error_product);
    if (error_sum || error_product) {
      error = true;
      return 0;
    } else {
      error = false;
      return ans;
    }
  }
}

inline long safeUpperRound(long a, long b, il::io_t, bool& error) {
  IL_EXPECT_FAST(a >= 0);
  IL_EXPECT_FAST(b > 0);

  const unsigned long q =
      static_cast<unsigned long>(a) / static_cast<unsigned long>(b);
  const unsigned long r =
      static_cast<unsigned long>(a) % static_cast<unsigned long>(b);
  if (r == 0) {
    error = false;
    return a;
  } else {
    bool error_sum = false;
    bool error_product = false;
    const long q_plus_one = il::safeSum(
        static_cast<long>(q), static_cast<long>(1), il::io, error_sum);
    const long ans = il::safeProduct(q_plus_one, b, il::io, error_product);
    if (error_sum || error_product) {
      error = true;
      return 0;
    } else {
      error = false;
      return ans;
    }
  }
}

inline long long safeUpperRound(long long a, long long b, il::io_t,
                                bool& error) {
  IL_EXPECT_FAST(a >= 0);
  IL_EXPECT_FAST(b > 0);

  const unsigned long long q =
      static_cast<unsigned long long>(a) / static_cast<unsigned long long>(b);
  const unsigned long long r =
      static_cast<unsigned long long>(a) % static_cast<unsigned long long>(b);
  if (r == 0) {
    error = false;
    return a;
  } else {
    bool error_sum = false;
    bool error_product = false;
    const long long q_plus_one =
        il::safeSum(static_cast<long long>(q), static_cast<long long>(1),
                    il::io, error_sum);
    const long long ans = il::safeProduct(q_plus_one, b, il::io, error_product);
    if (error_sum || error_product) {
      error = true;
      return 0;
    } else {
      error = false;
      return ans;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// convert
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
T1 safeConvert(T2 n, il::io_t, bool& error) {
  IL_UNUSED(n);
  IL_UNUSED(error);
  IL_UNREACHABLE;
}

template <>
inline int safeConvert(unsigned n, il::io_t, bool& error) {
  if (n > INT_MAX) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<int>(n);
  }
}

template <>
inline unsigned safeConvert(int n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<unsigned>(n);
  }
}

template <>
inline long safeConvert(unsigned long n, il::io_t, bool& error) {
  if (n > LONG_MAX) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<long>(n);
  }
}

template <>
inline unsigned long safeConvert(long n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<unsigned long>(n);
  }
}

template <>
inline long long safeConvert(unsigned long long n, il::io_t, bool& error) {
  if (n > LLONG_MAX) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<long long>(n);
  }
}

template <>
inline unsigned long long safeConvert(long long n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<unsigned long long>(n);
  }
}
}  // namespace il

#endif  // IL_SAFE_ARITHMETIC_H
