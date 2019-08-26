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

#ifndef IL_MATH_H
#define IL_MATH_H

#include <cmath>
#include <complex>

#include <il/core.h>

namespace il {

template <typename T>
struct epsilon {
  static constexpr T value = 1.0;
};

template <>
struct epsilon<double> {
  static constexpr double value = 1.1102230246251565404e-16;
};

template <>
struct epsilon<long double> {
  static constexpr double value = 2.7105054312137610850e-20;
};

const double pi = 3.1415926535897932385;
const std::complex<double> ii = std::complex<double>{0.0, 1.0};

template <typename T>
T min(T a, T b, T c, T d) {
  return min(min(a, b), min(c, d));
}

// template <typename T>
// T max(T a, T b, T c) {
//  auto temp = T{max(b, c)};
//  return a >= temp ? a : temp;
//}
//
// template <typename T>
// T max(T a, T b, T c, T d) {
//  return max(max(a, b), max(c, d));
//}

inline il::int_t floor(double x) {
  return static_cast<il::int_t>(std::floor(x));
}

inline int abs(int x) { return x >= 0 ? x : -x; }

inline il::int_t abs(il::int_t x) { return x >= 0 ? x : -x; }

inline float abs(float x) { return x >= 0 ? x : -x; }

inline double abs(double x) { return x >= 0 ? x : -x; }

inline float real(float x) { return x; }

inline double real(double x) { return x; }

inline double real(std::complex<double> x) { return x.real(); }

inline bool isFinite(double x) { return std::isfinite(x); }

inline bool isFinite(std::complex<double> x) {
  return std::isfinite(x.real()) && std::isfinite(x.imag());
}

inline double conjugate(double x) { return x; }

inline std::complex<double> conjugate(std::complex<double> z) {
  return std::conj(z);
}

inline float abs(std::complex<float> x) {
  const float re = x.real();
  const float im = x.imag();
  return std::sqrt(re * re + im * im);
}

inline double abs(std::complex<double> x) {
  const double re = x.real();
  const double im = x.imag();
  return std::sqrt(re * re + im * im);
}

inline il::int_t floor(il::int_t a, int b) { return (a / b) * b; }

inline il::int_t floor(il::int_t a, il::int_t b) { return (a / b) * b; }
// Template for pow(x,N) where N is a positive il::int_t constant.
// General case, N is not a power of 2:
template <bool IsPowerOf2, il::int_t N, typename T>
class powN {
 public:
  static T p(T x) {
// Remove right-most 1-bit in binary representation of N:
#define N1 (N & (N - 1))
    return powN<(N1 & (N1 - 1)) == 0, N1, T>::p(x) *
           powN<true, N - N1, T>::p(x);
#undef N1
  }
};

// Partial template specialization for N a power of 2
template <il::int_t N, typename T>
class powN<true, N, T> {
 public:
  static T p(T x) {
    return powN<true, N / 2, T>::p(x) * powN<true, N / 2, T>::p(x);
  }
};

// Full template specialization for N = 1. This ends the recursion
template <typename T>
class powN<true, 1, T> {
 public:
  static T p(T x) { return x; }
};

// Full template specialization for N = 0
// This is used only for avoiding infinite loop if powN is
// erroneously called with IsPowerOf2 = false where it should be true.
template <typename T>
class powN<true, 0, T> {
 public:
  static T p(T x) {
    (void)x;

    return 1;
  }
};

// Function template for x to the power of N
template <il::int_t N, typename T>
static T ipow(T x) {
  // (N & N-1) == 0 if N is a power of 2
  return powN<(N & (N - 1)) == 0, N, T>::p(x);
}

template <typename T>
T ipow(T x, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  T ans = 1;
  while (n != 0) {
    if (n & 1) {
      ans *= x;
    }
    x *= x;
    n >>= 1;
  }
  return ans;
}

inline unsigned int previous_power_of_2_32(unsigned int x) {
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x -= (x >> 1);

  return x;
}

inline unsigned int next_power_of_2_32(unsigned int x) {
  x -= 1;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x += 1;

  return x;
}

inline std::uint64_t previous_power_of_2_64(std::uint64_t x) {
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x -= (x >> 1);

  return x;
}

inline std::uint64_t next_power_of_2_64(std::uint64_t x) {
  x -= 1;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x += 1;

  return x;
}

// From http://chessprogramming.wikispaces.com/BitScan
static const int table_log2_64[64] = {
    63, 0,  58, 1,  59, 47, 53, 2,  60, 39, 48, 27, 54, 33, 42, 3,
    61, 51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4,
    62, 57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21,
    56, 45, 25, 31, 35, 16, 9,  12, 44, 24, 15, 8,  23, 7,  6,  5};

inline int nextLog2(std::size_t x) {
  //  x |= x >> 1;
  //  x |= x >> 2;
  //  x |= x >> 4;
  //  x |= x >> 8;
  //  x |= x >> 16;
  //  x |= x >> 32;
  //  const il::int_t index =
  //      static_cast<il::int_t>((x * 0x07EDD5E59A4E28C2) >> 58);
  //
  //  return table_log2_64[index];
  std::size_t power = 1;
  int k = 0;
  while (power < x) {
    power *= 2;
    k += 1;
  }
  return k;
}

static const int table_log2_32[32] = {
    0, 9,  1,  10, 13, 21, 2,  29, 11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7,  19, 27, 23, 6,  26, 5,  4, 31};

inline int next_log2_32(unsigned int x) {
  //  x |= x >> 1;
  //  x |= x >> 2;
  //  x |= x >> 4;
  //  x |= x >> 8;
  //  x |= x >> 16;
  //  const int32_t index = static_cast<int>(x * 0x07C4ACDD) >> 27;
  //
  //  return table_log2_32[index];
  unsigned int power = 1;
  int k = 0;
  while (power < x) {
    power *= 2;
    k += 1;
  }

  return k;
}
}  // namespace il

#endif  // IL_MATH_H
