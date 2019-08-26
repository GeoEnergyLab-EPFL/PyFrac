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

#ifndef IL_BASE_H
#define IL_BASE_H

// <cstddef> is needed for std::size_t and std::ptrdiff_t
#include <cstddef>
// <climits> is needed for LONG_MAX
#include <climits>
// <cstdlib> is needed for std::abort()
#include <cstdlib>

#include <complex>

////////////////////////////////////////////////////////////////////////////////
// Configuration
////////////////////////////////////////////////////////////////////////////////

//#define IL_BLAS_ATLAS

////////////////////////////////////////////////////////////////////////////////
// Multiple platforms
////////////////////////////////////////////////////////////////////////////////

#if defined(WIN32) || defined(_WIN32) || \
    defined(__WIN32) && !defined(__CYGWIN__)
#define IL_WINDOWS
#else
#define IL_UNIX
#endif

#if _WIN32 || _WIN64
#if _WIN64
#define IL_64_BIT
#else
#define IL_32_BIT
#endif
#endif

// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__
#define IL_64_BIT
#else
#define IL_32_BIT
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
// Assertions
////////////////////////////////////////////////////////////////////////////////

namespace il {
struct AbortException {
  AbortException() { (void)0; }
};

inline void abort() { std::abort(); }

}  // namespace il

#ifdef NDEBUG
#define IL_ASSERT(condition) ((void)0)
#else
#define IL_ASSERT(condition) (condition) ? ((void)0) : il::abort();
#endif

// Use this when the expectation is fast to compute compared to the function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_FAST(condition) \
  (condition) ? ((void)0) : throw il::AbortException {}
#elif NDEBUG
#define IL_EXPECT_FAST(condition) ((void)0)
#else
#define IL_EXPECT_FAST(condition) (condition) ? ((void)0) : il::abort();
#endif

#ifdef NDEBUG
#define IL_EXPECT_FAST_NOTHROW(condition) ((void)0)
#else
#define IL_EXPECT_FAST_NOTHROW(condition) (condition) ? ((void)0) : il::abort();
#endif

// Use this when the the expectation is as expensive to compute as the function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_MEDIUM(condition) \
  (condition) ? ((void)0) : throw il::AbortException {}
#elif NDEBUG
#define IL_EXPECT_MEDIUM(condition) ((void)0)
#else
#define IL_EXPECT_MEDIUM(condition) (condition) ? ((void)0) : il::abort();
#endif

// Use this when the the expectation is more expensive to compute than the
// function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_SLOW(condition) \
  (condition) ? ((void)0) : throw il::AbortException {}
#elif NDEBUG
#define IL_EXPECT_SLOW(condition) ((void)0)
#else
#define IL_EXPECT_SLOW(condition) (condition) ? ((void)0) : il::abort();
#endif

// This one is not check and can contain code that is not run
#define IL_EXPECT_AXIOM(message) ((void)0)

#ifdef IL_UNIT_TEST
#define IL_ENSURE(condition) \
  (condition) ? ((void)0) : throw il::AbortException {}
#elif NDEBUG
#define IL_ENSURE(condition) ((void)0)
#else
#define IL_ENSURE(condition) (condition) ? ((void)0) : il::abort();
#endif

#define IL_UNREACHABLE il::abort()

#define IL_UNUSED(var) ((void)var)

#ifndef NDEBUG
#define IL_DEFAULT_VALUE
#endif

#ifndef NDEBUG
#define IL_DEBUGGER_HELPERS
#endif

#ifndef NDEBUG
#define IL_DEBUG_CLASS
#endif

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

inline constexpr unsigned char operator"" _uchar(unsigned long long n) {
  return static_cast<unsigned char>(n);
}

////////////////////////////////////////////////////////////////////////////////
// Namespace il
////////////////////////////////////////////////////////////////////////////////

namespace il {

typedef std::ptrdiff_t int_t;
typedef std::size_t uint_t;

#ifdef IL_64_BIT
#define IL_INTEGER_MAX 9223372036854775807
#else
#define IL_INTEGER_MAX 2147483647
#endif

struct Range {
  il::int_t begin;
  il::int_t end;
};

template <typename T>
T max(T a, T b) {
  return a >= b ? a : b;
}

template <typename T>
T max(T a, int b) {
  return a >= b ? a : b;
}

template <typename T>
T max(int a, T b) {
  return a >= b ? a : b;
}

template <typename T>
T max(T a, T b, T c) {
  return max(max(a, b), c);
}

template <typename T>
T max(T a, T b, T c, T d) {
  return max(max(a, b), max(c, d));
}

template <typename T>
T max(T a, T b, T c, T d, T e) {
  return max(max(a, b), max(c, d), e);
}

template <typename T>
T max(T a, T b, T c, T d, T e, T f) {
  return max(max(a, b), max(c, d), max(e, f));
}

template <typename T>
T min(T a, T b) {
  return a <= b ? a : b;
}

////////////////////////////////////////////////////////////////////////////////
// For arrays
////////////////////////////////////////////////////////////////////////////////

struct dummy_t {};

struct io_t {};
const io_t io{};

struct value_t {};
const value_t value{};

struct unsafe_t {};
const unsafe_t unsafe{};

struct safe_t {};
const safe_t safe{};

struct emplace_t {};
const emplace_t emplace{};

struct align_t {};
const align_t align{};

struct unit_t {};
const unit_t unit{};

struct spot_t {
  il::int_t index;
#ifdef IL_DEBUG_CLASS
  std::size_t signature;
#endif
  spot_t() {
    index = -1;
#ifdef IL_DEBUG_CLASS
    signature = -1;
#endif
  };
  spot_t(il::int_t _index) {
    index = _index;
#ifdef IL_DEBUG_CLASS
    signature = -1;
#endif
  }
  bool isValid() const { return index >= 0; };
};

inline bool operator!=(il::spot_t i0, il::spot_t i1) {
  return i0.index != i1.index;
}

// class spot_t {
// private:
//  il::int_t i_;
//#ifdef IL_DEBUG_CLASS
//  std::size_t hash_;
//#endif
//
// public:
//#ifdef IL_DEBUG_CLASS
//  spot_t(il::int_t i, std::size_t hash);
//#else
//  spot_t(il::int_t i);
//#endif
//#ifdef IL_DEBUG_CLASS
//  void SetIndex(il::int_t i, std::size_t hash);
//#else
//  void SetIndex(il::int_t i);
//#endif
//  il::int_t index() const;
//  bool isValid() const;
//#ifdef IL_DEBUG_CLASS
//  std::size_t hash() const;
//#endif
//};
//
//#ifdef IL_DEBUG_CLASS
// inline spot_t::spot_t(il::int_t i, std::size_t hash) {
//  i_ = i;
//  hash_ = hash;
//}
//#else
// inline spot_t::spot_t(il::int_t i) { i_ = i; }
//#endif
//
//#ifdef IL_DEBUG_CLASS
// inline void spot_t::SetIndex(il::int_t i, std::size_t hash) {
//  i_ = i;
//  hash_ = hash;
//}
//#else
// inline void spot_t::SetIndex(il::int_t i) { i_ = i; }
//#endif
//
// inline il::int_t spot_t::index() const { return i_; }
//
// inline bool spot_t::isValid() const { return i_ >= 0; }
//
// inline std::size_t spot_t::hash() const { return hash_; }
//
// inline bool operator==(il::spot_t i0, il::spot_t i1) {
//#ifdef IL_DEBUG_CLASS
//  return i0.index() == i1.index() && i0.hash() == i1.hash();
//#else
//  return i0.index() == i1.index();
//#endif
//}
//
// inline bool operator!=(il::spot_t i0, il::spot_t i1) {
//#ifdef IL_DEBUG_CLASS
//  return i0.index() != i1.index() || i0.hash() != i1.hash();
//#else
//  return i0.index() != i1.index();
//#endif
//}

////////////////////////////////////////////////////////////////////////////////
// Default values for containers in debug mode
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct isTrivial {
  static constexpr bool value = false;
};

template <typename T>
T defaultValue() {
  return T{};
}

template <>
struct isTrivial<bool> {
  static constexpr bool value = true;
};

template <>
inline bool defaultValue<bool>() {
  return false;
}

template <>
struct isTrivial<char> {
  static constexpr bool value = true;
};

template <>
inline char defaultValue<char>() {
  return '\0';
}

template <>
struct isTrivial<signed char> {
  static constexpr bool value = true;
};

template <>
inline signed char defaultValue<signed char>() {
#if SCHAR_MAX == 127
  return 123;
#endif
}

template <>
struct isTrivial<unsigned char> {
  static constexpr bool value = true;
};

template <>
inline unsigned char defaultValue<unsigned char>() {
#if SCHAR_MAX == 127
  return 123;
#endif
}

template <>
struct isTrivial<short> {
  static constexpr bool value = true;
};

template <>
inline short defaultValue<short>() {
#if SHRT_MAX == 32767
  return 12345;
#endif
}

template <>
struct isTrivial<unsigned short> {
  static constexpr bool value = true;
};

template <>
inline unsigned short defaultValue<unsigned short>() {
#if SHRT_MAX == 32767
  return 12345;
#endif
}

template <>
struct isTrivial<int> {
  static constexpr bool value = true;
};

template <>
inline int defaultValue<int>() {
#if INT_MAX == 2147483647
  return 1234567891;
#endif
}

template <>
struct isTrivial<unsigned int> {
  static constexpr bool value = true;
};

template <>
inline unsigned int defaultValue<unsigned int>() {
#if INT_MAX == 2147483647
  return 1234567891;
#endif
}

template <>
struct isTrivial<long> {
  static constexpr bool value = true;
};

template <>
inline long defaultValue<long>() {
#if LONG_MAX == 2147483647
  return 1234567891;
#elif LONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct isTrivial<unsigned long> {
  static constexpr bool value = true;
};

template <>
inline unsigned long defaultValue<unsigned long>() {
#if LONG_MAX == 2147483647
  return 1234567891;
#elif LONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct isTrivial<long long> {
  static constexpr bool value = true;
};

template <>
inline long long defaultValue<long long>() {
#if LLONG_MAX == 2147483647
  return 1234567891;
#elif LLONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct isTrivial<unsigned long long> {
  static constexpr bool value = true;
};

template <>
inline unsigned long long defaultValue<unsigned long long>() {
#if LLONG_MAX == 2147483647
  return 1234567891;
#elif LLONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct isTrivial<float> {
  static constexpr bool value = true;
};

template <>
inline float defaultValue<float>() {
#ifdef IL_UNIX
  return 0.0f / 0.0f;
#else
  return 0.0f;
#endif
}

template <>
struct isTrivial<double> {
  static constexpr bool value = true;
};

template <>
inline double defaultValue<double>() {
#ifdef IL_UNIX
  return 0.0 / 0.0;
#else
  return 0.0;
#endif
}

template <>
struct isTrivial<long double> {
  static constexpr bool value = true;
};

template <>
inline long double defaultValue<long double>() {
#ifdef IL_UNIX
  return 0.0l / 0.0l;
#else
  return 0.0l;
#endif
}

template <>
struct isTrivial<std::complex<float>> {
  static constexpr bool value = true;
};

template <>
inline std::complex<float> defaultValue<std::complex<float>>() {
#ifdef IL_UNIX
  return std::complex<float>{0.0f / 0.0f, 0.0f / 0.0f};
#else
  return std::complex<float>{0.0f, 0.0f};
#endif
}

template <>
struct isTrivial<std::complex<double>> {
  static constexpr bool value = true;
};

template <>
inline std::complex<double> defaultValue<std::complex<double>>() {
#ifdef IL_UNIX
  return std::complex<double>{0.0 / 0.0, 0.0 / 0.0};
#else
  return std::complex<double>{0.0, 0.0};
#endif
}

}  // namespace il

#endif  // IL_BASE_H
