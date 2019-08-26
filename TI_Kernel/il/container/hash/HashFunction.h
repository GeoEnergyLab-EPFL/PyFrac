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

#ifndef IL_HASHFUNCTION_H
#define IL_HASHFUNCTION_H

#include <limits>

#include <il/String.h>
#include <il/core.h>

namespace il {

template <typename T>
class HashFunction {
 public:
  //  static inline bool isEmpty(const T& val);
  //  static inline bool isTombstone(const T& val);
  //  static inline void constructEmpty(il::io_t, T* val);
  //  static inline void constructTombstone(il::io_t, T* val);
  //  static std::size_t hash(const T& val, int p);
  //  static bool isEqual(const T& val0, const T& val1);
};
template <>
class HashFunction<int> {
 public:
  static inline bool isEmpty(int val) {
    return val == std::numeric_limits<int>::min();
  }
  static inline bool isTombstone(int val) {
    return val == std::numeric_limits<int>::min() + 1;
  }
  static inline void constructEmpty(il::io_t, int* val) {
    *val = std::numeric_limits<int>::min();
  }
  static inline void constructTombstone(il::io_t, int* val) {
    *val = std::numeric_limits<int>::min() + 1;
  }
  static std::size_t hash(int val, int p) {
#ifdef IL_64_BIT
    const std::size_t knuth = 11133510565745311;
    const std::size_t y = static_cast<std::size_t>(val);

    return (y * knuth) >> (64 - p);
#else
    return static_cast<std::size_t>(val);
#endif
  }
  static bool isEqual(int val0, int val1) { return val0 == val1; }
};

template <>
class HashFunction<long> {
 public:
  static inline bool isEmpty(long val) {
    return val == std::numeric_limits<long>::min();
  }
  static inline bool isTombstone(long val) {
    return val == std::numeric_limits<long>::min() + 1;
  }
  static inline void constructEmpty(il::io_t, long* val) {
    *val = std::numeric_limits<long>::min();
  }
  static inline void constructTombstone(il::io_t, long* val) {
    *val = std::numeric_limits<long>::min() + 1;
  }
  // Note that a 32-bit hash would be
  //
  // const std::size_t knuth = 2654435769;
  // const std::size_t y = val;
  // return (y * knuth) >> (32 - p);
  //
  // And a basic hash would be
  //
  // const std::size_t mask = (1 << p) - 1;
  // return static_cast<std::size_t>(val) & mask;
  static std::size_t hash(long val, int p) {
#ifdef IL_64_BIT
    const std::size_t knuth = 11133510565745311;
    const std::size_t y = val;

    return (y * knuth) >> (64 - p);
#else
    return static_cast<std::size_t>(val);
#endif
  }
  static bool isEqual(long val0, long val1) { return val0 == val1; }
};

template <>
class HashFunction<il::String> {
 public:
  static constexpr il::int_t max_small_size_ =
      static_cast<il::int_t>(3 * sizeof(std::size_t) - 2);
  static inline bool isEmpty(const il::String& s) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&s);
    const unsigned char value = p[max_small_size_ + 1];
    return value == 0x1F_uchar;
  }
  static inline bool isTombstone(const il::String& s) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&s);
    const unsigned char value = p[max_small_size_ + 1];
    return value == 0x1E_uchar;
  }
  static inline void constructEmpty(il::io_t, il::String* s) {
    unsigned char* p = reinterpret_cast<unsigned char*>(s);
    p[max_small_size_ + 1] = 0x1F_uchar;
  }
  static inline void constructTombstone(il::io_t, il::String* s) {
    unsigned char* p = reinterpret_cast<unsigned char*>(s);
    p[max_small_size_ + 1] = 0x1E_uchar;
  }
  static std::size_t hash(const il::String& s, int p) {
    const std::size_t mask = (1 << p) - 1;
    const il::int_t n = s.size();
    const char* p_string = s.asCString();
    std::size_t hash = 5381;
    for (il::int_t i = 0; i < n; ++i) {
      hash = ((hash << 5) + hash) + p_string[i];
    }
    return hash & mask;
  }
  static std::size_t hash(const char* s, il::int_t n, int p) {
    const std::size_t mask = (1 << p) - 1;
    std::size_t hash = 5381;
    for (il::int_t i = 0; i < n; ++i) {
      hash = ((hash << 5) + hash) + s[i];
    }
    return hash & mask;
  }
  template <il::int_t m>
  static std::size_t hash(const char (&s)[m], int p) {
    const std::size_t mask = (1 << p) - 1;
    const il::int_t n = m - 1;
    std::size_t hash = 5381;
    for (il::int_t i = 0; i < n; ++i) {
      hash = ((hash << 5) + hash) + s[i];
    }
    return hash & mask;
  }
  static bool isEqual(const il::String& s0, const il::String& s1) {
    const il::int_t n0 = s0.size();
    const il::int_t n1 = s1.size();
    if (n0 != n1) {
      return false;
    }

    const char* p0 = s0.asCString();
    const char* p1 = s1.asCString();
    il::int_t i = 0;
    while (i < n0 && p0[i] == p1[i]) {
      ++i;
    }
    return i == n0;
  }
  static bool isEqual(const il::String& s0, const char* s1, il::int_t n) {
    const il::int_t n0 = s0.size();
    const il::int_t n1 = n;
    if (n0 != n1) {
      return false;
    }

    const char* p0 = s0.asCString();
    il::int_t i = 0;
    while (i < n0 && p0[i] == s1[i]) {
      ++i;
    }
    return i == n0;
  }
  template <il::int_t m>
  static bool isEqual(const il::String& s0, const char (&s1)[m]) {
    const il::int_t n0 = s0.size();
    const il::int_t n1 = m - 1;
    if (n0 != n1) {
      return false;
    }

    const char* p0 = s0.asCString();
    il::int_t i = 0;
    while (i < n0 && p0[i] == s1[i]) {
      ++i;
    }
    return i == n0;
  }
};
}  // namespace il

#endif  // IL_HASHFUNCTION_H
