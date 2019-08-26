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

#ifndef IL_STRING_H
#define IL_STRING_H

// <cstring> is needed for memcpy
#include <cstring>
// <utility> is needed for std::move
#include <utility>

#include <iostream>

#include <il/core.h>
#include <il/core/memory/allocate.h>

namespace il {

// LargeString is defined this way (64-bit system, little endian as on Linux
// and Windows)
//
// - For small string optimization: 24 chars which makes 24 bytes
//   The first 22 chars can be used to store the string. Another char is used
//   to store the termination character '\0'. The last char is used to store
//   the length of the string and its type: ascii, utf8, wtf8 or byte
// - For large strings: 1 pointer and 2 std::size_t which make 24 bytes
//
// The last bit (the most significant ones as we are on a little endian
// system), is used to know if the string is a small string or no (0 for small
// string and 1 for large string).
//
// The 2 bits before are used to know the kind of the string. We use (little
// endian):
// 00: ascii
// 10: utf8
// 01: wtf8
// 11: byte

enum class StringType : unsigned char {
  Ascii = 0x00_uchar,
  Utf8 = 0x20_uchar,
  Wtf8 = 0x40_uchar,
  Byte = 0x60_uchar
};

StringType joinType(StringType s0, StringType s1);
StringType joinType(StringType s0, StringType s1, StringType s2);
StringType joinType(StringType s0, StringType s1, StringType s2, StringType s3);
StringType joinType(StringType s0, StringType s1, StringType s2, StringType s3,
                    StringType s4);
StringType joinType(StringType s0, StringType s1, StringType s2, StringType s3,
                    StringType s4, StringType s5);
StringType joinType(StringType s0, StringType s1, StringType s2, StringType s3,
                    StringType s4, StringType s5, StringType s6);
bool isRune(int rune);
bool isAscii(const char* s, il::int_t n);
bool isUtf8(const char* s, il::int_t n);

class StringView;

// Invariants
// - If the capacity is > 22, the memory is allocated on the heap
//   Otherwise, we use small string optimization
class String {
 private:
  struct LargeString {
    char* data;
    il::int_t size;
    std::size_t capacity;
  };
  union {
    char small_[sizeof(LargeString)];
    LargeString large_;
  };
#ifdef IL_DEBUGGER_HELPERS
  static const std::size_t debug_small_large_ =
      (static_cast<std::size_t>(1) << (8 * sizeof(std::size_t) - 3));
#endif

 public:
  String();
  explicit String(char c);
  template <il::int_t m>
  String(const char (&s)[m]);
  template <il::int_t m>
  explicit String(il::StringType, const char (&s)[m]);
  explicit String(il::StringType type, const char* data, il::int_t n);
  explicit String(const il::StringView& s);
  String(const il::String& s);
  String(il::String&& s);
  template <il::int_t m>
  String& operator=(const char (&data)[m]);
  String& operator=(const il::String& s);
  String& operator=(il::String&& s);
  ~String();
  il::int_t size() const;
  bool isSmall() const;
  bool isEmpty() const;
  il::int_t capacity() const;
  void Reserve(il::int_t r);
  il::StringType type() const;
  bool isUtf8() const;
  bool isAscii() const;
  bool isRuneBoundary(il::int_t i) const;
  void Append(char c);
  void Append(int rune);
  void Append(il::int_t n, char c);
  void Append(il::int_t n, int rune);
  template <il::int_t m>
  void Append(const char (&data)[m]);
  template <il::int_t m>
  void Append(il::StringType type, const char (&data)[m]);
  void Append(il::StringType type, const char* data, il::int_t n);
  void Append(const StringView& s0);
  void Append(const StringView& s0, const StringView& s1);
  void Append(const StringView& s0, const StringView& s1, const StringView& s2);
  void Insert(il::int_t i, char c);
  //  void Insert(il::int_t i, int rune);
  void Insert(il::int_t i, il::int_t n, char c);
  //  void Insert(il::int_t i, il::int_t n, int rune);
  template <il::int_t m>
  void Insert(il::int_t i, const char (&s)[m]);
  void Insert(il::int_t i, il::StringType t, const char* s, il::int_t n);
  void Insert(il::int_t i, const StringView& s0);
  il::int_t count(char c) const;
  //  il::int_t count(int rune);
  //  il::int_t count(const StringView& s);
  il::int_t search(const StringView& s) const;
  bool found(il::int_t i) const;
  //  void replace(const StringView& s_sold, const StringView& s_new);
  il::String substring(il::int_t i0, il::int_t i1) const;
  il::StringView subview(il::int_t i0, il::int_t i1) const;
  il::StringView view() const;
  void Clear();
  bool startsWith(const StringView& s) const;
  bool endsWith(const StringView& s) const;
  bool endsWith(char c) const;

  const char* asCString() const;
  bool isEqual(const char* s) const;
  bool isEqual(const char* s, il::int_t n) const;
  const char* data() const;
  char* Data();

  explicit String(il::unsafe_t, il::int_t n);
  void Grow(il::unsafe_t, il::int_t n_to_copy, il::int_t n);
  void SetInvariant(il::unsafe_t, il::StringType type, il::int_t n);

 private:
  il::int_t smallSize() const;
  il::int_t largeCapacity() const;
  // Does not set the '\0'. You are responsible for that.
  void setSmall(il::StringType type, il::int_t n);
  // Does not set the '\0'. You are responsible for that.
  void setLarge(il::StringType type, il::int_t n, il::int_t r);
  il::int_t static constexpr sizeCString(const char* data);
  il::int_t static nextCapacity(il::int_t n);
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(LargeString) - 2);
};

class StringView {
 protected:
  const char* data_;
  const char* size_;
  il::StringType type_;

 public:
  StringView();
  template <il::int_t m>
  StringView(const char (&s)[m]);
  StringView(const il::String& s);
  explicit StringView(il::StringType t, const char* s, il::int_t n);

  il::int_t size() const;
  bool isEmpty() const;
  il::StringType type() const;
  bool operator==(const char* string) const;

  bool startsWith(const char* data) const;
  bool startsWith(char c) const;
  bool startsWithNewLine() const;
  bool startsWithDigit() const;

  bool endsWith(const char* data) const;
  bool endsWith(char c) const;
  bool endsWithNewLine() const;
  bool endsWithDigit() const;

  bool has(il::int_t i, const char* data) const;
  bool has(il::int_t i, char c) const;
  bool hasSpace(il::int_t i) const;
  bool hasDigit(il::int_t i) const;
  bool hasHexaDecimal(il::int_t i) const;
  bool hasNewLine(il::int_t i) const;

  il::int_t nextChar(il::int_t i, char c) const;
  il::int_t nextDigit(il::int_t i) const;
  il::int_t nextNewLine(il::int_t i) const;

  void removePrefix(il::int_t i1);
  void removeSuffix(il::int_t i0);
  void trimPrefix();
  void trimSuffix();
  StringView subview(il::int_t i0, il::int_t i1) const;
  int rune(il::int_t i) const;
  il::int_t nextRune(il::int_t i) const;

  const char& operator[](il::int_t i) const;
  const char& back(il::int_t i) const;
  const char* asCString() const;
  const char* data() const;
};

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

inline bool isRune(int rune) {
  const unsigned int code_point_max = 0x0010FFFFu;
  const unsigned int lead_surrogate_min = 0x0000D800u;
  const unsigned int lead_surrogate_max = 0x0000DBFFu;
  const unsigned int urune = static_cast<unsigned int>(rune);
  return urune <= code_point_max &&
         (urune < lead_surrogate_min || urune > lead_surrogate_max);
}

inline bool isAscii(const char* s, il::int_t n) {
  bool ans = true;
  il::int_t i = 0;
  while (ans && i < n) {
    if (static_cast<unsigned char>(s[i]) >= 128) {
      ans = false;
    }
    ++i;
  }
  return ans;
}

inline il::StringType joinType(il::StringType t0, il::StringType t1) {
  return static_cast<il::StringType>(
      il::max(static_cast<unsigned char>(t0), static_cast<unsigned char>(t1)));
}

inline il::StringType joinType(il::StringType t0, il::StringType t1,
                               il::StringType t2) {
  return static_cast<il::StringType>(il::max(static_cast<unsigned char>(t0),
                                             static_cast<unsigned char>(t1),
                                             static_cast<unsigned char>(t2)));
}

inline il::StringType joinType(il::StringType t0, il::StringType t1,
                               il::StringType t2, il::StringType t3) {
  return static_cast<il::StringType>(
      il::max(static_cast<unsigned char>(t0), static_cast<unsigned char>(t1),
              static_cast<unsigned char>(t2), static_cast<unsigned char>(t3)));
}

inline il::StringType joinType(il::StringType t0, il::StringType t1,
                               il::StringType t2, il::StringType t3,
                               il::StringType t4) {
  return static_cast<il::StringType>(
      il::max(static_cast<unsigned char>(t0), static_cast<unsigned char>(t1),
              static_cast<unsigned char>(t2), static_cast<unsigned char>(t3),
              static_cast<unsigned char>(t4)));
}

inline il::StringType joinType(il::StringType t0, il::StringType t1,
                               il::StringType t2, il::StringType t3,
                               il::StringType t4, il::StringType t5) {
  return static_cast<il::StringType>(
      il::max(static_cast<unsigned char>(t0), static_cast<unsigned char>(t1),
              static_cast<unsigned char>(t2), static_cast<unsigned char>(t3),
              static_cast<unsigned char>(t4), static_cast<unsigned char>(t5)));
}

////////////////////////////////////////////////////////////////////////////////
// String
////////////////////////////////////////////////////////////////////////////////

inline String::String() {
  small_[0] = '\0';
  setSmall(il::StringType::Ascii, 0);
}

inline String::String(char c) {
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(c) < 128);

  small_[0] = c;
  small_[1] = '\0';
  setSmall(il::StringType::Ascii, 1);
}

template <il::int_t m>
inline String::String(const char (&s)[m]) {
  const il::int_t n = m - 1;
  if (n <= max_small_size_) {
    std::memcpy(small_, s, static_cast<std::size_t>(m));
    setSmall(il::StringType::Utf8, n);
  } else {
    const il::int_t r = nextCapacity(n);
    large_.data = il::allocateArray<char>(r + 1);
    std::memcpy(large_.data, s, static_cast<std::size_t>(m));
    setLarge(il::StringType::Utf8, n, r);
  }
}

template <il::int_t m>
inline String::String(il::StringType type, const char (&s)[m]) {
  IL_EXPECT_MEDIUM(type == il::StringType::Ascii ||
                   type == il::StringType::Utf8);

  const il::int_t n = m - 1;
  const bool is_ascii = (type == il::StringType::Ascii);
  if (n <= max_small_size_) {
    std::memcpy(small_, s, static_cast<std::size_t>(m));
    setSmall(is_ascii ? il::StringType::Ascii : il::StringType::Utf8, n);
  } else {
    const il::int_t r = nextCapacity(n);
    large_.data = il::allocateArray<char>(r + 1);
    std::memcpy(large_.data, s, static_cast<std::size_t>(m));
    setLarge(is_ascii ? il::StringType::Ascii : il::StringType::Utf8, n, r);
  }
}

inline String::String(il::StringType type, const char* s, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= max_small_size_) {
    std::memcpy(small_, s, static_cast<std::size_t>(n));
    small_[n] = '\0';
    setSmall(type, n);
  } else {
    const il::int_t r = nextCapacity(n);
    large_.data = il::allocateArray<char>(r + 1);
    std::memcpy(large_.data, s, static_cast<std::size_t>(n));
    large_.data[n] = '\0';
    setLarge(type, n, r);
  }
}

inline String::String(const il::StringView& s)
    : String{s.type(), s.data(), s.size()} {}

inline String::String(const String& s) : String{s.type(), s.data(), s.size()} {}

inline String::String(String&& s) {
  const il::int_t size = s.size();
  const il::StringType type = s.type();
  if (size <= max_small_size_) {
    std::memcpy(small_, s.asCString(), size + 1);
    setSmall(type, size);
  } else {
    large_.data = s.large_.data;
    large_.size = s.large_.size;
    large_.capacity = s.large_.capacity;
  }
  s.small_[0] = '\0';
  s.setSmall(il::StringType::Ascii, 0);
}

template <il::int_t m>
inline String& String::operator=(const char (&data)[m]) {
  const il::int_t size = m - 1;
  const il::StringType type = il::StringType::Utf8;
  if (size <= max_small_size_) {
    if (!isSmall()) {
      il::deallocate(large_.data);
    }
    std::memcpy(small_, data, size + 1);
    setSmall(type, size);
  } else {
    const il::int_t old_r = capacity();
    if (size <= old_r) {
      std::memcpy(large_.data, data, size + 1);
      setLarge(type, size, old_r);
      large_.size = size;
    } else {
      if (!isSmall()) {
        il::deallocate(large_.data);
      }
      const il::int_t new_r = nextCapacity(size);
      large_.data = il::allocateArray<char>(new_r + 1);
      std::memcpy(large_.data, data, size + 1);
      setLarge(type, size, new_r);
    }
  }
  return *this;
}

inline String& String::operator=(const String& s) {
  const il::int_t size = s.size();
  const il::StringType type = s.type();
  if (size <= max_small_size_) {
    if (!isSmall()) {
      il::deallocate(large_.data);
    }
    std::memcpy(small_, s.data(), size + 1);
    setSmall(type, size);
  } else {
    const il::int_t old_r = capacity();
    if (size <= old_r) {
      std::memcpy(large_.data, s.data(), size + 1);
      setLarge(type, size, old_r);
    } else {
      if (!isSmall()) {
        il::deallocate(large_.data);
      }
      const il::int_t new_r = nextCapacity(size);
      large_.data = il::allocateArray<char>(new_r + 1);
      std::memcpy(large_.data, s.data(), size + 1);
      setLarge(type, size, new_r);
    }
  }
  return *this;
}

inline String& String::operator=(String&& s) {
  if (this != &s) {
    const il::int_t size = s.size();
    const il::StringType type = s.type();
    if (size <= max_small_size_) {
      if (!isSmall()) {
        il::deallocate(large_.data);
      }
      std::memcpy(small_, s.data(), size + 1);
      setSmall(type, size);
    } else {
      large_.data = s.large_.data;
      large_.size = s.large_.size;
      large_.capacity = s.large_.capacity;
      s.small_[0] = '\0';
      s.setSmall(il::StringType::Ascii, 0);
    }
  }
  return *this;
}

inline String::~String() {
  if (!isSmall()) {
    il::deallocate(large_.data);
  }
}

inline il::int_t String::size() const {
  if (isSmall()) {
    return static_cast<il::int_t>(
        static_cast<unsigned char>(small_[max_small_size_ + 1]) & 0x1F_uchar);
  } else {
    return large_.size;
  }
}

inline bool String::isEmpty() const { return size() == 0; }

inline il::int_t String::capacity() const {
  if (isSmall()) {
    return max_small_size_;
  } else {
    return static_cast<il::int_t>(
        (static_cast<std::size_t>(large_.capacity) &
         ~(static_cast<std::size_t>(0xD0) << ((sizeof(std::size_t) - 1) * 8)))
        << 3);
  }
}

inline void String::Reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  const il::int_t old_capacity = capacity();
  if (r <= old_capacity) {
    return;
  }

  const bool old_small = isSmall();
  const il::StringType type = this->type();
  const il::int_t size = this->size();
  const il::int_t new_r = nextCapacity(r);
  char* new_data = il::allocateArray<char>(new_r + 1);
  std::memcpy(new_data, asCString(), static_cast<std::size_t>(size) + 1);
  if (!old_small) {
    il::deallocate(large_.data);
  }
  large_.data = new_data;
  setLarge(type, size, new_r);
}

inline il::StringType String::type() const {
  return static_cast<il::StringType>(
      static_cast<unsigned char>(small_[max_small_size_ + 1]) & 0x60_uchar);
}

inline bool String::isUtf8() const {
  const il::StringType t = type();
  return t == il::StringType::Ascii || t == il::StringType::Utf8;
}

inline bool String::isAscii() const { return type() == il::StringType::Ascii; }

inline bool String::isRuneBoundary(il::int_t i) const {
  IL_EXPECT_MEDIUM(type() == il::StringType::Ascii ||
                   type() == il::StringType::Utf8);
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <=
                   static_cast<std::size_t>(size()));

  const char* data = this->data();
  const unsigned char c = static_cast<unsigned char>(data[i]);
  return c < 0x80_uchar || ((c & 0xC0_uchar) != 0x80_uchar);
}

inline void String::Append(char c) {
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(c) < 128);

  const il::int_t old_n = size();
  const il::StringType t = type();
  Grow(il::unsafe, old_n, old_n + 1);
  char* p = Data();
  p[old_n] = c;
  p[old_n + 1] = '\0';
  SetInvariant(il::unsafe, t, old_n + 1);
}

inline void String::Append(int rune) {
  IL_EXPECT_MEDIUM(isRune(rune));

  const unsigned int urune = static_cast<unsigned int>(rune);
  const il::int_t old_n = size();
  const il::StringType t = type();
  if (urune < 0x00000080u) {
    Grow(il::unsafe, old_n, old_n + 1);
    char* p = Data();
    p[old_n] = static_cast<char>(rune);
    p[old_n + 1] = '\0';
    SetInvariant(il::unsafe, t, old_n + 1);
  } else if (urune < 0x00000800u) {
    Grow(il::unsafe, old_n, old_n + 2);
    char* p = Data();
    p[old_n] = static_cast<char>((urune >> 6) | 0x000000C0u);
    p[old_n + 1] = static_cast<char>((urune & 0x0000003Fu) | 0x00000080u);
    p[old_n + 2] = '\0';
    SetInvariant(il::unsafe, il::joinType(t, il::StringType::Utf8), old_n + 2);
  } else if (urune < 0x00010000u) {
    Grow(il::unsafe, old_n, old_n + 3);
    char* p = Data();
    p[old_n] = static_cast<char>((urune >> 12) | 0x000000E0u);
    p[old_n + 1] =
        static_cast<char>(((urune >> 6) & 0x0000003Fu) | 0x00000080u);
    p[old_n + 2] = static_cast<char>((urune & 0x0000003Fu) | 0x00000080u);
    p[old_n + 3] = '\0';
    SetInvariant(il::unsafe, il::joinType(t, il::StringType::Utf8), old_n + 3);
  } else {
    Grow(il::unsafe, old_n, old_n + 4);
    char* p = Data();
    p[old_n] = static_cast<unsigned char>((urune >> 18) | 0x000000F0u);
    p[old_n + 1] =
        static_cast<unsigned char>(((urune >> 12) & 0x0000003Fu) | 0x00000080u);
    p[old_n + 2] =
        static_cast<unsigned char>(((urune >> 6) & 0x0000003Fu) | 0x00000080u);
    p[old_n + 3] =
        static_cast<unsigned char>((urune & 0x0000003Fu) | 0x00000080u);
    p[old_n + 4] = '\0';
    SetInvariant(il::unsafe, il::joinType(t, il::StringType::Utf8), old_n + 4);
  }
}

inline void String::Append(il::int_t n, char c) {
  IL_EXPECT_FAST(static_cast<unsigned char>(c) < 128);

  const il::int_t old_n = size();
  const il::StringType t = type();
  Grow(il::unsafe, old_n, old_n + n);
  char* p = Data();
  for (il::int_t k = 0; k < n; ++k) {
    p[old_n + k] = c;
  }
  p[old_n + n] = '\0';
  SetInvariant(il::unsafe, t, old_n + n);
}

inline void String::Append(il::int_t n, int rune) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(isRune(rune));

  const unsigned int urune = static_cast<unsigned int>(rune);
  const il::int_t old_n = size();
  const il::StringType t = type();
  if (urune < 0x00000080u) {
    Grow(il::unsafe, old_n, old_n + n);
    char* p = Data();
    for (il::int_t k = 0; k < n; ++k) {
      p[old_n + k] = static_cast<char>(rune);
    }
    p[old_n + n] = '\0';
    SetInvariant(il::unsafe, t, old_n + 1);
  } else if (urune < 0x00000800u) {
    Grow(il::unsafe, old_n, old_n + 2 * n);
    char* p = Data();
    for (il::int_t k = 0; k < n; ++k) {
      p[old_n + 2 * k] = static_cast<char>((urune >> 6) | 0x000000C0u);
      p[old_n + 2 * k + 1] =
          static_cast<char>((urune & 0x0000003Fu) | 0x00000080u);
    }
    p[old_n + 2 * n] = '\0';
    SetInvariant(il::unsafe, il::joinType(t, il::StringType::Utf8),
                 old_n + 2 * n);
  } else if (urune < 0x00010000u) {
    Grow(il::unsafe, old_n, old_n + 3);
    char* p = Data();
    for (il::int_t k = 0; k < n; ++k) {
      p[old_n + 3 * k] = static_cast<char>((urune >> 12) | 0x000000E0u);
      p[old_n + 3 * k + 1] =
          static_cast<char>(((urune >> 6) & 0x0000003Fu) | 0x00000080u);
      p[old_n + 3 * k + 2] =
          static_cast<char>((urune & 0x0000003Fu) | 0x00000080u);
    }
    p[old_n + 3 * n] = '\0';
    SetInvariant(il::unsafe, il::joinType(t, il::StringType::Utf8),
                 old_n + 3 * n);
  } else {
    Grow(il::unsafe, old_n, old_n + 4 * n);
    char* p = Data();
    for (il::int_t k = 0; k < n; ++k) {
      p[old_n + 4 * k] =
          static_cast<unsigned char>((urune >> 18) | 0x000000F0u);
      p[old_n + 4 * k + 1] = static_cast<unsigned char>(
          ((urune >> 12) & 0x0000003Fu) | 0x00000080u);
      p[old_n + 4 * k + 2] = static_cast<unsigned char>(
          ((urune >> 6) & 0x0000003Fu) | 0x00000080u);
      p[old_n + 4 * k + 3] =
          static_cast<unsigned char>((urune & 0x0000003Fu) | 0x00000080u);
    }
    p[old_n + 4 * n] = '\0';
    SetInvariant(il::unsafe, il::joinType(t, il::StringType::Utf8),
                 old_n + 4 * n);
  }
}

template <il::int_t m>
inline void String::Append(const char (&s0)[m]) {
  const il::int_t old_n = size();
  const il::int_t n0 = m - 1;
  const il::StringType t = joinType(type(), il::StringType::Utf8);
  Grow(il::unsafe, old_n, old_n + n0);
  char* p = Data();
  std::memcpy(p + old_n, s0, n0);
  p[old_n + n0] = '\0';
  SetInvariant(il::unsafe, t, old_n + n0);
}

template <il::int_t m>
inline void String::Append(il::StringType t0, const char (&s0)[m]) {
  const il::int_t old_n = size();
  const il::int_t n0 = m - 1;
  const il::StringType t = joinType(type(), t0);
  Grow(il::unsafe, old_n, old_n + n0);
  char* p = Data();
  std::memcpy(p + old_n, s0, n0);
  p[old_n + n0] = '\0';
  SetInvariant(il::unsafe, t, old_n + n0);
}

inline void String::Append(il::StringType t0, const char* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_AXIOM("data must point to an array of length at least n");

  const il::int_t old_n = size();
  const il::StringType t = joinType(type(), t0);
  Grow(il::unsafe, old_n, old_n + n);
  char* p = this->Data();
  std::memcpy(p + old_n, data, n);
  p[old_n + n] = '\0';
  SetInvariant(il::unsafe, t, old_n + n);
}

inline void String::Append(const StringView& s0) {
  const il::int_t old_n = size();
  const il::int_t n0 = s0.size();
  const il::StringType t = joinType(type(), s0.type());
  Grow(il::unsafe, old_n, old_n + n0);
  char* p = Data();
  std::memcpy(p + old_n, s0.data(), n0);
  p[old_n + n0] = '\0';
  SetInvariant(il::unsafe, t, old_n + n0);
}

inline void String::Append(const StringView& s0, const StringView& s1) {
  const il::int_t old_n = size();
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::StringType t = joinType(type(), s0.type(), s1.type());
  Grow(il::unsafe, old_n, old_n + n0 + n1);
  char* p = Data();
  std::memcpy(p + old_n, s0.data(), n0);
  std::memcpy(p + old_n + n0, s1.data(), n1);
  p[old_n + n0 + n1] = '\0';
  SetInvariant(il::unsafe, t, old_n + n0 + n1);
}

inline void String::Append(const StringView& s0, const StringView& s1,
                           const StringView& s2) {
  const il::int_t old_n = size();
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::int_t n2 = s2.size();
  const il::StringType t = joinType(type(), s0.type(), s1.type(), s2.type());
  Grow(il::unsafe, old_n, old_n + n0 + n1 + n2);
  char* p = Data();
  std::memcpy(p + old_n, s0.data(), n0);
  std::memcpy(p + old_n + n0, s1.data(), n1);
  std::memcpy(p + old_n + n0 + n1, s2.data(), n2);
  p[old_n + n0 + n1 + n2] = '\0';
  SetInvariant(il::unsafe, t, old_n + n0 + n1 + n2);
}

inline void String::Insert(il::int_t i, char c) {
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(c) < 128);
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <=
                   static_cast<std::size_t>(size()));
  if (type() == il::StringType::Utf8) {
    IL_EXPECT_MEDIUM(isRuneBoundary(i));
  }

  const il::int_t old_size = size();
  const il::int_t old_capacity = capacity();
  const bool old_is_small = isSmall();
  char* p = Data();
  if (old_size + 1 <= old_capacity) {
    for (il::int_t k = old_size; k >= i + 1; k--) {
      p[k] = p[k - 1];
    }
    p[i] = c;
    if (old_is_small) {
      setSmall(type(), old_size + 1);
    } else {
      setLarge(type(), old_size + 1, old_capacity);
    }
  } else {
    const il::int_t r = nextCapacity(old_size + 1);
    char* new_p = il::allocateArray<char>(r + 1);
    std::memcpy(new_p, p, i);
    new_p[i] = c;
    std::memcpy(new_p + i + 1, p + i, old_size - i);
    large_.data = new_p;
    setLarge(type(), old_size + 1, r);
  }
}

// inline void String::Insert(il::int_t i, int rune) {}

inline void String::Insert(il::int_t i, il::int_t n, char c) {
  IL_EXPECT_FAST(static_cast<unsigned char>(c) < 128);
  IL_EXPECT_FAST(static_cast<std::size_t>(i) <= static_cast<std::size_t>(n));
  if (type() == il::StringType::Utf8) {
    IL_EXPECT_FAST(isRuneBoundary(i));
  }

  const il::int_t old_size = size();
  const il::int_t old_capacity = capacity();
  const bool old_is_small = isSmall();
  char* p = Data();
  if (old_size + n <= old_capacity) {
    for (il::int_t k = old_size + (n - 1); k >= i + n; k--) {
      p[k] = p[k - n];
    }
    for (il::int_t k = i; k < i + n; ++k) {
      p[k] = c;
    }
    if (old_is_small) {
      setSmall(type(), old_size + n);
    } else {
      setLarge(type(), old_size + n, old_capacity);
    }
  } else {
    const il::int_t r = nextCapacity(old_size + n);
    char* new_p = il::allocateArray<char>(r + 1);
    std::memcpy(new_p, p, i);
    for (il::int_t k = i; k < i + n; ++k) {
      new_p[k] = c;
    }
    std::memcpy(new_p + i + n, p + i, old_size - i);
    large_.data = new_p;
    setLarge(type(), old_size + n, r);
  }
}

// inline void String::Insert(il::int_t i, il::int_t n, int rune) {}

template <il::int_t m>
inline void String::Insert(il::int_t i, const char (&s)[m]) {
  IL_EXPECT_FAST(static_cast<std::size_t>(i) <=
                 static_cast<std::size_t>(size()));
  if (type() == il::StringType::Utf8) {
    IL_EXPECT_FAST(isRuneBoundary(i));
  }

  const il::int_t n = m - 1;
  const il::int_t old_size = size();
  const il::int_t old_capacity = capacity();
  const bool old_is_small = isSmall();
  char* p = Data();
  if (old_size + n <= old_capacity) {
    for (il::int_t k = old_size + (n - 1); k >= i + n; k--) {
      p[k] = p[k - n];
    }
    std::memcpy(p + i, s, n);
    if (old_is_small) {
      setSmall(type(), old_size + n);
    } else {
      setLarge(type(), old_size + n, old_capacity);
    }
  } else {
    const il::int_t r = nextCapacity(old_size + n);
    char* new_p = il::allocateArray<char>(r + 1);
    std::memcpy(new_p, p, i);
    std::memcpy(new_p + i, s, n);
    std::memcpy(new_p + i + n, p + i, old_size - i);
    large_.data = new_p;
    setLarge(type(), old_size + n, r);
  }
}

inline void String::Insert(il::int_t i, il::StringType t, const char* s,
                           il::int_t n) {
  IL_EXPECT_FAST(static_cast<std::size_t>(i) <= static_cast<std::size_t>(n));
  if (type() == il::StringType::Utf8) {
    IL_EXPECT_FAST(isRuneBoundary(i));
  }

  const il::int_t old_size = size();
  const il::int_t old_capacity = capacity();
  const bool old_is_small = isSmall();
  const il::StringType t_new = il::joinType(type(), t);
  char* p = Data();
  if (old_size + n <= old_capacity) {
    for (il::int_t k = old_size + (n - 1); k >= i + n; k--) {
      p[k] = p[k - n];
    }
    std::memcpy(p + i, s, n);
    if (old_is_small) {
      setSmall(t_new, old_size + n);
    } else {
      setLarge(t_new, old_size + n, old_capacity);
    }
  } else {
    const il::int_t r = nextCapacity(old_size + n);
    char* new_p = il::allocateArray<char>(r + 1);
    std::memcpy(new_p, p, i);
    std::memcpy(new_p + i, s, n);
    std::memcpy(new_p + i + n, p + i, old_size - i);
    large_.data = new_p;
    setLarge(t_new, old_size + n, r);
  }
}

inline void String::Insert(il::int_t i, const StringView& s0) {
  Insert(i, s0.type(), s0.data(), s0.size());
}

inline il::int_t String::count(char c) const {
  il::int_t ans = 0;
  const char* p = data();
  const il::int_t n = size();
  for (il::int_t i = 0; i < n; ++i) {
    if (p[i] == c) {
      ++ans;
    }
  }
  return ans;
}

inline il::int_t String::search(const StringView& s) const {
  const char* a = s.data();
  const char* b = data();
  const il::int_t n = s.size();
  const il::int_t m = size();
  il::int_t k = 0;
  bool found = false;
  while (!found && k + n <= m) {
    il::int_t i = 0;
    found = true;
    while (found && i < n) {
      if (a[i] != b[k + i]) {
        found = false;
      }
      ++i;
    }
    if (found) {
      return k;
    }
    ++k;
  }
  return -1;
}

inline bool String::found(il::int_t i) const { return i >= 0; }

inline il::String String::substring(il::int_t i0, il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(i0 <= i1);

  if (type() == il::StringType::Utf8) {
    IL_EXPECT_MEDIUM(isRuneBoundary(i0));
    IL_EXPECT_MEDIUM(isRuneBoundary(i1));
  }
  return il::String{type(), small_ + i0, i1 - i0};
}

inline il::StringView String::subview(il::int_t i0, il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(i0 <= i1);

  if (type() == il::StringType::Utf8) {
    IL_EXPECT_MEDIUM(isRuneBoundary(i0));
    IL_EXPECT_MEDIUM(isRuneBoundary(i1));
  }
  return il::StringView{type(), small_ + i0, i1 - i0};
}

inline il::StringView String::view() const {
  return il::StringView{type(), data(), size()};
}

inline void String::Clear() {
  if (isSmall()) {
    small_[0] = '\0';
    setSmall(il::StringType::Ascii, 0);
  } else {
    large_.data[0] = '\0';
    setLarge(il::StringType::Ascii, 0, capacity());
  }
}

inline bool String::startsWith(const StringView& s) const {
  const il::int_t n = s.size();
  const il::int_t m = size();
  if (n > m) {
    return false;
  }
  const char* a = s.data();
  const char* b = data();
  bool ans = true;
  il::int_t i = 0;
  while (ans && i < n) {
    if (a[i] != b[i]) {
      ans = false;
    }
    ++i;
  }
  return ans;
}

inline bool String::endsWith(const StringView& s) const {
  const il::int_t n = s.size();
  const il::int_t m = size();
  if (n > m) {
    return false;
  }
  const char* a = s.data();
  const char* b = data();
  bool ans = true;
  il::int_t i = 0;
  while (ans && i < n) {
    if (a[i] != b[m - n + i]) {
      ans = false;
    }
    ++i;
  }
  return ans;
}

inline bool String::endsWith(char c) const {
  const il::int_t n = size();
  if (n == 0) {
    return false;
  }
  const char* s = data();
  if (*s == 'c') {
    return true;
  } else {
    return false;
  }
}

inline const char* String::asCString() const {
  return isSmall() ? small_ : large_.data;
}

inline bool String::isEqual(const char* s) const {
  bool ans = true;
  const char* p = data();
  il::int_t i = 0;
  while (ans && s[i] != '\0') {
    if (p[i] != s[i]) {
      ans = false;
    }
    ++i;
  }
  return ans;
}

inline bool String::isEqual(const char* s, il::int_t n) const {
  bool ans = true;
  const char* p = data();
  il::int_t i = 0;
  while (ans && i < n) {
    if (p[i] != s[i]) {
      ans = false;
    }
    ++i;
  }
  return ans;
}

inline const char* String::data() const {
  return isSmall() ? small_ : large_.data;
}

inline char* String::Data() { return isSmall() ? small_ : large_.data; }

inline String::String(il::unsafe_t, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= max_small_size_) {
    setSmall(il::StringType::Byte, n);
  } else {
    const il::int_t r = nextCapacity(n);
    large_.data = il::allocateArray<char>(r + 1);
    setLarge(il::StringType::Byte, n, r);
  }
}

inline void String::Grow(il::unsafe_t, il::int_t n_to_copy, il::int_t n) {
  IL_EXPECT_FAST(n_to_copy >= 0);
  IL_EXPECT_FAST(n_to_copy <= n);

  const bool old_is_small = isSmall();
  char* old_data = Data();
  const il::int_t old_capacity = capacity();
  if (n <= max_small_size_) {
    if (!old_is_small) {
      std::memcpy(small_, old_data, n_to_copy);
    }
    setSmall(il::StringType::Byte, n);
  } else if (n <= old_capacity) {
    setLarge(il::StringType::Byte, n, old_capacity);
  } else {
    const il::int_t r = nextCapacity(n);
    char* p = il::allocateArray<char>(r + 1);
    std::memcpy(p, old_data, n_to_copy);
    large_.data = p;
    if (!old_is_small) {
      il::deallocate(old_data);
    }
    setLarge(il::StringType::Byte, n, r);
  }
}

inline void String::SetInvariant(il::unsafe_t, il::StringType type,
                                 il::int_t n) {
  if (isSmall()) {
    setSmall(type, n);
  } else {
    setLarge(type, n, capacity());
  }
}

inline il::int_t String::smallSize() const {
  return small_[max_small_size_ + 1] & static_cast<unsigned char>(0x1F);
}

inline il::int_t String::largeCapacity() const {
  constexpr unsigned char category_extract_mask = 0xD0_uchar;
  constexpr std::size_t capacity_extract_mask =
      ~(static_cast<std::size_t>(category_extract_mask)
        << ((sizeof(std::size_t) - 1) * 8));
  return static_cast<il::int_t>(
      (static_cast<std::size_t>(large_.capacity) & capacity_extract_mask) << 3);
}

inline void String::setSmall(il::StringType type, il::int_t n) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(max_small_size_));

  unsigned char value = static_cast<unsigned char>(n);
  value |= static_cast<unsigned char>(type);
  small_[max_small_size_ + 1] = value;
}

inline void String::setLarge(il::StringType type, il::int_t n, il::int_t r) {
  IL_EXPECT_MEDIUM(n >= 0);
  IL_EXPECT_MEDIUM(n <= r);

  large_.size = n;
  large_.capacity =
      (static_cast<std::size_t>(r) >> 3) |
      (static_cast<std::size_t>(static_cast<unsigned char>(type))
       << ((sizeof(std::size_t) - 1) * 8)) |
      (static_cast<std::size_t>(0x80_uchar) << ((sizeof(std::size_t) - 1) * 8));
}

// Return the next integer modulo 4
inline il::int_t String::nextCapacity(il::int_t r) {
  return static_cast<il::int_t>(
      (static_cast<std::size_t>(r) + static_cast<std::size_t>(7)) &
      ~static_cast<std::size_t>(7));
}

////////////////////////////////////////////////////////////////////////////////

inline il::int_t constexpr String::sizeCString(const char* data) {
  return data[0] == '\0' ? 0 : sizeCString(data + 1) + 1;
}

constexpr il::int_t mySizeCString(const char* s, int i) {
  return *s == '\0' ? i : mySizeCString(s + 1, i + 1);
}

constexpr bool myIsAscii(const char* s) {
  return (*s == '\0') ||
         ((static_cast<unsigned char>(*s) < static_cast<unsigned char>(128))
              ? myIsAscii(s + 1)
              : false);
}

inline il::int_t cstringSizeType(const char* s, il::io_t, bool& is_ascii) {
  is_ascii = true;
  il::int_t n = 0;
  while (s[n] != '\0') {
    if ((static_cast<unsigned char>(s[n]) & 0x80_uchar) != 0x00_uchar) {
      is_ascii = false;
    }
    ++n;
  }
  return n;
}

inline il::int_t size(const char* data) {
  il::int_t i = 0;
  while (data[i] != '\0') {
    ++i;
  }
  return i;
}

inline constexpr bool isAscii(const char* data) {
  return data[0] == '\0' ? true
                         : ((static_cast<unsigned char>(data[0]) &
                             0x80_uchar) == 0x00_uchar) &&
                               isAscii(data + 1);
}

// template <il::int_t m>
// inline bool isAscii(const char (&data)[m]) {
//  bool ans = true;
//  for (il::int_t i = 0; i < m - 1; ++i) {
//    if ((static_cast<unsigned char>(data[i]) & 0x80_uchar) != 0x00_uchar) {
//      ans = false;
//    }
//  }
//  return ans;
//}

inline constexpr bool auxIsUtf8(const char* s, int nbBytes, int pos,
                                bool surrogate) {
  return (pos == 0)
             // If it starts with the null character, the string is valid
             ? ((*s == 0x00)
                    ? true
                    // Otherwise, let's check if is starts with an ASCII
                    // character
                    : ((*s & 0x80) == 0
                           // In this case, check the rest of the string
                           ? auxIsUtf8(s + 1, 0, 0, false)
                           // Otherwise, it might start with a 2-byte sequence
                           : ((*s & 0xD0) == 0xB0
                                  // In this case, check the rest of the string
                                  ? auxIsUtf8(s + 1, 2, 1, false)
                                  // Otherwise, it might start with a 3-byte
                                  // sequence
                                  : ((*s & 0xF0) == 0xD0
                                         // In this case, check the rest of the
                                         // string
                                         ? auxIsUtf8(
                                               s + 1, 3, 1,
                                               (static_cast<unsigned char>(
                                                    *s) == 0xED))
                                         // Otherwise, it might start with a
                                         // 4-byte sequence
                                         : ((*s & 0xF8) == 0xF0
                                                ? auxIsUtf8(s + 1, 4, 1, false)
                                                : false)))))
             // In the case where we are scanning the second byte of a multibyte
             // sequence
             : ((pos == 1)
                    ? ((*s & 0xC0) == 0x80
                           ? (nbBytes == 2 ? ((*s & 0xA0) != 0xA0)
                                           : auxIsUtf8(s + 1, nbBytes, pos + 1,
                                                       surrogate))
                           : false)
                    // In the case where we are scanning the third byte of a
                    // multibyte sequence
                    : ((pos == 2)
                           ? ((*s & 0xC0) == 0x80
                                  ? (nbBytes == 3
                                         ? true
                                         : auxIsUtf8(s + 1, nbBytes, pos + 1,
                                                     surrogate))
                                  : false)
                           // In the case where we are scanning the
                           // fourth byte of a multibyte sequence
                           : ((pos == 3) ? ((*s & 0xC0) == 0x80) : false)));
}

inline constexpr bool isUtf8(const char* s) {
  return auxIsUtf8(s, 0, 0, false);
}

////////////////////////////////////////////////////////////////////////////////
// StringView
////////////////////////////////////////////////////////////////////////////////

inline StringView::StringView() {
  data_ = nullptr;
  size_ = nullptr;
  type_ = il::StringType::Ascii;
}

template <il::int_t m>
inline StringView::StringView(const char (&s)[m]) {
  data_ = const_cast<char*>(s);
  size_ = const_cast<char*>(s) + (m - 1);
  type_ = il::StringType::Utf8;
}

inline StringView::StringView(const il::String& s) {
  data_ = s.data();
  size_ = data_ + s.size();
  type_ = s.type();
}

inline StringView::StringView(il::StringType t, const char* data, il::int_t n) {
  IL_EXPECT_MEDIUM(n >= 0);

  data_ = const_cast<char*>(data);
  size_ = const_cast<char*>(data) + n;
  type_ = t;
}

inline il::int_t StringView::size() const { return size_ - data_; }

inline bool String::isSmall() const {
  return (static_cast<unsigned char>(small_[max_small_size_ + 1]) &
          0x80_uchar) == 0x00_uchar;
}

inline bool StringView::isEmpty() const { return size_ == data_; }

inline il::StringType StringView::type() const { return type_; }

inline const char& StringView::operator[](il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i];
}

inline bool StringView::hasSpace(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return (data_[i] == ' ') || (data_[i] == '\f') || (data_[i] == '\n') ||
         (data_[i] == '\r') || (data_[i] == '\t') || (data_[i] == '\v');
}

inline bool StringView::hasDigit(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i] >= '0' && data_[i] <= '9';
}

inline bool StringView::hasHexaDecimal(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return (data_[i] >= '0' && data_[i] <= '9') ||
         (data_[i] >= 'a' && data_[i] <= 'f') ||
         (data_[i] >= 'A' && data_[i] <= 'F');
}

inline bool StringView::startsWithDigit() const {
  IL_EXPECT_MEDIUM(size() > 0);

  return data_[0] >= '0' && data_[0] <= '9';
}

inline bool StringView::hasNewLine(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i] == '\n' ||
         (i + 1 < size() && data_[i] == '\r' && data_[i + 1] == '\n');
}

inline il::int_t StringView::nextChar(il::int_t i, char c) const {
  while (i < size() && data_[i] != c) {
    ++i;
  }
  return i < size() ? i : -1;
}

inline il::int_t StringView::nextDigit(il::int_t i) const {
  while (i < size() && !(data_[i] >= '0' && data_[i] <= '9')) {
    ++i;
  }
  return i < size() ? i : -1;
}

inline int StringView::rune(il::int_t i) const {
  unsigned int ans = 0;
  const unsigned char* data =
      reinterpret_cast<const unsigned char*>(this->data());
  if ((data[i] & 0x80u) == 0) {
    ans = static_cast<unsigned int>(data[i]);
  } else if ((data[i] & 0xE0u) == 0xC0u) {
    ans = (static_cast<unsigned int>(data[i] & 0x1Fu) << 6) +
          static_cast<unsigned int>(data[i + 1] & 0x3Fu);
  } else if ((data[i] & 0xF0u) == 0xE0u) {
    ans = (static_cast<unsigned int>(data[i] & 0x0Fu) << 12) +
          (static_cast<unsigned int>(data[i + 1] & 0x3Fu) << 6) +
          static_cast<unsigned int>(data[i + 2] & 0x3Fu);
  } else {
    ans = (static_cast<unsigned int>(data[i] & 0x07u) << 18) +
          (static_cast<unsigned int>(data[i + 1] & 0x3Fu) << 12) +
          (static_cast<unsigned int>(data[i + 2] & 0x3Fu) << 6) +
          static_cast<unsigned int>(data[i + 3] & 0x3Fu);
  }
  return static_cast<int>(ans);
}

inline il::int_t StringView::nextRune(il::int_t i) const {
  do {
    ++i;
  } while (i < size() && ((data_[i] & 0xC0u) == 0x80u));
  return i;
}

inline const char& StringView::back(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return size_[-(1 + i)];
}

inline bool StringView::startsWithNewLine() const {
  return (size() > 0 && data_[0] == '\n') ||
         (size() > 1 && data_[0] == '\r' && data_[1] == '\n');
}

inline void StringView::removePrefix(il::int_t i1) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <=
                   static_cast<std::size_t>(size()));

  data_ += i1;
}

inline void StringView::removeSuffix(il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <=
                   static_cast<std::size_t>(size()));

  size_ = data_ + i;
}

inline void StringView::trimPrefix() {
  il::int_t i = 0;
  while (i < size() && (data_[i] == ' ' || data_[i] == '\t')) {
    ++i;
  }
  data_ += i;
}

inline StringView StringView::subview(il::int_t i0, il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(i0 <= i1);

  return StringView{type_, data_ + i0, i1 - i0};
}

// inline StringView StringView::suffix(il::int_t n) const {
//  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
//                   static_cast<std::size_t>(size()));
//
//  return StringView{data_ + size() - n, size()};
//}
//
// inline StringView StringView::prefix(il::int_t n) const {
//  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
//                   static_cast<std::size_t>(size()));
//
//  return StringView{data_, n};
//}

inline bool StringView::operator==(const char* string) const {
  bool match = true;
  il::int_t k = 0;
  while (match && k < size() && string[k] != '\0') {
    if (data_[k] != string[k]) {
      match = false;
    }
    ++k;
  }
  return match;
}

inline const char* StringView::asCString() const {
  return reinterpret_cast<const char*>(data_);
}

inline const char* StringView::data() const { return data_; }

inline il::String join(const il::StringView& s0, const il::StringView& s1) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::StringType t = il::joinType(s0.type(), s1.type());
  il::String ans{il::unsafe, n0 + n1};
  char* p = ans.Data();
  std::memcpy(p, s0.data(), n0);
  std::memcpy(p + n0, s1.data(), n1);
  p[n0 + n1] = '\0';
  ans.SetInvariant(il::unsafe, t, n0 + n1);
  return ans;
}

inline il::String join(const il::StringView& s0, const il::StringView& s1,
                       const il::StringView& s2) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::int_t n2 = s2.size();
  const il::StringType t = il::joinType(s0.type(), s1.type(), s2.type());
  il::String ans{il::unsafe, n0 + n1 + n2};
  char* p = ans.Data();
  std::memcpy(p, s0.data(), n0);
  std::memcpy(p + n0, s1.data(), n1);
  std::memcpy(p + n0 + n1, s2.data(), n2);
  p[n0 + n1 + n2] = '\0';
  ans.SetInvariant(il::unsafe, t, n0 + n1 + n2);
  return ans;
}

inline il::String join(const il::StringView& s0, const il::StringView& s1,
                       const il::StringView& s2, const il::StringView& s3) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::int_t n2 = s2.size();
  const il::int_t n3 = s3.size();
  const il::StringType t =
      il::joinType(s0.type(), s1.type(), s2.type(), s3.type());
  il::String ans{il::unsafe, n0 + n1 + n2 + n3};
  char* p = ans.Data();
  std::memcpy(p, s0.data(), n0);
  std::memcpy(p + n0, s1.data(), n1);
  std::memcpy(p + n0 + n1, s2.data(), n2);
  std::memcpy(p + n0 + n1 + n2, s3.data(), n3);
  p[n0 + n1 + n2 + n3] = '\0';
  ans.SetInvariant(il::unsafe, t, n0 + n1 + n2 + n3);
  return ans;
}

inline il::String join(const il::StringView& s0, const il::StringView& s1,
                       const il::StringView& s2, const il::StringView& s3,
                       const il::StringView& s4) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::int_t n2 = s2.size();
  const il::int_t n3 = s3.size();
  const il::int_t n4 = s4.size();
  const il::StringType t =
      il::joinType(s0.type(), s1.type(), s2.type(), s3.type(), s4.type());
  il::String ans{il::unsafe, n0 + n1 + n2 + n3 + n4};
  char* p = ans.Data();
  std::memcpy(p, s0.data(), n0);
  std::memcpy(p + n0, s1.data(), n1);
  std::memcpy(p + n0 + n1, s2.data(), n2);
  std::memcpy(p + n0 + n1 + n2, s3.data(), n3);
  std::memcpy(p + n0 + n1 + n2 + n3, s4.data(), n4);
  p[n0 + n1 + n2 + n3 + n4] = '\0';
  ans.SetInvariant(il::unsafe, t, n0 + n1 + n2 + n3 + n4);
  return ans;
}

inline il::String join(const il::StringView& s0, const il::StringView& s1,
                       const il::StringView& s2, const il::StringView& s3,
                       const il::StringView& s4, const il::StringView& s5) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::int_t n2 = s2.size();
  const il::int_t n3 = s3.size();
  const il::int_t n4 = s4.size();
  const il::int_t n5 = s5.size();
  const il::StringType t = il::joinType(s0.type(), s1.type(), s2.type(),
                                        s3.type(), s4.type(), s5.type());
  il::String ans{il::unsafe, n0 + n1 + n2 + n3 + n4 + n5};
  char* p = ans.Data();
  std::memcpy(p, s0.data(), n0);
  std::memcpy(p + n0, s1.data(), n1);
  std::memcpy(p + n0 + n1, s2.data(), n2);
  std::memcpy(p + n0 + n1 + n2, s3.data(), n3);
  std::memcpy(p + n0 + n1 + n2 + n3, s4.data(), n4);
  std::memcpy(p + n0 + n1 + n2 + n3 + n4, s5.data(), n5);
  p[n0 + n1 + n2 + n3 + n4 + n5] = '\0';
  ans.SetInvariant(il::unsafe, t, n0 + n1 + n2 + n3 + n4 + n5);
  return ans;
}

inline bool operator<(const il::String& s0, const il::String& s1) {
  const int compare = std::strcmp(s0.data(), s1.data());
  return compare < 0;
}

inline bool operator<(const char* s0, const il::String& s1) {
  const int compare = std::strcmp(s0, s1.data());
  return compare < 0;
}

inline bool operator==(const il::String& s0, const il::String& s1) {
  const il::int_t s0_size = s0.size();
  const il::int_t s1_size = s1.size();
  if (s0_size != s1_size) {
    return false;
  } else {
    const char* p0 = s0.data();
    const char* p1 = s1.data();
    for (il::int_t i = 0; i < s0_size; ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

inline bool operator==(const char* s0, const il::String& s1) {
  const il::int_t s0_size = il::size(s0);
  const il::int_t s1_size = s1.size();
  if (s0_size != s1_size) {
    return false;
  } else {
    const char* p0 = s0;
    const char* p1 = s1.data();
    for (il::int_t i = 0; i < s0_size; ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

inline bool operator!=(const char* s0, const il::String& s1) {
  return !(s0 == s1);
}

inline bool operator==(const il::String& s0, const char* s1) {
  return (s1 == s0);
}

inline bool operator!=(const il::String& s0, const char* s1) {
  return !(s1 == s0);
}

inline il::String toString(const std::string& s) {
  return il::String{il::StringType::Byte, s.c_str(),
                    static_cast<il::int_t>(s.size())};
}

inline il::String toString(const char* s) {
  return il::String{il::StringType::Byte, s, il::size(s)};
}

inline std::ostream& operator<<(std::ostream& os, const String& s) {
  return os << s.data();
}

inline std::ostream& operator<<(std::ostream& os, const StringView& s) {
  return os << s.data();
}

}  // namespace il

#endif  // IL_STRING_H
