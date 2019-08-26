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

#ifndef IL_UTF16String_H
#define IL_UTF16String_H

// <cstring> is needed for memcpy
#include <cstring>

#include <il/core.h>
#include <il/core/memory/allocate.h>

namespace il {

// LargeString is defined this way (64-bit system, little endian as on Linux
// and Windows)
//
// - For small string optimization: 24 chars which makes 24 bytes
// - For large strings: 1 pointer and 2 std::size_t which make 24 bytes
//
// The last 2 bits (the most significant ones as we are on a little endian
// system), are used to know if:
//
// 00: small size optimization
// 01: large size (the string is on the heap)
// 10: not a String object, used as empty key for hash tables
// 11: not a String object, used as tombstone key for hash tables

#ifdef IL_WINDOWS
#define IL_UTF16CHAR wchar_t
#else
#define IL_UTF16CHAR char16_t
#endif

class UTF16String {
 private:
  struct LargeUTF16String {
    unsigned short* data;
    std::size_t size;
    std::size_t capacity;
  };
  union {
    unsigned short data_[sizeof(LargeUTF16String) / 2];
    LargeUTF16String large_;
  };

 public:
  UTF16String();
  UTF16String(const IL_UTF16CHAR* data);
  UTF16String(const il::UTF16String& s);
  UTF16String(il::UTF16String&& s);
  UTF16String& operator=(const il::UTF16String& s);
  UTF16String& operator=(il::UTF16String&& s);
  ~UTF16String();
  il::int_t size() const;
  il::int_t capacity() const;
  bool isSmall() const;
  void Reserve(il::int_t r);
  void Append(const UTF16String& s);
  void Append(const IL_UTF16CHAR* data);
  void Append(char c);
  void Append(int cp);
  void Append(il::int_t n, char c);
  void Append(il::int_t n, int cp);
  il::int_t nextRune(il::int_t i) const;
  int rune(il::int_t i) const;
  bool isEmpty() const;
  const unsigned short* data() const;
  const IL_UTF16CHAR* asWString() const;
  bool operator==(const il::UTF16String& other) const;

 private:
  void SetSmallSize(il::int_t n);
  void SetLargeCapacity(il::int_t r);
  unsigned short* Data();
  void Append(const IL_UTF16CHAR*, il::int_t n);
  bool validRune(int cp) const;
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(LargeUTF16String) / 2 - 1);
};

inline UTF16String::UTF16String() {
  data_[0] = static_cast<unsigned short>('\0');
  SetSmallSize(0);
}

inline UTF16String::UTF16String(const IL_UTF16CHAR* data) {
  IL_EXPECT_AXIOM("data is a UTF-16 null terminated string");

  il::int_t size = 0;
  while (data[size] != static_cast<IL_UTF16CHAR>('\0')) {
    ++size;
  }
  if (size <= max_small_size_) {
    std::memcpy(data_, data, 2 * (static_cast<std::size_t>(size) + 1));
    SetSmallSize(size);
  } else {
    large_.data = il::allocateArray<unsigned short>(size + 1);
    std::memcpy(large_.data, data, 2 * (static_cast<std::size_t>(size) + 1));
    large_.size = static_cast<std::size_t>(size);
    SetLargeCapacity(size);
  }
}

inline UTF16String::UTF16String(const UTF16String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.data(), 2 * (static_cast<std::size_t>(size) + 1));
    SetSmallSize(size);
  } else {
    large_.data = il::allocateArray<unsigned short>(size + 1);
    std::memcpy(large_.data, s.data(),
                2 * (static_cast<std::size_t>(size) + 1));
    large_.size = static_cast<std::size_t>(size);
    SetLargeCapacity(size);
  }
}

inline UTF16String::UTF16String(UTF16String&& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.Data(), 2 * (static_cast<std::size_t>(size) + 1));
    SetSmallSize(size);
  } else {
    large_.data = s.large_.data;
    large_.size = s.large_.size;
    large_.capacity = s.large_.capacity;
    s.data_[0] = static_cast<unsigned short>('\0');
    s.SetSmallSize(0);
  }
}

inline UTF16String& UTF16String::operator=(const UTF16String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    if (!isSmall()) {
      il::deallocate(large_.data);
    }
    std::memcpy(data_, s.data(), 2 * (static_cast<std::size_t>(size) + 1));
    SetSmallSize(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, s.data(),
                  2 * (static_cast<std::size_t>(size) + 1));
      large_.size = static_cast<std::size_t>(size);
    } else {
      if (!isSmall()) {
        il::deallocate(large_.data);
      }
      large_.data = il::allocateArray<unsigned short>(size + 1);
      std::memcpy(large_.data, s.data(),
                  2 * (static_cast<std::size_t>(size) + 1));
      large_.size = static_cast<std::size_t>(size);
      SetLargeCapacity(size);
    }
  }
  return *this;
}

inline UTF16String& UTF16String::operator=(UTF16String&& s) {
  if (this != &s) {
    const il::int_t size = s.size();
    if (size <= max_small_size_) {
      if (!isSmall()) {
        il::deallocate(large_.data);
      }
      std::memcpy(data_, s.Data(), 2 * (static_cast<std::size_t>(size) + 1));
      SetSmallSize(size);
    } else {
      large_.data = s.large_.data;
      large_.size = s.large_.size;
      large_.capacity = s.large_.capacity;
      s.data_[0] = static_cast<unsigned short>('\0');
      s.SetSmallSize(0);
    }
  }
  return *this;
}

inline UTF16String::~UTF16String() {
  if (!isSmall()) {
    il::deallocate(large_.data);
  }
}

inline il::int_t UTF16String::size() const {
  if (isSmall()) {
    return max_small_size_ - static_cast<il::int_t>(data_[max_small_size_]);
  } else {
    return static_cast<il::int_t>(large_.size);
  }
}

inline il::int_t UTF16String::capacity() const {
  if (isSmall()) {
    return max_small_size_;
  } else {
    const unsigned char category_extract_mask = 0xC0;
    const std::size_t capacity_extract_mask =
        ~(static_cast<std::size_t>(category_extract_mask)
          << ((sizeof(std::size_t) - 1) * 8));
    return static_cast<il::int_t>(large_.capacity & capacity_extract_mask);
  }
}

inline void UTF16String::Reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  const bool old_is_small = isSmall();
  const il::int_t old_capacity = capacity();
  if (r <= old_capacity) {
    return;
  }

  const il::int_t old_size = size();
  unsigned short* new_data = il::allocateArray<unsigned short>(r + 1);
  std::memcpy(new_data, data(), 2 * (static_cast<std::size_t>(old_size) + 1));
  if (!old_is_small) {
    il::deallocate(large_.data);
  }
  large_.data = new_data;
  large_.size = old_size;
  SetLargeCapacity(r);
}

inline void UTF16String::Append(const UTF16String& s) {
  Append(reinterpret_cast<const IL_UTF16CHAR*>(s.data()), s.size());
}

inline void UTF16String::Append(const IL_UTF16CHAR* data) {
  il::int_t size = 0;
  while (data[size] != static_cast<unsigned short>('\0')) {
    ++size;
  }
  Append(data, size);
}

inline void UTF16String::Append(char c) {
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(c) < 128);

  const il::int_t old_size = size();
  const il::int_t new_size = old_size + 1;
  const il::int_t new_capacity = il::max(new_size, 2 * old_size);
  Reserve(new_capacity);
  unsigned short* data = Data() + old_size;
  data[0] = static_cast<unsigned short>(c);
  data[1] = static_cast<unsigned short>('\0');
  if (isSmall()) {
    SetSmallSize(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void UTF16String::Append(il::int_t n, char c) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(static_cast<unsigned char>(c) < 128);

  const il::int_t old_size = size();
  const il::int_t new_size = old_size + 1;
  const il::int_t new_capacity = il::max(new_size, 2 * old_size);
  Reserve(new_capacity);
  unsigned short* data = Data() + old_size;
  for (il::int_t i = 0; i < n; ++i) {
    data[i] = static_cast<unsigned short>(c);
  }
  data[n] = static_cast<unsigned short>('\0');
  if (isSmall()) {
    SetSmallSize(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void UTF16String::Append(int cp) {
  IL_EXPECT_MEDIUM(validRune(cp));

  const unsigned int ucp = static_cast<unsigned int>(cp);
  const il::int_t old_size = size();
  il::int_t new_size;
  if (ucp < 0x00010000u) {
    new_size = old_size + 1;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    Reserve(new_capacity);
    unsigned short* data = Data() + size();
    data[0] = static_cast<unsigned short>(ucp);
    data[1] = static_cast<unsigned short>('\0');
  } else {
    new_size = old_size + 2;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    Reserve(new_capacity);
    unsigned short* data = Data() + size();
    const unsigned int a = ucp - 0x00010000u;
    data[0] = static_cast<unsigned short>(a >> 10) + 0xD800u;
    data[1] = static_cast<unsigned short>(a & 0x3FF) + 0xDC00u;
    data[2] = static_cast<unsigned short>('\0');
  }
  if (isSmall()) {
    SetSmallSize(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void UTF16String::Append(il::int_t n, int cp) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(validRune(cp));

  IL_UNUSED(n);
  IL_UNUSED(cp);
}

inline const unsigned short* UTF16String::data() const {
  if (isSmall()) {
    return reinterpret_cast<const unsigned short*>(data_);
  } else {
    return reinterpret_cast<const unsigned short*>(large_.data);
  }
}

inline const IL_UTF16CHAR* UTF16String::asWString() const {
  if (isSmall()) {
    return reinterpret_cast<const IL_UTF16CHAR*>(data_);
  } else {
    return reinterpret_cast<const IL_UTF16CHAR*>(large_.data);
  }
}

inline bool UTF16String::isEmpty() const { return size() == 0; }

inline il::int_t UTF16String::nextRune(il::int_t i) const {
  const unsigned short* local_data = data();
  if (local_data[i] < 0xD800u || local_data[i] >= 0xDC00u) {
    return i + 1;
  } else {
    return i + 2;
  }
}

inline int UTF16String::rune(il::int_t i) const {
  const unsigned short* local_data = data();
  if (local_data[i] < 0xD800u || local_data[i] >= 0xDC00u) {
    return static_cast<int>(local_data[i]);
  } else {
    unsigned int a = static_cast<unsigned int>(local_data[i]);
    unsigned int b = static_cast<unsigned int>(local_data[i + 1]);
    return static_cast<int>(((a & 0x03FFu) << 10) + (b & 0x03FFu) +
                            0x00010000u);
  }
}

inline bool UTF16String::operator==(const il::UTF16String& other) const {
  if (size() != other.size()) {
    return false;
  } else {
    const unsigned short* p0 = data();
    const unsigned short* p1 = other.data();
    for (il::int_t i = 0; i < size(); ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

inline bool UTF16String::isSmall() const {
  const unsigned short category_extract_mask = 0xC000;
  return (data_[max_small_size_] & category_extract_mask) == 0;
}

inline void UTF16String::SetSmallSize(il::int_t n) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(max_small_size_));

  data_[max_small_size_] = static_cast<unsigned short>(max_small_size_ - n);
}

inline void UTF16String::SetLargeCapacity(il::int_t r) {
  large_.capacity =
      static_cast<std::size_t>(r) |
      (static_cast<std::size_t>(0x80) << ((sizeof(std::size_t) - 1) * 8));
}

inline unsigned short* UTF16String::Data() {
  if (isSmall()) {
    return data_;
  } else {
    return large_.data;
  }
}

inline void UTF16String::Append(const IL_UTF16CHAR* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_AXIOM("data must point to an array of length at least n");

  const il::int_t old_size = size();
  const il::int_t new_capacity = il::max(old_size + n, 2 * old_size);
  Reserve(new_capacity);

  if (isSmall()) {
    std::memcpy(data_ + old_size, data, 2 * static_cast<std::size_t>(n));
    data_[old_size + n] = static_cast<unsigned short>('\0');
    SetSmallSize(old_size + n);
  } else {
    std::memcpy(large_.data + old_size, data, 2 * static_cast<std::size_t>(n));
    large_.data[old_size + n] = static_cast<unsigned short>('\0');
    large_.size = old_size + n;
  }
}

inline bool UTF16String::validRune(int cp) const {
  const unsigned int code_point_max = 0x0010FFFFu;
  const unsigned int lead_surrogate_min = 0x0000D800u;
  const unsigned int lead_surrogate_max = 0x0000DBFFu;
  const unsigned int ucp = static_cast<unsigned int>(cp);
  return ucp <= code_point_max &&
         (ucp < lead_surrogate_min || ucp > lead_surrogate_max);
}
}  // namespace il

#endif  // IL_UTF16String_H
