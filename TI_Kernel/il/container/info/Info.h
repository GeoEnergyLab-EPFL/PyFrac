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

#ifndef IL_INFO_H
#define IL_INFO_H

#include <il/core.h>
#include <cstring>

namespace il {

class Info {
 private:
  struct Large {
    unsigned char* data;
    std::size_t size;
    std::size_t capacity;
  };
  union {
    unsigned char small_[sizeof(Large)];
    Large large_;
  };
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(Large)) - 1;
  constexpr static unsigned char category_extract_mask_ = 0x80;
  constexpr static std::size_t capacity_extract_mask_ =
      ~(static_cast<std::size_t>(0x80) << (8 * (sizeof(std::size_t) - 1)));

 public:
  Info();
  Info(const Info& other);
  Info(Info&& other);
  Info& operator=(const Info& other);
  Info& operator=(Info&& other);

  bool isEmpty() const;
  void clear();

  void Set(const char* key, bool value);
  void Set(const char* key, int value);
#ifdef IL_64_BIT
  void Set(const char* key, il::int_t value);
#endif
  void Set(const char* key, double value);
  void Set(const char* key, const char* value);

  bool toBool(const char* key) const;
  il::int_t toInteger(const char* key) const;
  double toDouble(const char* key) const;
  const char* asCString(const char* key) const;

  il::int_t search(const char* key) const;
  bool found(il::int_t i) const;

  bool isBool(il::int_t i) const;
  bool isInteger(il::int_t i) const;
  bool isDouble(il::int_t i) const;
  bool isString(il::int_t i) const;

  bool toBool(il::int_t i) const;
  il::int_t toInteger(il::int_t i) const;
  double toDouble(il::int_t i) const;
  const char* asCString(il::int_t i) const;

 private:
  il::int_t size() const;
  il::int_t capacity() const;
  const unsigned char* data() const;
  unsigned char* data();
  void Resize(il::int_t n);

  bool isSmall() const;
  void SetSmallSize(il::int_t n);
  void SetLargeCapacity(il::int_t r);
  il::int_t largeCapacity() const;

  void Set(int value, unsigned char* data);
#ifdef IL_64_BIT
  void Set(il::int_t value, unsigned char* data);
#endif
  void Set(double value, unsigned char* data);

  int get_int(il::int_t i) const;
  il::int_t getInteger(il::int_t i) const;
};

inline Info::Info() { SetSmallSize(0); }

inline Info::Info(const Info& other) {
  const il::int_t size = other.size();
  if (size <= max_small_size_) {
    std::memcpy(small_, other.data(), size);
    SetSmallSize(size);
  } else {
    large_.data = new unsigned char[size];
    std::memcpy(large_.data, other.data(), size);
    large_.size = size;
    SetLargeCapacity(size);
  }
}

inline Info::Info(Info&& other) {
  const il::int_t size = other.size();
  if (size <= max_small_size_) {
    std::memcpy(small_, other.data(), size);
    SetSmallSize(size);
  } else {
    large_ = other.large_;
    other.SetSmallSize(0);
  }
}

inline Info& Info::operator=(const Info& other) {
  const il::int_t size = other.size();
  if (size <= max_small_size_) {
    if (!isSmall()) {
      delete[] large_.data;
    }
    std::memcpy(small_, other.data(), size);
    SetSmallSize(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, other.data(), size);
      large_.size = size;
    } else {
      if (!isSmall()) {
        delete[] large_.data;
      }
      large_.data = new unsigned char[size];
      std::memcpy(large_.data, other.data(), size);
      large_.size = size;
      SetLargeCapacity(size);
    }
  }
  return *this;
}

inline Info& Info::operator=(Info&& other) {
  if (this != &other) {
    const il::int_t size = other.size();
    if (size <= max_small_size_) {
      if (!isSmall()) {
        delete[] large_.data;
      }
      std::memcpy(small_, other.data(), size);
      SetSmallSize(size);
    } else {
      large_ = other.large_;
      other.SetSmallSize(0);
    }
  }
  return *this;
}

inline void Info::Set(const char* key, int value) {
  const int key_length = static_cast<int>(strlen(key));
  const int n = key_length + 2 + 2 * static_cast<int>(sizeof(int));
  il::int_t i = size();
  Resize(i + n);

  unsigned char* p = data();

  Set(n, p + i);
  i += sizeof(int);

  for (il::int_t j = 0; j < key_length + 1; ++j) {
    p[i + j] = key[j];
  }
  i += key_length + 1;

  p[i] = 3;
  ++i;

  Set(value, p + i);
}

#ifdef IL_64_BIT
inline void Info::Set(const char* key, il::int_t value) {
  const int key_length = static_cast<int>(strlen(key));
  const int n =
      key_length + 2 + static_cast<int>(sizeof(int) + sizeof(il::int_t));
  il::int_t i = size();
  Resize(i + n);

  unsigned char* p = data();

  Set(n, p + i);
  i += sizeof(int);

  for (il::int_t j = 0; j < key_length + 1; ++j) {
    p[i + j] = key[j];
  }
  i += key_length + 1;

  p[i] = 0;
  ++i;

  Set(value, p + i);
}
#endif

inline void Info::Set(const char* key, double value) {
  const int key_length = static_cast<int>(strlen(key));
  const int n = key_length + 2 + static_cast<int>(sizeof(int) + sizeof(double));
  il::int_t i = size();
  Resize(i + n);

  unsigned char* p = data();

  Set(n, p + i);
  i += sizeof(int);

  for (il::int_t j = 0; j < key_length + 1; ++j) {
    p[i + j] = key[j];
  }
  i += key_length + 1;

  p[i] = 1;
  ++i;

  Set(value, p + i);
}

inline void Info::Set(const char* key, const char* value) {
  const int key_length = static_cast<int>(strlen(key));
  const int value_length = static_cast<int>(strlen(value));
  const int n = key_length + value_length + 3 + static_cast<int>(sizeof(int));
  il::int_t i = size();
  Resize(i + n);

  unsigned char* p = data();

  Set(n, p + i);
  i += sizeof(int);

  for (il::int_t j = 0; j < key_length + 1; ++j) {
    p[i + j] = key[j];
  }
  i += key_length + 1;

  p[i] = 2;
  ++i;

  for (il::int_t j = 0; j < value_length + 1; ++j) {
    p[i + j] = value[j];
  }
}

inline il::int_t Info::search(const char* key) const {
  const unsigned char* p = data();
  il::int_t i = 0;
  bool found = false;
  il::int_t j;
  while (!found && i < size()) {
    const int step = get_int(i);

    j = 0;
    while (key[j] != '\0' && p[i + sizeof(int) + j] == key[j]) {
      ++j;
    }
    if (p[i + sizeof(int) + j] == '\0') {
      found = true;
    } else {
      i += step;
    }
  }

  return found ? i + static_cast<il::int_t>(sizeof(int)) + j + 1 : -1;
}

inline bool Info::found(il::int_t i) const { return i >= 0; }

inline bool Info::isInteger(il::int_t i) const { return data()[i] == 0; }

inline bool Info::isDouble(il::int_t i) const { return data()[i] == 1; }

inline bool Info::isString(il::int_t i) const { return data()[i] == 2; }

inline il::int_t Info::toInteger(il::int_t i) const {
  IL_EXPECT_FAST(isInteger(i));

  const unsigned char* p = data();
  const il::int_t n = static_cast<il::int_t>(sizeof(il::int_t));
  union {
    il::int_t local_value;
    unsigned char local_raw[n];
  };
  for (il::int_t j = 0; j < n; ++j) {
    local_raw[j] = p[i + 1 + j];
  }
  return local_value;
}

inline double Info::toDouble(il::int_t i) const {
  IL_EXPECT_FAST(isDouble(i));

  const unsigned char* p = data();
  const il::int_t n = static_cast<il::int_t>(sizeof(double));
  union {
    double local_value;
    unsigned char local_raw[n];
  };
  for (il::int_t j = 0; j < n; ++j) {
    local_raw[j] = p[i + 1 + j];
  }
  return local_value;
}

inline const char* Info::asCString(il::int_t i) const {
  IL_EXPECT_FAST(isString(i));

  return reinterpret_cast<const char*>(data()) + i + 1;
}

inline il::int_t Info::toInteger(const char* key) const {
  const il::int_t i = search(key);
  IL_ENSURE(found(i));

  return toInteger(i);
}

inline double Info::toDouble(const char* key) const {
  const il::int_t i = search(key);
  IL_ENSURE(found(i));

  return toDouble(i);
}

inline const char* Info::asCString(const char* key) const {
  const il::int_t i = search(key);
  IL_ENSURE(found(i));

  return asCString(i);
}

inline void Info::Set(int value, unsigned char* data) {
  const il::int_t n = static_cast<il::int_t>(sizeof(int));
  union {
    int local_value;
    unsigned char local_raw[n];
  };
  local_value = value;
  for (int j = 0; j < n; ++j) {
    data[j] = local_raw[j];
  }
}

#ifdef IL_64_BIT
inline void Info::Set(il::int_t value, unsigned char* data) {
  const il::int_t n = static_cast<il::int_t>(sizeof(il::int_t));
  union {
    il::int_t local_value;
    unsigned char local_raw[n];
  };
  local_value = value;
  for (il::int_t j = 0; j < n; ++j) {
    data[j] = local_raw[j];
  }
}
#endif

inline void Info::Set(double value, unsigned char* data) {
  const il::int_t n = static_cast<il::int_t>(sizeof(double));
  union {
    double local_value;
    unsigned char local_raw[n];
  };
  local_value = value;
  for (il::int_t j = 0; j < n; ++j) {
    data[j] = local_raw[j];
  }
}

inline int Info::get_int(il::int_t i) const {
  const unsigned char* p = data();
  const il::int_t n = static_cast<il::int_t>(sizeof(int));
  union {
    int local_value;
    unsigned char local_raw[n];
  };
  for (il::int_t j = 0; j < n; ++j) {
    local_raw[j] = p[i + j];
  }
  return local_value;
}

inline il::int_t Info::getInteger(il::int_t i) const {
  const unsigned char* p = data();
  const il::int_t n = static_cast<il::int_t>(sizeof(il::int_t));
  union {
    il::int_t local_value;
    unsigned char local_raw[n];
  };
  for (il::int_t j = 0; j < n; ++j) {
    local_raw[j] = p[i + j];
  }
  return local_value;
}

inline il::int_t Info::size() const {
  return isSmall() ? small_[max_small_size_] : large_.size;
}

inline il::int_t Info::capacity() const {
  return isSmall() ? max_small_size_ : largeCapacity();
}

inline bool Info::isSmall() const {
  return (small_[max_small_size_] & category_extract_mask_) == 0;
}

inline void Info::SetSmallSize(il::int_t n) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(max_small_size_));

  small_[max_small_size_] = static_cast<unsigned char>(n);
}

inline void Info::SetLargeCapacity(il::int_t r) {
  large_.capacity =
      static_cast<std::size_t>(r) |
      (static_cast<std::size_t>(0x80) << ((sizeof(std::size_t) - 1) * 8));
}

inline il::int_t Info::largeCapacity() const {
  return large_.capacity & capacity_extract_mask_;
}

inline const unsigned char* Info::data() const {
  return isSmall() ? small_ : large_.data;
}

inline unsigned char* Info::data() { return isSmall() ? small_ : large_.data; }

inline bool Info::isEmpty() const { return size() == 0; }

inline void Info::Resize(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  const il::int_t old_size = size();
  if (isSmall()) {
    if (n <= max_small_size_) {
      SetSmallSize(n);
    } else {
      unsigned char* new_data = new unsigned char[n];
      std::memcpy(new_data, small_, old_size);
      large_.data = new_data;
      large_.size = n;
      SetLargeCapacity(n);
    }
  } else {
    if (n <= capacity()) {
      large_.size = n;
    } else {
      unsigned char* new_data = new unsigned char[n];
      std::memcpy(new_data, large_.data, old_size);
      delete[] large_.data;
      large_.data = new_data;
      large_.size = n;
      SetLargeCapacity(n);
    }
  }
}

inline void Info::clear() {
  if (!isSmall()) {
    delete[] large_.data;
  }
  SetSmallSize(0);
}

}  // namespace il

#endif  // IL_INFO_H
