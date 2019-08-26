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

#ifndef IL_DYNAMIC_H
#define IL_DYNAMIC_H

#include <il/Type.h>

namespace il {

class Dynamic {
 private:
  il::Type type_;
  std::uint64_t data_;

 public:
  Dynamic();
  Dynamic(bool value);
  Dynamic(unsigned char value);
  Dynamic(signed char value);
  Dynamic(unsigned short value);
  Dynamic(short value);
#ifdef IL_64_BIT
  Dynamic(int value);
  Dynamic(il::int_t value);
#else
  Dynamic(int value);
#endif
  Dynamic(float value);
  Dynamic(double value);
  template <il::int_t m>
  Dynamic(const char (&value)[m]);
  Dynamic(il::StringType t, const char *value, il::int_t n);
  Dynamic(const il::String &value);
  Dynamic(il::String &&value);
  Dynamic(const il::Array<il::Dynamic> &value);
  Dynamic(il::Array<il::Dynamic> &&value);
  Dynamic(const il::Map<il::String, il::Dynamic> &value);
  Dynamic(il::Map<il::String, il::Dynamic> &&value);
  Dynamic(const il::MapArray<il::String, il::Dynamic> &value);
  Dynamic(il::MapArray<il::String, il::Dynamic> &&value);
  Dynamic(const il::Array<unsigned char> &value);
  Dynamic(il::Array<unsigned char> &&value);
  Dynamic(const il::Array<signed char> &value);
  Dynamic(il::Array<signed char> &&value);
  Dynamic(const il::Array<unsigned short> &value);
  Dynamic(il::Array<unsigned short> &&value);
  Dynamic(const il::Array<short> &value);
  Dynamic(il::Array<short> &&value);
  Dynamic(const il::Array<unsigned int> &value);
  Dynamic(il::Array<unsigned int> &&value);
  Dynamic(const il::Array<int> &value);
  Dynamic(il::Array<int> &&value);
#ifdef IL_64_BIT
  Dynamic(const il::Array<il::uint_t> &value);
  Dynamic(il::Array<il::uint_t> &&value);
  Dynamic(const il::Array<il::int_t> &value);
  Dynamic(il::Array<il::int_t> &&value);
#endif
  Dynamic(const il::Array<double> &value);
  Dynamic(il::Array<double> &&value);
  Dynamic(const il::Array2D<double> &value);
  Dynamic(il::Array2D<double> &&value);
  Dynamic(const il::Array2D<unsigned char> &value);
  Dynamic(il::Array2D<unsigned char> &&value);
  Dynamic(il::Type type);
  Dynamic(const il::Dynamic &other);
  Dynamic(il::Dynamic &&other);
  ~Dynamic();
  il::Dynamic &operator=(bool value);
  il::Dynamic &operator=(unsigned char value);
  il::Dynamic &operator=(signed char value);
  il::Dynamic &operator=(unsigned short value);
  il::Dynamic &operator=(short value);
#ifdef IL_64_BIT
  il::Dynamic &operator=(int value);
  il::Dynamic &operator=(il::int_t value);
#else
  il::Dynamic &operator=(int value);
#endif
  il::Dynamic &operator=(float value);
  il::Dynamic &operator=(double value);
  il::Dynamic &operator=(const il::String &value);
  il::Dynamic &operator=(il::String &&value);
  il::Dynamic &operator=(const il::Dynamic &other);
  il::Dynamic &operator=(il::Dynamic &&other);
  il::Type type() const;
  template <typename T>
  bool is() const;
  template <typename T>
  const T &as() const;
  template <typename T>
  T to() const;
  template <typename T>
  T &As();

 private:
  bool isStackAllocated() const;
  void CopyData(il::Type type, void *p);
  void ReleaseMemory();
};

inline Dynamic::Dynamic() { type_ = il::Type::Void; }

inline Dynamic::Dynamic(bool value) {
  type_ = il::Type::Bool;
  *reinterpret_cast<bool *>(&data_) = value;
}

inline Dynamic::Dynamic(unsigned char value) {
  type_ = il::Type::UInt8;
  *reinterpret_cast<unsigned char *>(&data_) = value;
}

inline Dynamic::Dynamic(signed char value) {
  type_ = il::Type::Int8;
  *reinterpret_cast<signed char *>(&data_) = value;
}

inline Dynamic::Dynamic(unsigned short value) {
  type_ = il::Type::UInt16;
  *reinterpret_cast<unsigned short *>(&data_) = value;
}

inline Dynamic::Dynamic(short value) {
  type_ = il::Type::Int16;
  *reinterpret_cast<short *>(&data_) = value;
}

#ifdef IL_64_BIT
inline Dynamic::Dynamic(int value) {
  type_ = il::Type::Integer;
  *reinterpret_cast<il::int_t *>(&data_) = static_cast<il::int_t>(value);
}

inline Dynamic::Dynamic(il::int_t value) {
  type_ = il::Type::Integer;
  *reinterpret_cast<il::int_t *>(&data_) = value;
}
#else
inline Dynamic::Dynamic(int value) {
  type_ = il::Type::Integer;
  *reinterpret_cast<int *>(&data_) = value;
}
#endif

inline Dynamic::Dynamic(float value) {
  type_ = il::Type::Single;
  *reinterpret_cast<float *>(&data_) = value;
}

inline Dynamic::Dynamic(double value) {
  type_ = il::Type::Double;
  *reinterpret_cast<double *>(&data_) = value;
}

template <il::int_t m>
Dynamic::Dynamic(const char (&value)[m]) {
  type_ = il::Type::UnicodeString;
  *reinterpret_cast<il::String **>(&data_) = new il::String{value};
}

inline Dynamic::Dynamic(il::StringType t, const char *value, il::int_t n) {
  type_ = il::Type::UnicodeString;
  *reinterpret_cast<il::String **>(&data_) = new il::String{t, value, n};
}

inline Dynamic::Dynamic(const il::String &value) {
  type_ = il::Type::UnicodeString;
  *reinterpret_cast<il::String **>(&data_) = new il::String{value};
}

inline Dynamic::Dynamic(il::String &&value) {
  type_ = il::Type::UnicodeString;
  *reinterpret_cast<il::String **>(&data_) = new il::String{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<il::Dynamic> &value) {
  type_ = il::Type::ArrayOfDynamic;
  *reinterpret_cast<il::Array<il::Dynamic> **>(&data_) =
      new il::Array<il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::Array<il::Dynamic> &&value) {
  type_ = il::Type::ArrayOfDynamic;
  *reinterpret_cast<il::Array<il::Dynamic> **>(&data_) =
      new il::Array<il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Map<il::String, il::Dynamic> &value) {
  type_ = il::Type::MapStringToDynamic;
  *reinterpret_cast<il::Map<il::String, il::Dynamic> **>(&data_) =
      new il::Map<il::String, il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::Map<il::String, il::Dynamic> &&value) {
  type_ = il::Type::MapStringToDynamic;
  *reinterpret_cast<il::Map<il::String, il::Dynamic> **>(&data_) =
      new il::Map<il::String, il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(const il::MapArray<il::String, il::Dynamic> &value) {
  type_ = il::Type::MapArrayStringToDynamic;
  *reinterpret_cast<il::MapArray<il::String, il::Dynamic> **>(&data_) =
      new il::MapArray<il::String, il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::MapArray<il::String, il::Dynamic> &&value) {
  type_ = il::Type::MapArrayStringToDynamic;
  *reinterpret_cast<il::MapArray<il::String, il::Dynamic> **>(&data_) =
      new il::MapArray<il::String, il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<unsigned char> &value) {
  type_ = il::Type::ArrayOfUInt8;
  *reinterpret_cast<il::Array<unsigned char> **>(&data_) =
      new il::Array<unsigned char>{value};
}

inline Dynamic::Dynamic(il::Array<unsigned char> &&value) {
  type_ = il::Type::ArrayOfUInt8;
  *reinterpret_cast<il::Array<unsigned char> **>(&data_) =
      new il::Array<unsigned char>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<signed char> &value) {
  type_ = il::Type::ArrayOfInt8;
  *reinterpret_cast<il::Array<signed char> **>(&data_) =
      new il::Array<signed char>{value};
}

inline Dynamic::Dynamic(il::Array<signed char> &&value) {
  type_ = il::Type::ArrayOfInt8;
  *reinterpret_cast<il::Array<signed char> **>(&data_) =
      new il::Array<signed char>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<unsigned short> &value) {
  type_ = il::Type::ArrayOfUInt16;
  *reinterpret_cast<il::Array<unsigned short> **>(&data_) =
      new il::Array<unsigned short>{value};
}

inline Dynamic::Dynamic(il::Array<unsigned short> &&value) {
  type_ = il::Type::ArrayOfUInt16;
  *reinterpret_cast<il::Array<unsigned short> **>(&data_) =
      new il::Array<unsigned short>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<short> &value) {
  type_ = il::Type::ArrayOfInt16;
  *reinterpret_cast<il::Array<short> **>(&data_) = new il::Array<short>{value};
}

inline Dynamic::Dynamic(const il::Array<unsigned int> &value) {
  type_ = il::Type::ArrayOfInt32;
  *reinterpret_cast<il::Array<unsigned int> **>(&data_) =
      new il::Array<unsigned int>{value};
}

inline Dynamic::Dynamic(il::Array<unsigned int> &&value) {
  type_ = il::Type::ArrayOfInt32;
  *reinterpret_cast<il::Array<unsigned int> **>(&data_) =
      new il::Array<unsigned int>{value};
}

inline Dynamic::Dynamic(const il::Array<int> &value) {
  type_ = il::Type::ArrayOfInt32;
  *reinterpret_cast<il::Array<int> **>(&data_) = new il::Array<int>{value};
}

inline Dynamic::Dynamic(il::Array<int> &&value) {
  type_ = il::Type::ArrayOfInt32;
  *reinterpret_cast<il::Array<int> **>(&data_) = new il::Array<int>{value};
}

#ifdef IL_64_BIT
inline Dynamic::Dynamic(const il::Array<il::uint_t> &value) {
  type_ = il::Type::ArrayOfUInt64;
  *reinterpret_cast<il::Array<il::uint_t> **>(&data_) =
      new il::Array<il::uint_t>{value};
}

inline Dynamic::Dynamic(il::Array<il::uint_t> &&value) {
  type_ = il::Type::ArrayOfUInt64;
  *reinterpret_cast<il::Array<il::uint_t> **>(&data_) =
      new il::Array<il::uint_t>{value};
}

inline Dynamic::Dynamic(const il::Array<il::int_t> &value) {
  type_ = il::Type::ArrayOfInt64;
  *reinterpret_cast<il::Array<il::int_t> **>(&data_) =
      new il::Array<il::int_t>{value};
}

inline Dynamic::Dynamic(il::Array<il::int_t> &&value) {
  type_ = il::Type::ArrayOfInt64;
  *reinterpret_cast<il::Array<il::int_t> **>(&data_) =
      new il::Array<il::int_t>{value};
}
#endif

inline Dynamic::Dynamic(const il::Array<double> &value) {
  type_ = il::Type::ArrayOfDouble;
  *reinterpret_cast<il::Array<double> **>(&data_) =
      new il::Array<double>{value};
}

inline Dynamic::Dynamic(il::Array<double> &&value) {
  type_ = il::Type::ArrayOfDouble;
  *reinterpret_cast<il::Array<double> **>(&data_) =
      new il::Array<double>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array2D<double> &value) {
  type_ = il::Type::Array2DOfDouble;
  *reinterpret_cast<il::Array2D<double> **>(&data_) =
      new il::Array2D<double>{value};
}

inline Dynamic::Dynamic(il::Array2D<double> &&value) {
  type_ = il::Type::Array2DOfDouble;
  *reinterpret_cast<il::Array2D<double> **>(&data_) =
      new il::Array2D<double>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array2D<unsigned char> &value) {
  type_ = il::Type::Array2DOfUInt8;
  *reinterpret_cast<il::Array2D<unsigned char> **>(&data_) =
      new il::Array2D<unsigned char>{value};
}

inline Dynamic::Dynamic(il::Array2D<unsigned char> &&value) {
  type_ = il::Type::Array2DOfUInt8;
  *reinterpret_cast<il::Array2D<unsigned char> **>(&data_) =
      new il::Array2D<unsigned char>{std::move(value)};
}

inline Dynamic::Dynamic(il::Type value) {
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(value) < 13);

  type_ = value;
  data_ = 0;
}

inline Dynamic::~Dynamic() {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
}

inline il::Dynamic &Dynamic::operator=(bool value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::Bool;
  *reinterpret_cast<bool *>(&data_) = value;
  return *this;
}

inline il::Dynamic &Dynamic::operator=(unsigned char value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::UInt8;
  *reinterpret_cast<unsigned char *>(&data_) = value;
  return *this;
}

inline il::Dynamic &Dynamic::operator=(signed char value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::Int8;
  *reinterpret_cast<signed char *>(&data_) = value;
  return *this;
}

inline il::Dynamic &Dynamic::operator=(unsigned short value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::UInt16;
  *reinterpret_cast<unsigned short *>(&data_) = value;
  return *this;
}

inline il::Dynamic &Dynamic::operator=(short value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::Int16;
  *reinterpret_cast<short *>(&data_) = value;
  return *this;
}

#ifdef IL_64_BIT
inline il::Dynamic &Dynamic::operator=(int value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::Int64;
  *reinterpret_cast<il::int_t *>(&data_) = static_cast<il::int_t>(value);
  return *this;
}

inline il::Dynamic &Dynamic::operator=(il::int_t value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::Int64;
  *reinterpret_cast<il::int_t *>(&data_) = value;
  return *this;
}
#else
inline il::Dynamic &Dynamic::operator=(int value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::Int32;
  *reinterpret_cast<int *>(&data_) = value;
  return *this;
}
#endif

inline il::Dynamic &Dynamic::operator=(float value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::Float32;
  *reinterpret_cast<float *>(&data_) = value;
  return *this;
}

inline il::Dynamic &Dynamic::operator=(double value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::Float64;
  *reinterpret_cast<double *>(&data_) = value;
  return *this;
}

inline il::Dynamic &Dynamic::operator=(const il::String &value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::UnicodeString;
  *reinterpret_cast<il::String **>(&data_) = new il::String{value};
  return *this;
}

inline il::Dynamic &Dynamic::operator=(il::String &&value) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = il::Type::UnicodeString;
  *reinterpret_cast<il::String **>(&data_) = new il::String{std::move(value)};
  return *this;
}

inline Dynamic::Dynamic(const il::Dynamic &other) {
  type_ = other.type_;
  if (isStackAllocated()) {
    data_ = other.data_;
  } else {
    CopyData(type_, reinterpret_cast<void *>(other.data_));
  }
}

inline Dynamic::Dynamic(il::Dynamic &&other) {
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = il::Type::Void;
}

inline il::Dynamic &Dynamic::operator=(const il::Dynamic &other) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = other.type_;
  if (isStackAllocated()) {
    data_ = other.data_;
  } else {
    CopyData(type_, reinterpret_cast<void *>(other.data_));
  }
  return *this;
}

inline il::Dynamic &Dynamic::operator=(il::Dynamic &&other) {
  if (!isStackAllocated()) {
    ReleaseMemory();
  }
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = il::Type::Void;
  return *this;
}

inline il::Type Dynamic::type() const { return type_; }

template <typename T>
bool Dynamic::is() const {
  return type_ == il::typeId<T>();
}

template <typename T>
const T &Dynamic::as() const {
  IL_EXPECT_MEDIUM(type_ == il::typeId<T>());
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(type_) >= 13);

  if (isStackAllocated()) {
    return *reinterpret_cast<const T *>(&data_);
  } else {
    return *reinterpret_cast<const T *>(data_);
  }
}

template <typename T>
T Dynamic::to() const {
  IL_EXPECT_MEDIUM(type_ == il::typeId<T>());
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(type_) <= 11);

  return *reinterpret_cast<const T *>(&data_);
}

template <typename T>
T &Dynamic::As() {
  IL_EXPECT_MEDIUM(type_ == il::typeId<T>());

  if (isStackAllocated()) {
    return *reinterpret_cast<T *>(&data_);
  } else {
    return *reinterpret_cast<T *>(data_);
  }
}

inline bool Dynamic::isStackAllocated() const {
  return static_cast<unsigned char>(type_) < 13;
}

inline void Dynamic::CopyData(il::Type type, void *p) {
  switch (type) {
    case il::Type::UnicodeString:
      *reinterpret_cast<il::String **>(&data_) =
          new il::String{*reinterpret_cast<il::String *>(p)};
      break;
    case il::Type::ArrayOfDynamic:
      *reinterpret_cast<il::Array<il::Dynamic> **>(&data_) =
          new il::Array<il::Dynamic>{
              *reinterpret_cast<il::Array<il::Dynamic> *>(p)};
      break;
    case il::Type::MapStringToDynamic:
      *reinterpret_cast<il::Map<il::String, il::Dynamic> **>(&data_) =
          new il::Map<il::String, il::Dynamic>{
              *reinterpret_cast<il::Map<il::String, il::Dynamic> *>(p)};
      break;
    case il::Type::MapArrayStringToDynamic:
      *reinterpret_cast<il::MapArray<il::String, il::Dynamic> **>(&data_) =
          new il::MapArray<il::String, il::Dynamic>{
              *reinterpret_cast<il::MapArray<il::String, il::Dynamic> *>(p)};
      break;
    case il::Type::ArrayOfBool:
      *reinterpret_cast<il::Array<bool> **>(&data_) =
          new il::Array<bool>{*reinterpret_cast<il::Array<bool> *>(p)};
      break;
    case il::Type::ArrayOfUInt8:
      *reinterpret_cast<il::Array<unsigned char> **>(&data_) =
          new il::Array<unsigned char>{
              *reinterpret_cast<il::Array<unsigned char> *>(p)};
      break;
    case il::Type::ArrayOfInt8:
      *reinterpret_cast<il::Array<signed char> **>(&data_) =
          new il::Array<signed char>{
              *reinterpret_cast<il::Array<signed char> *>(p)};
      break;
    case il::Type::ArrayOfUInt16:
      *reinterpret_cast<il::Array<unsigned short> **>(&data_) =
          new il::Array<unsigned short>{
              *reinterpret_cast<il::Array<unsigned short> *>(p)};
      break;
    case il::Type::ArrayOfInt16:
      *reinterpret_cast<il::Array<short> **>(&data_) =
          new il::Array<short>{*reinterpret_cast<il::Array<short> *>(p)};
      break;
    case il::Type::ArrayOfUInt32:
      *reinterpret_cast<il::Array<unsigned int> **>(&data_) =
          new il::Array<unsigned int>{
              *reinterpret_cast<il::Array<unsigned int> *>(p)};
      break;
    case il::Type::ArrayOfInt32:
      *reinterpret_cast<il::Array<int> **>(&data_) =
          new il::Array<int>{*reinterpret_cast<il::Array<int> *>(p)};
      break;
#ifdef IL_64_BIT
    case il::Type::ArrayOfUInteger:
      *reinterpret_cast<il::Array<std::size_t> **>(&data_) =
          new il::Array<std::size_t>{
              *reinterpret_cast<il::Array<std::size_t> *>(p)};
      break;
    case il::Type::ArrayOfInteger:
      *reinterpret_cast<il::Array<il::int_t> **>(&data_) =
          new il::Array<il::int_t>{
              *reinterpret_cast<il::Array<il::int_t> *>(p)};
      break;
#endif
    case il::Type::ArrayOfFloat:
      *reinterpret_cast<il::Array<float> **>(&data_) =
          new il::Array<float>{*reinterpret_cast<il::Array<float> *>(p)};
      break;
    case il::Type::ArrayOfDouble:
      *reinterpret_cast<il::Array<double> **>(&data_) =
          new il::Array<double>{*reinterpret_cast<il::Array<double> *>(p)};
      break;
    case il::Type::ArrayOfString:
      *reinterpret_cast<il::Array<il::String> **>(&data_) =
          new il::Array<il::String>{
              *reinterpret_cast<il::Array<il::String> *>(p)};
      break;
    default:
      IL_UNREACHABLE;
  }
}

inline void Dynamic::ReleaseMemory() {
  switch (type_) {
    case il::Type::UnicodeString:
      delete reinterpret_cast<il::String *>(data_);
      break;
    case il::Type::ArrayOfDynamic:
      delete reinterpret_cast<il::Array<il::Dynamic> *>(data_);
      break;
    case il::Type::MapStringToDynamic:
      delete reinterpret_cast<il::Map<il::String, il::Dynamic> *>(data_);
      break;
    case il::Type::MapArrayStringToDynamic:
      delete reinterpret_cast<il::MapArray<il::String, il::Dynamic> *>(data_);
      break;
    case il::Type::ArrayOfBool:
      delete reinterpret_cast<il::Array<bool> *>(data_);
      break;
    case il::Type::ArrayOfUInt8:
      delete reinterpret_cast<il::Array<unsigned char> *>(data_);
      break;
    case il::Type::ArrayOfInt8:
      delete reinterpret_cast<il::Array<signed char> *>(data_);
      break;
    case il::Type::ArrayOfUInt16:
      delete reinterpret_cast<il::Array<unsigned short> *>(data_);
      break;
    case il::Type::ArrayOfInt16:
      delete reinterpret_cast<il::Array<short> *>(data_);
      break;
    case il::Type::ArrayOfUInt32:
      delete reinterpret_cast<il::Array<unsigned int> *>(data_);
      break;
    case il::Type::ArrayOfInt32:
      delete reinterpret_cast<il::Array<int> *>(data_);
      break;
#ifdef IL_64_BIT
    case il::Type::ArrayOfUInteger:
      delete reinterpret_cast<il::Array<std::size_t> *>(data_);
      break;
    case il::Type::ArrayOfInteger:
      delete reinterpret_cast<il::Array<il::int_t> *>(data_);
      break;
#endif
    case il::Type::ArrayOfFloat:
      delete reinterpret_cast<il::Array<float> *>(data_);
      break;
    case il::Type::ArrayOfDouble:
      delete reinterpret_cast<il::Array<double> *>(data_);
      break;
    case il::Type::ArrayOfString:
      delete reinterpret_cast<il::Array<il::String> *>(data_);
      break;
    case il::Type::Array2DOfDouble:
      delete reinterpret_cast<il::Array2D<double> *>(data_);
      break;
    default:
      IL_UNREACHABLE;
  }
}

}  // namespace il

#endif  // IL_DYNAMIC_H
