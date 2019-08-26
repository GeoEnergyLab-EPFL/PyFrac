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

#ifndef IL_TYPE_H
#define IL_TYPE_H

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>
#include <il/Map.h>
#include <il/MapArray.h>
#include <il/String.h>

namespace il {

enum class Type : unsigned char {
  Void = 12,
  Bool = 0,
  UInt8 = 1,
  Int8 = 2,
  UChar = 1,  // C type (unsigned char)
  SChar = 2,  // C type (signed char, not char)
  UInt16 = 3,
  Int16 = 4,
  UShort = 3,  // C type (unsigned short)
  Short = 4,   // C type (shor)
  UInt32 = 5,
  Int32 = 6,
  UInt = 5,  // C type (unsigned int)
  Int = 6,   // C type (int)
  UInt64 = 7,
  Int64 = 8,
#ifdef IL_64_BIT
  UInteger = 7,  // C++ type (il::uint_t aka std::size_t)
  Integer = 8,   // C++ type (il::int_t aka std:ptrdiff_t)
#else
  UInteger = 5,  // C++ type (il::uint_t aka std::size_t)
  Integer = 6,   // C++ type (il::int_t aka std:ptrdiff_t)
#endif
  // TFp16 = 9,

  Float16,
  Float32 = 10,
  Float64 = 11,
  Single = 10,   // C type (float)
  Double = 11,  // C type (double)
//  FloatintPoint = 11,

  Complex64 = 17,
  Complex128 = 18,

  UnicodeString = 13,
  ArrayOfDynamic = 14,
  MapStringToDynamic = 15,
  MapArrayStringToDynamic = 16,

  ArrayOfBool = 20,
  ArrayOfUInt8 = 21,
  ArrayOfInt8 = 22,
  ArrayOfUInt16 = 23,
  ArrayOfInt16 = 24,
  ArrayOfUInt32 = 25,
  ArrayOfInt32 = 26,
  ArrayOfUInt = 25,
  ArrayOfInt = 26,
  ArrayOfUInt64 = 27,
  ArrayOfInt64 = 28,
#ifdef IL_64_BIT
  ArrayOfUInteger = 27,
  ArrayOfInteger = 28,
#else
  ArrayOfUInteger = 25,
  ArrayOfInteger = 26,
#endif
  ArrayOfFp32 = 30,
  ArrayOfFp64 = 31,
  ArrayOfFloat = 30,
  ArrayOfDouble = 31,
  ArrayOfFloatingPoint = 31,
  ArrayOfString = 32,
  ArrayOfStruct = 33,

  ArrayViewOfBool = 20,
  ArrayViewOfUInt8 = 21,
  ArrayViewOfInt8 = 22,
  ArrayViewOfUInt16 = 23,
  ArrayViewOfInt16 = 24,
  ArrayViewOfUInt32 = 25,
  ArrayViewOfInt32 = 26,
  ArrayViewOfUInt = 25,
  ArrayViewOfInt = 26,
  ArrayViewOfUInt64 = 27,
  ArrayViewOfInt64 = 28,
#ifdef IL_64_BIT
  ArrayViewOfUInteger = 27,
  ArrayViewOfInteger = 28,
#else
  ArrayViewOfUInteger = 25,
  ArrayViewOfInteger = 26,
#endif
  ArrayViewOfFp32 = 30,
  ArrayViewOfFp64 = 31,
  ArrayViewOfFloat = 30,
  ArrayViewOfDouble = 31,
  ArrayViewOfFloatingPoint = 31,
  ArrayViewOfString = 32,

  ArrayEditOfBool = 20,
  ArrayEditOfUInt8 = 21,
  ArrayEditOfInt8 = 22,
  ArrayEditOfUInt16 = 23,
  ArrayEditOfInt16 = 24,
  ArrayEditOfUInt32 = 25,
  ArrayEditOfInt32 = 26,
  ArrayEditOfUInt = 25,
  ArrayEditOfInt = 26,
  ArrayEditOfUInt64 = 27,
  ArrayEditOfInt64 = 28,
#ifdef IL_64_BIT
  ArrayEditOfUInteger = 27,
  ArrayEditOfInteger = 28,
#else
  ArrayEditOfUInteger = 25,
  ArrayEditOfInteger = 26,
#endif
  ArrayEditOfFp32 = 30,
  ArrayEditOfFp64 = 31,
  ArrayEditOfFloat = 30,
  ArrayEditOfDouble = 31,
  ArrayEditOfFloatingPoint = 31,
  ArrayEditOfString = 32,

  Array2DOfBool = 40,
  Array2DOfUInt8 = 41,
  Array2DOfInt8 = 42,
  Array2DOfUInt16 = 43,
  Array2DOfInt16 = 44,
  Array2DOfUInt32 = 45,
  Array2DOfInt32 = 46,
  Array2DOfUInt = 45,
  Array2DOfInt = 46,
  Array2DOfUInt64 = 47,
  Array2DOfInt64 = 48,
#ifdef IL_64_BIT
  Array2DOfUInteger = 47,
  Array2DOfInteger = 48,
#else
  Array2DOfUInteger = 45,
  Array2DOfInteger = 46,
#endif
  // TArray2DOfFp16 = 49,
  Array2DOfFp32 = 50,
  Array2DOfFp64 = 51,
  Array2DOfFloat = 50,
  Array2DOfDouble = 51,
  Array2DOfFloatingPoint = 51,
  Array2DOfString = 52,

  Array2COfBool = 60,
  Array2COfUInt8 = 61,
  Array2COfInt8 = 62,
  Array2COfUInt16 = 63,
  Array2COfInt16 = 64,
  Array2COfUInt32 = 65,
  Array2COfInt32 = 66,
  Array2COfUInt = 65,
  Array2COfInt = 66,
  Array2COfUInt64 = 67,
  Array2COfInt64 = 68,
#ifdef IL_64_BIT
  Array2COfUInteger = 67,
  Array2COfInteger = 68,
#else
  Array2COfUInteger = 65,
  Array2COfInteger = 66,
#endif
  // TArray2COfFp16 = 69,
  Array2COfFp32 = 70,
  Array2COfFp64 = 71,
  Array2COfFloat = 70,
  Array2COfDouble = 71,
  Array2COfFloatingPoint = 71,
  Array2COfString = 72,

};

inline il::int_t sizeOf(il::Type type) {
  switch(type) {
    case il::Type::Complex64:
      return 8;
    case il::Type::Complex128:
      return 16;
    default:
      IL_UNREACHABLE;
  }
}

class Dynamic;

template <typename T>
il::Type typeId() {
  return il::Type::Void;
}

template <>
inline il::Type typeId<bool>() {
  return il::Type::Bool;
}

template <>
inline il::Type typeId<int>() {
  return il::Type::Int;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::int_t>() {
  return il::Type::Integer;
}
#endif

template <>
inline il::Type typeId<float>() {
  return il::Type::Single;
}

template <>
inline il::Type typeId<double>() {
  return il::Type::Double;
}

template <>
inline il::Type typeId<il::String>() {
  return il::Type::UnicodeString;
}

template <>
inline il::Type typeId<il::Array<il::Dynamic>>() {
  return il::Type::ArrayOfDynamic;
}

template <>
inline il::Type typeId<il::Map<il::String, il::Dynamic>>() {
  return il::Type::MapStringToDynamic;
}

template <>
inline il::Type typeId<il::MapArray<il::String, il::Dynamic>>() {
  return il::Type::MapArrayStringToDynamic;
}

template <>
inline il::Type typeId<il::Array<unsigned char>>() {
  return il::Type::ArrayOfUInt8;
}

template <>
inline il::Type typeId<il::Array<signed char>>() {
  return il::Type::ArrayOfInt8;
}

template <>
inline il::Type typeId<il::Array<unsigned short>>() {
  return il::Type::ArrayOfUInt16;
}

template <>
inline il::Type typeId<il::Array<short>>() {
  return il::Type::ArrayOfInt16;
}

template <>
inline il::Type typeId<il::Array<unsigned int>>() {
  return il::Type::ArrayOfUInt32;
}

template <>
inline il::Type typeId<il::Array<int>>() {
  return il::Type::ArrayOfInt32;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::Array<std::size_t>>() {
  return il::Type::ArrayOfUInt64;
}

template <>
inline il::Type typeId<il::Array<il::int_t>>() {
  return il::Type::ArrayOfInt64;
}
#endif

template <>
inline il::Type typeId<il::Array<float>>() {
  return il::Type::ArrayOfFloat;
}

template <>
inline il::Type typeId<il::Array<double>>() {
  return il::Type::ArrayOfDouble;
}

template <>
inline il::Type typeId<il::Array2D<unsigned char>>() {
  return il::Type::Array2DOfUInt8;
}

template <>
inline il::Type typeId<il::Array2D<signed char>>() {
  return il::Type::Array2DOfInt8;
}

template <>
inline il::Type typeId<il::Array2D<unsigned short>>() {
  return il::Type::Array2DOfUInt16;
}

template <>
inline il::Type typeId<il::Array2D<short>>() {
  return il::Type::Array2DOfInt16;
}

template <>
inline il::Type typeId<il::Array2D<unsigned int>>() {
  return il::Type::Array2DOfUInt32;
}

template <>
inline il::Type typeId<il::Array2D<int>>() {
  return il::Type::Array2DOfInt32;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::Array2D<std::size_t>>() {
  return il::Type::Array2DOfUInt64;
}

template <>
inline il::Type typeId<il::Array2D<il::int_t>>() {
  return il::Type::Array2DOfInt64;
}
#endif

template <>
inline il::Type typeId<il::Array2D<float>>() {
  return il::Type::Array2DOfFloat;
}

template <>
inline il::Type typeId<il::Array2D<double>>() {
  return il::Type::Array2DOfDouble;
}

template <>
inline il::Type typeId<il::Array2C<signed char>>() {
  return il::Type::Array2COfInt8;
}

template <>
inline il::Type typeId<il::Array2C<unsigned short>>() {
  return il::Type::Array2COfUInt16;
}

template <>
inline il::Type typeId<il::Array2C<short>>() {
  return il::Type::Array2COfInt16;
}

template <>
inline il::Type typeId<il::Array2C<unsigned int>>() {
  return il::Type::Array2COfUInt32;
}

template <>
inline il::Type typeId<il::Array2C<int>>() {
  return il::Type::Array2COfInt32;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::Array2C<std::size_t>>() {
  return il::Type::Array2COfUInt64;
}

template <>
inline il::Type typeId<il::Array2C<il::int_t>>() {
  return il::Type::Array2COfInt64;
}
#endif

template <>
inline il::Type typeId<il::Array2C<float>>() {
  return il::Type::Array2COfFloat;
}

template <>
inline il::Type typeId<il::Array2C<double>>() {
  return il::Type::Array2COfDouble;
}

}  // namespace il

#endif  // IL_TYPE_H
