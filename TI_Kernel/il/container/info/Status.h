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

#ifndef IL_STATUS_H
#define IL_STATUS_H

#include <il/Info.h>

namespace il {

enum class ErrorDomain : unsigned short {
  Filesystem = 0,
  Binary = 4,
  Parse = 1,
  Overflow = 2,
  FloatingPoint = 3,
  Matrix = 5,
  Unimplemented = 126,
  Undefined = 127
};

enum class Error : unsigned short {
  FilesystemFileNotFound = 0 * 256 + 0,
  FilesystemDirectoryNotFound = 0 * 256 + 1,
  FilesystemNoReadAccess = 0 * 256 + 2,
  FilesystemNoWriteAccess = 0 * 256 + 3,
  FilesystemCanNotCloseFile = 0 * 256 + 4,
  FilesystemCanNotWriteToFile = 0 * 256 + 5,
  FilesystemFileNotLongEnough = 0 * 256 + 6,
  //
  BinaryFileWrongFormat = 4 * 256 + 0,
  BinaryFileWrongType = 4 * 256 + 1,
  BinaryFileWrongRank = 4 * 256 + 2,
  BinaryFileWrongEndianness = 4 * 256 + 3,
  //
  ParseBool = 1 * 256 + 0,
  ParseNumber = 1 * 256 + 11,
  ParseInt = 1 * 256 + 1,
  ParseIntOverflow = 1 * 256 + 2,
  ParseInteger = 1 * 256 + 3,
  ParseIntegerOverflow = 1 * 256 + 4,
  ParseFloat = 1 * 256 + 5,
  ParseDouble = 1 * 256 + 6,
  ParseString = 1 * 256 + 7,
  ParseUnclosedArray = 1 * 256 + 8,
  ParseUnidentifiedTrailingCharacter = 1 * 256 + 9,
  ParseCanNotDetermineType = 1 * 256 + 10,
  ParseHeterogeneousArray = 1 * 256 + 12,
  ParseArray = 1 * 256 + 13,
  ParseTable = 1 * 256 + 14,
  ParseDuplicateKey = 1 * 256 + 14,
  ParseKey = 1 * 256 + 15,
  ParseValue = 1 * 256 + 16,
  //
  OverflowInt = 2 * 256 + 0,
  OverflowInteger = 2 * 256 + 1,
  //
  FloatingPointNonNegative = 3 * 256 + 0,
  FloatingPointPositive = 3 * 256 + 1,
  FloatingPointNonPositive = 3 * 256 + 2,
  FloatingPointNegative = 3 * 256 + 3,
  //
  MatrixSingular = 5 * 256 + 0,
  MatrixEigenValueNoConvergence = 5 * 256 + 1,
  //
  Unimplemented = 126 * 256 + 0,
  //
  Undefined = 127 * 256 + 0
};

inline il::ErrorDomain domain(il::Error error) {
  return static_cast<il::ErrorDomain>(static_cast<short>(error) >> 8);
}

#define IL_SET_SOURCE(status) status.setSource(__FILE__, __LINE__)

class Status {
 private:
  bool ok_;
  bool to_check_;
  Error error_;
  il::Info info_;

 public:
  Status();
  Status(const Status& other) = delete;
  Status(Status&& other);
  Status& operator=(const Status& other) = delete;
  Status& operator=(Status&& other);
  ~Status();
  void SetOk();
  void SetError(il::Error error);
  void SetInfo(const char* key, int value);
#ifdef IL_64_BIT
  void SetInfo(const char* key, il::int_t value);
#endif
  void SetInfo(const char* key, double value);
  void SetInfo(const char* key, const char* value);
  void setSource(const char* file, il::int_t line);
  il::int_t toInteger(const char* key) const;
  double toDouble(const char* key) const;
  const char* asCString(const char* key) const;
  void Rearm();
  bool Ok();
  void AbortOnError();
  void IgnoreError();
  il::Error error() const;
  il::ErrorDomain domain() const;
};

inline Status::Status() : info_{} {
  ok_ = true;
  to_check_ = false;
  error_ = il::Error::Undefined;
}

inline Status::Status(Status&& other) {
  ok_ = other.ok_;
  to_check_ = true;
  error_ = other.error_;
  info_ = other.info_;

  other.to_check_ = false;
}

inline Status& Status::operator=(Status&& other) {
  IL_EXPECT_MEDIUM(!to_check_);

  ok_ = other.ok_;
  to_check_ = true;
  error_ = other.error_;
  info_ = other.info_;

  other.to_check_ = false;

  return *this;
}

inline Status::~Status() { IL_EXPECT_MEDIUM(!to_check_); }

inline void Status::SetError(il::Error error) {
  IL_EXPECT_MEDIUM(!to_check_);

  ok_ = false;
  to_check_ = true;
  error_ = error;
  info_.clear();
}

inline void Status::SetInfo(const char* key, int value) {
  IL_EXPECT_MEDIUM(to_check_);
  IL_EXPECT_MEDIUM(!ok_);

  info_.Set(key, value);
}

#ifdef IL_64_BIT
inline void Status::SetInfo(const char* key, il::int_t value) {
  IL_EXPECT_MEDIUM(to_check_);
  IL_EXPECT_MEDIUM(!ok_);

  info_.Set(key, value);
}
#endif

inline void Status::SetInfo(const char* key, double value) {
  IL_EXPECT_MEDIUM(to_check_);
  IL_EXPECT_MEDIUM(!ok_);

  info_.Set(key, value);
}

inline void Status::SetInfo(const char* key, const char* value) {
  IL_EXPECT_MEDIUM(to_check_);
  IL_EXPECT_MEDIUM(!ok_);

  info_.Set(key, value);
}

inline void Status::setSource(const char* file, il::int_t line) {
  SetInfo("source_file", file);
  SetInfo("source_line", line);
}

inline il::int_t Status::toInteger(const char* key) const {
  IL_EXPECT_MEDIUM(!ok_);

  return info_.toInteger(key);
}

inline double Status::toDouble(const char* key) const {
  IL_EXPECT_MEDIUM(!ok_);

  return info_.toDouble(key);
}

inline const char* Status::asCString(const char* key) const {
  IL_EXPECT_MEDIUM(!ok_);

  return info_.asCString(key);
}

inline void Status::SetOk() {
  IL_EXPECT_MEDIUM(!to_check_);

  ok_ = true;
  to_check_ = true;
}

inline void Status::Rearm() {
  IL_EXPECT_MEDIUM(!to_check_);

  to_check_ = true;
}

inline bool Status::Ok() {
  IL_EXPECT_MEDIUM(to_check_);

  to_check_ = false;
  return ok_;
}

inline void Status::AbortOnError() {
  IL_EXPECT_MEDIUM(to_check_);

  to_check_ = false;
  if (!ok_) {
    il::abort();
  }
}

inline void Status::IgnoreError() {
  IL_EXPECT_MEDIUM(to_check_);

  to_check_ = false;
}

inline il::Error Status::error() const {
  IL_EXPECT_MEDIUM(!to_check_);

  return error_;
}
}  // namespace il

#endif  // IL_STATUS_H
