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

#ifndef IL_NUMPY_H
#define IL_NUMPY_H

#include <string>

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/SparseMatrixCSR.h>
#include <il/Status.h>
#include <il/String.h>
#include <il/io/io_base.h>

#ifdef IL_WINDOWS
#include <il/UTF16String.h>
#include <il/unicode.h>
#endif

namespace il {

template <typename T>
struct numpyType {
  static constexpr const char* value = "";
};

template <>
struct numpyType<int> {
  static constexpr const char* value = "i4";
};

template <>
struct numpyType<double> {
  static constexpr const char* value = "f8";
};

struct NumpyInfo {
  il::String type;
  il::Array<il::int_t> shape;
  bool fortran_order;
};

NumpyInfo getNumpyInfo(il::io_t, std::FILE* fp, il::Status& status);
void saveNumpyInfo(const NumpyInfo& numpy_info, il::io_t, std::FILE* fp,
                   il::Status& status);

template <typename T>
class SaveHelper<il::Array<T>> {
 public:
  static void save(const il::Array<T>& v, const il::String& filename, il::io_t,
                   il::Status& status) {
#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.asCString(), "wb");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE* file = _wfopen(filename_utf16.asWString(), L"wb");
#endif
    if (!file) {
      status.SetError(il::Error::FilesystemFileNotFound);
      IL_SET_SOURCE(status);
      return;
    }

    il::NumpyInfo numpy_info;
    numpy_info.shape = il::Array<il::int_t>{il::value, {v.size()}};
    numpy_info.type = il::String{il::StringType::Ascii, il::numpyType<T>::value,
                                 il::size(il::numpyType<T>::value)};
    numpy_info.fortran_order = true;

    il::Status info_status{};
    il::saveNumpyInfo(numpy_info, il::io, file, info_status);
    if (!info_status.Ok()) {
      const int error = std::fclose(file);
      if (error != 0) {
        il::abort();
      }
      status = std::move(info_status);
      return;
    }

    std::size_t written = std::fwrite(v.data(), sizeof(T),
                                      static_cast<std::size_t>(v.size()), file);
    if (static_cast<il::int_t>(written) != v.size()) {
      status.SetError(il::Error::FilesystemCanNotWriteToFile);
      IL_SET_SOURCE(status);
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.SetError(il::Error::FilesystemCanNotCloseFile);
      IL_SET_SOURCE(status);
      return;
    }

    status.SetOk();
    return;
  }
};

template <typename T>
class SaveHelper<il::Array2D<T>> {
 public:
  static void save(const il::Array2D<T>& A, const il::String& filename,
                   il::io_t, il::Status& status) {
#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.asCString(), "wb");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE* file = _wfopen(filename_utf16.asWString(), L"wb");
#endif
    if (!file) {
      status.SetError(il::Error::FilesystemFileNotFound);
      IL_SET_SOURCE(status);
      return;
    }

    il::NumpyInfo numpy_info;
    numpy_info.shape = il::Array<il::int_t>{il::value, {A.size(0), A.size(1)}};
    numpy_info.type = il::String{il::StringType::Ascii, il::numpyType<T>::value,
                                 il::size(il::numpyType<T>::value)};
    numpy_info.fortran_order = true;

    il::Status info_status{};
    il::saveNumpyInfo(numpy_info, il::io, file, info_status);
    if (!info_status.Ok()) {
      const int error = std::fclose(file);
      if (error != 0) {
        il::abort();
      }
      status = std::move(info_status);
      return;
    }

    std::size_t written =
        std::fwrite(A.data(), sizeof(T),
                    static_cast<std::size_t>(A.size(0) * A.size(1)), file);
    if (static_cast<il::int_t>(written) != A.size(0) * A.size(1)) {
      status.SetError(il::Error::FilesystemCanNotWriteToFile);
      IL_SET_SOURCE(status);
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.SetError(il::Error::FilesystemCanNotCloseFile);
      IL_SET_SOURCE(status);
      return;
    }

    status.SetOk();
    return;
  }
};

template <typename T>
class LoadHelper<il::Array<T>> {
 public:
  static il::Array<T> load(const il::String& filename, il::io_t,
                           il::Status& status) {
    il::Array<T> v{};

#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.asCString(), "r+b");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE* file = _wfopen(filename_utf16.asWString(), L"r+b");
#endif
    if (!file) {
      status.SetError(il::Error::FilesystemFileNotFound);
      IL_SET_SOURCE(status);
      return v;
    }

    il::Status info_status{};
    il::NumpyInfo numpy_info = il::getNumpyInfo(il::io, file, info_status);
    if (!info_status.Ok()) {
      status = std::move(info_status);
      return v;
    }

    if (!(numpy_info.type.isEqual(il::numpyType<T>::value))) {
      status.SetError(il::Error::BinaryFileWrongType);
      IL_SET_SOURCE(status);
      return v;
    } else if (numpy_info.shape.size() != 1) {
      status.SetError(il::Error::BinaryFileWrongRank);
      IL_SET_SOURCE(status);
      return v;
    }

    v.Resize(numpy_info.shape[0]);
    const std::size_t read = fread(v.Data(), sizeof(T), v.size(), file);
    if (static_cast<il::int_t>(read) != v.size()) {
      status.SetError(il::Error::BinaryFileWrongFormat);
      IL_SET_SOURCE(status);
      return v;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.SetError(il::Error::FilesystemCanNotCloseFile);
      IL_SET_SOURCE(status);
      return v;
    }

    status.SetOk();
    return v;
  }
};

template <typename T>
class LoadHelper<il::Array2D<T>> {
 public:
  static il::Array2D<T> load(const il::String& filename, il::io_t,
                             il::Status& status) {
    il::Array2D<T> v{};

#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.asCString(), "r+b");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE* file = _wfopen(filename_utf16.asWString(), L"r+b");
#endif
    if (!file) {
      status.SetError(il::Error::FilesystemFileNotFound);
      IL_SET_SOURCE(status);
      return v;
    }

    il::Status info_status{};
    il::NumpyInfo numpy_info = il::getNumpyInfo(il::io, file, info_status);
    if (!info_status.Ok()) {
      status = std::move(info_status);
      return v;
    }

    if (!(numpy_info.type.isEqual(il::numpyType<T>::value))) {
      status.SetError(il::Error::BinaryFileWrongType);
      IL_SET_SOURCE(status);
      return v;
    } else if (numpy_info.shape.size() != 2) {
      status.SetError(il::Error::BinaryFileWrongRank);
      IL_SET_SOURCE(status);
      return v;
    } else if (!numpy_info.fortran_order) {
      status.SetError(il::Error::BinaryFileWrongEndianness);
      IL_SET_SOURCE(status);
      return v;
    }

    v.Resize(numpy_info.shape[0], numpy_info.shape[1]);
    const il::int_t n = v.size(0) * v.size(1);
    const std::size_t read = fread(v.Data(), sizeof(T), n, file);
    if (static_cast<il::int_t>(read) != n) {
      status.SetError(il::Error::BinaryFileWrongFormat);
      IL_SET_SOURCE(status);
      return v;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.SetError(il::Error::FilesystemCanNotCloseFile);
      IL_SET_SOURCE(status);
      return v;
    }

    status.SetOk();
    return v;
  }
};

template <>
class LoadHelper<il::SparseMatrixCSR<il::int_t, double>> {
 public:
  static il::SparseMatrixCSR<il::int_t, double> load(const il::String& filename,
                                                     il::io_t,
                                                     il::Status& status) {
    il::Status local_status{};
    il::String filename_row = filename;
    filename_row.Append(".row");
    auto row =
        il::load<il::Array<il::int_t>>(filename_row, il::io, local_status);
    local_status.AbortOnError();

    il::String filename_column = filename;
    filename_column.Append(".column");
    auto column =
        il::load<il::Array<il::int_t>>(filename_column, il::io, local_status);
    local_status.AbortOnError();

    il::String filename_element = filename;
    filename_element.Append(".element");
    auto element =
        il::load<il::Array<double>>(filename_element, il::io, local_status);
    local_status.AbortOnError();

    const il::int_t n = row.size() - 1;
    status.SetOk();
    return il::SparseMatrixCSR<il::int_t, double>{
        n, n, std::move(column), std::move(row), std::move(element)};
  }
};
}  // namespace il

#endif  // IL_NUMPY_H
