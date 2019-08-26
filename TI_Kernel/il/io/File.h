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

#ifndef IL_FILE_H
#define IL_FILE_H

#include <cstdio>

#include <il/Array.h>
#include <il/Status.h>
#include <il/String.h>
#ifdef IL_WINDOWS
#include <il/UTF16String.h>
#include <il/unicode.h>
#endif

namespace il {

enum class FileMode : unsigned char { Read, Write, ReadWrite };

class File {
 private:
  std::FILE* file_;
  il::FileMode mode_;
  il::Array<unsigned char> buffer_;
  il::int_t next_;
  il::int_t end_buffer_;
  il::int_t nb_bytes_read_;
  bool opened_;

 public:
  File(const il::String& filename, il::FileMode mode, il::int_t buffer_size,
       il::io_t, il::Status& status);
  File(const il::String& filename, il::FileMode mode, il::io_t,
       il::Status& status);
  ~File();
  template <il::int_t nb_objects, typename T>
  void Read(il::io_t, T* p, il::Status& status);
  void Read(il::int_t nb_objects, std::size_t nb_bytes, il::io_t, void* p,
            il::Status& status);
  void Write(const void* p, il::int_t nb_objects, std::size_t nb_bytes,
             il::io_t, il::Status& status);
  void Skip(std::size_t nb_bytes, il::io_t, il::Status& status);
  void Close(il::io_t, il::Status& status);
};

inline File::File(const il::String& filename, il::FileMode mode,
                  il::int_t buffer_size, il::io_t, il::Status& status)
    : buffer_{buffer_size} {
#ifdef IL_UNIX
  il::String s_mode{};
  switch (mode) {
    case il::FileMode::Read:
      s_mode = "rb";
      break;
    case il::FileMode::Write:
      s_mode = "wb";
      break;
    default:
      IL_UNREACHABLE;
  }
  file_ = std::fopen(filename.asCString(), s_mode.asCString());
  if (file_ == nullptr) {
    status.SetError(il::Error::FilesystemFileNotFound);
    return;
  }
#else  // Windows case
  il::UTF16String s_mode{};
  switch (mode) {
    case il::FileMode::Read:
      s_mode = L"rb";
      break;
    case il::FileMode::Write:
      s_mode = L"wb";
      break;
    default:
      IL_UNREACHABLE;
  }
  il::UTF16String filename_utf16 = il::toUtf16(filename);
  errno_t error_nb =
      _wfopen_s(&file_, filename_utf16.asWString(), s_mode.asWString());
  if (error_nb != 0) {
    status.SetError(il::Error::FilesystemFileNotFound);
    return;
  }
#endif

  mode_ = mode;
  next_ = 0;
  end_buffer_ = 0;
  nb_bytes_read_ = 0;
  opened_ = true;
  status.SetOk();
}

inline File::File(const il::String& filename, il::FileMode mode, il::io_t,
                  il::Status& status)
    : File{filename, mode, 0, il::io, status} {}

inline File::~File() { IL_EXPECT_MEDIUM(!opened_); }

template <il::int_t nb_objects, typename T>
inline void File::Read(il::io_t, T* p, il::Status& status) {
  IL_EXPECT_MEDIUM(opened_);
  IL_EXPECT_MEDIUM(mode_ == il::FileMode::Read ||
                   mode_ == il::FileMode::ReadWrite);
  IL_EXPECT_MEDIUM(nb_objects >= 0);

  const il::int_t nb_bytes_to_read =
      nb_objects * static_cast<il::int_t>(sizeof(T));
  if (next_ + nb_bytes_to_read <= end_buffer_) {
    std::memcpy(p, buffer_.data() + next_, nb_bytes_to_read);
    next_ += nb_bytes_to_read;
    nb_bytes_read_ += nb_bytes_to_read;
    status.SetOk();
  } else {
    Read(nb_objects, sizeof(T), il::io, p, status);
  }
};

inline void File::Read(il::int_t nb_objects, std::size_t nb_bytes, il::io_t,
                       void* p, il::Status& status) {
  IL_EXPECT_MEDIUM(opened_);
  IL_EXPECT_MEDIUM(mode_ == il::FileMode::Read ||
                   mode_ == il::FileMode::ReadWrite);
  IL_EXPECT_MEDIUM(nb_bytes > 0);
  IL_EXPECT_MEDIUM(nb_objects >= 0);

  const il::int_t nb_bytes_from_buffer = il::min(
      end_buffer_ - next_, nb_objects * static_cast<il::int_t>(nb_bytes));
  if (nb_bytes_from_buffer > 0) {
    std::memcpy(p, buffer_.data() + next_, nb_bytes_from_buffer);
    next_ += nb_bytes_from_buffer;
    nb_bytes_read_ += nb_bytes_from_buffer;
  }
  const il::int_t nb_bytes_to_read =
      nb_objects * static_cast<il::int_t>(nb_bytes) - nb_bytes_from_buffer;
  if (nb_bytes_to_read > 0) {
    if (nb_bytes_to_read > buffer_.size()) {
      // On the Windows platform (tested with Visual Studio 2015), reading 1
      // element of n bytes is faster than reading n elements of 1 byte. As a
      // consequence, I we try to do that first and if we fail because we are at
      // the end of the file, we try to read n elements of 1 byte.
      const std::size_t nb_bytes_read =
          std::fread(p, nb_bytes_to_read, 1, file_);
      nb_bytes_read_ += nb_bytes_read;
      if (nb_bytes_read != nb_bytes_to_read) {
        status.SetError(il::Error::FilesystemFileNotLongEnough);
        return;
      }
    } else {
      // Here, if we try to read a full line of buffer. The cursor position has
      // to be saved because if fread fails, its position is undefined.
      const long previous_position = std::ftell(file_);
      const std::size_t full_buffer_read = std::fread(
          buffer_.data(), static_cast<std::size_t>(buffer_.size()), 1, file_);
      if (full_buffer_read == 1) {
        std::memcpy(static_cast<unsigned char*>(p) + nb_bytes_from_buffer,
                    buffer_.data(), static_cast<std::size_t>(nb_bytes_to_read));
        next_ = nb_bytes_to_read;
        end_buffer_ = buffer_.size();
        nb_bytes_read_ += nb_bytes_to_read;
      } else {
        const int error = std::fseek(file_, previous_position, SEEK_SET);
        IL_EXPECT_MEDIUM(error == 0);
        next_ = 0;
        end_buffer_ = 0;
        const std::size_t chunk_read =
            std::fread(static_cast<unsigned char*>(p) + nb_bytes_from_buffer,
                       nb_bytes_to_read, 1, file_);
        nb_bytes_read_ += nb_bytes_to_read;
        if (chunk_read != 1) {
          status.SetError(il::Error::FilesystemFileNotLongEnough);
          return;
        }
      }
    }
  }

  status.SetOk();
}

inline void File::Skip(std::size_t nb_bytes, il::io_t, il::Status& status) {
  IL_EXPECT_MEDIUM(opened_);

  const il::int_t n = static_cast<std::size_t>(nb_bytes);
  const il::int_t nb_bytes_from_buffer = il::min(end_buffer_ - next_, n);
  next_ += nb_bytes_from_buffer;
  nb_bytes_read_ += nb_bytes_from_buffer;
  const il::int_t nb_bytes_to_read = n - nb_bytes_from_buffer;
  if (nb_bytes_to_read > 0) {
    const int error =
        std::fseek(file_, static_cast<long>(nb_bytes_to_read), SEEK_CUR);
    next_ = 0;
    end_buffer_ = 0;
    nb_bytes_read_ += nb_bytes_to_read;
    if (error != 0) {
      status.SetError(il::Error::FilesystemFileNotLongEnough);
      return;
    }
  }

  status.SetOk();
  return;
}

inline void File::Write(const void* p, il::int_t nb_objects,
                        std::size_t nb_bytes, il::io_t, il::Status& status) {
  IL_EXPECT_MEDIUM(opened_);
  IL_EXPECT_MEDIUM(mode_ == il::FileMode::Write ||
                   mode_ == il::FileMode::ReadWrite);
  IL_EXPECT_MEDIUM(nb_objects >= 0);

  const il::int_t nb_objects_written =
      std::fwrite(p, nb_bytes, static_cast<std::size_t>(nb_objects), file_);
  if (nb_objects_written != nb_objects) {
    status.SetError(il::Error::Undefined);
    return;
  }

  status.SetOk();
}

inline void File::Close(il::io_t, il::Status& status) {
  IL_EXPECT_MEDIUM(opened_);

  const int error = std::fclose(file_);
  if (error != 0) {
    status.SetError(il::Error::FilesystemCanNotCloseFile);
    return;
  }

  opened_ = false;
  status.SetOk();
}

}  // namespace il

#endif  // IL_FILE_H
