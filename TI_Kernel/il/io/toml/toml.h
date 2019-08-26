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

#ifndef IL_TOML_H
#define IL_TOML_H

#include <il/Dynamic.h>
#include <il/Status.h>
#include <il/core.h>

#include <il/io/io_base.h>

#include <il/Array.h>
#include <il/String.h>
#include <il/container/string/algorithmString.h>

#ifdef IL_WINDOWS
#include <il/UTF16String.h>
#include <il/unicode.h>
#endif

namespace il {

class TomlParser {
 private:
  static const il::int_t max_line_length_ = 200;
  char buffer_line_[max_line_length_ + 1];
  il::int_t line_number_;
  std::FILE *file_;

 public:
  TomlParser();
  il::MapArray<il::String, il::Dynamic> parse(const il::String &filename,
                                              il::io_t, il::Status &status);
  il::StringView skipWhitespaceAndComments(il::StringView string, il::io_t,
                                           il::Status &status);
  il::Dynamic parseValue(il::io_t, il::StringView &string, il::Status &status);
  il::Dynamic parseArray(il::io_t, il::StringView &string, il::Status &status);
  il::Dynamic parseValueArray(il::Type value_type, il::io_t,
                              il::StringView &string, il::Status &status);
  il::Dynamic parseObjectArray(il::Type object_type, char delimiter, il::io_t,
                               il::StringView &string, il::Status &status);
  il::Dynamic parseInlineTable(il::io_t, il::StringView &string,
                               il::Status &status);
  void parseKeyValue(il::io_t, il::StringView &string,
                     il::MapArray<il::String, il::Dynamic> &toml,
                     il::Status &status);
  void checkEndOfLineOrComment(il::StringView string, il::io_t,
                               il::Status &status);
  il::String currentLine() const;
  il::Dynamic parseNumber(il::io_t, il::StringView &string, il::Status &status);
  il::Type parseType(il::StringView string, il::io_t, il::Status &status);
  il::Dynamic parseBool(il::io_t, il::StringView &string, Status &status);
  il::String parseStringLiteral(char delimiter, il::io_t,
                                il::StringView &string, il::Status &status);
  il::String parseEscapeCode(il::io_t, il::StringView &string,
                             il::Status &status);
  il::String parseKey(char end, il::io_t, il::StringView &string,
                      il::Status &status);
  void parseTable(il::io_t, il::StringView &string,
                  il::MapArray<il::String, il::Dynamic> *&toml,
                  il::Status &status);
  void parseSingleTable(il::io_t, il::StringView &string,
                        il::MapArray<il::String, il::Dynamic> *&toml,
                        il::Status &status);
  void parseTableArray(il::io_t, il::StringView &string,
                       il::MapArray<il::String, il::Dynamic> *&toml,
                       il::Status &status);
  il::Dynamic parseString(il::io_t, il::StringView &string, il::Status &status);

 private:
  static bool containsDigit(char c);
};

inline void save_array(const il::Array<il::Dynamic> &array, il::io_t,
                       std::FILE *file, il::Status &status) {
  const int error0 = std::fputs("[ ", file);
  IL_UNUSED(error0);
  for (il::int_t j = 0; j < array.size(); ++j) {
    if (j > 0) {
      const int error1 = std::fputs(", ", file);
      IL_UNUSED(error1);
    }
    switch (array[j].type()) {
      case il::Type::Bool: {
        if (array[j].to<bool>()) {
          const int error2 = std::fputs("true", file);
          IL_UNUSED(error2);
        } else {
          const int error2 = std::fputs("false", file);
          IL_UNUSED(error2);
        }
      } break;
      case il::Type::Integer: {
        const int error2 = std::fprintf(file, "%td", array[j].to<il::int_t>());
        IL_UNUSED(error2);
      } break;
      case il::Type::Double: {
        const int error2 = std::fprintf(file, "%e", array[j].to<double>());
        IL_UNUSED(error2);
      } break;
      case il::Type::UnicodeString: {
        const int error2 = std::fputs("\"", file);
        const int error3 =
            std::fputs(array[j].as<il::String>().asCString(), file);
        const int error4 = std::fputs("\"", file);
        IL_UNUSED(error2);
        IL_UNUSED(error3);
        IL_UNUSED(error4);
      } break;
      case il::Type::ArrayOfDynamic: {
        save_array(array[j].as<il::Array<il::Dynamic>>(), il::io, file, status);
        if (!status.Ok()) {
          status.Rearm();
          return;
        }
      } break;
      default:
        IL_UNREACHABLE;
    }
  }
  const int error5 = std::fputs(" ]", file);
  IL_UNUSED(error5);
  status.SetOk();
}

template <typename M>
inline void save_aux(const M &toml, const il::String &name, il::io_t,
                     std::FILE *file, il::Status &status) {
  int error = 0;
  // Add an object that sets the error on destruction

  for (il::spot_t i = toml.spotBegin(); i != toml.spotEnd(); i = toml.next(i)) {
    const il::Dynamic &value = toml.value(i);
    const il::Type type = value.type();
    if (type != il::Type::MapArrayStringToDynamic &&
        type != il::Type::MapStringToDynamic) {
      error = std::fputs(toml.key(i).asCString(), file);
      if (error == EOF) return;
      error = std::fputs(" = ", file);
      if (error == EOF) return;
      switch (type) {
        case il::Type::Bool:
          if (value.to<bool>()) {
            error = std::fputs("true", file);
          } else {
            error = std::fputs("false", file);
          }
          if (error == EOF) return;
          break;
        case il::Type::Integer:
          error = std::fprintf(file, "%td", value.to<il::int_t>());
          if (error == EOF) return;
          break;
        case il::Type::Double:
          error = std::fprintf(file, "%e", value.to<double>());
          if (error == EOF) return;
          break;
        case il::Type::UnicodeString:
          error = std::fputs("\"", file);
          if (error == EOF) return;
          error = std::fputs(value.as<il::String>().asCString(), file);
          if (error == EOF) return;
          error = std::fputs("\"", file);
          if (error == EOF) return;
          break;
        case il::Type::ArrayOfDouble: {
          const il::Array<double> &v = value.as<il::Array<double>>();
          error = std::fputs("[ ", file);
          if (error == EOF) return;
          for (il::int_t i = 0; i < v.size(); ++i) {
            std::fprintf(file, "%e", v[i]);
            if (i + 1 < v.size()) {
              error = std::fputs(", ", file);
              if (error == EOF) return;
            }
          }
          error = std::fputs(" ]", file);
          if (error == EOF) return;
        } break;
        case il::Type::Array2DOfDouble: {
          const il::Array2D<double> &v = value.as<il::Array2D<double>>();
          error = std::fputs("[ ", file);
          if (error == EOF) return;
          for (il::int_t i = 0; i < v.size(0); ++i) {
            error = std::fputs("[ ", file);
            if (error == EOF) return;
            for (il::int_t j = 0; j < v.size(1); ++j) {
              std::fprintf(file, "%e", v(i, j));
              if (j + 1 < v.size(1)) {
                error = std::fputs(", ", file);
                if (error == EOF) return;
              }
            }
            error = std::fputs(" ]", file);
            if (error == EOF) return;
            if (i + 1 < v.size(0)) {
              error = std::fputs(", ", file);
              if (error == EOF) return;
            }
          }
          error = std::fputs(" ]", file);
          if (error == EOF) return;
        } break;
        case il::Type::ArrayOfDynamic: {
          save_array(value.as<il::Array<il::Dynamic>>(), il::io, file, status);
          status.AbortOnError();
        } break;
        default:
          IL_UNREACHABLE;
      }
      error = std::fputs("\n", file);
      if (error == EOF) return;
    } else if (type == il::Type::MapArrayStringToDynamic) {
      error = std::fputs("\n[", file);
      if (error == EOF) return;
      if (name.size() != 0) {
        error = std::fputs(name.asCString(), file);
        if (error == EOF) return;
        error = std::fputs(".", file);
        if (error == EOF) return;
      }
      error = std::fputs(toml.key(i).asCString(), file);
      if (error == EOF) return;
      error = std::fputs("]\n", file);
      if (error == EOF) return;
      save_aux(value.as<il::MapArray<il::String, il::Dynamic>>(), toml.key(i),
               il::io, file, status);
      if (!status.Ok()) {
        status.Rearm();
        return;
      }
    } else if (type == il::Type::MapStringToDynamic) {
      error = std::fputs("\n[", file);
      if (error == EOF) return;
      if (name.size() != 0) {
        error = std::fputs(name.asCString(), file);
        if (error == EOF) return;
        error = std::fputs(".", file);
        if (error == EOF) return;
      }
      error = std::fputs(toml.key(i).asCString(), file);
      if (error == EOF) return;
      error = std::fputs("]\n", file);
      if (error == EOF) return;
      save_aux(value.as<il::Map<il::String, il::Dynamic>>(), toml.key(i),
               il::io, file, status);
      if (!status.Ok()) {
        status.Rearm();
        return;
      }
    }
  }

  status.SetOk();
  return;
}

template <>
class SaveHelper<il::Map<il::String, il::Dynamic>> {
 public:
  static void save(const il::Map<il::String, il::Dynamic> &toml,
                   const il::String &filename, il::io_t, il::Status &status) {
#ifdef IL_UNIX
    std::FILE *file = std::fopen(filename.asCString(), "wb");
    if (!file) {
      status.SetError(il::Error::FilesystemFileNotFound);
      return;
    }
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE *file;
    errno_t error_nb = _wfopen_s(&file, filename_utf16.asWString(), L"wb");
    if (error_nb != 0) {
      status.SetError(il::Error::FilesystemFileNotFound);
      return;
    }
#endif

    il::String root_name{};
    save_aux(toml, root_name, il::io, file, status);
    if (!status.Ok()) {
      status.Rearm();
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.SetError(il::Error::FilesystemCanNotCloseFile);
      return;
    }

    status.SetOk();
    return;
  }
};

template <>
class SaveHelperToml<il::MapArray<il::String, il::Dynamic>> {
 public:
  static void save(const il::MapArray<il::String, il::Dynamic> &toml,
                   const il::String &filename, il::io_t, il::Status &status) {
#ifdef IL_UNIX
    std::FILE *file = std::fopen(filename.asCString(), "wb");
    if (!file) {
      status.SetError(il::Error::FilesystemFileNotFound);
      return;
    }
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE *file;
    errno_t error_nb = _wfopen_s(&file, filename_utf16.asWString(), L"wb");
    if (error_nb != 0) {
      status.SetError(il::Error::FilesystemFileNotFound);
      return;
    }
#endif

    il::String root_name{};
    save_aux(toml, root_name, il::io, file, status);
    if (!status.Ok()) {
      status.Rearm();
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.SetError(il::Error::FilesystemCanNotCloseFile);
      return;
    }

    status.SetOk();
    return;
  }
};

il::MapArray<il::String, il::Dynamic> parse(const il::String &filename,
                                            il::io_t, il::Status &status);

template <>
class LoadHelperToml<il::MapArray<il::String, il::Dynamic>> {
 public:
  static il::MapArray<il::String, il::Dynamic> load(const il::String &filename,
                                                    il::io_t,
                                                    il::Status &status) {
    il::TomlParser parser{};
    return parser.parse(filename, il::io, status);
  }
};
}  // namespace il

#endif  // IL_TOML_H