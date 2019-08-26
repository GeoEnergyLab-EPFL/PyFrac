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

#include <il/container/string/algorithmString.h>
#include <il/io/toml/toml.h>

namespace il {

TomlParser::TomlParser() {}

il::StringView TomlParser::skipWhitespaceAndComments(il::StringView string,
                                                     il::io_t,
                                                     il::Status& status) {
  string.trimPrefix();
  while (string.isEmpty() || string[0] == '#' || string.startsWithNewLine()) {
    const char* error = std::fgets(buffer_line_, max_line_length_ + 1, file_);
    if (error == nullptr) {
      IL_SET_SOURCE(status);
      status.SetError(il::Error::ParseUnclosedArray);
      status.SetInfo("line", line_number_);
      return string;
    }
    ++line_number_;
    string = il::StringView{buffer_line_};
    string.trimPrefix();
  }

  status.SetOk();
  return string;
}

bool TomlParser::containsDigit(char c) { return c >= '0' && c <= '9'; }

void TomlParser::checkEndOfLineOrComment(il::StringView string, il::io_t,
                                         il::Status& status) {
  if (!string.isEmpty() && (!string.startsWithNewLine()) &&
      (string[0] != '#')) {
    IL_SET_SOURCE(status);
    status.SetError(il::Error::ParseUnidentifiedTrailingCharacter);
    status.SetInfo("line", line_number_);
  } else {
    status.SetOk();
  }
}

il::String TomlParser::currentLine() const {
  return il::toString(line_number_);
}

il::Type TomlParser::parseType(il::StringView string, il::io_t,
                               il::Status& status) {
  if (string[0] == '"' || string[0] == '\'') {
    status.SetOk();
    return il::Type::UnicodeString;
  } else if (string.startsWithDigit() || string[0] == '-' || string[0] == '+') {
    il::int_t i = 0;
    if (string[0] == '-' || string[0] == '+') {
      ++i;
    }
    while (i < string.size() && string.hasDigit(i)) {
      ++i;
    }
    if (i < string.size() && string[i] == '.') {
      ++i;
      while (i < string.size() && string.hasDigit(i)) {
        ++i;
      }
      status.SetOk();
      return il::Type::Double;
    } else {
      status.SetOk();
      return il::Type::Integer;
    }
  } else if (string[0] == 't' || string[0] == 'f') {
    status.SetOk();
    return il::Type::Bool;
  } else if (string[0] == '[') {
    status.SetOk();
    return il::Type::ArrayOfDynamic;
  } else if (string[0] == '{') {
    status.SetOk();
    return il::Type::MapArrayStringToDynamic;
  } else {
    status.SetError(il::Error::ParseCanNotDetermineType);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return il::Type::Void;
  }
}

il::Dynamic TomlParser::parseBool(il::io_t, il::StringView& string,
                                  Status& status) {
  if (string.size() > 3 && string[0] == 't' && string[1] == 'r' &&
      string[2] == 'u' && string[3] == 'e') {
    status.SetOk();
    string.removePrefix(4);
    return il::Dynamic{true};
  } else if (string.size() > 4 && string[0] == 'f' && string[1] == 'a' &&
             string[2] == 'l' && string[3] == 's' && string[4] == 'e') {
    status.SetOk();
    string.removePrefix(5);
    return il::Dynamic{false};
  } else {
    status.SetError(il::Error::ParseBool);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return il::Dynamic{};
  }
}

il::Dynamic TomlParser::parseNumber(il::io_t, il::StringView& string,
                                    il::Status& status) {
  // Skip the +/- sign at the beginning of the string
  il::int_t i = 0;
  if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
    ++i;
  }

  // Check that there is no leading 0
  if (i + 1 < string.size() && string[i] == '0' &&
      !(string[i + 1] == '.' || string[i + 1] == ' ' ||
        string.hasNewLine(i + 1))) {
    status.SetError(il::Error::ParseNumber);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return il::Dynamic{};
  }

  // Skip the digits (there should be at least one) before the '.' ot the 'e'
  const il::int_t i_begin_number = i;
  while (i < string.size() && string.hasDigit(i)) {
    ++i;
    if (i < string.size() && string[i] == '_') {
      ++i;
      if (i == string.size() || (!string.hasDigit(i + 1))) {
        status.SetError(il::Error::ParseNumber);
        IL_SET_SOURCE(status);
        status.SetInfo("line", line_number_);
        return il::Dynamic{};
      }
    }
  }
  if (i == i_begin_number) {
    status.SetError(il::Error::ParseNumber);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return il::Dynamic{};
  }

  // Detecting if we are a floating point or an integer
  bool is_float;
  if (i < string.size() &&
      (string[i] == '.' || string[i] == 'e' || string[i] == 'E')) {
    is_float = true;

    const bool is_exponent = (string[i] == 'e' || string[i] == 'E');
    ++i;

    if (i == string.size()) {
      status.SetError(il::Error::ParseDouble);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return il::Dynamic{};
    }

    // Skip the +/- if we have an exponent
    if (is_exponent) {
      if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
        ++i;
      }
    }

    // Skip the digits (there should be at least one). Note that trailing 0
    // are accepted.
    const il::int_t i_begin_number = i;
    while (i < string.size() && string.hasDigit(i)) {
      ++i;
    }
    if (i == i_begin_number) {
      status.SetError(il::Error::ParseDouble);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return il::Dynamic{};
    }

    // If we were after the dot, we might have reached the exponent
    if (!is_exponent && i < string.size() &&
        (string[i] == 'e' || string[i] == 'E')) {
      ++i;

      // Skip the +/- and then the digits (there should be at least one)
      if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
        ++i;
      }
      const il::int_t i_begin_exponent = i;
      while (i < string.size() && string.hasDigit(i)) {
        ++i;
      }
      if (i == i_begin_exponent) {
        status.SetError(il::Error::ParseDouble);
        IL_SET_SOURCE(status);
        status.SetInfo("line", line_number_);
        return il::Dynamic{};
      }
    }
  } else {
    is_float = false;
  }

  // Remove the '_' in the number
  il::String number{};
  number.Reserve(i);
  for (il::int_t j = 0; j < i; ++j) {
    if (string[j] != '_') {
      number.Append(string[j]);
    }
  }
  il::StringView view{il::StringType::Byte, number.data(), i};

  if (is_float) {
    status.SetOk();
    string.removePrefix(i);
    double x = std::atof(view.asCString());
    return il::Dynamic{x};
  } else {
    status.SetOk();
    string.removePrefix(i);
    il::int_t n = std::atoll(view.asCString());
    return il::Dynamic{n};
  }
}

il::Dynamic TomlParser::parseString(il::io_t, il::StringView& string,
                                    il::Status& status) {
  IL_EXPECT_FAST(string[0] == '"' || string[0] == '\'');

  const char delimiter = string[0];

  il::Status parse_status{};
  il::Dynamic ans = parseStringLiteral(delimiter, il::io, string, parse_status);
  if (!parse_status.Ok()) {
    status = std::move(parse_status);
    return ans;
  }

  status.SetOk();
  return ans;
}

il::String TomlParser::parseStringLiteral(char delimiter, il::io_t,
                                          il::StringView& string,
                                          il::Status& status) {
  il::String ans{};

  string.removePrefix(1);
  while (!string.isEmpty()) {
    if (delimiter == '"' && string[0] == '\\') {
      il::Status parse_status{};
      ans.Append(parseEscapeCode(il::io, string, parse_status));
      if (!parse_status.Ok()) {
        status = std::move(parse_status);
        return ans;
      }
    } else if (string[0] == delimiter) {
      string.removePrefix(1);
      string = il::removeWhitespaceLeft(string);
      status.SetOk();
      return ans;
    } else {
      // TODO: I am not sure what will happen with a Unicode string
      ans.Append(string.rune(0));
      string.removePrefix(string.nextRune(0));
    }
  }

  status.SetError(il::Error::ParseString);
  IL_SET_SOURCE(status);
  status.SetInfo("line", line_number_);
  return ans;
}

il::String TomlParser::parseEscapeCode(il::io_t, il::StringView& string,
                                       il::Status& status) {
  IL_EXPECT_FAST(string.size() > 0 && string[0] == '\\');

  il::String ans{};
  il::int_t i = 1;
  if (i == string.size()) {
    status.SetError(il::Error::ParseString);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return ans;
  }

  char value;
  switch (string[i]) {
    case 'b':
      value = '\b';
      break;
    case 't':
      value = '\t';
      break;
    case 'n':
      value = '\n';
      break;
    case 'f':
      value = '\f';
      break;
    case 'r':
      value = '\r';
      break;
    case '"':
      value = '"';
      break;
    case '\\':
      value = '\\';
      break;
    case 'u':
    case 'U': {
      status.SetError(il::Error::ParseString);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return ans;
    } break;
    default:
      status.SetError(il::Error::ParseString);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return ans;
  }

  string.removePrefix(2);
  ans.Append(value);
  status.SetOk();
  return ans;
}

il::Dynamic TomlParser::parseArray(il::io_t, il::StringView& string,
                                   il::Status& status) {
  IL_EXPECT_FAST(!string.isEmpty() && string[0] == '[');

  il::Dynamic ans{il::Array<il::Dynamic>{}};
  il::Status parse_status{};

  string.removePrefix(1);
  string = skipWhitespaceAndComments(string, il::io, parse_status);
  if (!parse_status.Ok()) {
    status = std::move(parse_status);
    return ans;
  }

  if (!string.isEmpty() && string[0] == ']') {
    string.removePrefix(1);
    status.SetOk();
    return ans;
  }

  il::int_t i = 0;
  while (i < string.size() && (string[i] != ',') && (string[i] != ']') &&
         (string[i] != '#')) {
    ++i;
  }
  il::StringView value_string = string.subview(0, i);
  il::Type value_type = parseType(value_string, il::io, parse_status);
  if (!parse_status.Ok()) {
    status = std::move(parse_status);
    return ans;
  }

  switch (value_type) {
    case il::Type::Void:
    case il::Type::Bool:
    case il::Type::Integer:
    case il::Type::Double:
    case il::Type::UnicodeString: {
      ans = parseValueArray(value_type, il::io, string, parse_status);
      if (!parse_status.Ok()) {
        status = std::move(parse_status);
        return ans;
      }
      status.SetOk();
      return ans;
    } break;
    case il::Type::ArrayOfDynamic: {
      ans = parseObjectArray(il::Type::ArrayOfDynamic, '[', il::io, string,
                             parse_status);
      if (!parse_status.Ok()) {
        status = std::move(parse_status);
        return ans;
      }
      status.SetOk();
      return ans;
    }
    default:
      il::abort();
      return ans;
  }
}

il::Dynamic TomlParser::parseValueArray(il::Type value_type, il::io_t,
                                        il::StringView& string,
                                        il::Status& status) {
  il::Dynamic ans{il::Array<il::Dynamic>{}};
  il::Array<il::Dynamic>& array = ans.As<il::Array<il::Dynamic>>();
  il::Status parse_status{};

  while (!string.isEmpty() && (string[0] != ']')) {
    il::Dynamic value = parseValue(il::io, string, parse_status);
    if (!parse_status.Ok()) {
      status = std::move(parse_status);
      return ans;
    }

    if (value.type() == value_type) {
      array.Append(value);
    } else {
      status.SetError(il::Error::ParseHeterogeneousArray);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return ans;
    }

    string = skipWhitespaceAndComments(string, il::io, parse_status);
    if (!parse_status.Ok()) {
      status = std::move(parse_status);
      return ans;
    }

    if (string.isEmpty() || (string[0] != ',')) {
      break;
    }

    string.removePrefix(1);
    string = skipWhitespaceAndComments(string, il::io, parse_status);
    if (!parse_status.Ok()) {
      status = std::move(parse_status);
      return ans;
    }
  }

  if (!string.isEmpty()) {
    string.removePrefix(1);
  }

  status.SetOk();
  return ans;
}

il::Dynamic TomlParser::parseObjectArray(il::Type object_type, char delimiter,
                                         il::io_t, il::StringView& string,
                                         il::Status& status) {
  il::Dynamic ans{il::Array<il::Dynamic>{}};
  il::Array<il::Dynamic>& array = ans.As<il::Array<il::Dynamic>>();
  il::Status parse_status{};

  while (!string.isEmpty() && (string[0] != ']')) {
    if (string[0] != delimiter) {
      status.SetError(il::Error::ParseArray);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return ans;
    }

    if (object_type == il::Type::ArrayOfDynamic) {
      array.Append(parseArray(il::io, string, parse_status));
      if (!parse_status.Ok()) {
        status = std::move(parse_status);
        return ans;
      }
    } else {
      il::abort();
    }

    string = il::removeWhitespaceLeft(string);
    if (string.isEmpty() || (string[0] != ',')) {
      break;
    }
    string.removePrefix(1);
    string = il::removeWhitespaceLeft(string);
  }

  if (string.isEmpty() || (string[0] != ']')) {
    status.SetError(il::Error::ParseArray);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return ans;
  }
  string.removePrefix(1);
  status.SetOk();
  return ans;
}

il::Dynamic TomlParser::parseInlineTable(il::io_t, il::StringView& string,
                                         il::Status& status) {
  il::Dynamic ans = il::Dynamic{il::MapArray<il::String, il::Dynamic>{}};
  do {
    string.removePrefix(1);
    if (string.isEmpty()) {
      status.SetError(il::Error::ParseTable);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return ans;
    }
    string = il::removeWhitespaceLeft(string);
    il::Status parse_status{};
    parseKeyValue(il::io, string,
                  ans.As<il::MapArray<il::String, il::Dynamic>>(),
                  parse_status);
    if (!parse_status.Ok()) {
      status = std::move(parse_status);
      return ans;
    }
    string = il::removeWhitespaceLeft(string);
  } while (string[0] == ',');

  if (string.isEmpty() || (string[0] != '}')) {
    status.SetError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return ans;
  }

  string.removePrefix(1);
  string = il::removeWhitespaceLeft(string);

  status.SetOk();
  return ans;
}

void TomlParser::parseKeyValue(il::io_t, il::StringView& string,
                               il::MapArray<il::String, il::Dynamic>& toml,
                               il::Status& status) {
  il::Status parse_status{};
  il::String key = parseKey('=', il::io, string, parse_status);
  if (!parse_status.Ok()) {
    status = std::move(parse_status);
    return;
  }

  il::spot_t i = toml.search(key);
  if (toml.found(i)) {
    status.SetError(il::Error::ParseDuplicateKey);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return;
  }

  if (string.isEmpty() || (string[0] != '=')) {
    status.SetError(il::Error::ParseKey);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return;
  }
  string.removePrefix(1);
  string = il::removeWhitespaceLeft(string);

  il::Dynamic value = parseValue(il::io, string, parse_status);
  if (!parse_status.Ok()) {
    status = std::move(parse_status);
    return;
  }

  status.SetOk();
  toml.Set(key, value);
}

il::String TomlParser::parseKey(char end, il::io_t, il::StringView& string,
                                il::Status& status) {
  IL_EXPECT_FAST(end == '=' || end == '"' || end == '\'' || end == '@');

  il::String key{};
  string = il::removeWhitespaceLeft(string);
  IL_EXPECT_FAST(string.size() > 0);

  il::Status parse_status{};
  if (string[0] == '"') {
    //////////////////////////
    // We have a key in a ".."
    //////////////////////////
    string.removePrefix(1);
    while (string.size() > 0) {
      if (string[0] == '\\') {
        il::Status parse_status{};
        key.Append(parseEscapeCode(il::io, string, parse_status));
        if (!parse_status.Ok()) {
          status = std::move(parse_status);
          return key;
        }
      } else if (string[0] == '"') {
        string.removePrefix(1);
        string = il::removeWhitespaceLeft(string);
        status.SetOk();
        return key;
      } else {
        // Check what's going on with unicode
        key.Append(string[0]);
        string.removePrefix(1);
      }
    }
    status.SetError(il::Error::ParseString);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return key;
  } else {
    /////////////////////////////////////
    // We have a bare key: '..' or ->..<-
    /////////////////////////////////////
    // Search for the end of the key and move back to drop the whitespaces
    if (string[0] == '\'') {
      string.removePrefix(1);
    }
    il::int_t j = 0;
    if (end != '@') {
      while (j < string.size() && (string[j] != end)) {
        ++j;
      }
    } else {
      while (j < string.size() && (string[j] != '.') && (string[j] != ']')) {
        ++j;
      }
    }
    const il::int_t j_end = j;
    while (j > 0 && (string[j - 1] == ' ' || string[j - 1] == '\t')) {
      --j;
    }
    if (j == 0) {
      status.SetError(il::Error::ParseKey);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return key;
    }

    // Check if the key has forbidden characters
    il::StringView key_string = string.subview(0, j);
    string.removePrefix(j_end);

    for (il::int_t i = 0; i < key_string.size(); ++i) {
      if (key_string[i] == ' ' || key_string[i] == '\t') {
        status.SetError(il::Error::ParseKey);
        IL_SET_SOURCE(status);
        status.SetInfo("line", line_number_);
        return key;
      }
      if (key_string[i] == '#') {
        status.SetError(il::Error::ParseKey);
        IL_SET_SOURCE(status);
        status.SetInfo("line", line_number_);
        return key;
      }
      if (key_string[i] == '[' || key_string[i] == ']') {
        status.SetError(il::Error::ParseKey);
        IL_SET_SOURCE(status);
        status.SetInfo("line", line_number_);
        return key;
      }
    }

    key = il::String{il::StringType::Byte, key_string.data(), j};
    status.SetOk();
    return key;
  }
}

il::Dynamic TomlParser::parseValue(il::io_t, il::StringView& string,
                                   il::Status& status) {
  il::Dynamic ans{};

  // Check if there is a value
  if (string.isEmpty() || string[0] == '\n' || string[0] == '#') {
    status.SetError(il::Error::ParseValue);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return ans;
  }

  // Get the type of the value
  il::Status parse_status{};
  il::Type type = parseType(string, il::io, parse_status);
  if (!parse_status.Ok()) {
    status = std::move(parse_status);
    return ans;
  }

  switch (type) {
    case il::Type::Bool:
      ans = parseBool(il::io, string, parse_status);
      break;
    case il::Type::Integer:
    case il::Type::Double:
      ans = parseNumber(il::io, string, parse_status);
      break;
    case il::Type::UnicodeString:
      ans = parseString(il::io, string, parse_status);
      break;
    case il::Type::ArrayOfDynamic:
      ans = parseArray(il::io, string, parse_status);
      break;
    case il::Type::MapArrayStringToDynamic:
      ans = parseInlineTable(il::io, string, parse_status);
      break;
    default:
      IL_UNREACHABLE;
  }

  if (!parse_status.Ok()) {
    status = std::move(parse_status);
    return ans;
  }

  status.SetOk();
  return ans;
}

void TomlParser::parseTable(il::io_t, il::StringView& string,
                            il::MapArray<il::String, il::Dynamic>*& toml,
                            il::Status& status) {
  // Skip the '[' at the beginning of the table
  string.removePrefix(1);

  if (string.isEmpty()) {
    status.SetError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return;
  } else if (string[0] == '[') {
    parseTableArray(il::io, string, toml, status);
    return;
  } else {
    parseSingleTable(il::io, string, toml, status);
    return;
  }
}

void TomlParser::parseSingleTable(il::io_t, il::StringView& string,
                                  il::MapArray<il::String, il::Dynamic>*& toml,
                                  il::Status& status) {
  if (string.isEmpty() || string[0] == ']') {
    status.SetError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
  }

  il::String full_table_name{};
  bool inserted = false;
  while (!string.isEmpty() && (string[0] != ']')) {
    il::Status parse_status{};
    il::String table_name = parseKey('@', il::io, string, parse_status);
    if (!parse_status.Ok()) {
      status = std::move(parse_status);
      return;
    }
    if (table_name.isEmpty()) {
      status.SetError(il::Error::ParseTable);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return;
    }
    if (!full_table_name.isEmpty()) {
      full_table_name.Append('.');
    }
    full_table_name.Append(table_name);

    il::spot_t i = toml->search(table_name);
    if (toml->found(i)) {
      if (toml->value(i).is<il::MapArray<il::String, il::Dynamic>>()) {
        toml = &(toml->Value(i).As<il::MapArray<il::String, il::Dynamic>>());
      } else if (toml->value(i).is<il::Array<il::Dynamic>>()) {
        if (toml->value(i).as<il::Array<il::Dynamic>>().size() > 0 &&
            toml->value(i)
                .as<il::Array<il::Dynamic>>()
                .back()
                .is<il::MapArray<il::String, il::Dynamic>>()) {
          toml = &(toml->Value(i)
                       .As<il::Array<il::Dynamic>>()
                       .Back()
                       .As<il::MapArray<il::String, il::Dynamic>>());
        } else {
          status.SetError(il::Error::ParseDuplicateKey);
          IL_SET_SOURCE(status);
          status.SetInfo("line", line_number_);
          return;
        }
      } else {
        status.SetError(il::Error::ParseDuplicateKey);
        IL_SET_SOURCE(status);
        status.SetInfo("line", line_number_);
        return;
      }
    } else {
      inserted = true;
      toml->Set(table_name,
                il::Dynamic{il::MapArray<il::String, il::Dynamic>{}}, il::io,
                i);
      toml = &(toml->Value(i).As<il::MapArray<il::String, il::Dynamic>>());
    }

    string = il::removeWhitespaceLeft(string);
    while (!string.isEmpty() && string[0] == '.') {
      string.removePrefix(1);
    }
    string = il::removeWhitespaceLeft(string);
  }

  // TODO: One should check the redefinition of a table (line 1680)
  IL_UNUSED(inserted);

  string.removePrefix(1);
  string = il::removeWhitespaceLeft(string);
  if (!string.isEmpty() && (string[0] != '\n') && (string[0] != '#')) {
    il::abort();
  }
  status.SetOk();
}

void TomlParser::parseTableArray(il::io_t, il::StringView& string,
                                 il::MapArray<il::String, il::Dynamic>*& toml,
                                 il::Status& status) {
  string.removePrefix(1);
  if (string.isEmpty() || string[0] == ']') {
    status.SetError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return;
  }

  il::String full_table_name{};
  while (!string.isEmpty() && (string[0] != ']')) {
    il::Status parse_status{};
    il::String table_name = parseKey('@', il::io, string, parse_status);
    if (!parse_status.Ok()) {
      status = std::move(parse_status);
      return;
    }
    if (table_name.isEmpty()) {
      status.SetError(il::Error::ParseTable);
      IL_SET_SOURCE(status);
      status.SetInfo("line", line_number_);
      return;
    }
    if (!full_table_name.isEmpty()) {
      full_table_name.Append('.');
    }
    full_table_name.Append(table_name);

    string = il::removeWhitespaceLeft(string);
    il::spot_t i = toml->search(table_name);
    if (toml->found(i)) {
      il::Dynamic& b = toml->Value(i);
      if (!string.isEmpty() && string[0] == ']') {
        if (!b.is<il::Array<il::Dynamic>>()) {
          status.SetError(il::Error::ParseTable);
          IL_SET_SOURCE(status);
          status.SetInfo("line", line_number_);
          return;
        }
        il::Array<il::Dynamic>& v = b.As<il::Array<il::Dynamic>>();
        v.Append(il::Dynamic{il::MapArray<il::String, il::Dynamic>{}});
        toml = &(v.Back().As<il::MapArray<il::String, il::Dynamic>>());
      }
    } else {
      if (!string.isEmpty() && string[0] == ']') {
        toml->Set(table_name, il::Dynamic{il::Array<il::Dynamic>{}}, il::io, i);
        toml->Value(i).As<il::Array<il::Dynamic>>().Append(
            il::Dynamic{il::MapArray<il::String, il::Dynamic>{}});
        toml = &(toml->Value(i)
                     .As<il::Array<il::Dynamic>>()[0]
                     .As<il::MapArray<il::String, il::Dynamic>>());
      } else {
        toml->Set(table_name,
                  il::Dynamic{il::MapArray<il::String, il::Dynamic>{}}, il::io,
                  i);
        toml = &(toml->Value(i).As<il::MapArray<il::String, il::Dynamic>>());
      }
    }
  }

  if (string.isEmpty()) {
    status.SetError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return;
  }
  string.removePrefix(1);
  if (string.isEmpty()) {
    status.SetError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.SetInfo("line", line_number_);
    return;
  }
  string.removePrefix(1);

  string = il::removeWhitespaceLeft(string);
  il::Status parse_status{};
  checkEndOfLineOrComment(string, il::io, parse_status);
  if (!parse_status.Ok()) {
    status = std::move(parse_status);
    return;
  }

  status.SetOk();
  return;
}

il::MapArray<il::String, il::Dynamic> TomlParser::parse(
    const il::String& filename, il::io_t, il::Status& status) {
  il::MapArray<il::String, il::Dynamic> root_toml{};
  il::MapArray<il::String, il::Dynamic>* pointer_toml = &root_toml;

#ifdef IL_UNIX
  file_ = std::fopen(filename.asCString(), "r+b");
#else
  il::UTF16String filename_utf16 = il::toUtf16(filename);
  file_ = _wfopen(filename_utf16.asWString(), L"r+b");
#endif
  if (!file_) {
    status.SetError(il::Error::FilesystemFileNotFound);
    IL_SET_SOURCE(status);
    return root_toml;
  }

  line_number_ = 0;
  while (std::fgets(buffer_line_, max_line_length_ + 1, file_) != nullptr) {
    ++line_number_;

    il::StringView line{il::StringType::Byte, buffer_line_,
                        il::size(buffer_line_)};
    line = il::removeWhitespaceLeft(line);

    if (line.isEmpty() || line.startsWithNewLine() || line[0] == '#') {
      continue;
    } else if (line[0] == '[') {
      pointer_toml = &root_toml;
      il::Status parse_status{};
      parseTable(il::io, line, pointer_toml, parse_status);
      if (!parse_status.Ok()) {
        status = std::move(parse_status);
        std::fclose(file_);
        return root_toml;
      }
    } else {
      il::Status parse_status{};
      parseKeyValue(il::io, line, *pointer_toml, parse_status);
      if (!parse_status.Ok()) {
        status = std::move(parse_status);
        std::fclose(file_);
        return root_toml;
      }

      line = il::removeWhitespaceLeft(line);
      checkEndOfLineOrComment(line, il::io, parse_status);
      if (!parse_status.Ok()) {
        status = std::move(parse_status);
        return root_toml;
      }
    }
  }

  const int error = std::fclose(file_);
  if (error != 0) {
    status.SetError(il::Error::FilesystemCanNotCloseFile);
    IL_SET_SOURCE(status);
    return root_toml;
  }

  status.SetOk();
  return root_toml;
}
}  // namespace il
