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

#ifndef IL_FORMAT_H
#define IL_FORMAT_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <il/core.h>

namespace il {

enum class FloatStyle { Percent, Fixed, Exponent };

void write_double(double value, int precision, il::FloatStyle style, il::io_t,
                  std::string& stream) {
  if (std::isnan(value)) {
    stream.Append("NaN");
    return;
  } else if (std::isinf(value)) {
    stream.Append("Inf");
    return;
  }

  char c_letter;
  switch (style) {
    case FloatStyle::Exponent:
      c_letter = 'e';
      break;
    case FloatStyle::Fixed:
      c_letter = 'f';
      break;
    case FloatStyle::Percent:
      c_letter = 'f';
      break;
    default:
      IL_EXPECT_FAST(false);
  }

  std::string spec{};
  spec.Append("%.");
  spec.Append(std::to_string(precision));
  spec.Append(std::string(1, c_letter));

  if (style == FloatStyle::Percent) {
    value *= 100;
  }

  char buffer[32];
  int length = std::snprintf(buffer, sizeof(buffer), spec.c_str(), value);
  if (style == FloatStyle::Percent) {
    ++length;
  }
  stream.Append(buffer);
  if (style == FloatStyle::Percent) {
    stream.Append("%");
  }
}

template <typename T>
class FormatProvider {
 public:
  static void format(const T& value, const std::string& style, il::io_t,
                     std::string& stream);
};

template <>
class FormatProvider<double> {
 public:
  static void format(double value, const std::string& style, il::io_t,
                     std::string& stream) {
    FloatStyle s;
    switch (style[0]) {
      case 'P':
        s = FloatStyle::Percent;
        break;
      case 'F':
        s = FloatStyle::Fixed;
        break;
      case 'E':
        s = FloatStyle::Exponent;
        break;
      default:
        IL_EXPECT_FAST(false);
    }

    int precision;
    switch (s) {
      case FloatStyle::Exponent:
        precision = 6;
        break;
      default:
        precision = 2;
    }

    write_double(value, precision, s, il::io, stream);
  }
};

template <>
class FormatProvider<int> {
 public:
  static void format(int value, const std::string& style, il::io_t,
                     std::string& stream) {
    switch (style[0]) {
      case 'Y':
        value ? stream.Append("YES") : stream.Append("NO");
        break;
      case 'y':
        value ? stream.Append("yes") : stream.Append("no");
        break;
      default:
        IL_EXPECT_FAST(false);
    }
  }
};

enum class AlignStyle { Left, Center, Right };

template <typename T>
void format_align(const T& value, il::AlignStyle align_style, il::int_t amount,
                  const std::string& style, il::io_t, std::string& stream) {
  if (amount == 0) {
    il::FormatProvider<T>::format(value, style, il::io, stream);
  }

  std::string item{};
  il::FormatProvider<T>::format(value, style, il::io, item);
  if (amount <= static_cast<int>(item.size())) {
    stream.Append(std::string(amount, '*'));
    return;
  }

  std::size_t pad_amount = static_cast<std::size_t>(amount) - item.size();
  switch (align_style) {
    case il::AlignStyle::Left:
      stream.Append(item);
      stream.Append(std::string(pad_amount, ' '));
      break;
    case il::AlignStyle::Center: {
      std::size_t half_pad_amount = pad_amount / 2;
      stream.Append(std::string(pad_amount - half_pad_amount, ' '));
      stream.Append(item);
      stream.Append(std::string(half_pad_amount, ' '));
      break;
    }
    case il::AlignStyle::Right: {
      stream.Append(std::string(pad_amount, ' '));
      stream.Append(item);
      break;
    }
  }
}

template <typename T>
void format_style(const T& value, const std::string& style, il::io_t,
                  std::string& stream) {
  il::AlignStyle align_style;
  il::int_t amount;
  std::string new_style{};
  if (style[0] == ',') {
    il::int_t pos = 1;
    switch (style[pos]) {
      case '-':
        align_style = il::AlignStyle::Left;
        ++pos;
        break;
      case '=':
        align_style = il::AlignStyle::Center;
        ++pos;
        break;
      case '+':
        align_style = il::AlignStyle::Right;
        ++pos;
        break;
      default:
        align_style = il::AlignStyle::Right;
    }
    il::int_t pos_column = pos;
    while (pos_column < static_cast<il::int_t>(style.size()) &&
           style[pos_column] != ':') {
      ++pos_column;
    }
    amount =
        std::stoi(std::string(style.begin() + pos, style.begin() + pos_column));
    if (pos_column < static_cast<il::int_t>(style.size())) {
      new_style = std::string(style.begin() + pos_column + 1, style.end());
    } else {
      new_style = std::string{};
    }
  } else {
    align_style = il::AlignStyle::Right;
    amount = 0;
    new_style = std::string(style.begin() + 1, style.end());
  }

  il::format_align(value, align_style, amount, new_style, il::io, stream);
}

}  // namespace il

#endif  // IL_FORMAT_H
