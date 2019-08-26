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

#ifndef IL_UNICODE_H
#define IL_UNICODE_H

#include <il/container/string/String.h>
#include <il/core.h>
#include <iostream>
//#include <il/container/string/StringView.h>
#include <il/container/string/UTF16String.h>

namespace il {

// inline constexpr bool auxIsUtf8(const char* s, int nbBytes, int pos,
//                                bool surrogate) {
//  return (pos == 0)
//             // If it starts with the null character, the string is valid
//             ? ((*s == 0x00)
//                    ? true
//                    // Otherwise, let's check if is starts with an ASCII
//                    // character
//                    : ((*s & 0x80) == 0
//                           // In this case, check the rest of the string
//                           ? auxIsUtf8(s + 1, 0, 0, false)
//                           // Otherwise, it might start with a 2-byte sequence
//                           : ((*s & 0xD0) == 0xB0
//                                  // In this case, check the rest of the
//                                  string ? auxIsUtf8(s + 1, 2, 1, false)
//                                  // Otherwise, it might start with a 3-byte
//                                  // sequence
//                                  : ((*s & 0xF0) == 0xD0
//                                         // In this case, check the rest of
//                                         the
//                                         // string
//                                         ? auxIsUtf8(
//                                               s + 1, 3, 1,
//                                               (static_cast<unsigned char>(
//                                                    *s) == 0xED))
//                                         // Otherwise, it might start with a
//                                         // 4-byte sequence
//                                         : ((*s & 0xF8) == 0xF0
//                                                ? auxIsUtf8(s + 1, 4, 1,
//                                                false) : false)))))
//             // In the case where we are scanning the second byte of a
//             multibyte
//             // sequence
//             : ((pos == 1)
//                    ? ((*s & 0xC0) == 0x80
//                           ? (nbBytes == 2 ? ((*s & 0xA0) != 0xA0)
//                                           : auxIsUtf8(s + 1, nbBytes, pos +
//                                           1,
//                                                       surrogate))
//                           : false)
//                    // In the case where we are scanning the third byte of a
//                    // multibyte sequence
//                    : ((pos == 2)
//                           ? ((*s & 0xC0) == 0x80
//                                  ? (nbBytes == 3
//                                         ? true
//                                         : auxIsUtf8(s + 1, nbBytes, pos + 1,
//                                                     surrogate))
//                                  : false)
//                           // In the case where we are scanning the
//                           // fourth byte of a multibyte sequence
//                           : ((pos == 3) ? ((*s & 0xC0) == 0x80) : false)));
//}

// inline constexpr bool isUtf8(const char* s) {
//  return auxIsUtf8(s, 0, 0, false);
//}

inline il::UTF16String toUtf16(const il::String& string) {
  il::UTF16String ans{};
  il::StringView view{string.type(), string.data(), string.size()};

  for (il::int_t i = 0; i < view.size(); i = view.nextRune(i)) {
    ans.Append(view.rune(i));
  }
  ans.Append('\0');
  return ans;
}

inline il::String toUtf8(const il::UTF16String& string) {
  il::String ans{};

  for (il::int_t i = 0; i < string.size(); i = string.nextRune(i)) {
    ans.Append(string.rune(i));
  }
  ans.Append('\0');
  return ans;
}

enum Rune : int {
  Snowman = 0x2603,
  MahjongTileRedDragon = 0x0001F004,
  GrinningFace = 0x0001F600,
  GrinningFaceWithSmilingEyes = 0x0001F601,
  SmilingFaceWithHorns = 0x0001F608,
  ForAll = 0x2200,
  PartialDifferential = 0x2202,
  ThereExists = 0x2203,
  ThereDoesNotExist = 0x2204,
  EmptySet = 0x2205,
  Nabla = 0x2207,
  ElementOf = 0x2208,
  NotAnElementOf = 0x2209,
  RingOperator = 0x2218,
  Infinity = 0x221E,
  DoublestruckCapital_r = 0x211D,
  GreekSmallLetterAlpha = 0x03B1,
  GreekSmallLetterBeta = 0x03B2,
  GreekSmallLetterGamma = 0x03B3,
  GreekSmallLetterDelta = 0x03B4,
  GreekSmallLetterEpsilon = 0x03B5,
  GreekSmallLetterZeta = 0x03B6,
  GreekSmallLetterEta = 0x03B7,
  GreekSmallLetterTheta = 0x03B8,
  GreekSmallLetterIota = 0x03B9,
  GreekSmallLetterKappa = 0x03BA,
  reekSmallLetterLambda = 0x03BB,
  GreekSmallLetterMu = 0x03BC,
  GreekSmallLetterNu = 0x03BD,
  GreekSmallLetterXi = 0x03BE
};

}  // namespace il

#endif  // IL_UNICODE_H
