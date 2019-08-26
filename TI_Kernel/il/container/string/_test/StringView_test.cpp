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

#include <gtest/gtest.h>

#include <il/String.h>

TEST(StringView, Rune_0) {
  il::String s = "abc";
  il::StringView sv = s;

  const il::int_t i0 = 0;
  const il::int_t i1 = sv.nextRune(i0);
  const il::int_t i2 = sv.nextRune(i1);
  const il::int_t i3 = sv.nextRune(i2);

  ASSERT_TRUE(s.size() == 3 && sv.rune(i0) == 97 && sv.rune(i1) == 98 &&
              sv.rune(i2) == 99 && i3 == 3);
}

TEST(StringView, Rune_1) {
  il::String s = u8"nço";
  il::StringView sv = s;

  const il::int_t i0 = 0;
  const il::int_t i1 = sv.nextRune(i0);
  const il::int_t i2 = sv.nextRune(i1);
  const il::int_t i3 = sv.nextRune(i2);

  ASSERT_TRUE(s.size() == 4 && sv.rune(i0) == 110 && sv.rune(i1) == 231 &&
              sv.rune(i2) == 111 && i3 == 4);
}
