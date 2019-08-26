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

#ifndef IL_ALGORITHM_STRING_H
#define IL_ALGORITHM_STRING_H

#include <cstdio>

#include <il/String.h>

namespace il {

inline il::String toString(il::int_t n) {
  il::String ans{il::unsafe, 11};
  const il::int_t m = std::snprintf(ans.Data(), 11 + 1, "%td", n);
  ans.SetInvariant(il::unsafe, il::StringType::Ascii, m);

  IL_ENSURE(m > 0);
  return ans;
}

inline il::StringView removeWhitespaceLeft(StringView string) {
  il::int_t i = 0;
  while (i < string.size() && (string[i] == ' ' || string[i] == '\t')) {
    ++i;
  }
  string.removePrefix(i);

  return string;
}

inline il::int_t search(StringView a, StringView b) {
  const il::int_t na = a.size();
  const il::int_t nb = b.size();
  il::int_t k = 0;
  bool found = false;
  while (!found && k + na <= nb) {
    il::int_t i = 0;
    found = true;
    while (found && i < na) {
      if (a[i] != b[k + i]) {
        found = false;
      }
      ++i;
    }
    if (found) {
      return k;
    }
    ++k;
  }
  return -1;
}

inline il::StringView view(const char* s) {
  il::int_t i = 0;
  while (s[i] != '\0') {
    ++i;
  }
  return il::StringView{il::StringType::Byte, s, i};
}

inline il::int_t search(const char* a, StringView b) {
  return il::search(il::view(a), b);
}

inline il::int_t search(const String& a, const String& b) {
  return il::search(a.view(), b.view());
}

template <il::int_t m>
inline il::int_t search(const char (&s)[m], const String& b) {
  return il::search(StringView{s}, b.view());
}

inline il::int_t count(char c, StringView a) {
  IL_EXPECT_FAST(static_cast<unsigned char>(c) < 128);

  il::int_t ans = 0;
  for (il::int_t i = 0; i < a.size(); ++i) {
    if (a[i] == c) {
      ++ans;
    }
  }
  return ans;
}

}  // namespace il

#endif  // IL_ALGORITHM_STRING_H
