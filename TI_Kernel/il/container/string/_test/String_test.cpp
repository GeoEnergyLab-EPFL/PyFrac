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

TEST(String, default_constructor) {
  il::String s{};
  const char* p = s.asCString();

  ASSERT_TRUE(s.size() == 0 && s.isSmall() && s.capacity() == 22 &&
              p[0] == '\0');
}

TEST(String, c_string_constructor_0) {
  il::String s = "A quite small string !";
  const char* p = s.asCString();

  ASSERT_TRUE(s.size() == 22 && s.isSmall() && s.capacity() == 22 &&
              0 == std::strcmp(p, "A quite small string !"));
}

TEST(String, c_string_constructor_1) {
  il::String s = "A quite large string !!";
  const char* p = s.asCString();

  ASSERT_TRUE(s.size() == 23 && !s.isSmall() &&
              0 == std::strcmp(p, "A quite large string !!"));
}

TEST(String, c_string_constructor_2) {
  il::String s = "A quite\0 large string !!!";
  const char* p = s.asCString();

  ASSERT_TRUE(s.size() == 25 && !s.isSmall() &&
              0 == std::strcmp(p, "A quite\0 large string !!!"));
}

TEST(String, reserve_0) {
  il::String s{"A quite small string!"};

  s.Reserve(22);
  ASSERT_TRUE(s.size() == 21 && s.isSmall() && s.capacity() == 22 &&
              0 == std::strcmp(s.asCString(), "A quite small string!"));
}

TEST(String, reserve_1) {
  il::String s{"A quite small string !"};

  s.Reserve(24);
  ASSERT_TRUE(s.size() == 22 && !s.isSmall() && s.capacity() >= 24 &&
              0 == std::strcmp(s.asCString(), "A quite small string !"));
}

TEST(String, reserve_2) {
  il::String s{"A quite large string !!"};

  s.Reserve(30);
  ASSERT_TRUE(s.size() == 23 && !s.isSmall() && s.capacity() >= 30 &&
              0 == std::strcmp(s.asCString(), "A quite large string !!"));
}

TEST(String, append_0) {
  il::String s{"Hello"};
  s.Append(" world!");

  ASSERT_TRUE(s.size() == 12 && s.isSmall() && s.capacity() == 22 &&
              0 == std::strcmp(s.asCString(), "Hello world!"));
}

TEST(String, append_1) {
  il::String s = "Hello";
  s.Append(" world! I am so happy to be there");

  ASSERT_TRUE(s.size() == 38 && !s.isSmall() && s.capacity() >= 38 &&
              0 == std::strcmp(s.asCString(),
                               "Hello world! I am so happy to be there"));
}

TEST(String, append_2) {
  il::String s{"Hello"};
  s.Reserve(38);
  const char* p_before = s.asCString();
  s.Append(" world! I am so happy to be there");
  const char* p_after = s.asCString();

  ASSERT_TRUE(s.size() == 38 && !s.isSmall() && s.capacity() >= 38 &&
              p_before == p_after &&
              0 == std::strcmp(s.asCString(),
                               "Hello world! I am so happy to be there"));
}

TEST(String, append_3) {
  il::String s{"Hello world! I am so happy to be "};
  s.Append("there");

  ASSERT_TRUE(s.size() == 38 && !s.isSmall() && s.capacity() >= 38 &&
              0 == std::strcmp(s.asCString(),
                               "Hello world! I am so happy to be there"));
}

// TEST(String, append_4) {
//  il::String s = "HelloHelloHelloHello";
//  s.Append(s);
//
//  ASSERT_TRUE(s.size() == 40 &&
//              s == "HelloHelloHelloHelloHelloHelloHelloHello");
//}
