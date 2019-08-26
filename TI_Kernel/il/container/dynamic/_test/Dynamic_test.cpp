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

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <il/Dynamic.h>

TEST(Dynamic, implementation) {
  ASSERT_TRUE(sizeof(double) == 8 && sizeof(void*) == 8 &&
              sizeof(unsigned short) == 2 && sizeof(int) == 4 &&
              sizeof(long long) == 8 && sizeof(il::int_t) == 8);
}

TEST(Dynamic, default_constructor) {
  il::Dynamic a{};

  const bool ans = a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Void;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, bool_constructor_0) {
  il::Dynamic a = true;

  const bool ans = a.is<bool>() && a.to<bool>() && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, bool_constructor_1) {
  il::Dynamic a = false;

  const bool ans = a.is<bool>() && a.to<bool>() == false && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, integer_constructor_0) {
  il::Dynamic a = 3;

  const bool ans = a.is<il::int_t>() && a.to<il::int_t>() == 3 &&
                   !a.is<void>() && !a.is<bool>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Integer;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, integer_constructor_1) {
  il::Dynamic a = -3;

  const bool ans = a.is<il::int_t>() && a.to<il::int_t>() == -3 &&
                   !a.is<void>() && !a.is<bool>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Integer;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, integer_constructor_2) {
  const il::int_t n = (il::int_t{1} << 47) - 1;
  il::Dynamic a = n;

  const bool ans = a.is<il::int_t>() && a.to<il::int_t>() == n &&
                   !a.is<void>() && !a.is<bool>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Integer;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, integer_constructor_3) {
  const il::int_t n = -(il::int_t{1} << 47);
  il::Dynamic a = n;

  const bool ans = a.is<il::int_t>() && a.to<il::int_t>() == n &&
                   !a.is<void>() && !a.is<bool>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Integer;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, double_constructor_0) {
  const double x = 3.14159;
  il::Dynamic a = x;

  bool b0 = a.is<double>();
  IL_UNUSED(b0);

  const bool ans = a.is<double>() && a.to<double>() == x && !a.is<void>() &&
                   !a.is<bool>() && !a.is<il::int_t>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, double_constructor_1) {
  const double x = 0.0 / 0.0;
  il::Dynamic a = x;

  const bool ans = a.is<double>() && std::isnan(a.to<double>()) &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, double_constructor_2) {
  const double x = std::numeric_limits<double>::quiet_NaN();
  il::Dynamic a = x;

  const bool ans = a.is<double>() && std::isnan(a.to<double>()) &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, double_constructor_3) {
  const double x = std::numeric_limits<double>::signaling_NaN();
  il::Dynamic a = x;

  const bool ans = a.is<double>() && std::isnan(a.to<double>()) &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, double_constructor_4) {
  double x = std::numeric_limits<double>::signaling_NaN();
  x += 1.0;
  il::Dynamic a = x;

  const bool ans = a.is<double>() && std::isnan(a.to<double>()) &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, double_constructor_5) {
  const double x = std::numeric_limits<double>::infinity();
  il::Dynamic a = x;

  const bool ans = a.is<double>() && a.to<double>() == x && !a.is<void>() &&
                   !a.is<bool>() && !a.is<il::int_t>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, double_constructor_6) {
  const double x = -std::numeric_limits<double>::infinity();
  il::Dynamic a = x;

  const bool ans = a.is<double>() && a.to<double>() == x && !a.is<void>() &&
                   !a.is<bool>() && !a.is<il::int_t>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, string_constructor_0) {
  const il::String string{};
  il::Dynamic a = string;

  const bool ans = a.is<il::String>() && a.as<il::String>() == string &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::UnicodeString;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, string_constructor_1) {
  const il::String string = "Hello";
  il::Dynamic a = string;

  const bool ans = a.is<il::String>() && a.as<il::String>() == string &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::UnicodeString;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, string_constructor_2) {
  const char* c_string = "Hello";
  const il::String string{il::StringType::Ascii, c_string, il::size(c_string)};
  il::Dynamic a{il::StringType::Ascii, c_string, il::size(c_string)};

  const bool ans = a.is<il::String>() && a.as<il::String>() == string &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::UnicodeString;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_constructor_null) {
  il::Dynamic b{};
  il::Dynamic a{b};

  const bool ans = a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Void;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_constructor_bool_0) {
  il::Dynamic b = true;
  il::Dynamic a{b};

  const bool ans = a.is<bool>() && a.to<bool>() && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_constructor_bool_1) {
  il::Dynamic b = false;
  il::Dynamic a{b};

  const bool ans = a.is<bool>() && a.to<bool>() == false && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_constructor_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a{b};

  const bool ans = a.is<il::int_t>() && a.to<il::int_t>() == 3 &&
                   !a.is<void>() && !a.is<bool>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Integer;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_constructor_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a{b};

  const bool ans = a.is<double>() && a.to<double>() == x && !a.is<void>() &&
                   !a.is<bool>() && !a.is<il::int_t>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_constructor_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a{b};

  const bool ans = a.is<il::String>() && a.as<il::String>() == string &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::UnicodeString;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_constructor_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a{b};

  const bool ans = a.is<il::Array<il::Dynamic>>() &&
                   a.as<il::Array<il::Dynamic>>().size() == 3 &&
                   a.as<il::Array<il::Dynamic>>().capacity() == 3 &&
                   a.as<il::Array<il::Dynamic>>()[0].is<void>() &&
                   a.as<il::Array<il::Dynamic>>()[1].is<void>() &&
                   a.as<il::Array<il::Dynamic>>()[2].is<void>() &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::String>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::ArrayOfDynamic;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_constructor_hashmaparray_0) {
  il::MapArray<il::String, il::Dynamic> map{};
  map.Set(il::String{"Hello"}, il::Dynamic{5});
  map.Set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a{b};

  const bool ans =
      a.is<il::MapArray<il::String, il::Dynamic>>() &&
      a.as<il::MapArray<il::String, il::Dynamic>>().size() == 2 &&
      a.as<il::MapArray<il::String, il::Dynamic>>().search("Hello").isValid() &&
      a.as<il::MapArray<il::String, il::Dynamic>>()
          .search("World!")
          .isValid() &&
      !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() && !a.is<double>() &&
      !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
      a.type() == il::Type::MapArrayStringToDynamic;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_constructor_null) {
  il::Dynamic b{};
  il::Dynamic a = std::move(b);

  const bool ans = a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Void && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_constructor_bool_0) {
  il::Dynamic b = true;
  il::Dynamic a = std::move(b);

  const bool ans = a.is<bool>() && a.to<bool>() && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_constructor_bool_1) {
  il::Dynamic b = false;
  il::Dynamic a = std::move(b);

  const bool ans = a.is<bool>() && a.to<bool>() == false && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_constructor_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a = std::move(b);

  const bool ans = a.is<il::int_t>() && a.to<il::int_t>() == 3 &&
                   !a.is<void>() && !a.is<bool>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Integer && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_constructor_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a = std::move(b);

  const bool ans = a.is<double>() && a.to<double>() == x && !a.is<void>() &&
                   !a.is<bool>() && !a.is<il::int_t>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_constructor_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a = std::move(b);

  const bool ans = a.is<il::String>() && a.as<il::String>() == string &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::UnicodeString && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_constructor_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a = std::move(b);

  const bool ans = a.is<il::Array<il::Dynamic>>() &&
                   a.as<il::Array<il::Dynamic>>().size() == 3 &&
                   a.as<il::Array<il::Dynamic>>().capacity() == 3 &&
                   a.as<il::Array<il::Dynamic>>()[0].is<void>() &&
                   a.as<il::Array<il::Dynamic>>()[1].is<void>() &&
                   a.as<il::Array<il::Dynamic>>()[2].is<void>() &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::String>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::ArrayOfDynamic && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_constructor_hashmaparray_0) {
  il::MapArray<il::String, il::Dynamic> map{};
  map.Set(il::String{"Hello"}, il::Dynamic{5});
  map.Set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a = std::move(b);

  const bool ans =
      a.is<il::MapArray<il::String, il::Dynamic>>() &&
      a.as<il::MapArray<il::String, il::Dynamic>>().size() == 2 &&
      a.as<il::MapArray<il::String, il::Dynamic>>().search("Hello").isValid() &&
      a.as<il::MapArray<il::String, il::Dynamic>>()
          .search("World!")
          .isValid() &&
      !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() && !a.is<double>() &&
      !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
      a.type() == il::Type::MapArrayStringToDynamic && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_assignement_null) {
  il::Dynamic b{};
  il::Dynamic a{};
  a = b;

  const bool ans = a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Void;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_assignement_bool_0) {
  il::Dynamic b = true;
  il::Dynamic a{};
  a = b;

  const bool ans = a.is<bool>() && a.to<bool>() && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_assignement_bool_1) {
  il::Dynamic b = false;
  il::Dynamic a{};
  a = b;

  const bool ans = a.is<bool>() && a.to<bool>() == false && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_assignement_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a{};
  a = b;

  const bool ans = a.is<il::int_t>() && a.to<il::int_t>() == 3 &&
                   !a.is<void>() && !a.is<bool>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Integer;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_assignement_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a{};
  a = b;

  const bool ans = a.is<double>() && a.to<double>() == x && !a.is<void>() &&
                   !a.is<bool>() && !a.is<il::int_t>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_assignement_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a{};
  a = b;

  const bool ans = a.is<il::String>() && a.as<il::String>() == string &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::UnicodeString;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_assignement_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a{};
  a = b;

  const bool ans = a.is<il::Array<il::Dynamic>>() &&
                   a.as<il::Array<il::Dynamic>>().size() == 3 &&
                   a.as<il::Array<il::Dynamic>>().capacity() == 3 &&
                   a.as<il::Array<il::Dynamic>>()[0].is<void>() &&
                   a.as<il::Array<il::Dynamic>>()[1].is<void>() &&
                   a.as<il::Array<il::Dynamic>>()[2].is<void>() &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::String>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::ArrayOfDynamic;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, copy_assignement_hashmaparray_0) {
  il::MapArray<il::String, il::Dynamic> map{};
  map.Set(il::String{"Hello"}, il::Dynamic{5});
  map.Set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a{};
  a = b;

  const bool ans =
      a.is<il::MapArray<il::String, il::Dynamic>>() &&
      a.as<il::MapArray<il::String, il::Dynamic>>().size() == 2 &&
      a.as<il::MapArray<il::String, il::Dynamic>>().search("Hello").isValid() &&
      a.as<il::MapArray<il::String, il::Dynamic>>()
          .search("World!")
          .isValid() &&
      !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() && !a.is<double>() &&
      !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
      a.type() == il::Type::MapArrayStringToDynamic;
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_assignement_null) {
  il::Dynamic b{};
  il::Dynamic a{};
  a = std::move(b);

  const bool ans = a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Void && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_assignement_bool_0) {
  il::Dynamic b = true;
  il::Dynamic a{};
  a = std::move(b);

  const bool ans = a.is<bool>() && a.to<bool>() && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_assignement_bool_1) {
  il::Dynamic b = false;
  il::Dynamic a{};
  a = std::move(b);

  const bool ans = a.is<bool>() && a.to<bool>() == false && !a.is<void>() &&
                   !a.is<il::int_t>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Bool && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_assignement_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a{};
  a = std::move(b);

  const bool ans = a.is<il::int_t>() && a.to<il::int_t>() == 3 &&
                   !a.is<void>() && !a.is<bool>() && !a.is<double>() &&
                   !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Integer && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_assignement_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a{};
  a = std::move(b);

  const bool ans = a.is<double>() && a.to<double>() == x && !a.is<void>() &&
                   !a.is<bool>() && !a.is<il::int_t>() && !a.is<il::String>() &&
                   !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::Double && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_assignement_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a{};
  a = std::move(b);

  const bool ans = a.is<il::String>() && a.as<il::String>() == string &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::Array<il::Dynamic>>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::UnicodeString && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_assignement_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a{};
  a = std::move(b);

  const bool ans = a.is<il::Array<il::Dynamic>>() &&
                   a.as<il::Array<il::Dynamic>>().size() == 3 &&
                   a.as<il::Array<il::Dynamic>>().capacity() == 3 &&
                   a.as<il::Array<il::Dynamic>>()[0].is<void>() &&
                   a.as<il::Array<il::Dynamic>>()[1].is<void>() &&
                   a.as<il::Array<il::Dynamic>>()[2].is<void>() &&
                   !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() &&
                   !a.is<double>() && !a.is<il::String>() &&
                   !a.is<il::MapArray<il::String, il::Dynamic>>() &&
                   a.type() == il::Type::ArrayOfDynamic && b.is<void>();
  ASSERT_TRUE(ans);
}

TEST(Dynamic, move_assignement_hashmaparray_0) {
  il::MapArray<il::String, il::Dynamic> map{};
  map.Set(il::String{"Hello"}, il::Dynamic{5});
  map.Set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a{};
  a = std::move(b);

  const bool ans =
      a.is<il::MapArray<il::String, il::Dynamic>>() &&
      a.as<il::MapArray<il::String, il::Dynamic>>().size() == 2 &&
      a.as<il::MapArray<il::String, il::Dynamic>>().search("Hello").isValid() &&
      a.as<il::MapArray<il::String, il::Dynamic>>()
          .search("World!")
          .isValid() &&
      !a.is<void>() && !a.is<bool>() && !a.is<il::int_t>() && !a.is<double>() &&
      !a.is<il::String>() && !a.is<il::Array<il::Dynamic>>() &&
      a.type() == il::Type::MapArrayStringToDynamic && b.is<void>();
  ASSERT_TRUE(ans);
}
