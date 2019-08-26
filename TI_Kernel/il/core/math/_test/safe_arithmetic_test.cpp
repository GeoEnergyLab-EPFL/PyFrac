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

#include <il/core/math/safe_arithmetic.h>

TEST(safe_arithmetic, sum_int_0) {
  const int a = std::numeric_limits<int>::max();
  const int b = 1;
  bool error = false;

  const int sum = il::safeSum(a, b, il::io, error);
  IL_UNUSED(sum);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, sum_int_1) {
  const int a = std::numeric_limits<int>::min();
  const int b = -1;
  bool error = false;

  const int sum = il::safeSum(a, b, il::io, error);
  IL_UNUSED(sum);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, sum_int_2) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int sum = il::safeSum(a, b, il::io, error);
  IL_UNUSED(sum);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, sum_int_3) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int sum = il::safeSum(a, b, il::io, error);
  IL_UNUSED(sum);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_0) {
  const int a = std::numeric_limits<int>::max();
  const int b = -1;
  bool error = false;

  const int difference = il::safeDifference(a, b, il::io, error);
  IL_UNUSED(difference);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_1) {
  const int a = std::numeric_limits<int>::min();
  const int b = 1;
  bool error = false;

  const int difference = il::safeDifference(a, b, il::io, error);
  IL_UNUSED(difference);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_2) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::min();
  bool error = false;

  const int difference = il::safeDifference(a, b, il::io, error);
  IL_UNUSED(difference);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_3) {
  const int a = std::numeric_limits<int>::min();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int difference = il::safeDifference(a, b, il::io, error);
  IL_UNUSED(difference);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, product_int_0) {
  const int a = std::numeric_limits<int>::max() / 2 + 1;
  const int b = 2;
  bool error = false;

  const int product = il::safeProduct(a, b, il::io, error);
  IL_UNUSED(product);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, product_int_1) {
  const int a = std::numeric_limits<int>::min() / 2 - 1;
  const int b = 2;
  bool error = false;

  const int product = il::safeProduct(a, b, il::io, error);
  IL_UNUSED(product);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, product_int_2) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int product = il::safeProduct(a, b, il::io, error);
  IL_UNUSED(product);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, product_int_3) {
  const int a = std::numeric_limits<int>::min();
  const int b = std::numeric_limits<int>::min();
  bool error = false;

  const int product = il::safeProduct(a, b, il::io, error);
  IL_UNUSED(product);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, division_int_0) {
  const int a = 1;
  const int b = 0;
  bool error = false;

  const int quotient = il::safeDivision(a, b, il::io, error);
  IL_UNUSED(quotient);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, division_int_1) {
  const int a = std::numeric_limits<int>::min();
  const int b = -1;
  bool error = false;

  const int quotient = il::safeDivision(a, b, il::io, error);
  IL_UNUSED(quotient);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, safeConvert_0) {
  std::size_t a = std::numeric_limits<il::int_t>::max();
  a = a + 1;

  bool error = false;
  const il::int_t b = il::safeConvert<il::int_t>(a, il::io, error);
  IL_UNUSED(b);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, safeConvert_1) {
  const il::int_t a = -1;

  bool error = false;
  const std::size_t b = il::safeConvert<std::size_t>(a, il::io, error);
  IL_UNUSED(b);

  ASSERT_TRUE(error);
}
