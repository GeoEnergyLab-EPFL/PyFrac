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

#include <il/linearAlgebra/dense/blas/cross.h>
#include <il/math.h>

TEST(cross, cross2_0) {
  il::StaticArray<double, 2> x{il::value, {2.0, 1.0}};

  il::StaticArray<double, 2> y = il::cross(x);

  ASSERT_TRUE(y[0] == -1.0 && y[1] == 2.0);
}

TEST(cross, cross3_0) {
  il::StaticArray<double, 3> x{il::value, {1.0, 2.0, 3.0}};
  il::StaticArray<double, 3> y{il::value, {4.0, 5.0, 6.0}};

  il::StaticArray<double, 3> z = il::cross(x, y);

  const double error = il::max(il::abs(z[0] - (-3.0)), il::abs(z[1] - 6.0),
                               il::abs(z[2] - (-3.0)));

  ASSERT_TRUE(error <= 1.0e-15);
}
