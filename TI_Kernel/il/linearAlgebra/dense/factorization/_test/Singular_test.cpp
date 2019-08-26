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

#ifdef IL_BLAS
#include <il/algorithmArray.h>
#include <il/linearAlgebra/dense/blas/dot.h>
#include <il/linearAlgebra/dense/blas/blas_static.h>
#include <il/linearAlgebra/dense/factorization/Singular.h>

il::StaticArray2D<double, 3, 3> rotation1(double theta) {
  return il::StaticArray2D<double, 3, 3>{
      il::value,
      {{1.0, 0.0, 0.0},
       {0.0, std::cos(theta), std::sin(theta)},
       {0.0, -std::sin(theta), std::cos(theta)}}};
};

il::StaticArray2D<double, 3, 3> rotation2(double theta) {
  return il::StaticArray2D<double, 3, 3>{
      il::value,
      {{std::cos(theta), 0.0, std::sin(theta)},
       {0.0, 1.0, 0.0},
       {-std::sin(theta), 0.0, std::cos(theta)}}};
};

il::StaticArray2D<double, 3, 3> rotation3(double theta) {
  return il::StaticArray2D<double, 3, 3>{
      il::value,
      {{std::cos(theta), std::sin(theta), 0.0},
       {-std::sin(theta), std::cos(theta), 0.0},
       {0.0, 0.0, 1.0}}};
};

TEST(Singular, test0) {
  il::StaticArray2D<double, 3, 3> A{
      il::value, {{3.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 7.0}}};

  il::Status status{};
  il::StaticArray<double, 3> singular_value =
      il::singularValues(A, il::io, status);
  status.AbortOnError();
  il::sort(il::io, singular_value);

  const double epsilon = 1.0e-15;

  ASSERT_TRUE(il::abs(singular_value[0] - 3.0) <= epsilon &&
              il::abs(singular_value[1] - 5.0) <= epsilon &&
              il::abs(singular_value[2] - 7.0) <= epsilon);
}

TEST(Singular, test1) {
  il::StaticArray2D<double, 3, 3> A{
      il::value, {{3.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 7.0}}};
  const double theta = 1.0;
  A = il::dot(A, rotation1(theta));
  A = il::dot(rotation2(2 * theta), A);
  A = il::dot(A, rotation3(3 * theta));

  il::Status status{};
  il::StaticArray<double, 3> singular_value =
      il::singularValues(A, il::io, status);
  status.AbortOnError();
  il::sort(il::io, singular_value);

  const double epsilon = 1.0e-15;

  ASSERT_TRUE(il::abs(singular_value[0] - 3.0) <= epsilon &&
              il::abs(singular_value[1] - 5.0) <= epsilon &&
              il::abs(singular_value[2] - 7.0) <= epsilon);
}

#endif  // IL_BLAS
