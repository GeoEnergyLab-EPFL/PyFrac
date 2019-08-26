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

#include <il/math.h>
#include <il/linearAlgebra/dense/blas/dot.h>

#ifdef IL_BLAS
TEST(dot, vector_vector_f_0) {
  il::Array<float> x{il::value, {2.0f, 3.0f}};
  il::Array<float> y{il::value, {4.0f, 5.0f}};

  float alpha = il::dot(x, y);

  IL_EXPECT_FAST(il::abs(alpha - 23.0f) <= 1.0e-15f);
}

TEST(dot, vector_vector_f_1) {
  il::Array<float> x{il::value, {2.0f, 3.0f}};
  il::Array<float> y{il::value, {4.0f, 5.0f}};

  float alpha = il::dot(x.view(), y.view());

  IL_EXPECT_FAST(il::abs(alpha - 23.0f) <= 1.0e-15f);
}

TEST(dot, vector_vector_d_0) {
  il::Array<double> x{il::value, {2.0, 3.0}};
  il::Array<double> y{il::value, {4.0, 5.0}};

  double alpha = il::dot(x, y);

  IL_EXPECT_FAST(il::abs(alpha - 23.0) <= 1.0e-15);
}

TEST(dot, vector_vector_d_1) {
  il::Array<double> x{il::value, {2.0, 3.0}};
  il::Array<double> y{il::value, {4.0, 5.0}};

  double alpha = il::dot(x.view(), y.view());

  IL_EXPECT_FAST(il::abs(alpha - 23.0) <= 1.0e-15);
}

TEST(dot, vector_vector_cf_0) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x, y);

  IL_EXPECT_FAST(il::abs(alpha - std::complex<float>{14.875f, 24.5f}) <=
                 1.0e-15f);
}

TEST(dot, vector_vector_cf_1) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x.view(), y.view());

  IL_EXPECT_FAST(il::abs(alpha - std::complex<float>{14.875f, 24.5f}) <=
                 1.0e-15f);
}

TEST(dot, vector_vector_cf_2) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x, il::Dot::Star, y);

  IL_EXPECT_FAST(il::abs(alpha - std::complex<float>{31.125f, 0.5f}) <=
                 1.0e-15f);
}

TEST(dot, vector_vector_cf_3) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x.view(), il::Dot::Star, y.view());

  IL_EXPECT_FAST(il::abs(alpha - std::complex<float>{31.125f, 0.5f}) <=
                 1.0e-15f);
}

TEST(dot, vector_vector_cf_4) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x, y, il::Dot::Star);

  IL_EXPECT_FAST(il::abs(alpha - std::complex<float>{31.125f, -0.5f}) <=
                 1.0e-15f);
}

TEST(dot, vector_vector_cf_5) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x.view(), y.view(), il::Dot::Star);

  IL_EXPECT_FAST(il::abs(alpha - std::complex<float>{31.125f, -0.5f}) <=
                 1.0e-15f);
}

TEST(dot, vector_vector_cd_0) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x, y);

  IL_EXPECT_FAST(il::abs(alpha - std::complex<double>{14.875, 24.5}) <=
                 1.0e-15);
}

TEST(dot, vector_vector_cd_1) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x.view(), y.view());

  IL_EXPECT_FAST(il::abs(alpha - std::complex<double>{14.875, 24.5}) <=
                 1.0e-15);
}

TEST(dot, vector_vector_cd_2) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x, il::Dot::Star, y);

  IL_EXPECT_FAST(il::abs(alpha - std::complex<double>{31.125, 0.5}) <= 1.0e-15);
}

TEST(dot, vector_vector_cd_3) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x.view(), il::Dot::Star, y.view());

  IL_EXPECT_FAST(il::abs(alpha - std::complex<double>{31.125, 0.5}) <= 1.0e-15);
}

TEST(dot, vector_vector_cd_4) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x, y, il::Dot::Star);

  IL_EXPECT_FAST(il::abs(alpha - std::complex<double>{31.125, -0.5}) <=
                 1.0e-15);
}

TEST(dot, vector_vector_cd_5) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x.view(), y.view(), il::Dot::Star);

  IL_EXPECT_FAST(il::abs(alpha - std::complex<double>{31.125, -0.5}) <=
                 1.0e-15);
}

TEST(dot, matrix_vector_f_0) {
  il::Array2D<double> A{il::value, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}};
  il::Array<double> x{il::value, {1.0, 2.0, 3.0}};

  il::Array<double> y = il::dot(A, x);

  double error =
      il::max(il::abs(y[0] - 22.0) / 22.0, il::abs(y[1] - 28.0) / 28.0);

  IL_EXPECT_FAST(y.size() == 2 && error <= 1.0e-15);
}

TEST(dot, matrix_vector_c_0) {
  il::Array2C<double> A{il::value, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}};
  il::Array<double> x{il::value, {1.0, 2.0}};

  il::Array<double> y = il::dot(A, x);

  double error = il::max(il::abs(y[0] - 5.0) / 5.0, il::abs(y[1] - 11.0) / 11.0,
                         il::abs(y[2] - 17.0) / 17.0);

  IL_EXPECT_FAST(y.size() == 3 && error <= 1.0e-15);
}

TEST(dot, matrix_f_0) {
  il::Array2D<double> A{il::value, {{1.0, 2.0}, {3.0, 4.0}}};
  il::Array2D<double> B{il::value, {{5.0, 6.0}, {7.0, 8.0}}};

  il::Array2D<double> C = il::dot(A, B);

  double error =
      il::max(il::abs(C(0, 0) - 23.0) / 23.0, il::abs(C(1, 0) - 34.0) / 34.0,
              il::abs(C(0, 1) - 31.0) / 31.0, il::abs(C(1, 1) - 46.0) / 46.0);

  ASSERT_TRUE(C.size(0) == 2 && C.size(1) == 2 && error <= 1.0e-15);
}

TEST(dot, matrix_f_1) {
  il::Array2D<double> A{il::value, {{1.0}, {2.0}}};
  il::Array2D<double> B{il::value, {{3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}}};

  il::Array2D<double> C = il::dot(A, B);

  double error =
      il::max(il::abs(C(0, 0) - 11.0) / 11.0, il::abs(C(0, 1) - 17.0) / 17.0,
              il::abs(C(0, 2) - 23.0) / 23.0);

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && error <= 1.0e-15);
}

TEST(dot, matrix_f_simd_0) {
  il::Array2D<double> A{2, 2, il::align, 32};
  A(0, 0) = 1.0;
  A(1, 0) = 2.0;
  A(0, 1) = 3.0;
  A(1, 1) = 4.0;
  il::Array2D<double> B{2, 2, il::align, 32};
  B(0, 0) = 5.0;
  B(1, 0) = 6.0;
  B(0, 1) = 7.0;
  B(1, 1) = 8.0;

  il::Array2D<double> C = il::dot(A, B);

  double error =
      il::max(il::abs(C(0, 0) - 23.0) / 23.0, il::abs(C(1, 0) - 34.0) / 34.0,
              il::abs(C(0, 1) - 31.0) / 31.0, il::abs(C(1, 1) - 46.0) / 46.0);

  ASSERT_TRUE(C.size(0) == 2 && C.size(1) == 2 && error <= 1.0e-15);
}

TEST(dot, matrix_f_simd_1) {
  il::Array2D<double> A{1, 2, il::align, 32};
  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  il::Array2D<double> B{2, 3, il::align, 32};
  B(0, 0) = 3.0;
  B(1, 0) = 4.0;
  B(0, 1) = 5.0;
  B(1, 1) = 6.0;
  B(0, 2) = 7.0;
  B(1, 2) = 8.0;

  il::Array2D<double> C{il::dot(A, B)};

  double error =
      il::max(il::abs(C(0, 0) - 11.0) / 11.0, il::abs(C(0, 1) - 17.0) / 17.0,
              il::abs(C(0, 2) - 23.0) / 23.0);

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && error <= 1.0e-15);
}

TEST(dot, matrix_c_0) {
  il::Array2C<double> A{il::value, {{1.0, 2.0}, {3.0, 4.0}}};
  il::Array2C<double> B{il::value, {{5.0, 6.0}, {7.0, 8.0}}};

  il::Array2C<double> C = il::dot(A, B);

  double error =
      il::max(il::abs(C(0, 0) - 19.0) / 19.0, il::abs(C(0, 1) - 22.0) / 22.0,
              il::abs(C(1, 0) - 43.0) / 43.0, il::abs(C(1, 1) - 50.0) / 50.0);

  ASSERT_TRUE(C.size(0) == 2 && C.size(1) == 2 && error <= 1.0e-15);
}

TEST(dot, matrix_c_1) {
  il::Array2C<double> A{il::value, {{1.0, 2.0}}};
  il::Array2C<double> B{il::value, {{3.0, 5.0, 7.0}, {4.0, 6.0, 8.0}}};

  il::Array2C<double> C = il::dot(A, B);

  double error =
      il::max(il::abs(C(0, 0) - 11.0) / 11.0, il::abs(C(0, 1) - 17.0) / 17.0,
              il::abs(C(0, 2) - 23.0) / 23.0);

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && error <= 1.0e-15);
}

TEST(dot, matrix_c_simd_0) {
  il::Array2C<double> A{2, 2, il::align, 32};
  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  A(1, 0) = 3.0;
  A(1, 1) = 4.0;
  il::Array2C<double> B{2, 2, il::align, 32};
  B(0, 0) = 5.0;
  B(0, 1) = 6.0;
  B(1, 0) = 7.0;
  B(1, 1) = 8.0;

  il::Array2C<double> C = il::dot(A, B);

  double error =
      il::max(il::abs(C(0, 0) - 19.0) / 19.0, il::abs(C(0, 1) - 22.0) / 22.0,
              il::abs(C(1, 0) - 43.0) / 43.0, il::abs(C(1, 1) - 50.0) / 50.0);

  ASSERT_TRUE(C.size(0) == 2 && C.size(1) == 2 && error <= 1.0e-15);
}

TEST(dot, matrix_c_simd_1) {
  il::Array2C<double> A{1, 2, il::align, 32};
  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  il::Array2C<double> B{2, 3, il::align, 32};
  B(0, 0) = 3.0;
  B(0, 1) = 5.0;
  B(0, 2) = 7.0;
  B(1, 0) = 4.0;
  B(1, 1) = 6.0;
  B(1, 2) = 8.0;

  il::Array2C<double> C = il::dot(A, B);

  double error =
      il::max(il::abs(C(0, 0) - 11.0) / 11.0, il::abs(C(0, 1) - 17.0) / 17.0,
              il::abs(C(0, 2) - 23.0) / 23.0);

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && error <= 1.0e-15);
}
#endif // IL_BLAS
