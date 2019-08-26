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

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>
#include <il/linearAlgebra/dense/blas/blas.h>

#include <iostream>

#ifdef IL_BLAS
TEST(Blas, axpy_fp32) {
  il::Array<float> x{il::value, {1.0f, 2.0f}};
  il::Array<float> y{il::value, {4.0f, 3.0f}};
  float alpha = 3.0f;
  il::blas(alpha, x.view(), il::io, y.Edit());

  ASSERT_TRUE(y.size() == 2 && y[0] == 7.0f && y[1] == 9.0f);
}

TEST(Blas, axpy_fp64) {
  il::Array<double> x{il::value, {1.0, 2.0}};
  il::Array<double> y{il::value, {4.0, 3.0}};
  double alpha = 3.0;
  il::blas(alpha, x.view(), il::io, y.Edit());

  ASSERT_TRUE(y.size() == 2 && y[0] == 7.0 && y[1] == 9.0);
}

TEST(Blas, axpy_complex64) {
  il::Array<std::complex<float>> x{il::value, {{1.0f, 2.0f}, {3.0f, 7.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 5.0f}, {2.0f, -9.0f}}};
  std::complex<float> alpha{-5.0f, 7.0f};
  il::blas(alpha, x.view(), il::io, y.Edit());

  ASSERT_TRUE(y.size() == 2 && y[0].real() == -15.0f && y[0].imag() == 2.0f &&
              y[1].real() == -62.0f && y[1].imag() == -23.0f);
}

TEST(Blas, axpy_complex128) {
  il::Array<std::complex<double>> x{il::value, {{1.0, 2.0}, {3.0, 7.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 5.0}, {2.0, -9.0}}};
  std::complex<double> alpha{-5.0, 7.0};
  il::blas(alpha, x.view(), il::io, y.Edit());

  ASSERT_TRUE(y.size() == 2 && y[0].real() == -15.0 && y[0].imag() == 2.0 &&
              y[1].real() == -62.0 && y[1].imag() == -23.0);
}

TEST(Blas, axpby_fp32) {
  il::Array<float> x{il::value, {1.0f, 2.0f}};
  il::Array<float> y{il::value, {4.0f, 3.0f}};
  float alpha = 3.0f;
  float beta = 5.0f;
  il::blas(alpha, x.view(), beta, il::io, y.Edit());

  ASSERT_TRUE(y.size() == 2 && y[0] == 23.0f && y[1] == 21.0f);
}

TEST(Blas, axpby_fp64) {
  il::Array<double> x{il::value, {1.0, 2.0}};
  il::Array<double> y{il::value, {4.0, 3.0}};
  double alpha = 3.0;
  float beta = 5.0f;
  il::blas(alpha, x.view(), beta, il::io, y.Edit());

  ASSERT_TRUE(y.size() == 2 && y[0] == 23.0 && y[1] == 21.0);
}

TEST(Blas, axpby_complex64) {
  il::Array<std::complex<float>> x{il::value, {{1.0f, 2.0f}, {3.0f, 7.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 5.0f}, {2.0f, -9.0f}}};
  std::complex<float> alpha{-5.0f, 7.0f};
  std::complex<float> beta{3.0f, -2.0f};
  il::blas(alpha, x.view(), beta, il::io, y.Edit());

  ASSERT_TRUE(y.size() == 2 && y[0].real() == 3.0f && y[0].imag() == 4.0f &&
              y[1].real() == -76.0f && y[1].imag() == -45.0f);
}

TEST(Blas, axpby_complex128) {
  il::Array<std::complex<double>> x{il::value, {{1.0, 2.0}, {3.0, 7.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 5.0}, {2.0, -9.0}}};
  std::complex<double> alpha{-5.0, 7.0};
  std::complex<double> beta{3.0, -2.0};
  il::blas(alpha, x.view(), beta, il::io, y.Edit());

  ASSERT_TRUE(y.size() == 2 && y[0].real() == 3.0 && y[0].imag() == 4.0 &&
              y[1].real() == -76.0 && y[1].imag() == -45.0);
}

TEST(Blas, vector_vector_fp32) {
  il::Array<float> x{il::value, {2.0f, 3.0f}};
  il::Array<float> y{il::value, {4.0f, 5.0f}};

  float alpha = il::dot(x.view(), y.view());

  IL_EXPECT_FAST(alpha == 23.0f);
}

TEST(Blas, vector_vector_fp64) {
  il::Array<double> x{il::value, {2.0, 3.0}};
  il::Array<double> y{il::value, {4.0, 5.0}};

  double alpha = il::dot(x.view(), y.view());

  IL_EXPECT_FAST(alpha == 23.0);
}

TEST(Blas, vector_vector_complex64_0) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x.view(), y.view());

  IL_EXPECT_FAST(alpha.real() == 14.875f && alpha.imag() == 24.5f);
}

TEST(Blas, vector_vector_complex64_1) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x.view(), il::Dot::None, y.view());

  IL_EXPECT_FAST(alpha.real() == 14.875f && alpha.imag() == 24.5f);
}

TEST(Blas, vector_vector_complex64_2) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x.view(), il::Dot::Star, y.view());

  IL_EXPECT_FAST(alpha.real() == 31.125f && alpha.imag() == 0.5f);
}

TEST(Blas, vector_vector_complex64_3) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x.view(), y.view(), il::Dot::None);

  IL_EXPECT_FAST(alpha.real() == 14.875f && alpha.imag() == 24.5f);
}

TEST(Blas, vector_vector_complex64_4) {
  il::Array<std::complex<float>> x{il::value, {{2.0f, 0.5f}, {3.0f, 2.0f}}};
  il::Array<std::complex<float>> y{il::value, {{4.0f, 0.25f}, {5.0f, 4.0f}}};

  std::complex<float> alpha = il::dot(x.view(), y.view(), il::Dot::Star);

  IL_EXPECT_FAST(alpha.real() == 31.125f && alpha.imag() == -0.5f);
}

TEST(Blas, vector_vector_complex128_0) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x.view(), y.view());

  IL_EXPECT_FAST(alpha.real() == 14.875 && alpha.imag() == 24.5);
}

TEST(Blas, vector_vector_complex128_1) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x.view(), il::Dot::None, y.view());

  IL_EXPECT_FAST(alpha.real() == 14.875 && alpha.imag() == 24.5);
}

TEST(Blas, vector_vector_complex128_2) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x.view(), il::Dot::Star, y.view());

  IL_EXPECT_FAST(alpha.real() == 31.125 && alpha.imag() == 0.5);
}

TEST(Blas, vector_vector_complex128_3) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x.view(), y.view(), il::Dot::None);

  IL_EXPECT_FAST(alpha.real() == 14.875 && alpha.imag() == 24.5);
}

TEST(Blas, vector_vector_complex128_4) {
  il::Array<std::complex<double>> x{il::value, {{2.0, 0.5}, {3.0, 2.0}}};
  il::Array<std::complex<double>> y{il::value, {{4.0, 0.25}, {5.0, 4.0}}};

  std::complex<double> alpha = il::dot(x.view(), y.view(), il::Dot::Star);

  IL_EXPECT_FAST(alpha.real() == 31.125 && alpha.imag() == -0.5);
}

TEST(Blas, matrix2d_vector_float32_0) {
  il::Array2D<float> A{il::value, {{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0}}};
  A.Reserve(4, 5);
  il::Array<float> x{il::value, {7.0f, 8.0f, 9.0f}};
  il::Array<float> y{il::value, {10.0f, 11.0f}};
  const float alpha = 13.0f;
  const float beta = 14.0f;
  il::blas(alpha, A.view(), x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0] == 790.0f && y[1] == 1740.0f);
}

TEST(Blas, matrix2d_vector_float32_1) {
  il::Array2D<float> A{il::value, {{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0}}};
  A.Reserve(4, 5);
  il::Array<float> x{il::value, {7.0f, 8.0f, 9.0f}};
  il::Array<float> y{il::value, {10.0f, 11.0f}};
  const float alpha = 13.0f;
  const float beta = 14.0f;
  il::blas(alpha, A.view(), il::Dot::None, x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0] == 790.0f && y[1] == 1740.0f);
}

TEST(Blas, matrix2d_vector_float32_2) {
  il::Array2D<float> A{il::value, {{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0}}};
  A.Reserve(4, 5);
  il::Array<float> x{il::value, {7.0f, 8.0f}};
  il::Array<float> y{il::value, {9.0f, 10.0f, 11.0f}};
  const float alpha = 13.0f;
  const float beta = 14.0f;
  il::blas(alpha, A.view(), il::Dot::Transpose, x.view(), beta, il::io,
           y.Edit());

  IL_EXPECT_FAST(y[0] == 633.0f && y[1] == 842.0f && y[2] == 1051.0f);
}

TEST(Blas, matrix2d_vector_float64_0) {
  il::Array2D<double> A{il::value, {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}};
  A.Reserve(4, 5);
  il::Array<double> x{il::value, {7.0, 8.0, 9.0}};
  il::Array<double> y{il::value, {10.0, 11.0}};
  const double alpha = 13.0;
  const double beta = 14.0;
  il::blas(alpha, A.view(), x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0] == 790.0 && y[1] == 1740.0);
}

TEST(Blas, matrix2d_vector_float64_1) {
  il::Array2D<double> A{il::value, {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}};
  A.Reserve(4, 5);
  il::Array<double> x{il::value, {7.0, 8.0, 9.0}};
  il::Array<double> y{il::value, {10.0, 11.0}};
  const double alpha = 13.0;
  const double beta = 14.0;
  il::blas(alpha, A.view(), il::Dot::None, x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0] == 790.0 && y[1] == 1740.0);
}

TEST(Blas, matrix2d_vector_float64_2) {
  il::Array2D<double> A{il::value, {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}};
  A.Reserve(4, 5);
  il::Array<double> x{il::value, {7.0, 8.0}};
  il::Array<double> y{il::value, {9.0, 10.0, 11.0}};
  const double alpha = 13.0;
  const double beta = 14.0;
  il::blas(alpha, A.view(), il::Dot::Transpose, x.view(), beta, il::io,
           y.Edit());

  IL_EXPECT_FAST(y[0] == 633.0 && y[1] == 842.0 && y[2] == 1051.0);
}

TEST(Blas, matrix2d_vector_complex32_0) {
  il::Array2D<std::complex<float>> A{il::value,
                                     {{{1.0f, 2.0f}, {4.0f, 5.0f}},
                                      {{2.0f, 3.0f}, {5.0f, 6.0f}},
                                      {{3.0f, 4.0f}, {6.0f, 7.0f}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<float>> x{il::value,
                                   {{7.0f, 8.0f}, {8.0f, 9.0f}, {9.0f, 10.0f}}};
  il::Array<std::complex<float>> y{il::value, {{10.0f, 11.0f}, {11.0f, 12.0f}}};
  const std::complex<float> alpha{13.0f, 14.0f};
  const std::complex<float> beta{14.0f, 15.0f};
  il::blas(alpha, A.view(), x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0].real() == -2274.0f && y[0].imag() == 1532.0f &&
                 y[1].real() == -4534.0f && y[1].imag() == 3424.0f);
}

TEST(Blas, matrix2d_vector_complex32_1) {
  il::Array2D<std::complex<float>> A{il::value,
                                     {{{1.0f, 2.0f}, {4.0f, 5.0f}},
                                      {{2.0f, 3.0f}, {5.0f, 6.0f}},
                                      {{3.0f, 4.0f}, {6.0f, 7.0f}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<float>> x{il::value,
                                   {{7.0f, 8.0f}, {8.0f, 9.0f}, {9.0f, 10.0f}}};
  il::Array<std::complex<float>> y{il::value, {{10.0f, 11.0f}, {11.0f, 12.0f}}};
  const std::complex<float> alpha{13.0f, 14.0f};
  const std::complex<float> beta{14.0f, 15.0f};
  il::blas(alpha, A.view(), il::Dot::None, x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0].real() == -2274.0f && y[0].imag() == 1532.0f &&
                 y[1].real() == -4534.0f && y[1].imag() == 3424.0f);
}

TEST(Blas, matrix2d_vector_complex32_2) {
  il::Array2D<std::complex<float>> A{il::value,
                                     {{{1.0f, 2.0f}, {4.0f, 5.0f}},
                                      {{2.0f, 3.0f}, {5.0f, 6.0f}},
                                      {{3.0f, 4.0f}, {6.0f, 7.0f}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<float>> x{il::value,
                                   {{7.0f, 8.0f}, {8.0f, 9.0f}, {9.0f, 10.0f}}};
  il::Array<std::complex<float>> y{il::value, {{10.0f, 11.0f}, {11.0f, 12.0f}}};
  const std::complex<float> alpha{13.0f, 14.0f};
  const std::complex<float> beta{14.0f, 15.0f};
  il::blas(alpha, A.view(), il::Dot::Transpose, y.view(), beta, il::io,
           x.Edit());

  IL_EXPECT_FAST(x[0].real() == -2262.0f && x[0].imag() == 1567.0f &&
                 x[1].real() == -2905.0f && x[1].imag() == 2140.0f &&
                 x[2].real() == -3548.0f && x[2].imag() == 2713.0f);
}

TEST(Blas, matrix2d_vector_complex32_3) {
  il::Array2D<std::complex<float>> A{il::value,
                                     {{{1.0f, 2.0f}, {4.0f, 5.0f}},
                                      {{2.0f, 3.0f}, {5.0f, 6.0f}},
                                      {{3.0f, 4.0f}, {6.0f, 7.0f}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<float>> x{il::value,
                                   {{7.0f, 8.0f}, {8.0f, 9.0f}, {9.0f, 10.0f}}};
  il::Array<std::complex<float>> y{il::value, {{10.0f, 11.0f}, {11.0f, 12.0f}}};
  const std::complex<float> alpha{13.0f, 14.0f};
  const std::complex<float> beta{14.0f, 15.0f};
  il::blas(alpha, A.view(), il::Dot::Star, y.view(), beta, il::io, x.Edit());

  IL_EXPECT_FAST(x[0].real() == 1970.0f && x[0].imag() == 1913.0f &&
                 x[1].real() == 2513.0f && x[1].imag() == 2584.0f &&
                 x[2].real() == 3056.0f && x[2].imag() == 3255.0f);
}

TEST(Blas, matrix2d_vector_complex64_0) {
  il::Array2D<std::complex<double>> A{il::value,
                                      {{{1.0, 2.0}, {4.0, 5.0}},
                                       {{2.0, 3.0}, {5.0, 6.0}},
                                       {{3.0, 4.0}, {6.0, 7.0}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<double>> x{il::value,
                                    {{7.0, 8.0}, {8.0, 9.0}, {9.0, 10.0}}};
  il::Array<std::complex<double>> y{il::value, {{10.0, 11.0}, {11.0, 12.0}}};
  const std::complex<double> alpha{13.0, 14.0};
  const std::complex<double> beta{14.0, 15.0};
  il::blas(alpha, A.view(), x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0].real() == -2274.0 && y[0].imag() == 1532.0 &&
                 y[1].real() == -4534.0 && y[1].imag() == 3424.0);
}

TEST(Blas, matrix2d_vector_complex64_1) {
  il::Array2D<std::complex<double>> A{il::value,
                                      {{{1.0, 2.0}, {4.0, 5.0}},
                                       {{2.0, 3.0}, {5.0, 6.0}},
                                       {{3.0, 4.0}, {6.0, 7.0}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<double>> x{il::value,
                                    {{7.0, 8.0}, {8.0, 9.0}, {9.0, 10.0}}};
  il::Array<std::complex<double>> y{il::value, {{10.0, 11.0}, {11.0, 12.0}}};
  const std::complex<double> alpha{13.0, 14.0};
  const std::complex<double> beta{14.0, 15.0};
  il::blas(alpha, A.view(), il::Dot::None, x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0].real() == -2274.0 && y[0].imag() == 1532.0 &&
                 y[1].real() == -4534.0 && y[1].imag() == 3424.0);
}

TEST(Blas, matrix2d_vector_complex64_2) {
  il::Array2D<std::complex<double>> A{il::value,
                                      {{{1.0, 2.0}, {4.0, 5.0}},
                                       {{2.0, 3.0}, {5.0, 6.0}},
                                       {{3.0, 4.0}, {6.0, 7.0}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<double>> x{il::value,
                                    {{7.0, 8.0}, {8.0, 9.0}, {9.0, 10.0}}};
  il::Array<std::complex<double>> y{il::value, {{10.0, 11.0}, {11.0, 12.0}}};
  const std::complex<double> alpha{13.0, 14.0};
  const std::complex<double> beta{14.0, 15.0};
  il::blas(alpha, A.view(), il::Dot::Transpose, y.view(), beta, il::io,
           x.Edit());

  IL_EXPECT_FAST(x[0].real() == -2262.0 && x[0].imag() == 1567.0 &&
                 x[1].real() == -2905.0 && x[1].imag() == 2140.0 &&
                 x[2].real() == -3548.0 && x[2].imag() == 2713.0);
}

TEST(Blas, matrix2d_vector_complex64_3) {
  il::Array2D<std::complex<double>> A{il::value,
                                      {{{1.0, 2.0}, {4.0, 5.0}},
                                       {{2.0, 3.0}, {5.0, 6.0}},
                                       {{3.0, 4.0}, {6.0, 7.0}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<double>> x{il::value,
                                    {{7.0, 8.0}, {8.0, 9.0}, {9.0, 10.0}}};
  il::Array<std::complex<double>> y{il::value, {{10.0, 11.0}, {11.0, 12.0}}};
  const std::complex<double> alpha{13.0, 14.0};
  const std::complex<double> beta{14.0, 15.0};
  il::blas(alpha, A.view(), il::Dot::Star, y.view(), beta, il::io, x.Edit());

  IL_EXPECT_FAST(x[0].real() == 1970.0 && x[0].imag() == 1913.0 &&
                 x[1].real() == 2513.0 && x[1].imag() == 2584.0 &&
                 x[2].real() == 3056.0 && x[2].imag() == 3255.0);
}

TEST(Blas, matrix2c_vector_float32_0) {
  il::Array2C<float> A{il::value, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}};
  A.Reserve(4, 5);
  il::Array<float> x{il::value, {7.0f, 8.0f, 9.0f}};
  il::Array<float> y{il::value, {10.0f, 11.0f}};
  const float alpha = 13.0f;
  const float beta = 14.0f;
  il::blas(alpha, A.view(), x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0] == 790.0f && y[1] == 1740.0f);
}

TEST(Blas, matrix2c_vector_float32_1) {
  il::Array2C<float> A{il::value, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}};
  A.Reserve(4, 5);
  il::Array<float> x{il::value, {7.0f, 8.0f, 9.0f}};
  il::Array<float> y{il::value, {10.0f, 11.0f}};
  const float alpha = 13.0f;
  const float beta = 14.0f;
  il::blas(alpha, A.view(), il::Dot::None, x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0] == 790.0f && y[1] == 1740.0f);
}

TEST(Blas, matrix2c_vector_float32_2) {
  il::Array2C<float> A{il::value, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}};
  A.Reserve(4, 5);
  il::Array<float> x{il::value, {7.0f, 8.0f}};
  il::Array<float> y{il::value, {9.0f, 10.0f, 11.0f}};
  const float alpha = 13.0f;
  const float beta = 14.0f;
  il::blas(alpha, A.view(), il::Dot::Transpose, x.view(), beta, il::io,
           y.Edit());

  IL_EXPECT_FAST(y[0] == 633.0f && y[1] == 842.0f && y[2] == 1051.0f);
}

TEST(Blas, matrix2c_vector_float64_0) {
  il::Array2C<double> A{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  A.Reserve(4, 5);
  il::Array<double> x{il::value, {7.0, 8.0, 9.0}};
  il::Array<double> y{il::value, {10.0, 11.0}};
  const double alpha = 13.0;
  const double beta = 14.0;
  il::blas(alpha, A.view(), x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0] == 790.0 && y[1] == 1740.0);
}

TEST(Blas, matrix2c_vector_float64_1) {
  il::Array2C<double> A{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  A.Reserve(4, 5);
  il::Array<double> x{il::value, {7.0, 8.0, 9.0}};
  il::Array<double> y{il::value, {10.0, 11.0}};
  const double alpha = 13.0;
  const double beta = 14.0;
  il::blas(alpha, A.view(), il::Dot::None, x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0] == 790.0 && y[1] == 1740.0);
}

TEST(Blas, matrix2c_vector_float64_2) {
  il::Array2C<double> A{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  A.Reserve(4, 5);
  il::Array<double> x{il::value, {7.0, 8.0}};
  il::Array<double> y{il::value, {9.0, 10.0, 11.0}};
  const double alpha = 13.0;
  const double beta = 14.0;
  il::blas(alpha, A.view(), il::Dot::Transpose, x.view(), beta, il::io,
           y.Edit());

  IL_EXPECT_FAST(y[0] == 633.0 && y[1] == 842.0 && y[2] == 1051.0);
}

TEST(Blas, matrix2c_vector_complex32_0) {
  il::Array2C<std::complex<float>> A{
      il::value,
      {{{1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}},
       {{4.0f, 5.0f}, {5.0f, 6.0f}, {6.0f, 7.0f}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<float>> x{il::value,
                                   {{7.0f, 8.0f}, {8.0f, 9.0f}, {9.0f, 10.0f}}};
  il::Array<std::complex<float>> y{il::value, {{10.0f, 11.0f}, {11.0f, 12.0f}}};
  const std::complex<float> alpha{13.0f, 14.0f};
  const std::complex<float> beta{14.0f, 15.0f};
  il::blas(alpha, A.view(), x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0].real() == -2274.0f && y[0].imag() == 1532.0f &&
                 y[1].real() == -4534.0f && y[1].imag() == 3424.0f);
}

TEST(Blas, matrix2c_vector_complex32_1) {
  il::Array2C<std::complex<float>> A{
      il::value,
      {{{1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}},
       {{4.0f, 5.0f}, {5.0f, 6.0f}, {6.0f, 7.0f}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<float>> x{il::value,
                                   {{7.0f, 8.0f}, {8.0f, 9.0f}, {9.0f, 10.0f}}};
  il::Array<std::complex<float>> y{il::value, {{10.0f, 11.0f}, {11.0f, 12.0f}}};
  const std::complex<float> alpha{13.0f, 14.0f};
  const std::complex<float> beta{14.0f, 15.0f};
  il::blas(alpha, A.view(), il::Dot::None, x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0].real() == -2274.0f && y[0].imag() == 1532.0f &&
                 y[1].real() == -4534.0f && y[1].imag() == 3424.0f);
}

TEST(Blas, matrix2c_vector_complex32_2) {
  il::Array2C<std::complex<float>> A{
      il::value,
      {{{1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}},
       {{4.0f, 5.0f}, {5.0f, 6.0f}, {6.0f, 7.0f}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<float>> x{il::value,
                                   {{7.0f, 8.0f}, {8.0f, 9.0f}, {9.0f, 10.0f}}};
  il::Array<std::complex<float>> y{il::value, {{10.0f, 11.0f}, {11.0f, 12.0f}}};
  const std::complex<float> alpha{13.0f, 14.0f};
  const std::complex<float> beta{14.0f, 15.0f};
  il::blas(alpha, A.view(), il::Dot::Transpose, y.view(), beta, il::io,
           x.Edit());

  IL_EXPECT_FAST(x[0].real() == -2262.0f && x[0].imag() == 1567.0f &&
                 x[1].real() == -2905.0f && x[1].imag() == 2140.0f &&
                 x[2].real() == -3548.0f && x[2].imag() == 2713.0f);
}

TEST(Blas, matrix2c_vector_complex32_3) {
  il::Array2C<std::complex<float>> A{
      il::value,
      {{{1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}},
       {{4.0f, 5.0f}, {5.0f, 6.0f}, {6.0f, 7.0f}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<float>> x{il::value,
                                   {{7.0f, 8.0f}, {8.0f, 9.0f}, {9.0f, 10.0f}}};
  il::Array<std::complex<float>> y{il::value, {{10.0f, 11.0f}, {11.0f, 12.0f}}};
  const std::complex<float> alpha{13.0f, 14.0f};
  const std::complex<float> beta{14.0f, 15.0f};
  il::blas(alpha, A.view(), il::Dot::Star, y.view(), beta, il::io, x.Edit());

  IL_EXPECT_FAST(x[0].real() == 1970.0f && x[0].imag() == 1913.0f &&
                 x[1].real() == 2513.0f && x[1].imag() == 2584.0f &&
                 x[2].real() == 3056.0f && x[2].imag() == 3255.0f);
}

TEST(Blas, matrix2c_vector_complex64_0) {
  il::Array2C<std::complex<double>> A{il::value,
                                      {{{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}},
                                       {{4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<double>> x{il::value,
                                    {{7.0, 8.0}, {8.0, 9.0}, {9.0, 10.0}}};
  il::Array<std::complex<double>> y{il::value, {{10.0, 11.0}, {11.0, 12.0}}};
  const std::complex<double> alpha{13.0, 14.0};
  const std::complex<double> beta{14.0, 15.0};
  il::blas(alpha, A.view(), x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0].real() == -2274.0 && y[0].imag() == 1532.0 &&
                 y[1].real() == -4534.0 && y[1].imag() == 3424.0);
}

TEST(Blas, matrix2c_vector_complex64_1) {
  il::Array2C<std::complex<double>> A{il::value,
                                      {{{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}},
                                       {{4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<double>> x{il::value,
                                    {{7.0, 8.0}, {8.0, 9.0}, {9.0, 10.0}}};
  il::Array<std::complex<double>> y{il::value, {{10.0, 11.0}, {11.0, 12.0}}};
  const std::complex<double> alpha{13.0, 14.0};
  const std::complex<double> beta{14.0, 15.0};
  il::blas(alpha, A.view(), il::Dot::None, x.view(), beta, il::io, y.Edit());

  IL_EXPECT_FAST(y[0].real() == -2274.0 && y[0].imag() == 1532.0 &&
                 y[1].real() == -4534.0 && y[1].imag() == 3424.0);
}

TEST(Blas, matrix2c_vector_complex64_2) {
  il::Array2C<std::complex<double>> A{il::value,
                                      {{{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}},
                                       {{4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<double>> x{il::value,
                                    {{7.0, 8.0}, {8.0, 9.0}, {9.0, 10.0}}};
  il::Array<std::complex<double>> y{il::value, {{10.0, 11.0}, {11.0, 12.0}}};
  const std::complex<double> alpha{13.0, 14.0};
  const std::complex<double> beta{14.0, 15.0};
  il::blas(alpha, A.view(), il::Dot::Transpose, y.view(), beta, il::io,
           x.Edit());

  IL_EXPECT_FAST(x[0].real() == -2262.0 && x[0].imag() == 1567.0 &&
                 x[1].real() == -2905.0 && x[1].imag() == 2140.0 &&
                 x[2].real() == -3548.0 && x[2].imag() == 2713.0);
}

TEST(Blas, matrix2c_vector_complex64_3) {
  il::Array2C<std::complex<double>> A{il::value,
                                      {{{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}},
                                       {{4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0}}}};
  A.Reserve(4, 5);
  il::Array<std::complex<double>> x{il::value,
                                    {{7.0, 8.0}, {8.0, 9.0}, {9.0, 10.0}}};
  il::Array<std::complex<double>> y{il::value, {{10.0, 11.0}, {11.0, 12.0}}};
  const std::complex<double> alpha{13.0, 14.0};
  const std::complex<double> beta{14.0, 15.0};
  il::blas(alpha, A.view(), il::Dot::Star, y.view(), beta, il::io, x.Edit());

  IL_EXPECT_FAST(x[0].real() == 1970.0 && x[0].imag() == 1913.0 &&
                 x[1].real() == 2513.0 && x[1].imag() == 2584.0 &&
                 x[2].real() == 3056.0 && x[2].imag() == 3255.0);
}

TEST(Blas, matrix2d_matrix2d_float32_0) {
  il::Array2D<float> A{il::value, {{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2D<float> B{il::value,
                       {{7.0f, 11.0f, 15.0f},
                        {8.0f, 12.0f, 16.0f},
                        {9.0f, 13.0f, 17.0f},
                        {10.0f, 14.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2D<float> C{
      il::value,
      {{-1.0f, -5.0f}, {-2.0f, -6.0f}, {-3.0f, -7.0f}, {-4.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), B.view(), beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2d_matrix2d_float32_1) {
  il::Array2D<float> A{il::value, {{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2D<float> B{il::value,
                       {{7.0f, 11.0f, 15.0f},
                        {8.0f, 12.0f, 16.0f},
                        {9.0f, 13.0f, 17.0f},
                        {10.0f, 14.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2D<float> C{
      il::value,
      {{-1.0f, -5.0f}, {-2.0f, -6.0f}, {-3.0f, -7.0f}, {-4.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2d_matrix2d_float32_2) {
  il::Array2D<float> A{il::value, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2D<float> B{il::value,
                       {{7.0f, 11.0f, 15.0f},
                        {8.0f, 12.0f, 16.0f},
                        {9.0f, 13.0f, 17.0f},
                        {10.0f, 14.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2D<float> C{
      il::value,
      {{-1.0f, -5.0f}, {-2.0f, -6.0f}, {-3.0f, -7.0f}, {-4.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2d_matrix2d_float32_3) {
  il::Array2D<float> A{il::value, {{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2D<float> B{il::value,
                       {{7.0f, 8.0f, 9.0f, 10.0f},
                        {11.0f, 12.0f, 13.0f, 14.0f},
                        {15.0f, 16.0f, 17.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2D<float> C{
      il::value,
      {{-1.0f, -5.0f}, {-2.0f, -6.0f}, {-3.0f, -7.0f}, {-4.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Transpose, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2d_matrix2d_float32_4) {
  il::Array2D<float> A{il::value, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2D<float> B{il::value,
                       {{7.0f, 8.0f, 9.0f, 10.0f},
                        {11.0f, 12.0f, 13.0f, 14.0f},
                        {15.0f, 16.0f, 17.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2D<float> C{
      il::value,
      {{-1.0f, -5.0f}, {-2.0f, -6.0f}, {-3.0f, -7.0f}, {-4.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::Transpose,
           beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2d_matrix2d_float64_0) {
  il::Array2D<double> A{il::value, {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2D<double> B{il::value,
                        {{7.0, 11.0, 15.0},
                         {8.0, 12.0, 16.0},
                         {9.0, 13.0, 17.0},
                         {10.0, 14.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2D<double> C{
      il::value, {{-1.0, -5.0}, {-2.0, -6.0}, {-3.0, -7.0}, {-4.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), B.view(), beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2d_matrix2d_float64_1) {
  il::Array2D<double> A{il::value, {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2D<double> B{il::value,
                        {{7.0, 11.0, 15.0},
                         {8.0, 12.0, 16.0},
                         {9.0, 13.0, 17.0},
                         {10.0, 14.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2D<double> C{
      il::value, {{-1.0, -5.0}, {-2.0, -6.0}, {-3.0, -7.0}, {-4.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2d_matrix2d_float64_2) {
  il::Array2D<double> A{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2D<double> B{il::value,
                        {{7.0, 11.0, 15.0},
                         {8.0, 12.0, 16.0},
                         {9.0, 13.0, 17.0},
                         {10.0, 14.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2D<double> C{
      il::value, {{-1.0, -5.0}, {-2.0, -6.0}, {-3.0, -7.0}, {-4.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2d_matrix2d_float64_3) {
  il::Array2D<double> A{il::value, {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2D<double> B{il::value,
                        {{7.0, 8.0, 9.0, 10.0},
                         {11.0, 12.0, 13.0, 14.0},
                         {15.0, 16.0, 17.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2D<double> C{
      il::value, {{-1.0, -5.0}, {-2.0, -6.0}, {-3.0, -7.0}, {-4.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Transpose, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2d_matrix2d_float64_4) {
  il::Array2D<double> A{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2D<double> B{il::value,
                        {{7.0, 8.0, 9.0, 10.0},
                         {11.0, 12.0, 13.0, 14.0},
                         {15.0, 16.0, 17.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2D<double> C{
      il::value, {{-1.0, -5.0}, {-2.0, -6.0}, {-3.0, -7.0}, {-4.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::Transpose,
           beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2d_matrix2d_complex64_0) {
  il::Array2D<std::complex<float>> A{il::value,
                                     {{{1.0f, 1.5f}, {4.0f, 4.5f}},
                                      {{2.0f, 2.5f}, {5.0f, 5.5f}},
                                      {{3.0f, 3.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {11.0f, 11.5f}, {15.0f, 15.5f}},
       {{8.0f, 8.5f}, {12.0f, 12.5f}, {16.0f, 16.5f}},
       {{9.0f, 9.5f}, {13.0f, 13.5f}, {17.0f, 17.5f}},
       {{10.0f, 10.5f}, {14.0f, 14.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<float>> C{il::value,
                                     {{{-1.0f, -1.5f}, {-5.0f, -5.5f}},
                                      {{-2.0f, -2.5f}, {-6.0f, -6.5f}},
                                      {{-3.0f, -3.5f}, {-7.0f, -7.5f}},
                                      {{-4.0f, -4.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), B.view(), beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2d_matrix2d_complex64_1) {
  il::Array2D<std::complex<float>> A{il::value,
                                     {{{1.0f, 1.5f}, {4.0f, 4.5f}},
                                      {{2.0f, 2.5f}, {5.0f, 5.5f}},
                                      {{3.0f, 3.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {11.0f, 11.5f}, {15.0f, 15.5f}},
       {{8.0f, 8.5f}, {12.0f, 12.5f}, {16.0f, 16.5f}},
       {{9.0f, 9.5f}, {13.0f, 13.5f}, {17.0f, 17.5f}},
       {{10.0f, 10.5f}, {14.0f, 14.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<float>> C{il::value,
                                     {{{-1.0f, -1.5f}, {-5.0f, -5.5f}},
                                      {{-2.0f, -2.5f}, {-6.0f, -6.5f}},
                                      {{-3.0f, -3.5f}, {-7.0f, -7.5f}},
                                      {{-4.0f, -4.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2d_matrix2d_complex64_2) {
  il::Array2D<std::complex<float>> A{
      il::value,
      {{{1.0f, 1.5f}, {2.0f, 2.5f}, {3.0f, 3.5f}},
       {{4.0f, 4.5f}, {5.0f, 5.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {11.0f, 11.5f}, {15.0f, 15.5f}},
       {{8.0f, 8.5f}, {12.0f, 12.5f}, {16.0f, 16.5f}},
       {{9.0f, 9.5f}, {13.0f, 13.5f}, {17.0f, 17.5f}},
       {{10.0f, 10.5f}, {14.0f, 14.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<float>> C{il::value,
                                     {{{-1.0f, -1.5f}, {-5.0f, -5.5f}},
                                      {{-2.0f, -2.5f}, {-6.0f, -6.5f}},
                                      {{-3.0f, -3.5f}, {-7.0f, -7.5f}},
                                      {{-4.0f, -4.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2d_matrix2d_complex64_3) {
  il::Array2D<std::complex<float>> A{
      il::value,
      {{{1.0f, -1.5f}, {2.0f, -2.5f}, {3.0f, -3.5f}},
       {{4.0f, -4.5f}, {5.0f, -5.5f}, {6.0f, -6.5f}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {11.0f, 11.5f}, {15.0f, 15.5f}},
       {{8.0f, 8.5f}, {12.0f, 12.5f}, {16.0f, 16.5f}},
       {{9.0f, 9.5f}, {13.0f, 13.5f}, {17.0f, 17.5f}},
       {{10.0f, 10.5f}, {14.0f, 14.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<float>> C{il::value,
                                     {{{-1.0f, -1.5f}, {-5.0f, -5.5f}},
                                      {{-2.0f, -2.5f}, {-6.0f, -6.5f}},
                                      {{-3.0f, -3.5f}, {-7.0f, -7.5f}},
                                      {{-4.0f, -4.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::Star, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2d_matrix2d_complex64_4) {
  il::Array2D<std::complex<float>> A{il::value,
                                     {{{1.0f, 1.5f}, {4.0f, 4.5f}},
                                      {{2.0f, 2.5f}, {5.0f, 5.5f}},
                                      {{3.0f, 3.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {8.0f, 8.5f}, {9.0f, 9.5f}, {10.0f, 10.5f}},
       {{11.0f, 11.5f}, {12.0f, 12.5f}, {13.0f, 13.5f}, {14.0f, 14.5f}},
       {{15.0f, 15.5f}, {16.0f, 16.5f}, {17.0f, 17.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<float>> C{il::value,
                                     {{{-1.0f, -1.5f}, {-5.0f, -5.5f}},
                                      {{-2.0f, -2.5f}, {-6.0f, -6.5f}},
                                      {{-3.0f, -3.5f}, {-7.0f, -7.5f}},
                                      {{-4.0f, -4.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Transpose, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2d_matrix2d_complex64_5) {
  il::Array2D<std::complex<float>> A{il::value,
                                     {{{1.0f, 1.5f}, {4.0f, 4.5f}},
                                      {{2.0f, 2.5f}, {5.0f, 5.5f}},
                                      {{3.0f, 3.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<float>> B{
      il::value,
      {{{7.0f, -7.5f}, {8.0f, -8.5f}, {9.0f, -9.5f}, {10.0f, -10.5f}},
       {{11.0f, -11.5f}, {12.0f, -12.5f}, {13.0f, -13.5f}, {14.0f, -14.5f}},
       {{15.0f, -15.5f}, {16.0f, -16.5f}, {17.0f, -17.5f}, {18.0f, -18.5f}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<float>> C{il::value,
                                     {{{-1.0f, -1.5f}, {-5.0f, -5.5f}},
                                      {{-2.0f, -2.5f}, {-6.0f, -6.5f}},
                                      {{-3.0f, -3.5f}, {-7.0f, -7.5f}},
                                      {{-4.0f, -4.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Star, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2d_matrix2d_complex128_0) {
  il::Array2D<std::complex<double>> A{il::value,
                                      {{{1.0, 1.5}, {4.0, 4.5}},
                                       {{2.0, 2.5}, {5.0, 5.5}},
                                       {{3.0, 3.5}, {6.0, 6.5}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<double>> B{
      il::value,
      {{{7.0, 7.5}, {11.0, 11.5}, {15.0, 15.5}},
       {{8.0, 8.5}, {12.0, 12.5}, {16.0, 16.5}},
       {{9.0, 9.5}, {13.0, 13.5}, {17.0, 17.5}},
       {{10.0, 10.5}, {14.0, 14.5}, {18.0, 18.5}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<double>> C{il::value,
                                      {{{-1.0, -1.5}, {-5.0, -5.5}},
                                       {{-2.0, -2.5}, {-6.0, -6.5}},
                                       {{-3.0, -3.5}, {-7.0, -7.5}},
                                       {{-4.0, -4.5}, {-8.0, -8.5}}}};
  C.Reserve(7, 9);
  const std::complex<double> alpha = {0.5, 1.0};
  const std::complex<double> beta = {1.5, 2.0};
  il::blas(alpha, A.view(), B.view(), beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125 && C(0, 0).imag() == 59.25 &&
                 C(0, 1).real() == -189.875 && C(0, 1).imag() == 61.0 &&
                 C(0, 2).real() == -203.625 && C(0, 2).imag() == 62.75 &&
                 C(0, 3).real() == -217.375 && C(0, 3).imag() == 64.5 &&
                 C(1, 0).real() == -378.875 && C(1, 0).imag() == 142.0 &&
                 C(1, 1).real() == -410.625 && C(1, 1).imag() == 152.75 &&
                 C(1, 2).real() == -442.375 && C(1, 2).imag() == 163.5 &&
                 C(1, 3).real() == -474.125 && C(1, 3).imag() == 174.25);
}

TEST(Blas, matrix2d_matrix2d_complex128_1) {
  il::Array2D<std::complex<double>> A{il::value,
                                      {{{1.0, 1.5}, {4.0, 4.5}},
                                       {{2.0, 2.5}, {5.0, 5.5}},
                                       {{3.0, 3.5}, {6.0, 6.5}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<double>> B{
      il::value,
      {{{7.0, 7.5}, {11.0, 11.5}, {15.0, 15.5}},
       {{8.0, 8.5}, {12.0, 12.5}, {16.0, 16.5}},
       {{9.0, 9.5}, {13.0, 13.5}, {17.0, 17.5}},
       {{10.0, 10.5}, {14.0, 14.5}, {18.0, 18.5}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<double>> C{il::value,
                                      {{{-1.0, -1.5}, {-5.0, -5.5}},
                                       {{-2.0, -2.5}, {-6.0, -6.5}},
                                       {{-3.0, -3.5}, {-7.0, -7.5}},
                                       {{-4.0, -4.5}, {-8.0, -8.5}}}};
  C.Reserve(7, 9);
  const std::complex<double> alpha = {0.5, 1.0};
  const std::complex<double> beta = {1.5, 2.0};
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125 && C(0, 0).imag() == 59.25 &&
                 C(0, 1).real() == -189.875 && C(0, 1).imag() == 61.0 &&
                 C(0, 2).real() == -203.625 && C(0, 2).imag() == 62.75 &&
                 C(0, 3).real() == -217.375 && C(0, 3).imag() == 64.5 &&
                 C(1, 0).real() == -378.875 && C(1, 0).imag() == 142.0 &&
                 C(1, 1).real() == -410.625 && C(1, 1).imag() == 152.75 &&
                 C(1, 2).real() == -442.375 && C(1, 2).imag() == 163.5 &&
                 C(1, 3).real() == -474.125 && C(1, 3).imag() == 174.25);
}

TEST(Blas, matrix2d_matrix2d_complex128_2) {
  il::Array2D<std::complex<double>> A{il::value,
                                      {{{1.0, 1.5}, {2.0, 2.5}, {3.0, 3.5}},
                                       {{4.0, 4.5}, {5.0, 5.5}, {6.0, 6.5}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<double>> B{
      il::value,
      {{{7.0, 7.5}, {11.0, 11.5}, {15.0, 15.5}},
       {{8.0, 8.5}, {12.0, 12.5}, {16.0, 16.5}},
       {{9.0, 9.5}, {13.0, 13.5}, {17.0, 17.5}},
       {{10.0, 10.5}, {14.0, 14.5}, {18.0, 18.5}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<double>> C{il::value,
                                      {{{-1.0, -1.5}, {-5.0, -5.5}},
                                       {{-2.0, -2.5}, {-6.0, -6.5}},
                                       {{-3.0, -3.5}, {-7.0, -7.5}},
                                       {{-4.0, -4.5}, {-8.0, -8.5}}}};
  C.Reserve(7, 9);
  const std::complex<double> alpha = {0.5, 1.0};
  const std::complex<double> beta = {1.5, 2.0};
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125 && C(0, 0).imag() == 59.25 &&
                 C(0, 1).real() == -189.875 && C(0, 1).imag() == 61.0 &&
                 C(0, 2).real() == -203.625 && C(0, 2).imag() == 62.75 &&
                 C(0, 3).real() == -217.375 && C(0, 3).imag() == 64.5 &&
                 C(1, 0).real() == -378.875 && C(1, 0).imag() == 142.0 &&
                 C(1, 1).real() == -410.625 && C(1, 1).imag() == 152.75 &&
                 C(1, 2).real() == -442.375 && C(1, 2).imag() == 163.5 &&
                 C(1, 3).real() == -474.125 && C(1, 3).imag() == 174.25);
}

TEST(Blas, matrix2d_matrix2d_complex128_3) {
  il::Array2D<std::complex<double>> A{
      il::value,
      {{{1.0, -1.5}, {2.0, -2.5}, {3.0, -3.5}},
       {{4.0, -4.5}, {5.0, -5.5}, {6.0, -6.5}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<double>> B{
      il::value,
      {{{7.0, 7.5}, {11.0, 11.5}, {15.0, 15.5}},
       {{8.0, 8.5}, {12.0, 12.5}, {16.0, 16.5}},
       {{9.0, 9.5}, {13.0, 13.5}, {17.0, 17.5}},
       {{10.0, 10.5}, {14.0, 14.5}, {18.0, 18.5}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<double>> C{il::value,
                                      {{{-1.0, -1.5}, {-5.0, -5.5}},
                                       {{-2.0, -2.5}, {-6.0, -6.5}},
                                       {{-3.0, -3.5}, {-7.0, -7.5}},
                                       {{-4.0, -4.5}, {-8.0, -8.5}}}};
  C.Reserve(7, 9);
  const std::complex<double> alpha = {0.5, 1.0};
  const std::complex<double> beta = {1.5, 2.0};
  il::blas(alpha, A.view(), il::Dot::Star, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125 && C(0, 0).imag() == 59.25 &&
                 C(0, 1).real() == -189.875 && C(0, 1).imag() == 61.0 &&
                 C(0, 2).real() == -203.625 && C(0, 2).imag() == 62.75 &&
                 C(0, 3).real() == -217.375 && C(0, 3).imag() == 64.5 &&
                 C(1, 0).real() == -378.875 && C(1, 0).imag() == 142.0 &&
                 C(1, 1).real() == -410.625 && C(1, 1).imag() == 152.75 &&
                 C(1, 2).real() == -442.375 && C(1, 2).imag() == 163.5 &&
                 C(1, 3).real() == -474.125 && C(1, 3).imag() == 174.25);
}

TEST(Blas, matrix2d_matrix2d_complex128_4) {
  il::Array2D<std::complex<double>> A{il::value,
                                      {{{1.0, 1.5}, {4.0, 4.5}},
                                       {{2.0, 2.5}, {5.0, 5.5}},
                                       {{3.0, 3.5}, {6.0, 6.5}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<double>> B{
      il::value,
      {{{7.0, 7.5}, {8.0, 8.5}, {9.0, 9.5}, {10.0, 10.5}},
       {{11.0, 11.5}, {12.0, 12.5}, {13.0, 13.5}, {14.0, 14.5}},
       {{15.0, 15.5}, {16.0, 16.5}, {17.0, 17.5}, {18.0, 18.5}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<double>> C{il::value,
                                      {{{-1.0, -1.5}, {-5.0, -5.5}},
                                       {{-2.0, -2.5}, {-6.0, -6.5}},
                                       {{-3.0, -3.5}, {-7.0, -7.5}},
                                       {{-4.0, -4.5}, {-8.0, -8.5}}}};
  C.Reserve(7, 9);
  const std::complex<double> alpha = {0.5, 1.0};
  const std::complex<double> beta = {1.5, 2.0};
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Transpose, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125 && C(0, 0).imag() == 59.25 &&
                 C(0, 1).real() == -189.875 && C(0, 1).imag() == 61.0 &&
                 C(0, 2).real() == -203.625 && C(0, 2).imag() == 62.75 &&
                 C(0, 3).real() == -217.375 && C(0, 3).imag() == 64.5 &&
                 C(1, 0).real() == -378.875 && C(1, 0).imag() == 142.0 &&
                 C(1, 1).real() == -410.625 && C(1, 1).imag() == 152.75 &&
                 C(1, 2).real() == -442.375 && C(1, 2).imag() == 163.5 &&
                 C(1, 3).real() == -474.125 && C(1, 3).imag() == 174.25);
}

TEST(Blas, matrix2d_matrix2d_complex128_5) {
  il::Array2D<std::complex<double>> A{il::value,
                                      {{{1.0, 1.5}, {4.0, 4.5}},
                                       {{2.0, 2.5}, {5.0, 5.5}},
                                       {{3.0, 3.5}, {6.0, 6.5}}}};
  A.Reserve(5, 7);
  il::Array2D<std::complex<double>> B{
      il::value,
      {{{7.0, -7.5}, {8.0, -8.5}, {9.0, -9.5}, {10.0, -10.5}},
       {{11.0, -11.5}, {12.0, -12.5}, {13.0, -13.5}, {14.0, -14.5}},
       {{15.0, -15.5}, {16.0, -16.5}, {17.0, -17.5}, {18.0, -18.5}}}};
  B.Reserve(6, 8);
  il::Array2D<std::complex<double>> C{il::value,
                                      {{{-1.0, -1.5}, {-5.0, -5.5}},
                                       {{-2.0, -2.5}, {-6.0, -6.5}},
                                       {{-3.0, -3.5}, {-7.0, -7.5}},
                                       {{-4.0, -4.5}, {-8.0, -8.5}}}};
  C.Reserve(7, 9);
  const std::complex<double> alpha = {0.5, 1.0};
  const std::complex<double> beta = {1.5, 2.0};
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Star, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125 && C(0, 0).imag() == 59.25 &&
                 C(0, 1).real() == -189.875 && C(0, 1).imag() == 61.0 &&
                 C(0, 2).real() == -203.625 && C(0, 2).imag() == 62.75 &&
                 C(0, 3).real() == -217.375 && C(0, 3).imag() == 64.5 &&
                 C(1, 0).real() == -378.875 && C(1, 0).imag() == 142.0 &&
                 C(1, 1).real() == -410.625 && C(1, 1).imag() == 152.75 &&
                 C(1, 2).real() == -442.375 && C(1, 2).imag() == 163.5 &&
                 C(1, 3).real() == -474.125 && C(1, 3).imag() == 174.25);
}

/// To be continued from line 801

TEST(Blas, matrix2c_matrix2c_float32_0) {
  il::Array2C<float> A{il::value, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2C<float> B{il::value,
                       {{7.0f, 8.0f, 9.0f, 10.0f},
                        {11.0f, 12.0f, 13.0f, 14.0f},
                        {15.0f, 16.0f, 17.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2C<float> C{
      il::value, {{-1.0f, -2.0f, -3.0f, -4.0f}, {-5.0f, -6.0f, -7.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), B.view(), beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2c_matrix2c_float32_1) {
  il::Array2C<float> A{il::value, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2C<float> B{il::value,
                       {{7.0f, 8.0f, 9.0f, 10.0f},
                        {11.0f, 12.0f, 13.0f, 14.0f},
                        {15.0f, 16.0f, 17.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2C<float> C{
      il::value, {{-1.0f, -2.0f, -3.0f, -4.0f}, {-5.0f, -6.0f, -7.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2c_matrix2c_float32_2) {
  il::Array2C<float> A{il::value, {{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2C<float> B{il::value,
                       {{7.0f, 8.0f, 9.0f, 10.0f},
                        {11.0f, 12.0f, 13.0f, 14.0f},
                        {15.0f, 16.0f, 17.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2C<float> C{
      il::value, {{-1.0f, -2.0f, -3.0f, -4.0f}, {-5.0f, -6.0f, -7.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2c_matrix2c_float32_3) {
  il::Array2C<float> A{il::value, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2C<float> B{il::value,
                       {{7.0f, 11.0f, 15.0f},
                        {8.0f, 12.0f, 16.0f},
                        {9.0f, 13.0f, 17.0f},
                        {10.0f, 14.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2C<float> C{
      il::value, {{-1.0f, -2.0f, -3.0f, -4.0f}, {-5.0f, -6.0f, -7.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Transpose, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2c_matrix2c_float32_4) {
  il::Array2C<float> A{il::value, {{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}}};
  A.Reserve(5, 7);
  il::Array2C<float> B{il::value,
                       {{7.0f, 11.0f, 15.0f},
                        {8.0f, 12.0f, 16.0f},
                        {9.0f, 13.0f, 17.0f},
                        {10.0f, 14.0f, 18.0f}}};
  B.Reserve(6, 8);
  il::Array2C<float> C{
      il::value, {{-1.0f, -2.0f, -3.0f, -4.0f}, {-5.0f, -6.0f, -7.0f, -8.0f}}};
  C.Reserve(7, 9);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::Transpose,
           beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5f && C(0, 1) == 37.0f && C(0, 2) == 38.5f &&
                 C(0, 3) == 40.0f && C(1, 0) == 79.0f && C(1, 1) == 85.0f &&
                 C(1, 2) == 91.0f && C(1, 3) == 97.0f);
}

TEST(Blas, matrix2c_matrix2c_float64_0) {
  il::Array2C<double> A{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2C<double> B{il::value,
                        {{7.0, 8.0, 9.0, 10.0},
                         {11.0, 12.0, 13.0, 14.0},
                         {15.0, 16.0, 17.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2C<double> C{il::value,
                        {{-1.0, -2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), B.view(), beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2c_matrix2c_float64_1) {
  il::Array2C<double> A{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2C<double> B{il::value,
                        {{7.0, 8.0, 9.0, 10.0},
                         {11.0, 12.0, 13.0, 14.0},
                         {15.0, 16.0, 17.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2C<double> C{il::value,
                        {{-1.0, -2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2c_matrix2c_float64_2) {
  il::Array2C<double> A{il::value, {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2C<double> B{il::value,
                        {{7.0, 8.0, 9.0, 10.0},
                         {11.0, 12.0, 13.0, 14.0},
                         {15.0, 16.0, 17.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2C<double> C{il::value,
                        {{-1.0, -2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2c_matrix2c_float64_3) {
  il::Array2C<double> A{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2C<double> B{il::value,
                        {{7.0, 11.0, 15.0},
                         {8.0, 12.0, 16.0},
                         {9.0, 13.0, 17.0},
                         {10.0, 14.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2C<double> C{il::value,
                        {{-1.0, -2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Transpose, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2c_matrix2c_float64_4) {
  il::Array2C<double> A{il::value, {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}};
  A.Reserve(5, 7);
  il::Array2C<double> B{il::value,
                        {{7.0, 11.0, 15.0},
                         {8.0, 12.0, 16.0},
                         {9.0, 13.0, 17.0},
                         {10.0, 14.0, 18.0}}};
  B.Reserve(6, 8);
  il::Array2C<double> C{il::value,
                        {{-1.0, -2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0, -8.0}}};
  C.Reserve(7, 9);
  const double alpha = 0.5;
  const double beta = 1.5;
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::Transpose,
           beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0) == 35.5 && C(0, 1) == 37.0 && C(0, 2) == 38.5 &&
                 C(0, 3) == 40.0 && C(1, 0) == 79.0 && C(1, 1) == 85.0 &&
                 C(1, 2) == 91.0 && C(1, 3) == 97.0);
}

TEST(Blas, matrix2c_matrix2c_complex64_0) {
  il::Array2C<std::complex<float>> A{
      il::value,
      {{{1.0f, 1.5f}, {2.0f, 2.5f}, {3.0f, 3.5f}},
       {{4.0f, 4.5f}, {5.0f, 5.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2C<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {8.0f, 8.5f}, {9.0f, 9.5f}, {10.0f, 10.5f}},
       {{11.0f, 11.5f}, {12.0f, 12.5f}, {13.0f, 13.5f}, {14.0f, 14.5f}},
       {{15.0f, 15.5f}, {16.0f, 16.5f}, {17.0f, 17.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2C<std::complex<float>> C{
      il::value,
      {{{-1.0f, -1.5f}, {-2.0f, -2.5f}, {-3.0f, -3.5f}, {-4.0f, -4.5f}},
       {{-5.0f, -5.5f}, {-6.0f, -6.5f}, {-7.0f, -7.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), B.view(), beta, il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2c_matrix2c_complex64_1) {
  il::Array2C<std::complex<float>> A{
      il::value,
      {{{1.0f, 1.5f}, {2.0f, 2.5f}, {3.0f, 3.5f}},
       {{4.0f, 4.5f}, {5.0f, 5.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2C<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {8.0f, 8.5f}, {9.0f, 9.5f}, {10.0f, 10.5f}},
       {{11.0f, 11.5f}, {12.0f, 12.5f}, {13.0f, 13.5f}, {14.0f, 14.5f}},
       {{15.0f, 15.5f}, {16.0f, 16.5f}, {17.0f, 17.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2C<std::complex<float>> C{
      il::value,
      {{{-1.0f, -1.5f}, {-2.0f, -2.5f}, {-3.0f, -3.5f}, {-4.0f, -4.5f}},
       {{-5.0f, -5.5f}, {-6.0f, -6.5f}, {-7.0f, -7.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2c_matrix2c_complex64_2) {
  il::Array2C<std::complex<float>> A{il::value,
                                     {{{1.0f, 1.5f}, {4.0f, 4.5f}},
                                      {{2.0f, 2.5f}, {5.0f, 5.5f}},
                                      {{3.0f, 3.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2C<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {8.0f, 8.5f}, {9.0f, 9.5f}, {10.0f, 10.5f}},
       {{11.0f, 11.5f}, {12.0f, 12.5f}, {13.0f, 13.5f}, {14.0f, 14.5f}},
       {{15.0f, 15.5f}, {16.0f, 16.5f}, {17.0f, 17.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2C<std::complex<float>> C{
      il::value,
      {{{-1.0f, -1.5f}, {-2.0f, -2.5f}, {-3.0f, -3.5f}, {-4.0f, -4.5f}},
       {{-5.0f, -5.5f}, {-6.0f, -6.5f}, {-7.0f, -7.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::Transpose, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2c_matrix2c_complex64_3) {
  il::Array2C<std::complex<float>> A{il::value,
                                     {{{1.0f, -1.5f}, {4.0f, -4.5f}},
                                      {{2.0f, -2.5f}, {5.0f, -5.5f}},
                                      {{3.0f, -3.5f}, {6.0f, -6.5f}}}};
  A.Reserve(5, 7);
  il::Array2C<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {8.0f, 8.5f}, {9.0f, 9.5f}, {10.0f, 10.5f}},
       {{11.0f, 11.5f}, {12.0f, 12.5f}, {13.0f, 13.5f}, {14.0f, 14.5f}},
       {{15.0f, 15.5f}, {16.0f, 16.5f}, {17.0f, 17.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2C<std::complex<float>> C{
      il::value,
      {{{-1.0f, -1.5f}, {-2.0f, -2.5f}, {-3.0f, -3.5f}, {-4.0f, -4.5f}},
       {{-5.0f, -5.5f}, {-6.0f, -6.5f}, {-7.0f, -7.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::Star, B.view(), il::Dot::None, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2c_matrix2c_complex64_4) {
  il::Array2C<std::complex<float>> A{
      il::value,
      {{{1.0f, 1.5f}, {2.0f, 2.5f}, {3.0f, 3.5f}},
       {{4.0f, 4.5f}, {5.0f, 5.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2C<std::complex<float>> B{
      il::value,
      {{{7.0f, 7.5f}, {11.0f, 11.5f}, {15.0f, 15.5f}},
       {{8.0f, 8.5f}, {12.0f, 12.5f}, {16.0f, 16.5f}},
       {{9.0f, 9.5f}, {13.0f, 13.5f}, {17.0f, 17.5f}},
       {{10.0f, 10.5f}, {14.0f, 14.5f}, {18.0f, 18.5f}}}};
  B.Reserve(6, 8);
  il::Array2C<std::complex<float>> C{
      il::value,
      {{{-1.0f, -1.5f}, {-2.0f, -2.5f}, {-3.0f, -3.5f}, {-4.0f, -4.5f}},
       {{-5.0f, -5.5f}, {-6.0f, -6.5f}, {-7.0f, -7.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);
  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Transpose, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

TEST(Blas, matrix2c_matrix2c_complex64_5) {
  il::Array2C<std::complex<float>> A{
      il::value,
      {{{1.0f, 1.5f}, {2.0f, 2.5f}, {3.0f, 3.5f}},
       {{4.0f, 4.5f}, {5.0f, 5.5f}, {6.0f, 6.5f}}}};
  A.Reserve(5, 7);
  il::Array2C<std::complex<float>> B{
      il::value,
      {{{7.0f, -7.5f}, {11.0f, -11.5f}, {15.0f, -15.5f}},
       {{8.0f, -8.5f}, {12.0f, -12.5f}, {16.0f, -16.5f}},
       {{9.0f, -9.5f}, {13.0f, -13.5f}, {17.0f, -17.5f}},
       {{10.0f, -10.5f}, {14.0f, -14.5f}, {18.0f, -18.5f}}}};
  B.Reserve(6, 8);
  il::Array2C<std::complex<float>> C{
      il::value,
      {{{-1.0f, -1.5f}, {-2.0f, -2.5f}, {-3.0f, -3.5f}, {-4.0f, -4.5f}},
       {{-5.0f, -5.5f}, {-6.0f, -6.5f}, {-7.0f, -7.5f}, {-8.0f, -8.5f}}}};
  C.Reserve(7, 9);

  const std::complex<float> alpha = {0.5f, 1.0f};
  const std::complex<float> beta = {1.5f, 2.0f};
  il::blas(alpha, A.view(), il::Dot::None, B.view(), il::Dot::Star, beta,
           il::io, C.Edit());

  IL_EXPECT_FAST(C(0, 0).real() == -176.125f && C(0, 0).imag() == 59.25f &&
                 C(0, 1).real() == -189.875f && C(0, 1).imag() == 61.0f &&
                 C(0, 2).real() == -203.625f && C(0, 2).imag() == 62.75f &&
                 C(0, 3).real() == -217.375f && C(0, 3).imag() == 64.5f &&
                 C(1, 0).real() == -378.875f && C(1, 0).imag() == 142.0f &&
                 C(1, 1).real() == -410.625f && C(1, 1).imag() == 152.75f &&
                 C(1, 2).real() == -442.375f && C(1, 2).imag() == 163.5f &&
                 C(1, 3).real() == -474.125f && C(1, 3).imag() == 174.25f);
}

#endif
