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

#include <il/linearAlgebra/dense/factorization/linearSolve.h>

#ifdef IL_BLAS
TEST(linear_solve, square_matrix_0) {
  il::int_t test_passed{false};

  il::Array2D<double> A{3, 4, 0.0};
  for (il::int_t i = 0; i < 3; ++i) {
    A(i, i) = 1.0;
  }
  il::Array<double> y{3, 0.0};

  try {
    il::Status status{};
    il::Array<double> x{il::linearSolve(std::move(A), y, il::io, status)};
    status.IgnoreError();
  } catch (il::AbortException) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(linear_solve, square_matrix_1) {
  il::int_t test_passed{false};

  il::Array2D<double> A{4, 3, 0.0};
  for (il::int_t i = 0; i < 3; ++i) {
    A(i, i) = 1.0;
  }
  il::Array<double> y{4, 0.0};

  try {
    il::Status status{};
    il::Array<double> x{il::linearSolve(std::move(A), y, il::io, status)};
    status.IgnoreError();
  } catch (il::AbortException) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(linear_solve, size_y) {
  il::int_t test_passed{false};

  il::Array2D<double> A{3, 3, 0.0};
  for (il::int_t i = 0; i < 3; ++i) {
    A(i, i) = 1.0;
  }
  il::Array<double> y{4, 0.0};

  try {
    il::Status status{};
    il::Array<double> x{il::linearSolve(std::move(A), y, il::io, status)};
    status.IgnoreError();
  } catch (il::AbortException) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(linear_solve, c_order) {
  il::Array2C<double> A{il::value, {{1.0, 2.0}, {1.0, 4.0}}};
  il::Array<double> y{il::value, {5.0, 9.0}};

  il::Status status{};
  il::Array<double> x{il::linearSolve(A, y, il::io, status)};
  status.IgnoreError();

  ASSERT_TRUE(x.size() == 2 && x[0] == 1.0 && x[1] == 2.0);
}

TEST(linear_solve, f_order) {
  il::Array2D<double> A{il::value, {{1.0, 1.0}, {2.0, 4.0}}};
  il::Array<double> y{il::value, {5.0, 9.0}};

  il::Status status{};
  il::Array<double> x{il::linearSolve(A, y, il::io, status)};
  status.IgnoreError();

  ASSERT_TRUE(x.size() == 2 && x[0] == 1.0 && x[1] == 2.0);
}

TEST(linear_solve, singular_matrix_0) {
  il::Array2D<double> A{2, 2, 0.0};
  il::Array<double> y{il::value, {1.0, 1.0}};
  bool test_passed{false};

  il::Status status{};
  il::Array<double> x{il::linearSolve(A, y, il::io, status)};
  if (!status.Ok() && status.error() == il::Error::MatrixSingular) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(linear_solve, singular_matrix_1) {
  il::Array2D<double> A{2, 2, 1.0};
  il::Array<double> y{il::value, {1.0, 1.0}};
  bool test_passed{false};

  il::Status status{};
  il::Array<double> x{il::linearSolve(A, y, il::io, status)};
  if (!status.Ok() && status.error() == il::Error::MatrixSingular) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}
#endif
