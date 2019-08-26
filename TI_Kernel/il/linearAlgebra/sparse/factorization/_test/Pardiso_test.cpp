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

#include <il/linearAlgebra/sparse/blas/sparseBlas.h>
#include <il/linearAlgebra/sparse/factorization/Pardiso.h>
#include <il/linearAlgebra/sparse/factorization/_test/matrix/heat.h>

#include <il/benchmark/tools/timer/Timer.h>

#ifdef IL_MKL

TEST(Pardiso, base) {
  const double epsilon = 1.0e-15;

  il::Array<double> y{il::value, {3.0, 7.0}};

  il::Array<il::StaticArray<il::int_t, 2>> position{};
  position.Append(il::StaticArray<il::int_t, 2>{il::value, {0, 0}});
  position.Append(il::StaticArray<il::int_t, 2>{il::value, {0, 1}});
  position.Append(il::StaticArray<il::int_t, 2>{il::value, {1, 0}});
  position.Append(il::StaticArray<il::int_t, 2>{il::value, {1, 1}});

  il::Array<il::int_t> index{};
  il::SparseMatrixCSR<il::int_t, double> A{2, position, il::io, index};
  A[index[0]] = 1.0;
  A[index[1]] = 2.0;
  A[index[2]] = 3.0;
  A[index[3]] = 4.0;

  il::Pardiso solver{};
  solver.SymbolicFactorization(A);
  solver.NumericalFactorization(A);

  il::Array<double> x = solver.Solve(A, y);

  ASSERT_TRUE(il::abs(x[0] - 1.0) <= epsilon && il::abs(x[1] - 1.0) <= epsilon);
}

TEST(Pardiso, heat) {
  const il::int_t n = 10;

  il::SparseMatrixCSR<il::int_t, double> A = il::heat_1d<il::int_t>(n);
  il::Array<double> x_theory{n, 1.0};
  il::Array<double> y{n, 0.0};
  il::blas(1.0, A, x_theory, 0.0, il::io, y);

  il::Pardiso solver{};
  solver.SymbolicFactorization(A);
  solver.NumericalFactorization(A);
  il::Array<double> x = solver.Solve(A, y);

  double max_err = 0.0;
  for (il::int_t i = 0; i < n; ++i) {
    max_err = il::max(max_err, il::abs(x[i] - 1.0));
  }

  const double epsilon = 1.0e-15;
  ASSERT_TRUE(max_err <= epsilon);
}

TEST(Pardiso, heat2D) {
  const il::int_t n = 10;

  il::SparseMatrixCSR<il::int_t, double> A = il::heat_2d<il::int_t>(n);
  il::Array<double> x_theory{n * n, 1.0};
  il::Array<double> y{n * n, 0.0};
  il::blas(1.0, A, x_theory, 0.0, il::io, y);

  il::Pardiso solver{};
  solver.SymbolicFactorization(A);
  solver.NumericalFactorization(A);
  il::Array<double> x = solver.Solve(A, y);

  double max_err = 0.0;
  for (il::int_t i = 0; i < x.size(); ++i) {
    max_err = il::max(max_err, il::abs(x[i] - 1.0));
  }

  const double epsilon = 1.0e-15;
  ASSERT_TRUE(max_err <= 10 * epsilon);
}

TEST(Pardiso, heat3D) {
  const il::int_t n = 10;

  il::SparseMatrixCSR<il::int_t, double> A = il::heat3d<il::int_t, double>(n);
  il::Array<double> x_theory{n * n * n, 1.0};
  il::Array<double> y{n * n * n, 0.0};
  il::blas(1.0, A, x_theory, 0.0, il::io, y);

  il::Pardiso solver{};
  solver.SymbolicFactorization(A);
  solver.NumericalFactorization(A);
  il::Array<double> x = solver.Solve(A, y);

  double max_err = 0.0;
  for (il::int_t i = 0; i < x.size(); ++i) {
    max_err = il::max(max_err, il::abs(x[i] - 1.0));
  }

  const double epsilon = 1.0e-7;
  ASSERT_TRUE(max_err <= 10 * epsilon);
}

#endif  // IL_MKL
