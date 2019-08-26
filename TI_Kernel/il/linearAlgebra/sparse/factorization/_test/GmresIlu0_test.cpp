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
//#include <il/linearAlgebra/sparse/factorization/GmresIlu0.h>
#include <il/linearAlgebra/sparse/factorization/_test/matrix/heat.h>

#include <il/benchmark/tools/timer/Timer.h>

#ifdef IL_MKL

//TEST(GmresIlu0, base) {
//  const double epsilon = 1.0e-15;
//
//  il::Array<double> y{il::value, {3.0, 7.0}};
//
//  il::Array<il::StaticArray<int, 2>> position{};
//  position.Append(il::StaticArray<int, 2>{il::value, {0, 0}});
//  position.Append(il::StaticArray<int, 2>{il::value, {0, 1}});
//  position.Append(il::StaticArray<int, 2>{il::value, {1, 0}});
//  position.Append(il::StaticArray<int, 2>{il::value, {1, 1}});
//
//  il::Array<int> index{};
//  il::SparseMatrixCSR<int, double> A{2, position, il::io, index};
//  A[index[0]] = 1.0;
//  A[index[1]] = 2.0;
//  A[index[2]] = 3.0;
//  A[index[3]] = 4.0;
//
//  const double relative_precision = epsilon;
//  const il::int_t max_nb_iteration = 10;
//  const il::int_t nb_restart = 5;
//  il::GmresIlu0 solver{relative_precision, max_nb_iteration, nb_restart};
//  solver.computePreconditionner(il::io, A);
//  il::Array<double> x = solver.solve(y, il::io, A);
//
//  ASSERT_TRUE(il::abs(x[0] - 1.0) <= epsilon && il::abs(x[1] - 1.0) <= epsilon);
//}
//
//TEST(GmresIlu0, heat) {
//  const il::int_t n = 10;
//
//  il::SparseMatrixCSR<int, double> A = il::heat_1d<int>(n);
//  il::Array<double> x_theory{n, 1.0};
//  il::Array<double> y{n, 0.0};
//  il::blas(1.0, A, x_theory, 0.0, il::io, y);
//
//  const double relative_precision = 1.0e-15;
//  const il::int_t max_nb_iteration = 10;
//  const il::int_t nb_restart = 5;
//  il::GmresIlu0 solver{relative_precision, max_nb_iteration, nb_restart};
//  solver.computePreconditionner(il::io, A);
//  il::Array<double> x = solver.solve(y, il::io, A);
//
//  double max_err = 0.0;
//  for (il::int_t i = 0; i < n; ++i) {
//    max_err = il::max(max_err, il::abs(x[i] - 1.0));
//  }
//
//  ASSERT_TRUE(max_err <= 10 * relative_precision);
//}
//
//TEST(GmresIlu0, heat2D) {
//  const il::int_t n = 10;
//
//  il::SparseMatrixCSR<int, double> A = il::heat_2d<int>(n);
//  il::Array<double> x_theory{A.size(0), 1.0};
//  il::Array<double> y{A.size(0), 0.0};
//  il::blas(1.0, A, x_theory, 0.0, il::io, y);
//
//  const double relative_precision = 1.0e-7;
//  const il::int_t max_nb_iteration = 100;
//  const il::int_t nb_restart = 10;
//  il::GmresIlu0 solver{relative_precision, max_nb_iteration, nb_restart};
//  solver.computePreconditionner(il::io, A);
//  il::Array<double> x = solver.solve(y, il::io, A);
//  il::int_t nb_iteration = solver.nbIterations();
//  (void)nb_iteration;
//
//  double max_err = 0.0;
//  for (il::int_t i = 0; i < x.size(); ++i) {
//    max_err = il::max(max_err, il::abs(x[i] - 1.0));
//  }
//
//  ASSERT_TRUE(max_err <= 10 * relative_precision);
//}
//
//TEST(GmresIlu0, heat3D) {
//  const il::int_t n = 10;
//
//  il::SparseMatrixCSR<int, double> A = il::heat3d<int, double>(n);
//  il::Array<double> x_theory{A.size(0), 1.0};
//  il::Array<double> y{A.size(0), 0.0};
//  il::blas(1.0, A, x_theory, 0.0, il::io, y);
//
//  const double relative_precision = 1.0e-7;
//  const il::int_t max_nb_iteration = 100;
//  const il::int_t nb_restart = 10;
//  il::GmresIlu0 solver{relative_precision, max_nb_iteration, nb_restart};
//  solver.computePreconditionner(il::io, A);
//  il::Array<double> x = solver.solve(y, il::io, A);
//
//  double max_err = 0.0;
//  for (il::int_t i = 0; i < x.size(); ++i) {
//    max_err = il::max(max_err, il::abs(x[i] - 1.0));
//  }
//
//  ASSERT_TRUE(max_err <= 10 * relative_precision);
//}

#endif  // IL_MKL
