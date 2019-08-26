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

#include <algorithm>

#include <gtest/gtest.h>

#include <il/Array2D.h>

#ifdef IL_BLAS
#include <il/linearAlgebra/dense/factorization/Eigen.h>

bool complex_sort(std::complex<double> z0, std::complex<double> z1) {
  if (std::real(z0) == std::real(z1)) {
    return std::imag(z0) < std::imag(z1);
  } else {
    return std::real(z0) < std::real(z1);
  }
}

TEST(Eigen, test0) {
  il::Array2D<double> A{il::value,
                        {{0.0, 3.0, -2.0}, {2.0, -2.0, 2.0}, {-1.0, 0.0, 1.0}}};

  il::Status status{};
  il::Eigen<il::Array2D<double>> eigen_decomposition{A, il::io, status};
  status.AbortOnError();

  il::Array<std::complex<double>> ev = eigen_decomposition.eigenValue();
  std::sort(ev.Data(), ev.Data() + ev.size(), complex_sort);
  il::Array<std::complex<double>> result{il::value,
                                         {{-4, 0.0}, {1.0, 0.0}, {2.0, 0.0}}};
  const double epsilon = 1.0e-15;

  ASSERT_TRUE(ev.size() == 3 && std::abs(ev[0] - result[0]) <= epsilon &&
              std::abs(ev[1] - result[1]) <= epsilon &&
              std::abs(ev[2] - result[2]) <= epsilon);
}

TEST(Eigen, test1) {
  il::Array2D<double> A{il::value,
                        {{1.0, 0.0, -1.0}, {4.0, 6.0, 4.0}, {-2.0, -3.0, 0.0}}};

  il::Status status{};
  il::Eigen<il::Array2D<double>> eigen_decomposition{A, il::io, status};
  status.AbortOnError();

  il::Array<std::complex<double>> ev = eigen_decomposition.eigenValue();
  std::sort(ev.Data(), ev.Data() + ev.size(), complex_sort);
  il::Array<std::complex<double>> result{il::value,
                                         {{2.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}}};
  const double epsilon = 1.0e-5;

  ASSERT_TRUE(ev.size() == 3 && std::abs(ev[0] - result[0]) <= epsilon &&
              std::abs(ev[1] - result[1]) <= epsilon &&
              std::abs(ev[2] - result[2]) <= epsilon);
}

TEST(Eigen, test2) {
  il::Array2D<double> A{il::value,
                        {{2.0, 5.0, -3.0}, {2.0, 1.0, 4.0}, {-3.0, -5.0, 0.0}}};

  il::Status status{};
  il::Eigen<il::Array2D<double>> eigen_decomposition{A, il::io, status};
  status.AbortOnError();

  il::Array<std::complex<double>> ev = eigen_decomposition.eigenValue();
  std::sort(ev.Data(), ev.Data() + ev.size(), complex_sort);
  il::Array<std::complex<double>> result{il::value,
                                         {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}}};
  const double epsilon = 1.0e-4;
  ASSERT_TRUE(ev.size() == 3 && std::abs(ev[0] - result[0]) <= epsilon &&
              std::abs(ev[1] - result[1]) <= epsilon &&
              std::abs(ev[2] - result[2]) <= epsilon);
}

TEST(Eigen, test3) {
  il::Array2D<double> A{il::value,
                        {{0.0, 0.0, 0.0, -1.0},
                         {0.0, 0.0, 1.0, 0.0},
                         {0.0, -1.0, 0.0, 0.0},
                         {1, 0.0, 0.0, 0.0}}};

  il::Status status{};
  il::Eigen<il::Array2D<double>> eigen_decomposition{A, il::io, status};
  status.AbortOnError();

  il::Array<std::complex<double>> ev = eigen_decomposition.eigenValue();
  std::sort(ev.Data(), ev.Data() + ev.size(), complex_sort);
  il::Array<std::complex<double>> result{
      il::value, {{0.0, -1.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 1.0}}};
  const double epsilon = 1.0e-15;

  ASSERT_TRUE(ev.size() == 4 && std::abs(ev[0] - result[0]) <= epsilon &&
              std::abs(ev[1] - result[1]) <= epsilon &&
              std::abs(ev[2] - result[2]) <= epsilon &&
              std::abs(ev[3] - result[3]) <= epsilon);
}
#endif
