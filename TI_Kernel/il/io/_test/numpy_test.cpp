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

#include <il/io/numpy/numpy.h>

il::String filename = IL_FOLDER "/../gtest/tmp/b.npy";

TEST(numpy, array_0) {
  il::Array<int> v{il::value, {1, 2, 3}};

  il::Status save_status{};
  il::save(v, filename, il::io, save_status);

  il::Status load_status{};
  il::Array<int> w = il::load<il::Array<int>>(filename, il::io, load_status);

  ASSERT_TRUE(save_status.Ok() && load_status.Ok() && w.size() == 3 &&
              w[0] == v[0] && w[1] == v[1] && w[2] == v[2]);
}

TEST(numpy, array_1) {
  il::Array<double> v{il::value, {1.0, 2.0, 3.0}};

  il::Status save_status{};
  il::save(v, filename, il::io, save_status);

  il::Status load_status{};
  il::Array<double> w =
      il::load<il::Array<double>>(filename, il::io, load_status);

  ASSERT_TRUE(save_status.Ok() && load_status.Ok() && w.size() == 3 &&
              w[0] == v[0] && w[1] == v[1] && w[2] == v[2]);
}

TEST(numpy, array2d_0) {
  il::Array2D<int> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};

  il::Status save_status{};
  il::save(A, filename, il::io, save_status);

  il::Status load_status{};
  il::Array2D<int> B =
      il::load<il::Array2D<int>>(filename, il::io, load_status);

  ASSERT_TRUE(save_status.Ok() && load_status.Ok() && B.size(0) == A.size(0) &&
              B.size(1) == A.size(1) && B(0, 0) == A(0, 0) &&
              B(1, 0) == A(1, 0) && B(0, 1) == A(0, 1) && B(1, 1) == A(1, 1) &&
              B(0, 2) == A(0, 2) && B(1, 2) == A(1, 2));
}

TEST(numpy, array2d_1) {
  il::Array2D<double> A{il::value, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}};

  il::Status save_status{};
  il::save(A, filename, il::io, save_status);

  il::Status load_status{};
  il::Array2D<double> B =
      il::load<il::Array2D<double>>(filename, il::io, load_status);

  ASSERT_TRUE(save_status.Ok() && load_status.Ok() && B.size(0) == A.size(0) &&
              B.size(1) == A.size(1) && B(0, 0) == A(0, 0) &&
              B(1, 0) == A(1, 0) && B(0, 1) == A(0, 1) && B(1, 1) == A(1, 1) &&
              B(0, 2) == A(0, 2) && B(1, 2) == A(1, 2));
}
