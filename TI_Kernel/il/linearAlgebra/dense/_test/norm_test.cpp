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

#include <il/norm.h>

#ifdef IL_BLAS
TEST(norm_staticarray, L1) {
  il::StaticArray<double, 3> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::L1), 3.5);
}

TEST(norm_staticarray, L2) {
  il::StaticArray<double, 3> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::L2), std::sqrt(5.25));
}

TEST(norm_staticarray, Linf) {
  il::StaticArray<double, 3> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::Linf), 2.0);
}

TEST(norm_array, L1) {
  il::Array<double> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::L1), 3.5);
}

TEST(norm_array, L2) {
  il::Array<double> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::L2), std::sqrt(5.25));
}

TEST(norm_array, Linf) {
  il::Array<double> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::Linf), 2.0);
}
#endif
