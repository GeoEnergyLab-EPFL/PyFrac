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

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <il/Map.h>

TEST(Map, constructor_default_0) {
  il::Map<int, int> map{};

  ASSERT_TRUE(map.nbElements() == 0 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 0);
}

TEST(Map, constructor_initializer_list_0) {
  il::Map<int, int> map{il::value, {{0, 0}}};

  ASSERT_TRUE(map.nbElements() == 1 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 2);
}

TEST(Map, constructor_initializer_list_1) {
  il::Map<int, int> map{il::value, {{0, 0}, {1, 1}}};

  ASSERT_TRUE(map.nbElements() == 2 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 4);
}

TEST(Map, constructor_initializer_list_3) {
  il::Map<int, int> map{il::value, {{0, 0}, {1, 1}, {2, 2}}};

  ASSERT_TRUE(map.nbElements() == 3 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 4);
}

TEST(Map, constructor_initializer_list_4) {
  il::Map<int, int> map{il::value, {{0, 0}, {1, 1}, {2, 2}, {3, 3}}};

  ASSERT_TRUE(map.nbElements() == 4 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 8);
}
