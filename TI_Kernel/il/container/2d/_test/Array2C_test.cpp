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

#include <il/Array2C.h>
#include <il/container/1d/_test/Dummy_test.h>

TEST(Array2C, default_constructor) {
  il::Array2C<il::int_t> A{};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.data() == nullptr);
}

TEST(Array2C, size_constructor_0) {
  il::Array2C<il::int_t> A{0, 0};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.data() == nullptr);
}

TEST(Array2C, size_constructor_1) {
  il::Array2C<il::int_t> A{0, 3};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 3 &&
              A.capacity(1) == 3);
}

TEST(Array2C, size_constructor_2) {
  il::Array2C<il::int_t> A{3, 0};

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 0 &&
              A.capacity(1) == 1);
}

TEST(Array2C, size_constructor_3) {
  il::Array2C<il::int_t> A{3, 5};

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 5 &&
              A.capacity(1) == 5);
}

TEST(Array2C, alignment_constructor_0) {
  il::Array2C<double> A{3, 8, il::align, 32};

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 8 &&
              A.capacity(1) == 8);
}

TEST(Array2C, alignment_constructor_1) {
  il::Array2C<double> A{3, 9, il::align, 32};

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 9 &&
              A.capacity(1) == 12);
}

TEST(Array2C, alignment_constructor_2) {
  il::Array2C<double> A{3, 11, il::align, 32};

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 11 &&
              A.capacity(1) == 12);
}

TEST(Array2C, size_value_constructor_0) {
  const il::int_t x = 9;
  il::Array2C<il::int_t> A{0, 0, x};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.data() == nullptr);
}

TEST(Array2C, size_value_constructor_1) {
  const il::int_t x = 9;
  il::Array2C<il::int_t> A{0, 3, x};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 3 &&
              A.capacity(1) == 3);
}

TEST(Array2C, size_value_constructor_2) {
  const il::int_t x = 9;
  il::Array2C<il::int_t> A{3, 0, x};

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 0 &&
              A.capacity(1) == 1);
}

TEST(Array2C, size_value_constructor_3) {
  const il::int_t n = 3;
  const il::int_t m = 5;
  const il::int_t x = 9;
  il::Array2C<il::int_t> A{n, m, x};

  bool correct_elements{true};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      if (A(i, j) != x) {
        correct_elements = false;
      }
    }
  }

  ASSERT_TRUE(A.size(0) == n && A.capacity(0) == n && A.size(1) == m &&
              A.capacity(1) == m && correct_elements);
}

TEST(Array2C, initializer_list_constructor_0) {
  il::Array2C<il::int_t> A{il::value, {{}}};

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 0 &&
              A.capacity(1) == 1);
}

TEST(Array2C, initializer_list_constructor_1) {
  il::Array2C<il::int_t> A{il::value, {}};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.data() == nullptr);
}

TEST(Array2C, initializer_list_constructor_2) {
  il::Array2C<il::int_t> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A(0, 0) == 1 && A(0, 1) == 2 &&
              A(1, 0) == 3 && A(1, 1) == 4 && A(2, 0) == 5 && A(2, 1) == 6);
}

TEST(Array2C, copy_constructor) {
  const il::int_t n = 3;
  const il::int_t m = 5;
  il::Array2C<il::int_t> A{n, m};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      A(i, j) = i + 2 * j;
    }
  }
  A.Reserve(2 * n, 2 * m);

  il::Array2C<il::int_t> B{A};
  bool same_elements{true};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      if (B(i, j) != i + 2 * j) {
        same_elements = false;
      }
    }
  }

  ASSERT_TRUE(B.size(0) == n && B.capacity(0) == n && B.size(1) == m &&
              B.capacity(1) == m && same_elements);
}

TEST(Array2C, move_constructor) {
  const il::int_t n = 3;
  const il::int_t m = 5;
  il::Array2C<il::int_t> A{n, m};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      A(i, j) = i + 2 * j;
    }
  }
  A.Reserve(2 * n, 2 * m);
  const il::int_t* const old_A_data{A.data()};

  il::Array2C<il::int_t> B{std::move(A)};
  bool same_elements{true};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      if (B(i, j) != i + 2 * j) {
        same_elements = false;
      }
    }
  }

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.data() == nullptr &&
              B.data() == old_A_data && B.size(0) == n &&
              B.capacity(0) == 2 * n && B.size(1) == m &&
              B.capacity(1) == 2 * m && same_elements);
}

TEST(Array2C, copy_assignment_0) {
  const il::int_t n = 3;
  const il::int_t m = 5;
  il::Array2C<il::int_t> A{n, m};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      A(i, j) = i + 2 * j;
    }
  }
  A.Reserve(2 * n, 2 * m);

  il::Array2C<il::int_t> B{};
  B = A;
  bool same_elements{true};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      if (B(i, j) != i + 2 * j) {
        same_elements = false;
      }
    }
  }

  ASSERT_TRUE(B.size(0) == n && B.capacity(0) == n && B.size(1) == m &&
              B.capacity(1) == m && same_elements);
}

TEST(Array2C, copy_assignment_1) {
  const il::int_t n = 3;
  const il::int_t m = 5;
  il::Array2C<il::int_t> A{n, m};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      A(i, j) = i + 2 * j;
    }
  }
  A.Reserve(2 * n, 2 * m);

  il::Array2C<il::int_t> B{3 * n, 3 * m};
  B = A;
  bool same_elements{true};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      if (B(i, j) != i + 2 * j) {
        same_elements = false;
      }
    }
  }

  ASSERT_TRUE(B.size(0) == n && B.capacity(0) == 3 * n && B.size(1) == m &&
              B.capacity(1) == 3 * m && same_elements);
}

TEST(Array2C, copy_assignment_object_0) {
  Dummy::reset();

  il::Array2C<Dummy> A{3, 3};
  il::Array2C<Dummy> B{1, 2};
  A = B;

  ASSERT_TRUE(Dummy::destroyed.size() == 7 && Dummy::destroyed[0] == 8 &&
              Dummy::destroyed[1] == 7 && Dummy::destroyed[2] == 6 &&
              Dummy::destroyed[3] == 5 && Dummy::destroyed[4] == 4 &&
              Dummy::destroyed[5] == 3 && Dummy::destroyed[6] == 2);
}

TEST(Array2C, copy_assignment_object_1) {
  Dummy::reset();

  il::Array2C<Dummy> A{3, 3};
  il::Array2C<Dummy> B{1, 4};
  A = B;

  ASSERT_TRUE(Dummy::destroyed.size() == 9 && Dummy::destroyed[0] == 8 &&
              Dummy::destroyed[1] == 7 && Dummy::destroyed[2] == 6 &&
              Dummy::destroyed[3] == 5 && Dummy::destroyed[4] == 4 &&
              Dummy::destroyed[5] == 3 && Dummy::destroyed[6] == 2 &&
              Dummy::destroyed[7] == 1 && Dummy::destroyed[8] == 0);
}

TEST(Array2C, move_assignment) {
  const il::int_t n = 3;
  const il::int_t m = 5;
  il::Array2C<il::int_t> A{n, m};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      A(i, j) = i + 2 * j;
    }
  }
  A.Reserve(2 * n, 2 * m);
  const il::int_t* const old_A_data{A.data()};

  il::Array2C<il::int_t> B{3 * n, 3 * m};
  B = std::move(A);
  bool same_elements{true};
  for (il::int_t j = 0; j < m; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      if (B(i, j) != i + 2 * j) {
        same_elements = false;
      }
    }
  }

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.data() == nullptr &&
              B.data() == old_A_data && B.size(0) == n &&
              B.capacity(0) == 2 * n && B.size(1) == m &&
              B.capacity(1) == 2 * m && same_elements);
}

TEST(Array2C, move_assignment_object_0) {
  Dummy::reset();

  il::Array2C<Dummy> A{2, 2};
  il::Array2C<Dummy> B{1, 2};
  A = std::move(B);

  ASSERT_TRUE(Dummy::destroyed.size() == 4 && Dummy::destroyed[0] == 3 &&
              Dummy::destroyed[1] == 2 && Dummy::destroyed[2] == 1 &&
              Dummy::destroyed[3] == 0);
}

TEST(Array2C, destructor_object) {
  Dummy::reset();
  { il::Array2C<Dummy> A{2, 3}; }

  ASSERT_TRUE(Dummy::destroyed.size() == 6 && Dummy::destroyed[0] == 5 &&
              Dummy::destroyed[1] == 4 && Dummy::destroyed[2] == 3 &&
              Dummy::destroyed[3] == 2 && Dummy::destroyed[4] == 1 &&
              Dummy::destroyed[5] == 0);
}

TEST(Array2C, bounds_checking) {
  bool test_passed{true};
  const il::int_t n = 3;
  const il::int_t m = 5;
  il::Array2C<double> A{n, m};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)A(-1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(n, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, -1);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, m);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array2C, bounds_checking_default) {
  bool test_passed{true};
  il::Array2C<double> A{};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)A(-1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, -1);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array2C, bounds_checking_const) {
  bool test_passed{true};
  const il::int_t n = 3;
  const il::int_t m = 5;
  const il::Array2C<double> A{n, m};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)A(-1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(n, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, -1);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, m);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array2C, bounds_checking_default_const) {
  bool test_passed{true};
  const il::Array2C<double> A{};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)A(-1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, -1);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array2C, resize_0) {
  il::Array2C<il::int_t> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};
  const il::int_t* const old_A_data{A.data()};
  A.Resize(1, 2);

  ASSERT_TRUE(old_A_data == A.data() && A.size(0) == 1 && A.capacity(0) == 3 &&
              A.size(1) == 2 && A.capacity(1) == 2 && A(0, 0) == 1 &&
              A(0, 1) == 2);
}

TEST(Array2C, resize_1) {
  il::Array2C<il::int_t> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};
  A.Resize(1, 4);

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 4 &&
              A.capacity(1) == 4 && A(0, 0) == 1 && A(0, 1) == 2);
}

TEST(Array2C, resize_2) {
  il::Array2C<il::int_t> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};
  A.Resize(4, 1);

  ASSERT_TRUE(A.size(0) == 4 && A.capacity(0) == 4 && A.size(1) == 1 &&
              A.capacity(1) == 1 && A(0, 0) == 1 && A(1, 0) == 3 &&
              A(2, 0) == 5);
}

TEST(Array2C, resize_3) {
  il::Array2C<il::int_t> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};
  A.Resize(3, 4);

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 4 &&
              A.capacity(1) == 4 && A(0, 0) == 1 && A(0, 1) == 2 &&
              A(1, 0) == 3 && A(1, 1) == 4 && A(2, 0) == 5 && A(2, 1) == 6);
}

TEST(Array2C, resize_4) {
  il::Array2C<il::int_t> A{};
  A.Resize(4, 0);

  ASSERT_TRUE(A.size(0) == 4 && A.capacity(0) == 4 && A.size(1) == 0 &&
              A.capacity(1) == 1);
}

TEST(Array2C, resize_object_0) {
  Dummy::reset();
  il::Array2C<Dummy> A{2, 3};
  const Dummy* const old_A_data{A.data()};
  A.Resize(1, 1);

  ASSERT_TRUE(old_A_data == A.data() && A.size(0) == 1 && A.capacity(0) == 2 &&
              A.size(1) == 1 && A.capacity(1) == 3 && A(0, 0).id() == 0 &&
              Dummy::destroyed.size() == 5 && Dummy::destroyed[0] == 5 &&
              Dummy::destroyed[1] == 4 && Dummy::destroyed[2] == 3 &&
              Dummy::destroyed[3] == 2 && Dummy::destroyed[4] == 1);
}

TEST(Array2C, resize_object_1) {
  Dummy::reset();
  il::Array2C<Dummy> A{2, 3};
  A.Resize(4, 1);

  ASSERT_TRUE(A.size(0) == 4 && A.capacity(0) == 4 && A.size(1) == 1 &&
              A.capacity(1) == 1 && Dummy::destroyed.size() == 6 &&
              Dummy::destroyed[0] == 5 && Dummy::destroyed[1] == 4 &&
              Dummy::destroyed[2] == 2 && Dummy::destroyed[3] == 1 &&
              Dummy::destroyed[4] == -4 && Dummy::destroyed[5] == -1);
}

TEST(Array2C, reserve_0) {
  const il::int_t n = 3;
  const il::int_t m = 5;
  il::Array2C<il::int_t> A{n, m};
  const il::int_t* const old_A_data{A.data()};
  A.Reserve(n - 1, m - 1);

  ASSERT_TRUE(old_A_data == A.data() && A.size(0) == n && A.capacity(0) == n &&
              A.size(1) == m && A.capacity(1) == m);
}

TEST(Array2C, reserve_1) {
  il::Array2C<il::int_t> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};
  A.Reserve(3, 3);

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 2 &&
              A.capacity(1) == 3 && A(0, 0) == 1 && A(0, 1) == 2 &&
              A(1, 0) == 3 && A(1, 1) == 4 && A(2, 0) == 5 && A(2, 1) == 6);
}

TEST(Array2C, reserve_2) {
  il::Array2C<il::int_t> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};
  A.Reserve(2, 4);

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 2 &&
              A.capacity(1) == 4 && A(0, 0) == 1 && A(0, 1) == 2 &&
              A(1, 0) == 3 && A(1, 1) == 4 && A(2, 0) == 5 && A(2, 1) == 6);
}

TEST(Array2C, reserve_3) {
  il::Array2C<il::int_t> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};
  A.Reserve(3, 4);

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 2 &&
              A.capacity(1) == 4 && A(0, 0) == 1 && A(0, 1) == 2 &&
              A(1, 0) == 3 && A(1, 1) == 4 && A(2, 0) == 5 && A(2, 1) == 6);
}

TEST(Array2C, reserve_object_Order_c) {
  Dummy::reset();
  il::Array2C<Dummy> A{2, 3};
  A.Reserve(1, 5);

  ASSERT_TRUE(Dummy::destroyed.size() == 6 && Dummy::destroyed[0] == -6 &&
              Dummy::destroyed[1] == -5 && Dummy::destroyed[2] == -4 &&
              Dummy::destroyed[3] == -3 && Dummy::destroyed[4] == -2 &&
              Dummy::destroyed[5] == -1);
}
