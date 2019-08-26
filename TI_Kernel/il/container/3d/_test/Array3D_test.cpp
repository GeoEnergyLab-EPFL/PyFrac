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

#include <il/Array3D.h>
#include <il/container/1d/_test/Dummy_test.h>

TEST(Array3D, default_constructor) {
  il::Array3D<il::int_t> A{};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.size(2) == 0 && A.capacity(2) == 0 &&
              A.data() == nullptr);
}

TEST(Array3D, size_constructor_0) {
  il::Array3D<il::int_t> A{0, 0, 0};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.size(2) == 0 && A.capacity(2) == 0 &&
              A.data() == nullptr);
}

TEST(Array3D, size_constructor_1) {
  il::Array3D<il::int_t> A{0, 2, 3};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A.size(2) == 3 && A.capacity(2) == 3);
}

TEST(Array3D, size_constructor_2) {
  il::Array3D<il::int_t> A{1, 0, 3};

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 0 &&
              A.capacity(1) == 1 && A.size(2) == 3 && A.capacity(2) == 3);
}

TEST(Array3D, size_constructor_3) {
  il::Array3D<il::int_t> A{1, 2, 0};

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A.size(2) == 0 && A.capacity(2) == 1);
}

TEST(Array3D, size_constructor_4) {
  il::Array3D<il::int_t> A{1, 0, 0};

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 0 &&
              A.capacity(1) == 1 && A.size(2) == 0 && A.capacity(2) == 1);
}

TEST(Array3D, size_constructor_5) {
  il::Array3D<il::int_t> A{0, 2, 0};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A.size(2) == 0 && A.capacity(2) == 1);
}

TEST(Array3D, size_constructor_6) {
  il::Array3D<il::int_t> A{0, 0, 3};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 0 &&
              A.capacity(1) == 1 && A.size(2) == 3 && A.capacity(2) == 3);
}

TEST(Array3D, size_constructor_7) {
  il::Array3D<il::int_t> A{1, 2, 3};

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A.size(2) == 3 && A.capacity(2) == 3);
}

TEST(Array3D, size_value_constructor_0) {
  il::int_t x = 9;
  il::Array3D<il::int_t> A{0, 0, 0, x};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.size(2) == 0 && A.capacity(2) == 0 &&
              A.data() == nullptr);
}

TEST(Array3D, size_value_constructor_1) {
  il::int_t x = 9;
  il::Array3D<il::int_t> A{0, 2, 3, x};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A.size(2) == 3 && A.capacity(2) == 3);
}

TEST(Array3D, size_value_constructor_2) {
  il::int_t x = 9;
  il::Array3D<il::int_t> A{1, 0, 3, x};

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 0 &&
              A.capacity(1) == 1 && A.size(2) == 3 && A.capacity(2) == 3);
}

TEST(Array3D, size_value_constructor_3) {
  il::int_t x = 9;
  il::Array3D<il::int_t> A{1, 2, 0, x};

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A.size(2) == 0 && A.capacity(2) == 1);
}

TEST(Array3D, size_value_constructor_4) {
  il::int_t x = 9;
  il::Array3D<il::int_t> A{1, 0, 0, x};

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 0 &&
              A.capacity(1) == 1 && A.size(2) == 0 && A.capacity(2) == 1);
}

TEST(Array3D, size_value_constructor_5) {
  il::int_t x = 9;
  il::Array3D<il::int_t> A{0, 2, 0, x};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A.size(2) == 0 && A.capacity(2) == 1);
}

TEST(Array3D, size_value_constructor_6) {
  il::int_t x = 9;
  il::Array3D<il::int_t> A{0, 0, 3, x};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 0 &&
              A.capacity(1) == 1 && A.size(2) == 3 && A.capacity(2) == 3);
}

TEST(Array3D, size_value_constructor_7) {
  il::int_t x = 9;
  il::Array3D<il::int_t> A{1, 2, 3, x};

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A.size(2) == 3 && A.capacity(2) == 3);
}

TEST(Array3D, size_value_constructor_8) {
  const il::int_t n0 = 2;
  const il::int_t n1 = 3;
  const il::int_t n2 = 4;
  const il::int_t x = 9;
  il::Array3D<il::int_t> A{n0, n1, n2, x};

  bool correct_elements{true};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        if (A(i, j, k) != x) {
          correct_elements = false;
        }
      }
    }
  }

  ASSERT_TRUE(A.size(0) == n0 && A.capacity(0) == n0 && A.size(1) == n1 &&
              A.capacity(1) == n1 && A.size(2) == n2 && A.capacity(2) == n2 &&
              correct_elements);
}

TEST(Array3D, initializer_list_constructor_0) {
  il::Array3D<il::int_t> A{il::value, {{{}}}};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 1 &&
              A.capacity(1) == 1 && A.size(2) == 1 && A.capacity(2) == 1);
}

TEST(Array3D, initializer_list_constructor_1) {
  il::Array3D<il::int_t> A{il::value, {{}}};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 1 && A.size(1) == 0 &&
              A.capacity(1) == 1 && A.size(2) == 1 && A.capacity(2) == 1);
}

TEST(Array3D, initializer_list_constructor_2) {
  il::Array3D<il::int_t> A{il::value, {}};

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.size(2) == 0 && A.capacity(2) == 0 &&
              A.data() == nullptr);
}

TEST(Array3D, initializer_list_constructor_3) {
  il::Array3D<il::int_t> A{il::value, {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}};

  ASSERT_TRUE(A.size(0) == 2 && A.capacity(0) == 2 && A.size(1) == 2 &&
              A.capacity(1) == 2 && A.size(2) == 2 && A.capacity(2) == 2 &&
              A(0, 0, 0) == 1 && A(1, 0, 0) == 2 && A(0, 1, 0) == 3 &&
              A(1, 1, 0) == 4 && A(0, 0, 1) == 5 && A(1, 0, 1) == 6 &&
              A(0, 1, 1) == 7 && A(1, 1, 1) == 8);
}

TEST(Array3D, copy_constructor) {
  const il::int_t n0 = 2;
  const il::int_t n1 = 3;
  const il::int_t n2 = 4;
  il::Array3D<il::int_t> A{n0, n1, n2};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        A(i, j, k) = i + 17 * j + 23 * k;
      }
    }
  }
  A.Reserve(2 * n0, 2 * n1, 2 * n2);

  il::Array3D<il::int_t> B{A};
  bool same_elements{true};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        if (B(i, j, k) != i + 17 * j + 23 * k) {
          same_elements = false;
        }
      }
    }
  }

  ASSERT_TRUE(B.size(0) == n0 && B.capacity(0) == n0 && B.size(1) == n1 &&
              B.capacity(1) == n1 && B.size(2) == n2 && B.capacity(2) == n2 &&
              same_elements);
}

TEST(Array3D, move_constructor) {
  const il::int_t n0 = 2;
  const il::int_t n1 = 3;
  const il::int_t n2 = 4;
  il::Array3D<il::int_t> A{n0, n1, n2};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        A(i, j, k) = i + 17 * j + 23 * k;
      }
    }
  }
  A.Reserve(2 * n0, 2 * n1, 2 * n2);
  const il::int_t* const old_A_data{A.data()};

  il::Array3D<il::int_t> B{std::move(A)};
  bool same_elements{true};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        if (B(i, j, k) != i + 17 * j + 23 * k) {
          same_elements = false;
        }
      }
    }
  }

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.size(2) == 0 && A.capacity(2) == 0 &&
              A.data() == nullptr && B.data() == old_A_data &&
              B.size(0) == n0 && B.capacity(0) == 2 * n0 && B.size(1) == n1 &&
              B.capacity(1) == 2 * n1 && B.size(2) == n2 &&
              B.capacity(2) == 2 * n2 && same_elements);
}

TEST(Array3D, copy_assignment_0) {
  const il::int_t n0 = 2;
  const il::int_t n1 = 3;
  const il::int_t n2 = 4;
  il::Array3D<il::int_t> A{n0, n1, n2};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        A(i, j, k) = i + 17 * j + 23 * k;
      }
    }
  }
  A.Reserve(2 * n0, 2 * n1, 2 * n2);

  il::Array3D<il::int_t> B{};
  B = A;
  bool same_elements{true};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        if (B(i, j, k) != i + 17 * j + 23 * k) {
          same_elements = false;
        }
      }
    }
  }

  ASSERT_TRUE(B.size(0) == n0 && B.capacity(0) == n0 && B.size(1) == n1 &&
              B.capacity(1) == n1 && B.size(2) == n2 && B.capacity(2) == n2 &&
              same_elements);
}

TEST(Array3D, copy_assignment_1) {
  const il::int_t n0 = 2;
  const il::int_t n1 = 3;
  const il::int_t n2 = 4;
  il::Array3D<il::int_t> A{n0, n1, n2};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        A(i, j, k) = i + 17 * j + 23 * k;
      }
    }
  }
  A.Reserve(2 * n0, 2 * n1, 2 * n2);

  il::Array3D<il::int_t> B{3 * n0, 3 * n1, 3 * n2};
  B = A;
  bool same_elements{true};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        if (B(i, j, k) != i + 17 * j + 23 * k) {
          same_elements = false;
        }
      }
    }
  }

  ASSERT_TRUE(B.size(0) == n0 && B.capacity(0) == 3 * n0 && B.size(1) == n1 &&
              B.capacity(1) == 3 * n1 && B.size(2) == n2 &&
              B.capacity(2) == 3 * n2 && same_elements);
}

TEST(Array3D, copy_assignment_object_0) {
  Dummy::reset();

  il::Array3D<Dummy> A{2, 3, 2};
  il::Array3D<Dummy> B{1, 2, 1};
  A = B;

  ASSERT_TRUE(Dummy::destroyed.size() == 10 && Dummy::destroyed[0] == 11 &&
              Dummy::destroyed[1] == 10 && Dummy::destroyed[2] == 9 &&
              Dummy::destroyed[3] == 8 && Dummy::destroyed[4] == 7 &&
              Dummy::destroyed[5] == 6 && Dummy::destroyed[6] == 5 &&
              Dummy::destroyed[7] == 4 && Dummy::destroyed[8] == 3 &&
              Dummy::destroyed[9] == 1);
}

TEST(Array3D, copy_assignment_object_1) {
  Dummy::reset();

  il::Array3D<Dummy> A{2, 3, 2};
  il::Array3D<Dummy> B{1, 4, 1};
  A = B;

  ASSERT_TRUE(Dummy::destroyed.size() == 12 && Dummy::destroyed[0] == 11 &&
              Dummy::destroyed[1] == 10 && Dummy::destroyed[2] == 9 &&
              Dummy::destroyed[3] == 8 && Dummy::destroyed[4] == 7 &&
              Dummy::destroyed[5] == 6 && Dummy::destroyed[6] == 5 &&
              Dummy::destroyed[7] == 4 && Dummy::destroyed[8] == 3 &&
              Dummy::destroyed[9] == 2 && Dummy::destroyed[10] == 1 &&
              Dummy::destroyed[11] == 0);
}

TEST(Array3D, move_assignment) {
  const il::int_t n0 = 2;
  const il::int_t n1 = 3;
  const il::int_t n2 = 4;
  il::Array3D<il::int_t> A{n0, n1, n2};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        A(i, j, k) = i + 17 * j + 23 * k;
      }
    }
  }
  A.Reserve(2 * n0, 2 * n1, 2 * n2);
  const il::int_t* const old_A_data{A.data()};

  il::Array3D<il::int_t> B{3 * n0, 3 * n1, 3 * n2};
  B = std::move(A);
  bool same_elements{true};
  for (il::int_t k = 0; k < n2; ++k) {
    for (il::int_t j = 0; j < n1; ++j) {
      for (il::int_t i = 0; i < n0; ++i) {
        if (B(i, j, k) != i + 17 * j + 23 * k) {
          same_elements = false;
        }
      }
    }
  }

  ASSERT_TRUE(A.size(0) == 0 && A.capacity(0) == 0 && A.size(1) == 0 &&
              A.capacity(1) == 0 && A.size(2) == 0 && A.capacity(2) == 0 &&
              A.data() == nullptr && B.data() == old_A_data &&
              B.size(0) == n0 && B.capacity(0) == 2 * n0 && B.size(1) == n1 &&
              B.capacity(1) == 2 * n1 && B.size(2) == n2 &&
              B.capacity(2) == 2 * n2 && same_elements);
}

TEST(Array3D, move_assignment_object) {
  Dummy::reset();

  il::Array3D<Dummy> A{2, 1, 2};
  il::Array3D<Dummy> B{2, 2, 2};
  A = std::move(B);

  ASSERT_TRUE(Dummy::destroyed.size() == 4 && Dummy::destroyed[0] == 3 &&
              Dummy::destroyed[1] == 2 && Dummy::destroyed[2] == 1 &&
              Dummy::destroyed[3] == 0);
}

TEST(Array3D, destructor_object) {
  Dummy::reset();
  { il::Array3D<Dummy> A{2, 2, 2}; }

  ASSERT_TRUE(Dummy::destroyed.size() == 8 && Dummy::destroyed[0] == 7 &&
              Dummy::destroyed[1] == 6 && Dummy::destroyed[2] == 5 &&
              Dummy::destroyed[3] == 4 && Dummy::destroyed[4] == 3 &&
              Dummy::destroyed[5] == 2 && Dummy::destroyed[6] == 1 &&
              Dummy::destroyed[7] == 0);
}

TEST(Array3D, bounds_checking) {
  bool test_passed{true};
  const il::int_t n0 = 2;
  const il::int_t n1 = 3;
  const il::int_t n2 = 4;
  il::Array3D<double> A{n0, n1, n2};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)A(-1, 0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(n0, 0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, -1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, n1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0, -1);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0, n2);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array3D, bounds_checking_default) {
  bool test_passed{true};
  il::Array3D<double> A{};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)A(-1, 0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, -1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0, -1);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array3D, bounds_checking_const) {
  bool test_passed{true};
  const il::int_t n0 = 2;
  const il::int_t n1 = 3;
  const il::int_t n2 = 4;
  const il::Array3D<double> A{n0, n1, n2};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)A(-1, 0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(n0, 0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, -1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, n1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0, -1);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0, n2);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array3D, bounds_checking_default_const) {
  bool test_passed{true};
  const il::Array3D<double> A{};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)A(-1, 0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, -1, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0, -1);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)A(0, 0, 0);
  } catch (il::AbortException) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array3D, resize_0) {
  il::Array3D<il::int_t> A{il::value, {{{1, 2}, {3, 4}, {5, 6}}}};
  const il::int_t* const old_A_data{A.data()};
  A.Resize(1, 2, 1);

  ASSERT_TRUE(old_A_data == A.data() && A.size(0) == 1 && A.capacity(0) == 2 &&
              A.size(1) == 2 && A.capacity(1) == 3 && A.size(2) == 1 &&
              A.capacity(2) == 1 && A(0, 0, 0) == 1 && A(0, 1, 0) == 3);
}

TEST(Array3D, resize_1) {
  il::Array3D<il::int_t> A{
      il::value, {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}}};
  A.Resize(1, 4, 1);

  ASSERT_TRUE(A.size(0) == 1 && A.capacity(0) == 1 && A.size(1) == 4 &&
              A.capacity(1) == 4 && A.size(2) == 1 && A.capacity(2) == 1 &&
              A(0, 0, 0) == 1 && A(0, 1, 0) == 3 && A(0, 2, 0) == 5);
}

TEST(Array3D, resize_2) {
  il::Array3D<il::int_t> A{
      il::value, {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}}};
  A.Resize(3, 1, 2);

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 1 &&
              A.capacity(1) == 1 && A.size(2) == 2 && A.capacity(2) == 2 &&
              A(0, 0, 0) == 1 && A(1, 0, 0) == 2 && A(0, 0, 1) == 7 &&
              A(1, 0, 1) == 8);
}

TEST(Array3D, resize_3) {
  il::Array3D<il::int_t> A{
      il::value, {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}}};
  A.Resize(3, 4, 3);

  ASSERT_TRUE(A.size(0) == 3 && A.capacity(0) == 3 && A.size(1) == 4 &&
              A.capacity(1) == 4 && A.size(2) == 3 && A.capacity(2) == 3 &&
              A(0, 0, 0) == 1 && A(1, 0, 0) == 2 && A(0, 1, 0) == 3 &&
              A(1, 1, 0) == 4 && A(0, 2, 0) == 5 && A(1, 2, 0) == 6 &&
              A(0, 0, 1) == 7 && A(1, 0, 1) == 8 && A(0, 1, 1) == 9 &&
              A(1, 1, 1) == 10 && A(0, 2, 1) == 11 && A(1, 2, 1) == 12);
}

TEST(Array3D, resize_4) {
  il::Array3D<il::int_t> A{};
  A.Resize(4, 0, 3);

  ASSERT_TRUE(A.size(0) == 4 && A.capacity(0) == 4 && A.size(1) == 0 &&
              A.capacity(1) == 1 && A.size(2) == 3 && A.capacity(2) == 3);
}

TEST(Array3D, resize_object_0) {
  Dummy::reset();
  il::Array3D<Dummy> A{2, 3, 2};
  const Dummy* const old_A_data{A.data()};
  A.Resize(1, 2, 1);

  ASSERT_TRUE(old_A_data == A.data() && A.size(0) == 1 && A.capacity(0) == 2 &&
              A.size(1) == 2 && A.capacity(1) == 3 && A.size(2) == 1 &&
              A.capacity(2) == 2 && A(0, 0, 0).id() == 0 &&
              Dummy::destroyed.size() == 10 && Dummy::destroyed[0] == 11 &&
              Dummy::destroyed[1] == 10 && Dummy::destroyed[2] == 9 &&
              Dummy::destroyed[3] == 8 && Dummy::destroyed[4] == 7 &&
              Dummy::destroyed[5] == 6 && Dummy::destroyed[6] == 5 &&
              Dummy::destroyed[7] == 4 && Dummy::destroyed[8] == 3 &&
              Dummy::destroyed[9] == 1);
}

TEST(Array3D, resize_object_1) {
  Dummy::reset();
  il::Array3D<Dummy> A{2, 3, 1};
  A.Resize(4, 1, 1);

  ASSERT_TRUE(A.size(0) == 4 && A.capacity(0) == 4 && A.size(1) == 1 &&
              A.capacity(1) == 1 && A.size(2) == 1 && A.capacity(2) == 1 &&
              Dummy::destroyed.size() == 6 && Dummy::destroyed[0] == 5 &&
              Dummy::destroyed[1] == 4 && Dummy::destroyed[2] == 3 &&
              Dummy::destroyed[3] == 2 && Dummy::destroyed[4] == -2 &&
              Dummy::destroyed[5] == -1);
}

TEST(Array3D, reserve_0) {
  const il::int_t n0 = 3;
  const il::int_t n1 = 4;
  const il::int_t n2 = 5;
  il::Array3D<il::int_t> A{n0, n1, n2};
  const il::int_t* const old_A_data{A.data()};
  A.Reserve(n0 - 1, n1 - 1, n2 - 1);

  ASSERT_TRUE(old_A_data == A.data() && A.size(0) == n0 &&
              A.capacity(0) == n0 && A.size(1) == n1 && A.capacity(1) == n1 &&
              A.size(2) == n2 && A.capacity(2) == n2);
}

TEST(Array3D, reserve_1) {
  il::Array3D<il::int_t> A{
      il::value, {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}}};
  A.Reserve(3, 4, 3);

  ASSERT_TRUE(A.size(0) == 2 && A.capacity(0) == 3 && A.size(1) == 3 &&
              A.capacity(1) == 4 && A.size(2) == 2 && A.capacity(2) == 3 &&
              A(0, 0, 0) == 1 && A(1, 0, 0) == 2 && A(0, 1, 0) == 3 &&
              A(1, 1, 0) == 4 && A(0, 2, 0) == 5 && A(1, 2, 0) == 6 &&
              A(0, 0, 1) == 7 && A(1, 0, 1) == 8 && A(0, 1, 1) == 9 &&
              A(1, 1, 1) == 10 && A(0, 2, 1) == 11 && A(1, 2, 1) == 12);
}

TEST(Array3D, reserve_object) {
  Dummy::reset();
  il::Array3D<Dummy> A{2, 3, 1};
  A.Reserve(1, 5, 1);

  ASSERT_TRUE(Dummy::destroyed.size() == 6 && Dummy::destroyed[0] == -6 &&
              Dummy::destroyed[1] == -5 && Dummy::destroyed[2] == -4 &&
              Dummy::destroyed[3] == -3 && Dummy::destroyed[4] == -2 &&
              Dummy::destroyed[5] == -1);
}

TEST(Array3D, basic) {
  il::Array3D<il::int_t> A{2, 2, 2};
  A(0, 0, 0) = 0;
  A(1, 0, 0) = 1;
  A(0, 1, 0) = 2;
  A(1, 1, 0) = 3;
  A(0, 0, 1) = 4;
  A(1, 0, 1) = 5;
  A(0, 1, 1) = 6;
  A(1, 1, 1) = 7;

  const il::int_t* const data{A.data()};
  ASSERT_TRUE(data[0] == 0 && data[1] == 1 && data[2] == 2 && data[3] == 3 &&
              data[4] == 4 && data[5] == 5 && data[6] == 6 && data[7] == 7);
}
