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

#include <il/benchmark/memory/matrix_multiplication.h>

#include <immintrin.h>

namespace il {

void matrix_multiplication_0(const il::Array2C<double> &A,
                             const il::Array2C<double> &B,
                             il::Array2C<double> &C) {
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  for (il::int_t i = 0; i < C.size(0); ++i) {
    for (il::int_t j = 0; j < C.size(1); ++j) {
      C(i, j) = 0.0;
      for (il::int_t k = 0; k < A.size(1); ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

void matrix_multiplication_1(const il::Array2C<double> &A,
                             const il::Array2C<double> &B,
                             il::Array2C<double> &C) {
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  for (il::int_t i = 0; i < C.size(0); ++i) {
    for (il::int_t k = 0; k < A.size(1); ++k) {
      for (il::int_t j = 0; j < C.size(1); ++j) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

void matrix_multiplication_2(const il::Array2C<double> &A,
                             const il::Array2C<double> &B,
                             il::Array2C<double> &C) {
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  for (il::int_t i = 0; i < C.size(0); ++i) {
    for (il::int_t k = 0; k < A.size(1); ++k) {
#pragma omp simd
      for (il::int_t j = 0; j < C.size(1); ++j) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

void matrix_multiplication_3(const il::Array2C<double> &A,
                             const il::Array2C<double> &B,
                             il::Array2C<double> &C) {
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(A.size(1) == B.size(0));

#pragma omp parallel for
  for (il::int_t i = 0; i < C.size(0); ++i) {
    for (il::int_t k = 0; k < A.size(1); ++k) {
#pragma omp simd
      for (il::int_t j = 0; j < C.size(1); ++j) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

il::int_t minimum(il::int_t a, il::int_t b) { return a < b ? a : b; }

void matrix_multiplication_4(const il::Array2C<double> &A,
                             const il::Array2C<double> &B,
                             il::Array2C<double> &C) {
  const il::int_t n = A.size(0);
  IL_EXPECT_FAST(A.size(0) == n);
  IL_EXPECT_FAST(A.size(1) == n);
  IL_EXPECT_FAST(B.size(0) == n);
  IL_EXPECT_FAST(B.size(1) == n);
  IL_EXPECT_FAST(C.size(0) == n);
  IL_EXPECT_FAST(C.size(1) == n);

  const il::int_t block_size = 64;
#pragma omp parallel for
  for (il::int_t bk = 0; bk < n; bk += block_size) {
    for (il::int_t bj = 0; bj < n; bj += block_size) {
      for (il::int_t i = 0; i < n; ++i) {
        for (il::int_t k = bk; k < minimum(bk + block_size, n); ++k) {
#pragma omp simd
          for (il::int_t j = bj; j < minimum(bj + block_size, n); ++j) {
            C(i, j) += A(i, k) * B(k, j);
          }
        }
      }
    }
  }
}

void aux_matrix_multiplication(const double *a, const double *b, double *c,
                               il::int_t n0, il::int_t n1, il::int_t n2,
                               il::int_t lda, il::int_t ldb, il::int_t ldc) {
  const il::int_t tuning = 64;
  if (n0 <= tuning && n1 <= tuning && n2 <= tuning) {
    for (il::int_t i = 0; i < n0; ++i) {
      for (il::int_t k = 0; k < n1; ++k) {
#pragma omp simd
        for (il::int_t j = 0; j < n2; ++j) {
          c[i * ldc + j] += a[i * lda + k] * b[k * ldb + j];
        }
      }
    }
  } else {
    if (n1 >= n0 && n1 >= n2) {
      const il::int_t m1 = n1 / 2;
      aux_matrix_multiplication(a, b, c, n0, m1, n2, lda, ldb, ldc);
      aux_matrix_multiplication(a + m1, b + m1 * ldb, c, n0, n1 - m1, n2, lda,
                                ldb, ldc);
    } else if (n0 >= n1 && n0 >= n2) {
      const il::int_t m0 = n0 / 2;
      aux_matrix_multiplication(a, b, c, m0, n1, n2, lda, ldb, ldc);
      aux_matrix_multiplication(a + m0 * lda, b, c + m0 * ldc, n0 - m0, n1, n2,
                                lda, ldb, ldc);
    } else {
      const il::int_t m2 = n2 / 2;
      aux_matrix_multiplication(a, b, c, n0, n1, m2, lda, ldb, ldc);
      aux_matrix_multiplication(a, b + m2, c + m2, n0, n1, n2 - m2, lda, ldb,
                                ldc);
    }
  }
}

void matrix_multiplication_5(const il::Array2C<double> &A,
                             const il::Array2C<double> &B,
                             il::Array2C<double> &C) {
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  aux_matrix_multiplication(A.data(), B.data(), C.data(), A.size(0), A.size(1),
                            B.size(1), A.capacity(1), B.capacity(1),
                            C.capacity(1));
}

void matrix_multiplication_6(const il::Array2C<double> &A,
                             const il::Array2C<double> &B,
                             il::Array2C<double> &C) {
  IL_EXPECT_FAST(A.size(0) % 128 == 0);
  IL_EXPECT_FAST(A.size(1) % 128 == 0);
  IL_EXPECT_FAST(B.size(0) % 128 == 0);
  IL_EXPECT_FAST(A.alignment() % 32 == 0);
  IL_EXPECT_FAST(B.alignment() % 32 == 0);
  IL_EXPECT_FAST(C.alignment() % 32 == 0);

  const il::int_t m = A.size(0);
  const il::int_t p = A.size(1);
  const il::int_t n = B.size(1);
  const il::int_t i_block_size = 64;
  const il::int_t k_block_size = 128;
  const il::int_t j_block_size = 128;

  IL_EXPECT_FAST(i_block_size % 2 == 0);
  IL_EXPECT_FAST(j_block_size % 16 == 0);

#pragma omp parallel for collapse(3) schedule(guided)
  for (il::int_t ii = 0; ii < m; ii += i_block_size) {
    for (il::int_t jj = 0; jj < n; jj += j_block_size) {
      for (il::int_t kk = 0; kk < p; kk += k_block_size) {
        for (il::int_t j = jj; j < jj + j_block_size; j += 16) {
          for (il::int_t i = ii; i < ii + i_block_size; i += 2) {
            __m256d c0 = _mm256_load_pd(&C(i, j));
            __m256d c1 = _mm256_load_pd(&C(i, j + 4));
            __m256d c2 = _mm256_load_pd(&C(i, j + 8));
            __m256d c3 = _mm256_load_pd(&C(i, j + 12));
            __m256d c4 = _mm256_load_pd(&C(i + 1, j));
            __m256d c5 = _mm256_load_pd(&C(i + 1, j + 4));
            __m256d c6 = _mm256_load_pd(&C(i + 1, j + 8));
            __m256d c7 = _mm256_load_pd(&C(i + 1, j + 12));

            const double *b = &B(kk, j);
            const il::int_t ldb = B.capacity(1);
            const double *a_first = &A(i, kk);
            const double *a_second = &A(i + 1, kk);

            il::int_t kb = 0;
            for (il::int_t k = 0; k < k_block_size; ++k) {
              __m256d b0 = _mm256_load_pd(b + kb);
              __m256d b1 = _mm256_load_pd(b + kb + 4);
              __m256d b2 = _mm256_load_pd(b + kb + 8);
              __m256d b3 = _mm256_load_pd(b + kb + 12);

              __m256d a0 = _mm256_set1_pd(a_first[k]);
              __m256d a1 = _mm256_set1_pd(a_second[k]);

              c0 = _mm256_fmadd_pd(a0, b0, c0);
              c1 = _mm256_fmadd_pd(a0, b1, c1);
              c2 = _mm256_fmadd_pd(a0, b2, c2);
              c3 = _mm256_fmadd_pd(a0, b3, c3);
              c4 = _mm256_fmadd_pd(a1, b0, c4);
              c5 = _mm256_fmadd_pd(a1, b1, c5);
              c6 = _mm256_fmadd_pd(a1, b2, c6);
              c7 = _mm256_fmadd_pd(a1, b3, c7);

              kb += ldb;
            }
            _mm256_store_pd(&C(i, j), c0);
            _mm256_store_pd(&C(i, j + 4), c1);
            _mm256_store_pd(&C(i, j + 8), c2);
            _mm256_store_pd(&C(i, j + 12), c3);
            _mm256_store_pd(&C(i + 1, j), c4);
            _mm256_store_pd(&C(i + 1, j + 4), c5);
            _mm256_store_pd(&C(i + 1, j + 8), c6);
            _mm256_store_pd(&C(i + 1, j + 12), c7);
          }
        }
      }
    }
  }
}
}  // namespace il
