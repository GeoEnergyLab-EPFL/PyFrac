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

#ifndef IL_Pardiso_H
#define IL_Pardiso_H

#include <chrono>

#include <il/SparseMatrixCSR.h>
#include <il/container/1d/Array.h>

#ifdef IL_MKL

#include <mkl.h>

namespace il {

template <typename Int>
class PardisoSizeIntSelector {
 public:
  static void pardiso_int_t(_MKL_DSS_HANDLE_t pt, const Int *maxfct,
                            const Int *mnum, const Int *mtype, const Int *phase,
                            const Int *n, const void *a, const Int *ia,
                            const Int *ja, Int *perm, const Int *nrhs,
                            Int *iparm, const Int *msglvl, void *b, void *x,
                            Int *error);
};

template <typename Int>
void PardisoSizeIntSelector<Int>::pardiso_int_t(
    _MKL_DSS_HANDLE_t pt, const Int *maxfct, const Int *mnum, const Int *mtype,
    const Int *phase, const Int *n, const void *a, const Int *ia, const Int *ja,
    Int *perm, const Int *nrhs, Int *iparm, const Int *msglvl, void *b, void *x,
    Int *error) {
  (void)pt;
  (void)maxfct;
  (void)mnum;
  (void)mtype;
  (void)phase;
  (void)n;
  (void)a;
  (void)ia;
  (void)ja;
  (void)perm;
  (void)nrhs;
  (void)iparm;
  (void)msglvl;
  (void)b;
  (void)x;
  (void)error;
  IL_EXPECT_FAST(false);
}

template <>
void PardisoSizeIntSelector<int>::pardiso_int_t(
    _MKL_DSS_HANDLE_t pt, const int *maxfct, const int *mnum, const int *mtype,
    const int *phase, const int *n, const void *a, const int *ia, const int *ja,
    int *perm, const int *nrhs, int *iparm, const int *msglvl, void *b, void *x,
    int *error) {
  pardiso(pt, maxfct, mnum, mtype, phase, n, a, ia, ja, perm, nrhs, iparm,
          msglvl, b, x, error);
}

template <>
void PardisoSizeIntSelector<std::ptrdiff_t>::pardiso_int_t(
    _MKL_DSS_HANDLE_t pt, const std::ptrdiff_t *maxfct,
    const std::ptrdiff_t *mnum, const std::ptrdiff_t *mtype,
    const std::ptrdiff_t *phase, const std::ptrdiff_t *n, const void *a,
    const std::ptrdiff_t *ia, const std::ptrdiff_t *ja, std::ptrdiff_t *perm,
    const std::ptrdiff_t *nrhs, std::ptrdiff_t *iparm,
    const std::ptrdiff_t *msglvl, void *b, void *x, std::ptrdiff_t *error) {
  IL_EXPECT_FAST(sizeof(long long int) == sizeof(std::ptrdiff_t));
  pardiso_64(pt, reinterpret_cast<const long long int *>(maxfct),
             reinterpret_cast<const long long int *>(mnum),
             reinterpret_cast<const long long int *>(mtype),
             reinterpret_cast<const long long int *>(phase),
             reinterpret_cast<const long long int *>(n), a,
             reinterpret_cast<const long long int *>(ia),
             reinterpret_cast<const long long int *>(ja),
             reinterpret_cast<long long int *>(perm),
             reinterpret_cast<const long long int *>(nrhs),
             reinterpret_cast<long long int *>(iparm),
             reinterpret_cast<const long long int *>(msglvl), b, x,
             reinterpret_cast<long long int *>(error));
}

class Pardiso {
 private:
  il::int_t n_;
  il::int_t pardiso_nrhs_;
  il::int_t pardiso_max_fact_;
  il::int_t pardiso_mnum_;
  il::int_t pardiso_mtype_;
  il::int_t pardiso_msglvl_;
  il::int_t pardiso_iparm_[64];
  void *pardiso_pt_[64];
  bool is_symbolic_factorization_;
  bool is_numerical_factorization_;
  const double *matrix_element_;

 public:
  Pardiso();
  ~Pardiso();
  void SymbolicFactorization(const il::SparseMatrixCSR<il::int_t, double> &A);
  void NumericalFactorization(const il::SparseMatrixCSR<il::int_t, double> &A);
  il::Array<double> Solve(const il::SparseMatrixCSR<il::int_t, double> &A,
                          const il::Array<double> &y);
  il::Array<double> SolveIterative(
      const il::SparseMatrixCSR<il::int_t, double> &A,
      const il::Array<double> &y);

 private:
  void Release();
};

inline Pardiso::Pardiso() {
  n_ = 0;
  pardiso_nrhs_ = 1;

  // This is used to store multiple LU factorization using the same sparsity
  // pattern
  pardiso_max_fact_ = 1;
  pardiso_mnum_ = 1;

  // The following matrices are accepted
  // - 1: real and structurally symmetric
  // - 2: real and symmetric positive definite
  // - -2: real and symmetric indefinite
  // - 3: complex and structurally symmetric
  // - 4: complex and hermitian positive definite
  // - -4: complex and hermitian indefinite
  // - 6: complex and symmetric
  // - 11: real and nonsymmetric
  // - 13: complex and nonsymmetric
  pardiso_mtype_ = 11;

  pardiso_msglvl_ = 0;

  for (il::int_t i = 0; i < 64; ++i) {
    pardiso_iparm_[i] = 0;
  }

  // Default values
  // - 0: use default values
  // - 1: use values given in iparm
  pardiso_iparm_[0] = 1;

  // Fill-in reducing algorithm for the input matrix
  // - 0: use the minimum degree algorithm
  // - 2: use the METIS reordering scheme
  // - 3: use the parallel METIS reordering scheme. It can decrease the chrono
  // of
  //      computation on multicore computers, especially when the phase 1 takes
  //      significant chrono.
  pardiso_iparm_[1] = 2;

  // Although Pardiso is a direct solver, it allows you to be used as an
  // iterative solver using a previously computed LU factorization
  // - 0: use it as a direct solver
  // - 10 * L + K: Use an itera1tive solver with
  //   - K = 0: The direct solver is used
  //   - K = 1: CGS replaces the direct solver preconditionned by a previous LU
  //   - K = 2: CGS replaces the direct solver preconditionned by a previous
  //            Cholesky factorization
  //   - L: The Krylov subspace iteration is stopped when |dxi|/|dx0| <= 10^(-L)
  pardiso_iparm_[3] = 0;

  // Permutation
  // - 0: don't use a user permutation
  pardiso_iparm_[4] = 0;

  // Where to write the solution of A.x = y
  // - 0: write the array on x
  // - 1: overwrite the solution on y
  pardiso_iparm_[5] = 0;

  // Iterative refinement step
  // - 0: The solver uses two steps of iterative refinement when a perturbated
  //      pivot is used during the factorization
  // - n > 0: Performs at most n iterative refinements
  // - -n < 0: Performs at most n iterative refinements but the accumulation
  //   of the residual is computed using extended precision
  //
  // The number of iterations used is stored in iparm[6]
  pardiso_iparm_[7] = 0;

  // This parameter instructs pardiso_64 how to handle small pivots or zero
  // pivots
  // for unsymmetric matrices and symmetric matrices. Here, we use pivoting
  // perturbation of 1.0e-13.
  pardiso_iparm_[9] = 13;

  // Scaling
  // 0 : No Scaling vector
  // 1 : Scale the matrix so that the diagonal elements are equal to 1 and the
  //     absolute values of the off-diagonal entries are less or equal to 1.
  //     Note that in the analysis phase, you need to supply the numerical
  //     values of A in case of scaling.
  pardiso_iparm_[10] = 1;

  // Specify which system to solve
  // - 0: solve the problem A.x = b
  // - 1: solve the conjugate transpose problem AH.x = b based upon the
  //      factorization of A
  // - 2: solve the transpose problem AT.x = b based upon the factorization of A
  pardiso_iparm_[11] = 0;

  // improved accuracy using (non-)symmetric weighted  matchings
  // - 0: disable matching
  // - 1: enable matching, which is the default for nonsymmetric matrices
  //      In this case, you need to provide the values of A during the symbolic
  //      factorization
  pardiso_iparm_[12] = 1;

  // Report the number of nonzeros in the factors
  // - n < 0: enable report, -1 being the default values.
  // - n >= 0: disable the report
  pardiso_iparm_[17] = -1;

  // Report the number of MFLOPS necessary to factor the matrix A (it increases
  // the computation chrono)
  // - n < 0: enable report, -1 being the default values.
  // - n >= 0: disable the report
  pardiso_iparm_[18] = 0;

  // pivoting for symmetric indefinite matrices (useless for unsymmetric
  // matrices)
  pardiso_iparm_[20] = 0;

  // Parallel factorization control (new version)
  // - 0: uses the classic algorithm for factorization
  // - 1: uses a two-level factorization algorithm which generaly improves
  //      scalability in case of parallel factorization with many threads
  pardiso_iparm_[23] = 1;

  // parallel forward/backward solve control
  // - 0: uses a parallel algorithm for the forward/backward substitution
  // - 1: uses a serial algorithm for the forward/backward substitution
  pardiso_iparm_[24] = 0;

  // Matrix checker
  // - 0: does not check the sparse matrix representation
  // - 1: checks the sparse matrix representation. It checks if column are in
  //      increasing order in each row
  pardiso_iparm_[26] = 0;

  // Input/Output and working precision
  // - 0: double precision
  // - 1: single precision
  pardiso_iparm_[27] = 0;

  // Sparsity of the second member
  // - 0: don't expect a sparse second member
  // - Check the MKL documentation for other values
  pardiso_iparm_[30] = 0;

  // Type of indexing for the columns and the rows
  // - 0: Use Fortran indexing (starting at 1)
  // - 1: Use C indexing (starting at 0)
  pardiso_iparm_[34] = 1;

  // In-core / Out-of-core pardiso_64
  // - 0: Use In-core pardiso_64
  // - 1: Use Out-of-core pardiso_64. Use only if you need to solver very large
  //      problems that do not fit in memory. It uses the hard-drive to store
  //      the elements.
  pardiso_iparm_[59] = 0;

  for (il::int_t i = 0; i < 64; ++i) {
    pardiso_pt_[i] = nullptr;
  }

  is_symbolic_factorization_ = false;
  is_numerical_factorization_ = false;
  matrix_element_ = nullptr;
}

inline Pardiso::~Pardiso() { Release(); }

void Pardiso::SymbolicFactorization(
    const il::SparseMatrixCSR<il::int_t, double> &A) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));
  n_ = A.size(0);

  const il::int_t phase = 11;
  il::int_t error = 0;
  il::int_t i_dummy;

  if (is_numerical_factorization_) {
    Release();
  }
  PardisoSizeIntSelector<il::int_t>::pardiso_int_t(
      pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_, &phase,
      &n_, A.elementData(), A.rowData(), A.columnData(), &i_dummy,
      &pardiso_nrhs_, pardiso_iparm_, &pardiso_msglvl_, nullptr, nullptr,
      &error);
  IL_EXPECT_FAST(error == 0);

  is_symbolic_factorization_ = true;
  matrix_element_ = A.elementData();
}

void Pardiso::NumericalFactorization(
    const il::SparseMatrixCSR<il::int_t, double> &A) {
  IL_EXPECT_FAST(matrix_element_ = A.elementData());
  IL_EXPECT_FAST(is_symbolic_factorization_);
  IL_EXPECT_FAST(A.size(0) == n_);
  IL_EXPECT_FAST(A.size(1) == n_);

  const il::int_t phase = 22;
  il::int_t error = 0;
  il::int_t i_dummy;

  PardisoSizeIntSelector<il::int_t>::pardiso_int_t(
      pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_, &phase,
      &n_, A.elementData(), A.rowData(), A.columnData(), &i_dummy,
      &pardiso_nrhs_, pardiso_iparm_, &pardiso_msglvl_, nullptr, nullptr,
      &error);
  IL_EXPECT_FAST(error == 0);

  is_numerical_factorization_ = true;
}

inline il::Array<double> Pardiso::Solve(
    const il::SparseMatrixCSR<il::int_t, double> &A,
    const il::Array<double> &y) {
  IL_EXPECT_FAST(matrix_element_ = A.elementData());
  IL_EXPECT_FAST(is_numerical_factorization_);
  IL_EXPECT_FAST(A.size(0) == n_);
  IL_EXPECT_FAST(A.size(1) == n_);
  IL_EXPECT_FAST(y.size() == n_);
  il::Array<double> x{n_};

  const il::int_t phase = 33;
  il::int_t error = 0;
  il::int_t i_dummy;
  PardisoSizeIntSelector<il::int_t>::pardiso_int_t(
      pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_, &phase,
      &n_, A.elementData(), A.rowData(), A.columnData(), &i_dummy,
      &pardiso_nrhs_, pardiso_iparm_, &pardiso_msglvl_,
      const_cast<double *>(y.data()), x.Data(), &error);

  IL_EXPECT_FAST(error == 0);

  return x;
}

inline il::Array<double> Pardiso::SolveIterative(
    const il::SparseMatrixCSR<il::int_t, double> &A,
    const il::Array<double> &y) {
  IL_EXPECT_FAST(matrix_element_ = A.elementData());
  IL_EXPECT_FAST(is_numerical_factorization_);
  IL_EXPECT_FAST(A.size(0) == n_);
  IL_EXPECT_FAST(A.size(1) == n_);
  IL_EXPECT_FAST(y.size() == n_);
  il::Array<double> x{n_};

  const il::int_t old_solver = pardiso_iparm_[3];
  // 6 digits of accuracy using LU decomposition
  pardiso_iparm_[3] = 21;

  const il::int_t phase = 33;
  il::int_t error = 0;
  il::int_t i_dummy;
  PardisoSizeIntSelector<il::int_t>::pardiso_int_t(
      pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_, &phase,
      &n_, A.elementData(), A.rowData(), A.columnData(), &i_dummy,
      &pardiso_nrhs_, pardiso_iparm_, &pardiso_msglvl_,
      const_cast<double *>(y.data()), x.Data(), &error);
  IL_EXPECT_FAST(error == 0);

  pardiso_iparm_[3] = old_solver;

  return x;
}

inline void Pardiso::Release() {
  const il::int_t phase = -1;
  il::int_t error = 0;
  il::int_t i_dummy;
  if (is_symbolic_factorization_) {
    PardisoSizeIntSelector<il::int_t>::pardiso_int_t(
        pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_,
        &phase, &n_, nullptr, nullptr, nullptr, &i_dummy, &pardiso_nrhs_,
        pardiso_iparm_, &pardiso_msglvl_, nullptr, nullptr, &error);
    IL_EXPECT_FAST(error == 0);

    is_symbolic_factorization_ = false;
    is_numerical_factorization_ = false;
  }
}
}  // namespace il

#endif  // IL_MKL
#endif  // IL_Pardiso_H
