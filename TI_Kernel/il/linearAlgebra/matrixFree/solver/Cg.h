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

#ifndef IL_CG_H
#define IL_CG_H

#include <il/Array.h>
#include <il/StaticArray.h>
#include <il/Status.h>

#include "Gmres.h"
#include "mkl_blas.h"
#include "mkl_rci.h"

namespace il {

template <typename T>
class Cg {};

template <>
class Cg<double> {
 private:
  const il::FunctorArray<double>* A_;
  const il::FunctorArray<double>* B_;

  il::int_t max_nb_iterations_;
  double relative_precision_;
  double absolute_precision_;

  il::Array<double> x_;
  il::Array<double> y_;

  MKL_INT n_;
  MKL_INT rci_request_;
  il::StaticArray<MKL_INT, 128> ipar_;
  il::StaticArray<double, 128> dpar_;
  il::Array<double> tmp_;
  il::Array<double> tmp2_;

  double norm_residual_;
  il::int_t nb_iterations_;

 public:
  Cg(const il::FunctorArray<double>& A);
  Cg(const il::FunctorArray<double>& A, const il::FunctorArray<double>& B);

  il::Array<double> Solve(const il::Array<double>& y, il::io_t,
                          il::Status& status);
  il::Array<double> Solve(const il::Array<double>& y,
                          const il::Array<double>& x0, il::io_t,
                          il::Status& status);
  void Solve(il::ArrayView<double> y, il::io_t, il::ArrayEdit<double> x);
  void Solve(il::ArrayView<double> y, il::ArrayView<double> x0, il::io_t,
             il::ArrayEdit<double> x);

  void SetTrueNormForConvergence();
  void SetPreconditionnedNormForConvergence();

  void SetToSolve(il::ArrayView<double> y);
  void Next();
  void getSolution(il::io_t, il::ArrayEdit<double> x);
  double trueResidualNorm() const;
  double preconditionnedResidualNorm() const;
  double objectiveFunction() const;
  il::int_t nbIterations() const;

  void SetRelativePrecision(double relative_precision);
  void SetAbsolutionPrecision(double absolute_precision);
  void SetRelativeDivergence(double relative_divergence);
  void SetMaxNbIterations(il::int_t max_nb_iterations);

  double relativePrecision() const;
  double absolutePrecision() const;
  double relativeDivergence() const;
  il::int_t maxNbIterations() const;
};

Cg<double>::Cg(const il::FunctorArray<double>& A) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));

  il::int_t n = A.size(0);
  A_ = &A;
  B_ = nullptr;
  n_ = static_cast<MKL_INT>(n);
  x_.Resize(n);
  y_.Resize(n);
  tmp_.Resize(4 * n);
  tmp2_.Resize(n);
  relative_precision_ = 1.0e-6;
  absolute_precision_ = 0.0;
  max_nb_iterations_ = 100;
  norm_residual_ = -1.0;
  nb_iterations_ = -1;
}

Cg<double>::Cg(const il::FunctorArray<double>& A,
               const il::FunctorArray<double>& B) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));
  IL_EXPECT_FAST(B.size(0) == B.size(1));
  IL_EXPECT_FAST(A.size(0) == B.size(0));

  il::int_t n = A.size(0);
  A_ = &A;
  B_ = &B;
  n_ = static_cast<MKL_INT>(n);
  x_.Resize(n);
  y_.Resize(n);
  tmp_.Resize(4 * n);
  tmp2_.Resize(n);
  relative_precision_ = 1.0e-6;
  absolute_precision_ = 0.0;
  max_nb_iterations_ = 100;
  norm_residual_ = -1.0;
  nb_iterations_ = -1;
}

il::Array<double> Cg<double>::Solve(const il::Array<double>& y, il::io_t,
                                    il::Status& status) {
  IL_EXPECT_FAST(y.size() == n_);

  il::Array<double> x{n_, 0.0};
  dcg_init(&n_, x.data(), y.data(), &rci_request_, ipar_.Data(), dpar_.Data(),
           tmp_.Data());
  IL_ENSURE(rci_request_ == 0);
  //  ipar_[4] = static_cast<MKL_INT>(max_nb_iterations_);
  ipar_[7] = 0;
  ipar_[8] = 0;
  ipar_[9] = 1;
  ipar_[10] = (B_ == nullptr) ? 0 : 1;
  //  dpar_[0] = relative_precision_;
  //  dpar_[1] = absolute_precision_;
  dcg_check(&n_, x.data(), y.data(), &rci_request_, ipar_.Data(), dpar_.Data(),
            tmp_.Data());
  IL_ENSURE(rci_request_ == 0);

  bool stop_loop = false;
  bool max_nb_iterations_reached = false;
  bool has_converged = false;
  while (!stop_loop) {
    dcg(&n_, x.Data(), y.data(), &rci_request_, ipar_.Data(), dpar_.Data(),
        tmp_.Data());
    switch (rci_request_) {
      case 0: {
        // In this case, the solution was found with the required precision
        IL_UNREACHABLE;
      } break;
      case 1: {
        // In this case, we should compute the vector A.tmp_[0] and put it in
        // A.tmp_[n_]
        il::ArrayView<double> u{tmp_.data(), n_};
        il::ArrayEdit<double> v{tmp_.Data() + n_, n_};
        (*A_)(u, il::io, v);
      } break;
      case 2: {
        // In this case, we should do the user defined stopping test
        MKL_INT nb_iterations;
        dcg_get(&n_, x_.data(), y_.data(), &rci_request_, ipar_.data(),
                dpar_.data(), tmp_.data(), &nb_iterations);
        nb_iterations_ = nb_iterations;

        double norm_residual = std::sqrt(dpar_[4]);
        if (norm_residual <=
            relative_precision_ * std::sqrt(dpar_[2]) + absolute_precision_) {
          norm_residual_ = norm_residual;
          has_converged = true;
          stop_loop = true;
        } else if (nb_iterations_ == max_nb_iterations_) {
          max_nb_iterations_reached = true;
          stop_loop = true;
        }
      } break;
      case 3: {
        // Apply the preconditionner on tmp_[2 * n_] and store it in
        // tmp_[3 * n_];
        il::ArrayView<double> u{tmp_.data() + 2 * n_, n_};
        il::ArrayEdit<double> v{tmp_.Data() + 3 * n_, n_};
        (*B_)(u, il::io, v);
      } break;
      case -1: {
        IL_UNREACHABLE;
      } break;
      default: { IL_UNREACHABLE; } break;
    }
  }

  if (!max_nb_iterations_reached) {
    status.SetOk();
  } else {
    status.SetError(il::Error::Undefined);
  }

  return x;
}

void Cg<double>::SetToSolve(il::ArrayView<double> y) {
  IL_EXPECT_FAST(y.size() == n_);

  for (il::int_t i = 0; i < n_; ++i) {
    x_[i] = 0.0;
    y_[i] = y[i];
  };
  MKL_INT integer_one = 1;
  norm_residual_ = dnrm2(&n_, y_.data(), &integer_one);
  nb_iterations_ = 0;

  dcg_init(&n_, x_.data(), y_.data(), &rci_request_, ipar_.Data(), dpar_.Data(),
           tmp_.Data());
  IL_ENSURE(rci_request_ == 0);

  // Specifies the size of the problem. The dcg_init routine assigns this
  // number to n_. As a consequence, it does not need to be updated here.
  // ipar_[0] = n_;

  // The default value of 6 means that all messages are displayed on the
  // screen.
  // ipar_[1] = 6;

  // Do not touch ipar_[2]

  // Contains the current iteration number. The initial value is 0
  // ipar_[3] = 0;

  // Specifies the maximum number of iterations
  //  ipar_[4] = static_cast<MKL_INT>(max_nb_iterations_);

  // A value of 1, which is the default, does not output any error message
  // but gives a negative value for rci_request_
  // ipar_[5] = 1;

  // A value of 1, which is the default, does not output any warning message
  // but gives a negative value for rci_request_
  // ipar_[6] = 1;

  // At least one of the three (ipar_[7] -- ipar_[9]) parameters should be set
  // to 1

  // The default value is 1 which is related to the stopping criteria and the
  // maximum number of iterations
  ipar_[7] = 0;

  // The default value is 0 which is related to the stopping criteria and the
  // relative and absolute errors. Be careful, this case is
  // ||r_k||^2 <= eps_rel ||r_0||^2 + eps_abs
  // and contains squares
  ipar_[8] = 0;

  // The default value is 1 which is related to the stopping criteria and user
  // defined tests
  ipar_[9] = 1;

  // For a value of 0, runs the non-preconditioned algorithm
  // For a value of 1, runs the preconditioned algorithm
  ipar_[10] = (B_ == nullptr) ? 0 : 1;

  // Specifies the relative tolerance, the default value being 1.0e-6
  //  dpar_[0] = relative_precision_;

  // Specifies the absolute tolerance, the default value being 0.0
  //  dpar_[1] = absolute_precision_;

  // Specifies the square norm of the initial residual. The initial value is 0.0
  // dpar_[2] = 0.0;

  // Service variable: relative_tol * initial_residual + absolute_tol
  // The initial value is 0.0
  // dpar_[3] = 0.0;

  // Specifies the square norm of the current residual
  // dpar_[4] = 0.0;

  // Specifies the square norm of the residual from the previous step
  // dpar_[5] = 0.0;

  // Contains the alpha parameter of the conjugate gradient method
  // dpar_[6] = 0.0;

  // Contains the beta parameter of the conjugate gradient method
  // It is equal to dpar_[4] / dpar_[5]
  // dpar_[7] = 0.0;

  dcg_check(&n_, x_.data(), y_.data(), &rci_request_, ipar_.Data(),
            dpar_.Data(), tmp_.Data());
  IL_ENSURE(rci_request_ == 0);
}

void Cg<double>::Next() {
  bool stop_loop = false;
  il::int_t initial_nb_iterations = nb_iterations_;

  while (!stop_loop) {
    dcg(&n_, x_.Data(), y_.data(), &rci_request_, ipar_.Data(), dpar_.Data(),
        tmp_.Data());
    switch (rci_request_) {
      case 0: {
        // In this case, the solution was found with the required precision
        stop_loop = true;
      } break;
      case 1: {
        // In this case, we should compute the vector A.tmp_[0] and put it in
        // A.tmp_[n_]
        il::ArrayView<double> u{tmp_.data(), n_};
        il::ArrayEdit<double> v{tmp_.Data() + n_, n_};
        (*A_)(u, il::io, v);
      } break;
      case 2: {
        // In this case, we should do the user defined stopping test
        MKL_INT nb_iterations;
        dcg_get(&n_, x_.data(), y_.data(), &rci_request_, ipar_.data(),
                dpar_.data(), tmp_.data(), &nb_iterations);
        nb_iterations_ = nb_iterations;

        norm_residual_ = std::sqrt(dpar_[4]);
        if (nb_iterations_ == initial_nb_iterations + 1) {
          stop_loop = true;
        }
      } break;
      case 3: {
        // Apply the preconditionner on tmp_[2 * n_] and store it in
        // tmp_[3 * n_];
        il::ArrayView<double> u{tmp_.data() + 2 * n_, n_};
        il::ArrayEdit<double> v{tmp_.Data() + 3 * n_, n_};
        (*B_)(u, il::io, v);
      } break;
      default: { IL_ENSURE(false); } break;
    }
  }
}

double Cg<double>::trueResidualNorm() const { return norm_residual_; }

il::int_t Cg<double>::nbIterations() const { return nb_iterations_; }

void Cg<double>::getSolution(il::io_t, il::ArrayEdit<double> x) {
  for (il::int_t i = 0; i < n_; ++i) {
    x[i] = x_[i];
  }
}

void Cg<double>::SetRelativePrecision(double relative_precision) {
  IL_EXPECT_MEDIUM(relative_precision >= 0.0);

  relative_precision_ = relative_precision;
}

void Cg<double>::SetAbsolutionPrecision(double absolute_precision) {
  IL_EXPECT_MEDIUM(absolute_precision >= 0.0);

  absolute_precision_ = absolute_precision;
}

void Cg<double>::SetMaxNbIterations(il::int_t max_nb_iterations) {
  IL_EXPECT_MEDIUM(max_nb_iterations >= 0);

  max_nb_iterations_ = max_nb_iterations;
}

double Cg<double>::relativePrecision() const { return relative_precision_; }

double Cg<double>::absolutePrecision() const { return absolute_precision_; }

il::int_t Cg<double>::maxNbIterations() const { return max_nb_iterations_; }

}  // namespace il

#endif  // IL_CG_H
