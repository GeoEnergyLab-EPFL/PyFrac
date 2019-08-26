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

#ifndef IL_GMRES_H
#define IL_GMRES_H

#include <il/ArrayView.h>
#include <il/StaticArray.h>

#include "mkl_blas.h"
#include "mkl_rci.h"

namespace il {

template <typename T>
class FunctorArray {
 public:
  virtual il::int_t size(il::int_t d) const = 0;
  virtual void operator()(il::ArrayView<T> x, il::io_t,
                          il::ArrayEdit<T> y) const = 0;
};

// Most likely, this is a right-preconditionned implementation of GMRES that
// comes from the Saad paper: "A fleible Inner-outer preconditioned GMRES
// Algorithm". In order to solve:
//
// A.x = y
//
// We solve A.B.u = y where B is the inverse of our preconditioner.
// We seek to minimize ||A.B.u - y|| in Span(y, A.B.y, (A.B)^2.y, ...)
// x_0 = 0
// x_1 = Minimizes ||A.B.u - y|| in Span(y)
// x_2 = Minimizes ||A.B.u - y|| in Span(y, A.B.y)
// x_3 = Minimizes ||A.B.u - y|| in Span(y, (A.B)^2.y)
  //
template <typename T>
class Gmres {
 private:
  const FunctorArray<T>* m_;
  const FunctorArray<T>* cond_;
  il::StaticArray<int, 128> ipar_;
  il::StaticArray<double, 128> dpar_;
  int n_;
  il::Array<double> tmp_;
  il::Array<double> residual_;
  il::Array<double> trvec_;
  il::Array<double> yloc_;
  il::Array<double> ycopy_;
  il::Array<double> x_;

  il::int_t restart_iteration_;
  il::int_t max_nb_iterations_;
  double relative_precision_;
  double absolute_precision_;

  int RCI_request_;
  double norm_residual_;
  il::int_t nb_iterations_;
 public:
  Gmres(const il::FunctorArray<T>& m, il::int_t krylov_dim);
  Gmres(const il::FunctorArray<T>& m, const il::FunctorArray<T>& cond,
        il::int_t krylov_dim);
  void SetRelativePrecision(double relative_precision);
  double relativePrecision() const;
  void SetAbsolutePrecision(double absolute_precision);
  double absolutePrecision() const;
  void SetMaxNbIterations(il::int_t max_nb_iterations);
  il::int_t maxNbIterations() const;

  void SetToSolve(il::ArrayView<double> y);
  void SetToSolve(const il::Array<double>& y);
  void SetToSolve(il::ArrayView<double> y, il::ArrayView<double> x_init);

  double normResidual() const;
  void Next();
  void getSolution(il::io_t, il::ArrayEdit<double> x);
  void getSolution(il::io_t, il::Array<double>& x);


  void Solve(il::ArrayView<T> y, double relative_precision,
             il::int_t max_nb_iterations, il::io_t, il::ArrayEdit<T> x,
             il::Status& status);
  il::Array<T> solve(const il::Array<T>& y);
  il::int_t nbIterations() const;
};

template <typename T>
Gmres<T>::Gmres(const il::FunctorArray<T>& m, il::int_t krylov_dim) {
  IL_EXPECT_MEDIUM(m.size(1) == m.size(0));

  m_ = &m;
  cond_ = nullptr;

  n_ = m.size(1);
  restart_iteration_ = krylov_dim;
  max_nb_iterations_ = 100;

  const il::int_t tmp_size =
      (2 * krylov_dim + 1) * n_ + krylov_dim * (krylov_dim + 9) / 2 + 1;
  tmp_.Resize(tmp_size);
  residual_.Resize(n_);
  trvec_.Resize(n_);
  yloc_.Resize(n_);
  ycopy_.Resize(n_);
  x_.Resize(n_);
  nb_iterations_ = -1;
}

template <typename T>
Gmres<T>::Gmres(const il::FunctorArray<T>& m, const il::FunctorArray<T>& cond,
                il::int_t restart_iteration)
    : m_{&m}, cond_{&cond} {
  IL_EXPECT_MEDIUM(m.size(1) == m.size(0));
  IL_EXPECT_MEDIUM(m.size(0) == cond.size(0));
  IL_EXPECT_MEDIUM(cond.size(0) == cond.size(1));

  n_ = m.size(1);
  restart_iteration_ = restart_iteration;
  const il::int_t tmp_size = (2 * restart_iteration_ + 1) * n_ +
                             restart_iteration_ * (restart_iteration_ + 9) / 2 +
                             1;
  tmp_.Resize(tmp_size);
  residual_.Resize(n_);
  trvec_.Resize(n_);
  yloc_.Resize(n_);
  ycopy_.Resize(n_);
  x_.Resize(n_);
  nb_iterations_ = -1;
}

template <typename T>
void Gmres<T>::SetToSolve(const il::Array<double>& y) {
  this->SetToSolve(y.view());
}

template <typename T>
void Gmres<T>::SetToSolve(il::ArrayView<double> y) {
  IL_EXPECT_FAST(y.size() == n_);

  nb_iterations_ = 0;

  for (il::int_t i = 0; i < n_; ++i) {
    x_[i] = 0.0;
    yloc_[i] = y[i];
    ycopy_[i] = y[i];
  }

  int itercount = 0;

  const char l_char = 'L';
  const char n_char = 'N';
  const char u_char = 'U';
  const int one_int = 1;
  const double minus_one_double = -1.0;

  dfgmres_init(&n_, x_.data(), yloc_.data(), &RCI_request_, ipar_.Data(),
               dpar_.Data(), tmp_.Data());
  IL_EXPECT_FAST(RCI_request_ == 0);

  // ipar_[0]: The size of the matrix
  // ipar_[1]: The default value is 6 and specifies that the errors are reported
  //           to the screen
  // ipar_[2]: Contains the current stage of the computation. The initial value
  //           is 1
  // ipar_[3]: Contains the current iteration number. The initial value is 0
  // ipar_[4]: Specifies the maximum number of iterations. The default value is
  //           min(150, n)
  ipar_[4] = static_cast<int>(max_nb_iterations_);
  // ipar_[5]: If the value is not 0, is report error messages according to
  //           ipar_[1]. The default value is 1.
  // ipar_[6]: For Warnings.
  // ipar_[7]: If the value is not equal to 0, the dfgmres performs the stopping
  //           test ipar_[3] <= ipar 4. The default value is 1.
  // ipar_[8]: If the value is not equal to 0, the dfgmres performs the residual
  //           stopping test dpar_[4] <= dpar_[3]. The default value is 1.
  // ipar_[9]: For a used-defined stopping test
  // ipar_[10]: If the value is set to 0, the routine runs the
  //            non-preconditionned version of the algorithm. The default value
  //            is 0.
  ipar_[10] = (cond_ == nullptr) ? 0 : 1;
  // ipar_[13]: Contains the internal iteration counter that counts the number
  //            of iterations before the restart takes place. The initial value
  //            is 0.
  // ipar_[14]: Specifies the number of the non-restarted FGMRES iterations.
  //            To run the restarted version of the FGMRES method, assign the
  //            number of iterations to ipar_[14] before the restart.
  //            The default value is min(150, n) which means that by default,
  //            the non-restarted version of FGMRES method is used.
  ipar_[14] = static_cast<int>(restart_iteration_);
  // ipar_[30] = behaviour_zero_diagonal_;

  // dpar_[0]: Specifies the relative tolerance. The default value is 1.0e-6
  dpar_[0] = 1.0e-6;
  // dpar_[1]: Specifies the absolute tolerance. The default value is 1.0
  // dpar_[2]: Specifies the Euclidean norm of the initial residual. The initial
  //          value is 0.0
  // dpar_[3]: dpar_[0] * dpar_[2] + dpar_[1]
  //          The value for the residual under which the iteration is stopped
  //          (?)
  // dpar_[4]: Specifies the euclidean norm of the current residual.
  // dpar_[5]: Specifies the euclidean norm of the previsous residual
  // dpar_[6]: Contains the norm of the generated vector
  // dpar_[7]: Contains the tolerance for the norm of the generated vector
  // dpar_[30] = zero_diagonal_threshold_;
  // dpar_[31] = zero_diagonal_replacement_;

  dfgmres_check(&n_, x_.data(), yloc_.data(), &RCI_request_, ipar_.Data(),
                dpar_.Data(), tmp_.Data());
  IL_EXPECT_FAST(RCI_request_ == 0);

  norm_residual_ = dnrm2(&n_, ycopy_.data(), &one_int);
}


template <typename T>
double Gmres<T>::normResidual() const {
  return norm_residual_;
}

template <typename T>
void Gmres<T>::Next() {
  IL_EXPECT_FAST(RCI_request_ == 0);

  const char l_char = 'L';
  const char n_char = 'N';
  const char u_char = 'U';
  const int one_int = 1;
  const double minus_one_double = -1.0;
  bool stop_iteration = false;
  double y_norm = dnrm2(&n_, yloc_.data(), &one_int);
  int itercount = static_cast<il::int_t>(nb_iterations_);
  int begin_itercount = itercount;

  while (!stop_iteration) {
    // The beginning of the iteration
    dfgmres(&n_, x_.Data(), yloc_.Data(), &RCI_request_, ipar_.Data(),
            dpar_.Data(), tmp_.Data());
    switch (RCI_request_) {
      case 0:
        // In that case, the solution has been found with the right precision.
        // This occurs only if the stopping test is fully automatic.
        stop_iteration = true;
        break;
      case 1: {
        // This is a Free Matrix/Vector multiplication
        // It takes the input from tmp[ipar_[21]] and put it into tmp[ipar_[22]]
        il::ArrayView<double> my_x{tmp_.data() + ipar_[21] - 1, n_};
        il::ArrayEdit<double> my_y{tmp_.Data() + ipar_[22] - 1, n_};
        for (il::int_t i = 0; i < n_; ++i) {
          my_y[i] = 0.0;
        }
        (*m_)(my_x, il::io, my_y);
      } break;
      case 2: {
        ipar_[12] = 1;
        // Retrieve iteration number AND update sol
        dfgmres_get(&n_, x_.Data(), ycopy_.Data(), &RCI_request_, ipar_.data(),
                    dpar_.data(), tmp_.Data(), &itercount);
        // Compute the current true residual via MKL (Sparse) BLAS
        // routines. It multiplies the matrix A with yCopy and
        // store the result in residual.
        //--------------- To Change
        il::ArrayView<double> my_x = ycopy_.view();
        il::ArrayEdit<double> my_y = residual_.Edit();
        for (il::int_t i = 0; i < n_; ++i) {
          my_y[i] = 0.0;
        }
        (*m_)(my_x, il::io, my_y);
        // Compute: residual = A.(current x) - y
        // Note that A.(current x) is stored in residual before this operation
        daxpy(&n_, &minus_one_double, yloc_.data(), &one_int, residual_.Data(),
              &one_int);

        norm_residual_ = dnrm2(&n_, residual_.data(), &one_int);
        if (itercount == begin_itercount + 1) {
          stop_iteration = true;
        }


        // This number plays a critical role in the precision of the method
//        double error_ratio = norm_residual / y_norm;
//        if (norm_residual <= relative_precision_ * y_norm) {
//          stop_iteration = true;
//        }
      } break;
      case 3: {
        // If RCI_REQUEST=3, then apply the preconditioner on the
        // vector TMP(IPAR(22)) and put the result in vector
        // TMP(IPAR(23)). Here is the recommended usage of the
        // result produced by ILUT routine via standard MKL Sparse
        // Blas solver rout'ine mkl_dcsrtrsv
        il::ArrayView<double> my_x{tmp_.data() + ipar_[21] - 1, n_};
        il::ArrayEdit<double> my_y{tmp_.Data() + ipar_[22] - 1, n_};
        for (il::int_t i = 0; i < n_; ++i) {
          my_y[i] = my_x[i];
        }
        (*cond_)(my_x, il::io, my_y);
      } break;
      case 4:
        // If RCI_REQUEST=4, then check if the norm of the next
        // generated vector is not zero up to rounding and
        // computational errors. The norm is contained in DPAR(7)
        // parameter
        if (dpar_[6] == 0.0) {
          stop_iteration = true;
        }
        break;
      default:
        IL_EXPECT_FAST(false);
        break;
    }
  }
  nb_iterations_ = itercount;
}

template <typename T>
void Gmres<T>::getSolution(il::io_t, il::ArrayEdit<double> x) {
  ipar_[12] = 1;
  int itercount;
  for (il::int_t i = 0; i < n_; ++i) {
    x[i] = 0.0;
  }
  dfgmres_get(&n_, x.Data(), x.Data(), &RCI_request_, ipar_.data(),
              dpar_.data(), tmp_.Data(), &itercount);
}

template <typename T>
void Gmres<T>::getSolution(il::io_t, il::Array<double>& x) {
  this->getSolution(il::io, x.Edit());
}

template <typename T>
void Gmres<T>::Solve(il::ArrayView<T> y, double relative_precision,
                     il::int_t max_nb_iterations, il::io_t, il::ArrayEdit<T> x,
                     il::Status& status) {
  IL_EXPECT_FAST(y.size() == n_);
  IL_EXPECT_FAST(x.size() == n_);

  for (il::int_t i = 0; i < n_; ++i) {
    yloc_[i] = y[i];
    ycopy_[i] = y[i];
  }

  int itercount = 0;

  const char l_char = 'L';
  const char n_char = 'N';
  const char u_char = 'U';
  const int one_int = 1;
  const double minus_one_double = -1.0;

  dfgmres_init(&n_, x.data(), yloc_.data(), &RCI_request_, ipar_.Data(),
               dpar_.Data(), tmp_.Data());
  IL_EXPECT_FAST(RCI_request_ == 0);

  // ipar_[0]: The size of the matrix
  // ipar_[1]: The default value is 6 and specifies that the errors are reported
  //           to the screen
  // ipar_[2]: Contains the current stage of the computation. The initial value
  //           is 1
  // ipar_[3]: Contains the current iteration number. The initial value is 0
  // ipar_[4]: Specifies the maximum number of iterations. The default value is
  //           min(150, n)
  ipar_[4] = static_cast<int>(max_nb_iterations);
  // ipar_[5]: If the value is not 0, is report error messages according to
  //           ipar_[1]. The default value is 1.
  // ipar_[6]: For Warnings.
  // ipar_[7]: If the value is not equal to 0, the dfgmres performs the stopping
  //           test ipar_[3] <= ipar 4. The default value is 1.
  // ipar_[8]: If the value is not equal to 0, the dfgmres performs the residual
  //           stopping test dpar_[4] <= dpar_[3]. The default value is 1.
  // ipar_[9]: For a used-defined stopping test
  // ipar_[10]: If the value is set to 0, the routine runs the
  //            non-preconditionned version of the algorithm. The default value
  //            is 0.
  ipar_[10] = (cond_ == nullptr) ? 0 : 1;
  // ipar_[13]: Contains the internal iteration counter that counts the number
  //            of iterations before the restart takes place. The initial value
  //            is 0.
  // ipar_[14]: Specifies the number of the non-restarted FGMRES iterations.
  //            To run the restarted version of the FGMRES method, assign the
  //            number of iterations to ipar_[14] before the restart.
  //            The default value is min(150, n) which means that by default,
  //            the non-restarted version of FGMRES method is used.
  ipar_[14] = static_cast<int>(restart_iteration_);
  // ipar_[30] = behaviour_zero_diagonal_;

  // dpar_[0]: Specifies the relative tolerance. The default value is 1.0e-6
  dpar_[0] = 1.0e-6;
  // dpar_[1]: Specifies the absolute tolerance. The default value is 1.0
  // dpar_[2]: Specifies the Euclidean norm of the initial residual. The initial
  //          value is 0.0
  // dpar_[3]: dpar_[0] * dpar_[2] + dpar_[1]
  //          The value for the residual under which the iteration is stopped
  //          (?)
  // dpar_[4]: Specifies the euclidean norm of the current residual.
  // dpar_[5]: Specifies the euclidean norm of the previsous residual
  // dpar_[6]: Contains the norm of the generated vector
  // dpar_[7]: Contains the tolerance for the norm of the generated vector
  // dpar_[30] = zero_diagonal_threshold_;
  // dpar_[31] = zero_diagonal_replacement_;

  dfgmres_check(&n_, x.data(), yloc_.data(), &RCI_request_, ipar_.Data(),
                dpar_.Data(), tmp_.Data());
  IL_EXPECT_FAST(RCI_request_ == 0);
  bool stop_iteration = false;
  double y_norm = dnrm2(&n_, yloc_.data(), &one_int);
  while (!stop_iteration) {
    // The beginning of the iteration
    dfgmres(&n_, x.Data(), yloc_.Data(), &RCI_request_, ipar_.Data(),
            dpar_.Data(), tmp_.Data());
    switch (RCI_request_) {
      case 0:
        // In that case, the solution has been found with the right precision.
        // This occurs only if the stopping test is fully automatic.
        stop_iteration = true;
        break;
      case 1: {
        // This is a Free Matrix/Vector multiplication
        // It takes the input from tmp[ipar_[21]] and put it into tmp[ipar_[22]]
        il::ArrayView<double> my_x{tmp_.data() + ipar_[21] - 1, n_};
        il::ArrayEdit<double> my_y{tmp_.Data() + ipar_[22] - 1, n_};
        for (il::int_t i = 0; i < n_; ++i) {
          my_y[i] = 0.0;
        }
        (*m_)(my_x, il::io, my_y);
      } break;
      case 2: {
        ipar_[12] = 1;
        // Retrieve iteration number AND update sol
        dfgmres_get(&n_, x.Data(), ycopy_.Data(), &RCI_request_, ipar_.data(),
                    dpar_.data(), tmp_.Data(), &itercount);
        // Compute the current true residual via MKL (Sparse) BLAS
        // routines. It multiplies the matrix A with yCopy and
        // store the result in residual.
        //--------------- To Change
        il::ArrayView<double> my_x = ycopy_.view();
        il::ArrayEdit<double> my_y = residual_.Edit();
        for (il::int_t i = 0; i < n_; ++i) {
          my_y[i] = 0.0;
        }
        (*m_)(my_x, il::io, my_y);
        // Compute: residual = A.(current x) - y
        // Note that A.(current x) is stored in residual before this operation
        daxpy(&n_, &minus_one_double, yloc_.data(), &one_int, residual_.Data(),
              &one_int);
        // This number plays a critical role in the precision of the method
        double norm_residual = dnrm2(&n_, residual_.data(), &one_int);
        double error_ratio = norm_residual / y_norm;
        if (norm_residual <= relative_precision * y_norm) {
          stop_iteration = true;
        }
      } break;
      case 3: {
        // If RCI_REQUEST=3, then apply the preconditioner on the
        // vector TMP(IPAR(22)) and put the result in vector
        // TMP(IPAR(23)). Here is the recommended usage of the
        // result produced by ILUT routine via standard MKL Sparse
        // Blas solver rout'ine mkl_dcsrtrsv
        il::ArrayView<double> my_x{tmp_.data() + ipar_[21] - 1, n_};
        il::ArrayEdit<double> my_y{tmp_.Data() + ipar_[22] - 1, n_};
        for (il::int_t i = 0; i < n_; ++i) {
          my_y[i] = my_x[i];
        }
        (*cond_)(my_x, il::io, my_y);
      } break;
      case 4:
        // If RCI_REQUEST=4, then check if the norm of the next
        // generated vector is not zero up to rounding and
        // computational errors. The norm is contained in DPAR(7)
        // parameter
        if (dpar_[6] == 0.0) {
          stop_iteration = true;
        }
        break;
      default:
        IL_EXPECT_FAST(false);
        break;
    }
  }
  ipar_[12] = 0;
  dfgmres_get(&n_, x.Data(), yloc_.Data(), &RCI_request_, ipar_.data(),
              dpar_.data(), tmp_.Data(), &itercount);

  nb_iterations_ = itercount;
  status.SetOk();
}

template <typename T>
il::int_t Gmres<T>::nbIterations() const {
  return nb_iterations_;
}

}  // namespace il

#endif  // IL_GMRES_H
