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

#pragma once

#ifdef IL_MKL

#include "mkl_blas.h"
#include "mkl_rci.h"
#include "mkl_service.h"
#include "mkl_spblas.h"

#include <il/SparseMatrixCSR.h>
#include <il/container/1d/StaticArray.h>
#include <il/math.h>

namespace il {

class GmresIlu0 {
 private:
  il::int_t max_nb_iteration_;
  il::int_t restart_iteration_;
  double relative_precision_;
  bool preconditionner_computed_;
  il::int_t nb_iteration_;
  int behaviour_zero_diagonal_;
  double zero_diagonal_threshold_;
  double zero_diagonal_replacement_;
  il::StaticArray<int, 128> ipar_;
  il::StaticArray<double, 128> dpar_;
  il::Array<double> bilu0_;
  const double *matrix_element_;

 public:
  GmresIlu0();
  GmresIlu0(double relative_precision, int max_nb_iteration,
            int restart_iteration);
  void setRelativePrecision(double relative_precision);
  void setMaxNbIteration(il::int_t max_nb_iteration);
  void setRestartIteration(il::int_t restart_iteration);
  void computePreconditionner(il::io_t, il::SparseMatrixCSR<int, double> &A);
  il::Array<double> solve(const il::Array<double> &y, il::io_t,
                          il::SparseMatrixCSR<int, double> &A);
  il::int_t nbIterations() const;

 private:
  static void convertCToFortran(il::io_t, il::SparseMatrixCSR<int, double> &A);
  static void convertFortranToC(il::io_t, il::SparseMatrixCSR<int, double> &A);
};

inline GmresIlu0::GmresIlu0() : GmresIlu0{1.0e-3, 100, 20} {}

inline GmresIlu0::GmresIlu0(double relative_precision, int max_nb_iteration,
                            int restart_iteration)
    : ipar_{}, dpar_{}, bilu0_{} {
  behaviour_zero_diagonal_ = 0;
  zero_diagonal_threshold_ = 1.0e-16;
  zero_diagonal_replacement_ = 1.0e-10;
  relative_precision_ = relative_precision;
  max_nb_iteration_ = max_nb_iteration;
  restart_iteration_ = restart_iteration;
  preconditionner_computed_ = false;
  matrix_element_ = nullptr;
}

inline void GmresIlu0::setRelativePrecision(double relative_precision) {
  relative_precision_ = relative_precision;
}

inline void GmresIlu0::setMaxNbIteration(il::int_t max_nb_iteration) {
  max_nb_iteration_ = max_nb_iteration;
}

inline void GmresIlu0::setRestartIteration(il::int_t restart_iteration) {
  restart_iteration_ = restart_iteration;
}

inline void GmresIlu0::computePreconditionner(
    il::io_t, il::SparseMatrixCSR<int, double> &A) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));

  const int n = static_cast<int>(A.size(0));
  bilu0_.Resize(A.nbNonZeros());
  convertCToFortran(il::io, A);
  // In this example, specific for DCSRILU0 entries are set in turn:
  // ipar_[30]: Specifies how the routine operates when a zero diagonal
  //            element occurs during calculation. It his parameter is set to
  //            0, the the calculations are stopped and the routine returns a
  //            non-zero error value.
  ipar_[30] = behaviour_zero_diagonal_;
  // dpar_[30]: Specifies a small value, which is compared to the computed
  //            diagonal elements. When ipar_[30] is not 0, the diagonal
  //            elements less than dpar_[30] are set to dpar_[31]. The default
  //            value is 1.0e-16.
  dpar_[30] = zero_diagonal_threshold_;
  // dpar_[31]: See dpar_[30]
  dpar_[31] = zero_diagonal_replacement_;
  int ierr = 0;
  dcsrilu0(&n, A.elementData(), A.rowData(), A.columnData(), bilu0_.Data(),
           ipar_.data(), dpar_.data(), &ierr);
  IL_EXPECT_FAST(ierr == 0);
  preconditionner_computed_ = true;

  convertFortranToC(il::io, A);
  matrix_element_ = A.elementData();
}

inline il::Array<double> il::GmresIlu0::solve(
    const il::Array<double> &y, il::io_t, il::SparseMatrixCSR<int, double> &A) {
  IL_EXPECT_FAST(matrix_element_ == A.elementData());
  IL_EXPECT_FAST(preconditionner_computed_);
  IL_EXPECT_FAST(A.size(0) == y.size());

  const int n = static_cast<int>(A.size(0));

  convertCToFortran(il::io, A);
  il::Array<double> yloc = y;
  il::Array<double> ycopy = y;
  il::Array<double> x{n, 0.0};

  const il::int_t tmp_size = (2 * restart_iteration_ + 1) * n +
                             restart_iteration_ * (restart_iteration_ + 9) / 2 +
                             1;
  il::Array<double> tmp{tmp_size};
  il::Array<double> residual{n};
  il::Array<double> trvec{n};

  int itercount = 0;
  int RCI_request;

  const char l_char = 'L';
  const char n_char = 'N';
  const char u_char = 'U';
  const int one_int = 1;
  const double minus_one_double = -1.0;

  dfgmres_init(&n, x.data(), yloc.data(), &RCI_request, ipar_.Data(),
               dpar_.Data(), tmp.Data());
  IL_EXPECT_FAST(RCI_request == 0);

  // ipar_[0]: The size of the matrix
  // ipar_[1]: The default value is 6 and specifies that the errors are reported
  //           to the screen
  // ipar_[2]: Contains the current stage of the computation. The initial value
  //           is 1
  // ipar_[3]: Contains the current iteration number. The initial value is 0
  // ipar_[4]: Specifies the maximum number of iterations. The default value is
  //           min(150, n)
  ipar_[4] = static_cast<int>(max_nb_iteration_);
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
  ipar_[10] = 1;
  // ipar_[13]: Contains the internal iteration counter that counts the number
  //            of iterations before the restart takes place. The initial value
  //            is 0.
  // ipar_[14]: Specifies the number of the non-restarted FGMRES iterations.
  //            To run the restarted version of the FGMRES method, assign the
  //            number of iterations to ipar_[14] before the restart.
  //            The default value is min(150, n) which means that by default,
  //            the non-restarted version of FGMRES method is used.
  ipar_[14] = static_cast<int>(restart_iteration_);
  ipar_[30] = behaviour_zero_diagonal_;

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
  dpar_[30] = zero_diagonal_threshold_;
  dpar_[31] = zero_diagonal_replacement_;

  dfgmres_check(&n, x.data(), yloc.data(), &RCI_request, ipar_.Data(),
                dpar_.Data(), tmp.Data());
  IL_EXPECT_FAST(RCI_request == 0);
  bool stop_iteration = false;
  double y_norm = dnrm2(&n, yloc.data(), &one_int);
  while (!stop_iteration) {
    // The beginning of the iteration
    dfgmres(&n, x.Data(), yloc.Data(), &RCI_request, ipar_.Data(), dpar_.Data(),
            tmp.Data());
    switch (RCI_request) {
      case 0:
        // In that case, the solution has been found with the right precision.
        // This occurs only if the stopping test is fully automatic.
        stop_iteration = true;
        break;
      case 1:
        // This is a Sparse matrix/Vector multiplication
        mkl_dcsrgemv(&n_char, &n, A.elementData(), A.rowData(), A.columnData(),
                     &tmp[ipar_[21] - 1], &tmp[ipar_[22] - 1]);
        break;
      case 2:
        ipar_[12] = 1;
        // Retrieve iteration number AND update sol
        dfgmres_get(&n, x.Data(), ycopy.Data(), &RCI_request, ipar_.Data(),
                    dpar_.Data(), tmp.Data(), &itercount);
        // Compute the current true residual via MKL (Sparse) BLAS
        // routines. It multiplies the matrix A with yCopy and
        // store the result in residual.
        mkl_dcsrgemv(&n_char, &n, A.elementData(), A.rowData(), A.columnData(),
                     ycopy.data(), residual.Data());
        // Compute: residual = A.(current x) - y
        // Note that A.(current x) is stored in residual before this operation
        daxpy(&n, &minus_one_double, yloc.data(), &one_int, residual.Data(),
              &one_int);
        // This number plays a critical role in the precision of the method
        if (dnrm2(&n, residual.data(), &one_int) <=
            relative_precision_ * y_norm) {
          stop_iteration = true;
        }
        break;
      case 3:
        // If RCI_REQUEST=3, then apply the preconditioner on the
        // vector TMP(IPAR(22)) and put the result in vector
        // TMP(IPAR(23)). Here is the recommended usage of the
        // result produced by ILUT routine via standard MKL Sparse
        // Blas solver rout'ine mkl_dcsrtrsv
        mkl_dcsrtrsv(&l_char, &n_char, &u_char, &n, bilu0_.data(), A.rowData(),
                     A.columnData(), &tmp[ipar_[21] - 1], trvec.Data());
        mkl_dcsrtrsv(&u_char, &n_char, &n_char, &n, bilu0_.data(), A.rowData(),
                     A.columnData(), trvec.data(), &tmp[ipar_[22] - 1]);
        break;
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
  dfgmres_get(&n, x.Data(), yloc.Data(), &RCI_request, ipar_.data(),
              dpar_.data(), tmp.Data(), &itercount);

  nb_iteration_ = itercount;

  convertFortranToC(il::io, A);

  return x;
}

inline il::int_t il::GmresIlu0::nbIterations() const { return nb_iteration_; }

inline void GmresIlu0::convertCToFortran(il::io_t,
                                         il::SparseMatrixCSR<int, double> &A) {
  int n = static_cast<int>(A.size(0));
  int *row = A.RowData();
  for (int i = 0; i < n + 1; ++i) {
    row[i] += 1;
  }
  int *column = A.ColumnData();
  for (int k = 0; k < A.nbNonZeros(); ++k) {
    column[k] += 1;
  }
}

inline void GmresIlu0::convertFortranToC(il::io_t,
                                         il::SparseMatrixCSR<int, double> &A) {
  int n = static_cast<int>(A.size(0));
  int *row = A.RowData();
  for (int i = 0; i < n + 1; ++i) {
    row[i] -= 1;
  }
  int *column = A.ColumnData();
  for (int k = 0; k < A.nbNonZeros(); ++k) {
    column[k] -= 1;
  }
}
}  // namespace il

#endif  // IL_MKL
