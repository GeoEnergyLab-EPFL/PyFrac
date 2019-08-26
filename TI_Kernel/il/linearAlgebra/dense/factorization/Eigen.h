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

#ifndef IL_EIGEN_H
#define IL_EIGEN_H

#include <complex>

#include <il/Status.h>
#include <il/container/1d/Array.h>
#include <il/container/2d/Array2D.h>
#include <il/linearAlgebra/dense/norm.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif

namespace il {

template <typename MatrixType>
class Eigen {};

template <>
class Eigen<il::Array2D<double>> {
 private:
  il::Array<double> eigen_value_;
  il::Array<double> eigen_value_r_;
  il::Array<double> eigen_value_i_;

 public:
  // Compute eigen values of A
  Eigen(il::Array2D<double> A, il::io_t, il::Status& status);

  // Get the Eigen values
  // - The precision looks bad if the matrix can't be diagonalized in C
  il::Array<std::complex<double>> eigenValue() const;
};

Eigen<il::Array2D<double>>::Eigen(il::Array2D<double> A, il::io_t,
                                  il::Status& status)
    : eigen_value_{}, eigen_value_r_{}, eigen_value_i_{} {
  IL_EXPECT_FAST(A.size(0) > 0);
  IL_EXPECT_FAST(A.size(1) > 0);
  IL_EXPECT_FAST(A.size(0) == A.size(1));

  const int layout = LAPACK_COL_MAJOR;
  const lapack_int n = static_cast<lapack_int>(A.size(0));
  const lapack_int ilo = 1;
  const lapack_int ihi = n;
  const lapack_int lda = static_cast<lapack_int>(A.capacity(0));
  il::Array<double> tau{n > 1 ? (n - 1) : 1};
  lapack_int lapack_error =
      LAPACKE_dgehrd(layout, n, ilo, ihi, A.Data(), lda, tau.Data());
  IL_EXPECT_FAST(lapack_error == 0);

  const char job = 'E';
  const char compz = 'N';
  const lapack_int ldz = 1;
  il::Array<double> z{ldz * n};
  il::Array<double> w{n};
  il::Array<double> wr{n};
  il::Array<double> wi{n};
  lapack_error = LAPACKE_dhseqr(layout, job, compz, n, ilo, ihi, A.Data(), lda,
                                wr.Data(), wi.Data(), z.Data(), ldz);

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.SetOk();
    eigen_value_ = std::move(w);
    eigen_value_r_ = std::move(wr);
    eigen_value_i_ = std::move(wi);
  } else {
    status.SetError(il::Error::MatrixEigenValueNoConvergence);
    IL_SET_SOURCE(status);
  }
}

il::Array<std::complex<double>> Eigen<il::Array2D<double>>::eigenValue() const {
  il::Array<std::complex<double>> ans{eigen_value_r_.size()};

  for (il::int_t i = 0; i < ans.size(); ++i) {
    ans[i] = std::complex<double>{eigen_value_r_[i], eigen_value_i_[i]};
  }

  return ans;
}
}  // namespace il

#endif  // IL_EIGEN_H
