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

#ifndef IL_SOBOL_H
#define IL_SOBOL_H

#include <il/Array2C.h>

#ifdef IL_BLAS
#include <mkl.h>
#endif

namespace il {

inline il::Array2C<double> sobol(il::int_t nb_point, il::int_t dim, double a,
                                 double b) {
  il::Array2C<double> A{nb_point, dim};

  VSLStreamStatePtr stream;
  const MKL_INT brng = VSL_BRNG_SOBOL;
  int error_code;
  error_code = vslNewStream(&stream, brng, static_cast<MKL_INT>(dim));
  error_code = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, dim * nb_point,
                            A.data(), a, b);
  error_code = vslDeleteStream(&stream);

  return A;
}

}  // namespace il

#endif  // IL_SOBOL_H
