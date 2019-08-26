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

#ifndef IL_VECTOR_ADDITION_H
#define IL_VECTOR_ADDITION_H

#include <il/Array.h>
#include <il/benchmark/tools/timer/Benchmark.h>
#include <cstdio>

#ifdef IL_TBB
#include <tbb/tbb.h>
#endif

#ifdef IL_CILK
#include <cilk/cilk.h>
#endif

void vector_addition() {
  std::printf(
      "****************************************************************"
      "****************\n");
  std::printf("* Vector addition\n");
  std::printf(
      "****************************************************************"
      "****************\n");

  il::Array<il::int_t> size{
      il::value, {100, 1000, 10000, 100000, 1000000, 10000000, 100000000}};
  for (il::int_t n : size) {
    std::printf("Size of array: %td\n", n);

    auto vector_addition_serial = [&n](il::io_t, il::BState& state) {
      il::Array<double> v1{n, 0.0};
      il::Array<double> v2{n, 0.0};
      while (state.keep_running()) {
        for (il::int_t k = 0; k < v2.size(); ++k) {
          v2[k] += v1[k];
        }
      }
    };
    double time_serial{il::benchmark(vector_addition_serial) / n};
    std::printf("Serial: %7.3e s\n", time_serial);

#ifdef IL_OPENMP
    auto vector_addition_openmp = [&n](il::io_t, il::BState& state) {
      il::Array<double> v1{n, 0.0};
      il::Array<double> v2{n, 0.0};
      while (state.keep_running()) {
#pragma omp parallel for
        for (il::int_t k = 0; k < v2.size(); ++k) {
          v2[k] += v1[k];
        }
      }
    };
    double time_openmp{il::benchmark(vector_addition_openmp) / n};
    std::printf("OpenMP: %7.3e s, Ratio: %5.3f\n", time_openmp,
                time_serial / time_openmp);
#endif

#ifdef IL_TBB
    auto vector_addition_tbb = [&n](il::io_t, il::BState& state) {
      il::Array<double> v1{n, 0.0};
      il::Array<double> v2{n, 0.0};
      while (state.keep_running()) {
        tbb::parallel_for(
            tbb::blocked_range<il::int_t>(0, v2.size()),
            [=, &v1, &v2](const tbb::blocked_range<il::int_t>& range) {
              for (il::int_t k{range.begin()}; k < range.end(); ++k) {
                v2[k] += v1[k];
              }
            });
      }
    };
    double time_tbb{il::benchmark(vector_addition_tbb) / n};
    std::printf("   TBB: %7.3e s, Ratio: %5.3f\n", time_tbb,
                time_serial / time_tbb);
#endif

#ifdef IL_CILK
    auto vector_addition_cilk = [&n](il::io_t, il::BState& state) {
      il::Array<double> v1{n, 0.0};
      il::Array<double> v2{n, 0.0};
      while (state.keep_running()) {
        cilk_for(il::int_t k = 0; k < n; ++k) { v2[k] += v1[k]; }
      }
    };
    double time_cilk{il::benchmark(vector_addition_cilk) / n};
    std::printf("  Cilk: %7.3e s, Ratio: %5.3f\n", time_cilk,
                time_serial / time_cilk);
#endif

    std::printf("\n");
  }
}

#endif  // IL_VECTOR_ADDITION_H
