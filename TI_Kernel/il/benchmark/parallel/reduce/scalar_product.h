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

#ifndef IL_SCALAR_PRODUCT_H
#define IL_SCALAR_PRODUCT_H

#include <il/Array.h>
#include <il/benchmark/tools/memory/memory.h>
#include <il/benchmark/tools/timer/Benchmark.h>
#include <cstdio>

#ifdef IL_TBB
#include <tbb/tbb.h>
#endif

#ifdef IL_CILK
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#endif

void scalar_product() {
  std::printf(
      "****************************************************************"
      "****************\n");
  std::printf("* Scalar product\n");
  std::printf(
      "****************************************************************"
      "****************\n");

  il::Array<il::int_t> size{
      il::value, {100, 1000, 10000, 100000, 1000000, 10000000, 100000000}};
  for (il::int_t n : size) {
    std::printf("Size of array: %td\n", n);

    auto scalar_product_serial = [&n](il::io_t, il::BState& state) {
      il::Array<float> v1{n, 0.0};
      il::Array<float> v2{n, 0.0};
      float sum = 0.0;
      while (state.keep_running()) {
        for (il::int_t k = 0; k < v2.size(); ++k) {
          sum += v1[k] * v2[k];
        }
      }
      il::do_not_optimize(sum);
    };
    float time_serial{il::benchmark(scalar_product_serial) / n};
    std::printf("Serial: %7.3e s\n", time_serial);

    auto scalar_product_sse = [&n](il::io_t, il::BState& state) {
      il::Array<float> v1{n, 0.0};
      il::Array<float> v2{n, 0.0};
      float* v1_data{v1.data()};
      float* v2_data{v2.data()};
      float sum = 0.0;
      while (state.keep_running()) {
        __m128 res;
        __m128 prd;
        __m128 ma;
        __m128 mb;
        for (std::size_t k = 0; k < n; k += 4) {
          ma = _mm_loadu_ps(&v1_data[k]);
          mb = _mm_loadu_ps(&v2_data[k]);
          prd = _mm_mul_ps(ma, mb);
          res = _mm_add_ps(prd, res);
        }
        prd = _mm_setzero_ps();
        res = _mm_hadd_ps(res, prd);
        res = _mm_hadd_ps(res, prd);
        _mm_store_ss(&sum, res);
      }
      il::do_not_optimize(sum);
    };
    float time_sse{il::benchmark(scalar_product_sse) / n};
    std::printf("   SSE: %7.3e s, Ratio: %5.3f\n", time_sse,
                time_serial / time_sse);

#ifdef IL_OPENMP
    auto scalar_product_openmp = [&n](il::io_t, il::BState& state) {
      il::Array<float> v1{n, 0.0};
      il::Array<float> v2{n, 0.0};
      float sum = 0.0;
      while (state.keep_running()) {
#pragma omp parallel for reduction(+ : sum)
        for (il::int_t k = 0; k < v2.size(); ++k) {
          sum += v1[k] * v2[k];
        }
      }
      il::do_not_optimize(sum);
    };
    float time_openmp{il::benchmark(scalar_product_openmp) / n};
    std::printf("OpenMP: %7.3e s, Ratio: %5.3f\n", time_openmp,
                time_serial / time_openmp);
#endif

#ifdef IL_TBB
    auto scalar_product_tbb = [&n](il::io_t, il::BState& state) {
      il::Array<float> v1{n, 0.0};
      il::Array<float> v2{n, 0.0};
      float sum = 0.0;
      while (state.keep_running()) {
        sum += tbb::parallel_reduce(
            tbb::blocked_range<il::int_t>(0, n), float{0.0},
            [=, &v1, &v2](const tbb::blocked_range<il::int_t>& range,
                          float in) -> float {
              for (il::int_t k{range.begin()}; k < range.end(); ++k) {
                in += v1[k] * v2[k];
              }
              return in;
            },
            [](float sum1, float sum2) -> float { return sum1 + sum2; });
      }
      il::do_not_optimize(sum);
    };
    float time_tbb{il::benchmark(scalar_product_tbb) / n};
    std::printf("   TBB: %7.3e s, Ratio: %5.3f\n", time_tbb,
                time_serial / time_tbb);
#endif

#ifdef IL_CILK
    auto scalar_product_cilk = [&n](il::io_t, il::BState& state) {
      il::Array<float> v1{n, 0.0};
      il::Array<float> v2{n, 0.0};
      float sum = 0.0;
      while (state.keep_running()) {
        cilk::reducer_opadd<float> sum{0.0};
        cilk_for(il::int_t k = 0; k < n; ++k) { sum += v1[k] * v2[k]; }
        sum += float{sum.get_value()};
      }
      il::do_not_optimize(sum);
    };
    float time_cilk{il::benchmark(scalar_product_cilk) / n};
    std::printf("  Cilk: %7.3e s, Ratio: %5.3f\n", time_cilk,
                time_serial / time_cilk);
#endif

    std::printf("\n");
  }
}

#endif  // IL_SCALAR_PRODUCT_H
