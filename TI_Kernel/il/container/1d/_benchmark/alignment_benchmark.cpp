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

#include <il/Array.h>
#include <il/benchmark/tools/memory/memory.h>

#include <benchmark/benchmark.h>

static void BM_UNALIGNED(benchmark::State& state) {
  const int n = 35;
  while (state.KeepRunning()) {
    state.PauseTiming();
    il::Array<float> v{n, 0.0f, il::align, 32, 16};
    state.ResumeTiming();
    for (il::int_t i = 0; i < v.size(); ++i) {
      v[i] = (v[i] / 5.3f) * (v[i] * v[i] + v[i]) - (12.5f / (v[i] + 0.3f)) +
             (v[i] / (14.3f / (v[i] + 1.4f))) - (v[i] / 23.0f) +
             (14.8f / (2.4f + v[i]));
    }
  }
}

static void BM_ALIGNED(benchmark::State& state) {
  const int n = 35;
  while (state.KeepRunning()) {
    state.PauseTiming();
    il::Array<float> v{n, 0.0f, il::align, 32, 0};
    state.ResumeTiming();
    float* const w{v.data()};
#pragma omp simd aligned(w : 32)
    for (il::int_t i = 0; i < v.size(); ++i) {
      w[i] = (w[i] / 5.3f) * (w[i] * w[i] + w[i]) - (12.5f / (w[i] + 0.3f)) +
             (w[i] / (14.3f / (w[i] + 1.4f))) - (w[i] / 23.0f) +
             (14.8f / (2.4f + w[i]));
    }
  }
}

static void BM_ALIGNED_NOREMAINDER(benchmark::State& state) {
  const int n = 40;
  while (state.KeepRunning()) {
    //    state.PauseTiming();
    //    il::Array<float> v{n, 0.0f, il::align, 32, 0};
    float w[40] __attribute__((align(32, 0)));
    //    state.ResumeTiming();
    //    float* const w{v.data()};
    //    __assume(n % 8 == 0);
    //    __assume_aligned(w, 32);
    //#pragma omp simd aligned(w: 32)
    il::escape((void*)w);
    il::clobber();
    for (int i = 0; i < 40; ++i) {
      w[i] = (w[i] / 5.3f) * (w[i] * w[i] + w[i]) - (12.5f / (w[i] + 0.3f)) +
             (w[i] / (14.3f / (w[i] + 1.4f))) - (w[i] / 23.0f) +
             (14.8f / (2.4f + w[i]));
    }
    il::escape((void*)w);
    il::clobber();
  }
}

BENCHMARK(BM_UNALIGNED);
BENCHMARK(BM_ALIGNED);
BENCHMARK(BM_ALIGNED_NOREMAINDER);

BENCHMARK_MAIN()
