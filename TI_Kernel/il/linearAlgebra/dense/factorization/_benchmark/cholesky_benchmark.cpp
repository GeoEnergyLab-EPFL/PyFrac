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

#include <il/Array2D.h>
#include <il/LowerArray2D.h>
#include <il/linearAlgebra/factorization/Cholesky.h>
#include <il/linearAlgebra/factorization/LowerCholesky.h>
#include <il/math.h>

#include <benchmark/benchmark.h>

// icpc -std=c++11 -O3 -xHost -mkl=sequential -DNDEBUG -DIL_MKL
// -I/home/fayard/Documents/Projects/InsideLoop/InsideLoop
// -I/opt/gbenchmark/include
// -L/opt/gbenchmark/lib cholesky_benchmark.cpp -o main -lbenchmark
//
// The Cholesky factorization on a packed matrix is usually 2 times slower than
// with the full matrix.
// -> If you want to favor speed, use il::Array2D<double> which gives you the
//    full matrix and has a cost of n^2 elements for the memory
// -> If you want to favor low memory consumption, use il::LowerArray2D<double>
//    which gives you the half of the matrix and has a cost of n^2 / 2 elements
//    for the memory

static void BM_CHOLESKY_FULL(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const il::int_t n{state.range_x()};
    il::Array2D<double> B{n, n};
    for (il::int_t j = 0; j < n; ++j) {
      for (il::int_t i = 0; i < n; ++i) {
        B(i, j) = 1.0 / (1 + il::abs(i - j));
      }
    }
    state.ResumeTiming();
    il::Status status{};
    il::Cholesky C{std::move(B), il::io, status};
    status.AbortOnError();
  }
}

static void BM_CHOLESKY_PACKED(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const il::int_t n{state.range_x()};
    il::LowerArray2D<double> B{n};
    for (il::int_t j = 0; j < n; ++j) {
      for (il::int_t i{j}; i < n; ++i) {
        B(i, j) = 1.0 / (1 + il::abs(i - j));
      }
    }
    state.ResumeTiming();
    il::Status status{};
    il::LowerCholesky C{std::move(B), il::io, status};
    status.AbortOnError();
  }
}

BENCHMARK(BM_CHOLESKY_FULL)->Arg(100)->Arg(300)->Arg(1000)->Arg(3000);
BENCHMARK(BM_CHOLESKY_PACKED)->Arg(100)->Arg(300)->Arg(1000)->Arg(3000);

BENCHMARK_MAIN();
