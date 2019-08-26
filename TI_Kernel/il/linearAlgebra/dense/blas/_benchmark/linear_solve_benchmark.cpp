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

// icpc -std=c++11 -O3 -xHost -ansi-alias -mkl=parallel -DNDEBUG -DIL_MKL
//   linear_solve_benchmark.cpp -o main -lbenchmark

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>
#include <il/linearAlgebra.h>

#include <benchmark/benchmark.h>

// BM_LU_ARRAY2D (Fortran ordered) is usually faster than BM_LU_ARRAY_2C
// (C ordered)
// - 20% to 30% faster on OSX 10.11.2 with Intel compiler 16.0.1

static void BM_LU_ARRAY2D_MISALIGNED(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n, il::align, il::cacheline, 16};
    il::Array<double> y{n, il::align, il::cacheline, 16};
    for (il::int_t j = 0; j < n; ++j) {
      y[j] = 1.0;
      for (il::int_t i = 0; i < n; ++i) {
        A(i, j) = 1.0 / (i + j + 1);
      }
    }
    il::Status status{};
    il::Array<double> x{
        il::linearSolve(std::move(A), std::move(y), il::io, status)};
    status.IgnoreError();
  }
}

static void BM_LU_ARRAY2D(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n};
    il::Array<double> y{n};
    for (il::int_t j = 0; j < n; ++j) {
      y[j] = 1.0;
      for (il::int_t i = 0; i < n; ++i) {
        A(i, j) = 1.0 / (i + j + 1);
      }
    }
    il::Status status{};
    il::Array<double> x{
        il::linearSolve(std::move(A), std::move(y), il::io, status)};
    status.IgnoreError();
  }
}

static void BM_LU_ARRAY2D_COPY(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n};
    il::Array<double> y{n};
    for (il::int_t j = 0; j < n; ++j) {
      y[j] = 1.0;
      for (il::int_t i = 0; i < n; ++i) {
        A(i, j) = 1.0 / (i + j + 1);
      }
    }
    il::Status status{};
    il::Array<double> x{il::linearSolve(A, y, il::io, status)};
    status.IgnoreError();
  }
}

static void BM_LU_ARRAY2D_SIMD_ALIGNED(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n, il::align, il::cacheline, 32};
    il::Array<double> y{n, il::align, il::cacheline, 32};
    for (il::int_t j = 0; j < n; ++j) {
      y[j] = 1.0;
      for (il::int_t i = 0; i < n; ++i) {
        A(i, j) = 1.0 / (i + j + 1);
      }
    }
    il::Status status{};
    il::Array<double> x{
        il::linearSolve(std::move(A), std::move(y), il::io, status)};
    status.IgnoreError();
  }
}

static void BM_LU_ARRAY2D_CACHE_ALIGNED(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n, il::align, il::cacheline};
    il::Array<double> y{n, il::align, il::cacheline};
    for (il::int_t j = 0; j < n; ++j) {
      y[j] = 1.0;
      for (il::int_t i = 0; i < n; ++i) {
        A(i, j) = 1.0 / (i + j + 1);
      }
    }
    il::Status status{};
    il::Array<double> x{
        il::linearSolve(std::move(A), std::move(y), il::io, status)};
    status.IgnoreError();
  }
}

static void BM_LU_ARRAY2C_MISALIGNED(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n{state.range_x()};
    il::Array2C<double> A{n, n, il::align, il::cacheline, 16};
    il::Array<double> y{n, il::align, il::cacheline, 16};
    for (il::int_t i = 0; i < n; ++i) {
      y[i] = 1.0;
      for (il::int_t j = 0; j < n; ++j) {
        A(i, j) = 1.0 / (i + j + 1);
      }
    }
    il::Status status{};
    il::Array<double> x{
        il::linearSolve(std::move(A), std::move(y), il::io, status)};
    status.IgnoreError();
  }
}

static void BM_LU_ARRAY2C(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n{state.range_x()};
    il::Array2C<double> A{n, n};
    il::Array<double> y{n};
    for (il::int_t i = 0; i < n; ++i) {
      y[i] = 1.0;
      for (il::int_t j = 0; j < n; ++j) {
        A(i, j) = 1.0 / (i + j + 1);
      }
    }
    il::Status status{};
    il::Array<double> x{
        il::linearSolve(std::move(A), std::move(y), il::io, status)};
    status.IgnoreError();
  }
}

static void BM_LU_ARRAY2C_SIMD_ALIGNED(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n{state.range_x()};
    il::Array2C<double> A{n, n, il::align, il::cacheline, 32};
    il::Array<double> y{n, il::align, il::cacheline, 32};
    for (il::int_t i = 0; i < n; ++i) {
      y[i] = 1.0;
      for (il::int_t j = 0; j < n; ++j) {
        A(i, j) = 1.0 / (i + j + 1);
      }
    }
    il::Status status{};
    il::Array<double> x{
        il::linearSolve(std::move(A), std::move(y), il::io, status)};
    status.IgnoreError();
  }
}

static void BM_LU_ARRAY2C_CACHE_ALIGNED(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n{state.range_x()};
    il::Array2C<double> A{n, n, il::align, il::cacheline};
    il::Array<double> y{n, il::align, il::cacheline};
    for (il::int_t i = 0; i < n; ++i) {
      y[i] = 1.0;
      for (il::int_t j = 0; j < n; ++j) {
        A(i, j) = 1.0 / (i + j + 1);
      }
    }
    il::Status status{};
    il::Array<double> x{
        il::linearSolve(std::move(A), std::move(y), il::io, status)};
    status.IgnoreError();
  }
}

// BENCHMARK(BM_LU_ARRAY2D_MISALIGNED)->Arg(10)->Arg(16)->Arg(30)->Arg(64)->Arg(100)->Arg(128)->Arg(300)->Arg(512)->Arg(1000)->Arg(2048)->Arg(3000);
// BENCHMARK(BM_LU_ARRAY2D)->Arg(10)->Arg(16)->Arg(30)->Arg(64)->Arg(100)->Arg(128)->Arg(300)->Arg(512)->Arg(1000)->Arg(2048)->Arg(3000);
// BENCHMARK(BM_LU_ARRAY2D_COPY)->Arg(10)->Arg(16)->Arg(30)->Arg(64)->Arg(100)->Arg(128)->Arg(300)->Arg(512)->Arg(1000)->Arg(2048)->Arg(3000);
// BENCHMARK(BM_LU_ARRAY2D_SIMD_ALIGNED)->Arg(10)->Arg(16)->Arg(30)->Arg(64)->Arg(100)->Arg(128)->Arg(300)->Arg(512)->Arg(1000)->Arg(2048)->Arg(3000);
// BENCHMARK(BM_LU_ARRAY2D_CACHE_ALIGNED)->Arg(10)->Arg(16)->Arg(30)->Arg(64)->Arg(100)->Arg(128)->Arg(300)->Arg(512)->Arg(1000)->Arg(2048)->Arg(3000);
// BENCHMARK(BM_LU_ARRAY2C_MISALIGNED)->Arg(10)->Arg(16)->Arg(30)->Arg(64)->Arg(100)->Arg(128)->Arg(300)->Arg(512)->Arg(1000)->Arg(2048)->Arg(3000);
// BENCHMARK(BM_LU_ARRAY2C)->Arg(10)->Arg(16)->Arg(30)->Arg(64)->Arg(100)->Arg(128)->Arg(300)->Arg(512)->Arg(1000)->Arg(2048)->Arg(3000);
// BENCHMARK(BM_LU_ARRAY2C_SIMD_ALIGNED)->Arg(10)->Arg(16)->Arg(30)->Arg(64)->Arg(100)->Arg(128)->Arg(300)->Arg(512)->Arg(1000)->Arg(2048)->Arg(3000);
// BENCHMARK(BM_LU_ARRAY2C_CACHE_ALIGNED)->Arg(10)->Arg(16)->Arg(30)->Arg(64)->Arg(100)->Arg(128)->Arg(300)->Arg(512)->Arg(1000)->Arg(2048)->Arg(3000);

BENCHMARK(BM_LU_ARRAY2D)->Arg(3000);
BENCHMARK(BM_LU_ARRAY2C)->Arg(3000);

BENCHMARK_MAIN();
