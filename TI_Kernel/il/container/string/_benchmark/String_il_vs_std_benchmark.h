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

#include <benchmark/benchmark.h>

#include <il/String.h>
#include <string>

static void IlStringConstruct_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "123456789012345";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StdStringConstruct_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::string s0 = "123456789012345";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void IlStringConstruct_1(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "1234567890123456789012";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StdStringConstruct_1(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::string s0 = "1234567890123456789012";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void IlStringConstruct_2(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "12345678901234567890123";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StdStringConstruct_2(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::string s0 = "12345678901234567890123";
    benchmark::DoNotOptimize(s0.data());
  }
}

// static void IlStringAppend_0(benchmark::State& state) {
//  while (state.KeepRunning()) {
//    il::String s0 = "Hello";
//    il::String s1 = "world";
//    s0.Append(" ", s1, "!");
//    benchmark::DoNotOptimize(s0.data());
//  }
//}
//
// static void StdStringAppend_0(benchmark::State& state) {
//  while (state.KeepRunning()) {
//    std::string s0 = "Hello";
//    std::string s1 = "world";
//    s0 = s0 + " " + s1 + "!";
//    benchmark::DoNotOptimize(s0.data());
//  }
//}

BENCHMARK(IlStringConstruct_0);
BENCHMARK(StdStringConstruct_0);
BENCHMARK(IlStringConstruct_1);
BENCHMARK(StdStringConstruct_1);
BENCHMARK(IlStringConstruct_2);
BENCHMARK(StdStringConstruct_2);
// BENCHMARK(IlStringAppend_0);
// BENCHMARK(StdStringAppend_0);
