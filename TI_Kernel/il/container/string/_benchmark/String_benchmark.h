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

static void StringConstruct_Small_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "1234567890123456789012";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StringConstruct_Large(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "12345678901234567890123";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StringAppend_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "hello";
    il::String s1 = "world!";
    s0.Append(s1);
    benchmark::DoNotOptimize(s0.data());
    benchmark::DoNotOptimize(s1.data());
  }
}

static void StringAppend_1(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "hello";
    s0.Append("world!");
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StringJoin_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "hello";
    il::String s1 = "world";
    il::String s = il::join(s0, " ", s1, "!");

    benchmark::DoNotOptimize(s0.data());
    benchmark::DoNotOptimize(s1.data());
    benchmark::DoNotOptimize(s.data());
  }
}

// static void StringFFJoinLarge(benchmark::State& state) {
//  while (state.KeepRunning()) {
//    auto j1 = il::join("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");
//    auto j2 = il::join("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");
//    auto j3 = il::join("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");
//    auto j4 = il::join("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");
//    benchmark::DoNotOptimize(j1.data());
//    benchmark::DoNotOptimize(j2.data());
//    benchmark::DoNotOptimize(j3.data());
//    benchmark::DoNotOptimize(j4.data());
//  }
//}

BENCHMARK(StringConstruct_Small_0);
BENCHMARK(StringConstruct_Large);
BENCHMARK(StringAppend_0);
BENCHMARK(StringAppend_1);
BENCHMARK(StringJoin_0);
// BENCHMARK(StringFFJoinLarge);
