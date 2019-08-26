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

#include <string>
#include <unordered_map>

#include <il/Array.h>
#include <il/Map.h>
#include <il/String.h>

static void IlMapInt_Set(benchmark::State& state) {
  const il::int_t n = 100;
  il::Array<int> v{n};
  unsigned int k = 1234567891;
  for (il::int_t i = 0; i < n; ++i) {
    v[i] = static_cast<int>(k / 2);
    k = 93u * k + 117u;
  }

  while (state.KeepRunning()) {
    il::Map<int, int> map{n};
    for (il::int_t i = 0; i < n; ++i) {
      map.Set(v[i], 0);
    }
  }
}

static void StdMapInt_Set(benchmark::State& state) {
  const std::size_t n = 100;
  il::Array<int> v{n};
  unsigned int k = 1234567891;
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<int>(k / 2);
    k = 93u * k + 117u;
  }

  while (state.KeepRunning()) {
    std::unordered_map<int, int> map{n};
    for (std::size_t i = 0; i < n; ++i) {
      map[v[i]] = 0;
    }
  }
}

static void IlMapString_Set(benchmark::State& state) {
  const il::int_t n = 100;
  il::Array<il::String> v{n};
  unsigned int k = 1234567891;
  for (il::int_t i = 0; i < n; ++i) {
    v[i] = il::String{il::StringType::Byte, reinterpret_cast<const char*>(&k),
                      sizeof(unsigned int)};
    k = 93u * k + 117u;
  }

  while (state.KeepRunning()) {
    il::Map<il::String, int> map{n};
    for (il::int_t i = 0; i < n; ++i) {
      map.Set(v[i], 0);
    }
  }
}

static void StdMapString_Set(benchmark::State& state) {
  const std::size_t n = 100;
  il::Array<std::string> v{n};
  unsigned int k = 1234567891;
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = std::string{reinterpret_cast<const char*>(&k), sizeof(unsigned int)};
    k = 93u * k + 117u;
  }

  while (state.KeepRunning()) {
    std::unordered_map<std::string, int> map{n};
    for (std::size_t i = 0; i < n; ++i) {
      map[v[i]] = 0;
    }
  }
}

static void IlMapString_Search(benchmark::State& state) {
  const il::int_t n = 100;
  il::Array<il::String> v{n};
  unsigned int k = 1234567891;
  for (il::int_t i = 0; i < n; ++i) {
    v[i] = il::String{il::StringType::Byte, reinterpret_cast<const char*>(&k),
                      sizeof(unsigned int)};
    k = 93u * k + 117u;
  }
  il::Map<il::String, int> map{n};
  for (il::int_t i = 0; i < n; ++i) {
    map.Set(v[i], 0);
  }

  while (state.KeepRunning()) {
    int sum = 0;
    for (il::int_t i = 0; i < n; ++i) {
      const il::spot_t j = map.search(v[i]);
      sum += map.value(j);
    }
    benchmark::DoNotOptimize(&sum);
  }
}

static void StdMapString_Search(benchmark::State& state) {
  const std::size_t n = 100;
  il::Array<std::string> v{n};
  unsigned int k = 1234567891;
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = std::string{reinterpret_cast<const char*>(&k), sizeof(unsigned int)};
    k = 93u * k + 117u;
  }
  std::unordered_map<std::string, int> map{};
  for (std::size_t i = 0; i < n; ++i) {
    map[v[i]] = 0;
  }

  while (state.KeepRunning()) {
    int sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sum += map[v[i]];
    }
    benchmark::DoNotOptimize(&sum);
  }
}

BENCHMARK(IlMapInt_Set);
BENCHMARK(StdMapInt_Set);
BENCHMARK(IlMapString_Set);
BENCHMARK(StdMapString_Set);
BENCHMARK(IlMapString_Search);
BENCHMARK(StdMapString_Search);
