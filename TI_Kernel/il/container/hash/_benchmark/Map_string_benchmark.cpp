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

#include <iostream>

#include <benchmark/benchmark.h>

#include <il/Map.h>
#include <il/String.h>

static void BM_MapCString(benchmark::State& state) {
  il::Map<il::String, int> map{7};
  map.Set("aaa", 0);
  map.Set("bbb", 0);
  map.Set("ccc", 0);
  map.Set("ddd", 0);
  map.Set("eee", 0);
  map.Set("fff", 0);
  map.Set("ggg", 0);
  il::int_t i;
  while (state.KeepRunning()) {
    i += map.searchCString("aaa");
    i += map.searchCString("bbb");
    i += map.searchCString("ccc");
    i += map.searchCString("ddd");
    i += map.searchCString("eee");
    i += map.searchCString("fff");
    i += map.searchCString("ggg");
  }
  std::cout << i << std::endl;
}

static void BM_SetMapCString(benchmark::State& state) {
  il::Map<il::String, int> map{7};
  while (state.KeepRunning()) {
    map.setCString("aaa", 0);
    map.setCString("bbb", 0);
    map.setCString("ccc", 0);
    map.setCString("ddd", 0);
    map.setCString("eee", 0);
    map.setCString("fff", 0);
    map.setCString("ggg", 0);
  }
}

static void BM_Map(benchmark::State& state) {
  il::Map<il::String, int> map{7};
  map.Set("aaa", 0);
  map.Set("bbb", 0);
  map.Set("ccc", 0);
  map.Set("ddd", 0);
  map.Set("eee", 0);
  map.Set("fff", 0);
  map.Set("ggg", 0);
  il::int_t i;
  while (state.KeepRunning()) {
    i += map.search("aaa");
    i += map.search("bbb");
    i += map.search("ccc");
    i += map.search("ddd");
    i += map.search("eee");
    i += map.search("fff");
    i += map.search("ggg");
  }
  std::cout << i << std::endl;
}

static void BM_SetMap(benchmark::State& state) {
  il::Map<il::String, int> map{7};
  while (state.KeepRunning()) {
    map.Set("aaa", 0);
    map.Set("bbb", 0);
    map.Set("ccc", 0);
    map.Set("ddd", 0);
    map.Set("eee", 0);
    map.Set("fff", 0);
    map.Set("ggg", 0);
  }
}

BENCHMARK(BM_MapCString);
BENCHMARK(BM_Map);
BENCHMARK(BM_SetMapCString);
BENCHMARK(BM_SetMap);

BENCHMARK_MAIN();
