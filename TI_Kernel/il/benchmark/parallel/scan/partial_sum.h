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

#ifndef IL_PARTIAL_SUM_H
#define IL_PARTIAL_SUM_H

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

namespace il {

#ifdef IL_TBB
class Body {
 private:
  double sum_;
  const il::Array<double>& v_;
  il::Array<double>& p_;

 public:
  Body(const il::Array<double>& v, il::io_t, il::Array<double>& p)
      : v_{v}, p_{p} {
    sum_ = 0;
  };
  double get_sum() { return sum_; };
  template <typename Tag>
  void operator()(const tbb::blocked_range<il::int_t>& range, Tag) {
    double temp{sum_};
    for (il::int_t i{range.begin()}; i < range.end(); ++i) {
      temp += v_[i];
      if (Tag::is_final_scan()) {
        p_[i] = temp;
      }
      sum_ = temp;
    }
  }
  Body(Body& b, tbb::split) : v_{b.v_}, p_{b.p_} { sum_ = 0; }
  void reverse_join(Body& a) { sum_ += a.sum_; }
  void assign(Body& b) { sum_ = b.sum_; }
};
#endif

void partial_sum() {
  std::printf(
      "****************************************************************"
      "****************\n");
  std::printf("* Partial sum\n");
  std::printf(
      "****************************************************************"
      "****************\n");

  il::Array<il::int_t> size{
      il::value, {100, 1000, 10000, 100000, 1000000, 10000000, 100000000}};
  for (il::int_t n : size) {
    std::printf("Size of array: %td\n", n);

    auto partial_sum_serial = [&n](il::io_t, il::BState& state) {
      il::Array<double> v{n, 0.0};
      il::Array<double> p{n, 0.0};
      for (il::int_t k = 0; k < v.size(); ++k) {
        v[k] = 1.0 / (k + 1);
      }
      while (state.keep_running()) {
        p[0] = v[0];
        for (il::int_t k = 1; k < p.size(); ++k) {
          p[k] = p[k - 1] + v[k];
        }
      }
      il::do_not_optimize(p.data());
    };
    double time_serial{il::benchmark(partial_sum_serial) / n};
    std::printf("Serial: %7.3e s\n", time_serial);

#ifdef IL_TBB
    auto partial_sum_tbb = [&n](il::io_t, il::BState& state) {
      il::Array<double> v{n, 0.0};
      il::Array<double> p{n, 0.0};
      for (il::int_t k = 0; k < v.size(); ++k) {
        v[k] = 1.0 / (k + 1);
      }
      while (state.keep_running()) {
        Body body{v, il::io, p};
        tbb::parallel_scan(tbb::blocked_range<il::int_t>(0, n), body);
      }
      il::do_not_optimize(p.data());
    };
    double time_tbb{il::benchmark(partial_sum_tbb) / n};
    std::printf("   TBB: %7.3e s, Ratio: %5.3f\n", time_tbb,
                time_serial / time_tbb);
#endif

    std::printf("\n");
  }
}
}  // namespace il

#endif  // IL_PARTIAL_SUM_H
