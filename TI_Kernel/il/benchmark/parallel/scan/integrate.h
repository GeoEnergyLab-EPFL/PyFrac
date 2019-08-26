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

#ifndef IL_INTEGRATE_H
#define IL_INTEGRATE_H

#include <cstdio>

#include <il/Array.h>
#include <il/benchmark/tools/memory/memory.h>
#include <il/benchmark/tools/timer/Benchmark.h>
#include <il/math.h>

#ifdef IL_TBB
#include <tbb/tbb.h>
#endif

#ifdef IL_CILK
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#endif

namespace il {

#ifdef IL_TBB
class IntegrateBody {
 private:
  double sum_;
  il::Array<double>& f_;

 public:
  IntegrateBody(il::io_t, il::Array<double>& f) : f_{f} { sum_ = 0; };
  double get_sum() { return sum_; };
  template <typename Tag>
  void operator()(const tbb::blocked_range<il::int_t>& range, Tag) {
    const il::int_t n{f_.size()};
    double tmp{sum_};
    for (il::int_t i{range.begin()}; i < range.end(); ++i) {
      tmp += il::ipow<2>(static_cast<double>(i) / n);
      if (Tag::is_final_scan()) {
        f_[i] = tmp;
      }
      sum_ = tmp;
    }
  }
  IntegrateBody(IntegrateBody& b, tbb::split) : f_{b.f_} { sum_ = 0; }
  void reverse_join(IntegrateBody& a) { sum_ += a.sum_; }
  void assign(IntegrateBody& b) { sum_ = b.sum_; }
};
#endif

void integrate() {
  std::printf(
      "****************************************************************"
      "****************\n");
  std::printf("* Integration\n");
  std::printf(
      "****************************************************************"
      "****************\n");

  il::Array<il::int_t> size{
      il::value, {100, 1000, 10000, 100000, 1000000, 10000000, 100000000}};
  for (il::int_t n : size) {
    std::printf("Size of array: %td\n", n);

    auto integrate_serial = [&n](il::io_t, il::BState& state) {
      il::Array<double> f{n, 0.0};
      while (state.keep_running()) {
        f[0] = 0.0;
        for (il::int_t k = 1; k < f.size(); ++k) {
          f[k] += il::ipow<2>(static_cast<double>(k) / n);
        }
      }
    };
    double time_serial{il::benchmark(integrate_serial) / n};
    std::printf("Serial: %7.3e s\n", time_serial);

#ifdef IL_TBB
    auto integrate_tbb = [&n](il::io_t, il::BState& state) {
      il::Array<double> f{n, 0.0};
      while (state.keep_running()) {
        IntegrateBody IntegrateBody{il::io, f};
        tbb::parallel_scan(tbb::blocked_range<il::int_t>(0, n), IntegrateBody);
      }
    };
    double time_tbb{il::benchmark(integrate_tbb) / n};
    std::printf("   TBB: %7.3e s, Ratio: %5.3f\n", time_tbb,
                time_serial / time_tbb);
#endif

    std::printf("\n");
  }
}
}  // namespace il

#endif  // IL_INTEGRATE_H
