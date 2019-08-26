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

#ifndef IL_BENCHMARK_H
#define IL_BENCHMARK_H

#include <chrono>
#include <cstdio>

namespace il {

class BStateShort {
 private:
  il::int_t n_;

 public:
  BStateShort(il::int_t n) : n_{n} {};
  bool keep_running() {
    --n_;
    return n_ >= 0;
  };
};

template <typename P>
double benchmark_short(const P &program, double time_goal = 1.0) {
  double total_time = 0.0;
  il::int_t nb_iterations = 1;

  il::int_t count = 0;
  const il::int_t max_count = 20;
  const il::int_t growth_factor = 10;

  while (total_time <= 0.5 * time_goal && count <= max_count) {
    il::BStateShort state{nb_iterations};
    auto begin = std::chrono::high_resolution_clock::now();
    program(il::io, state);
    auto end = std::chrono::high_resolution_clock::now();
    total_time =
        1.0e-9 *
        (1 + std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                 .count());
    const il::int_t estimated_nb_iterations{
        static_cast<il::int_t>(time_goal / (total_time / nb_iterations))};
    nb_iterations = estimated_nb_iterations <= growth_factor * nb_iterations
                        ? estimated_nb_iterations
                        : growth_factor * nb_iterations;
    ++count;
  }

  return total_time / nb_iterations;
}

class BState {
 private:
  il::int_t n_;
  il::int_t k_;
  std::size_t time_;
  bool started_;
  std::chrono::time_point<std::chrono::high_resolution_clock> point_begin_;

 public:
  BState(il::int_t n) {
    n_ = n;
    k_ = n;
    time_ = 0;
    started_ = false;
  };
  bool keep_running() {
    if (!started_) {
      resume_timing();
    }
    --k_;
    const bool ans{k_ >= 0};
    if (!ans) {
      pause_timing();
    }
    return ans;
  };
  void resume_timing() {
    point_begin_ = std::chrono::high_resolution_clock::now();
    started_ = true;
  }
  void pause_timing() {
    std::chrono::time_point<std::chrono::high_resolution_clock> point_end{
        std::chrono::high_resolution_clock::now()};
    time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(point_end -
                                                                  point_begin_)
                 .count();
    started_ = false;
  }
  double time() const { return (1.0e-9 * time_) / n_; }
};

template <typename P>
double benchmark(const P &program, double time_goal = 1.0) {
  double total_time = 0.0;
  il::int_t nb_iterations = 1;

  il::int_t count = 0;
  const il::int_t max_count = 20;
  const il::int_t growth_factor = 10;

  while (total_time <= 0.5 * time_goal && count <= max_count) {
    il::BState state{nb_iterations};
    program(il::io, state);
    total_time = nb_iterations * (1.0e-9 + state.time());
    const il::int_t estimated_nb_iterations{
        static_cast<il::int_t>(time_goal / (total_time / nb_iterations))};
    nb_iterations = estimated_nb_iterations <= growth_factor * nb_iterations
                        ? estimated_nb_iterations
                        : growth_factor * nb_iterations;
    ++count;
  }

  return total_time / nb_iterations;
}

template <typename T>
void do_not_optimize(const T &value) {
  asm volatile("" : "+rm"(const_cast<T &>(value)));
}

}  // namespace il

#endif  // IL_BENCHMARK_H
