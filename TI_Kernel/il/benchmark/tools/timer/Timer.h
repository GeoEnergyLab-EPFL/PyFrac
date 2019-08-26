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

#ifndef IL_TIMER_H
#define IL_TIMER_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>

#include <il/core/core.h>

namespace il {

class Timer {
 private:
  bool launched_;
  double time_;
  std::chrono::time_point<std::chrono::high_resolution_clock> point_begin_;

 public:
  Timer();
  void Start();
  void Stop();
  void Reset();
  double time() const;
  void sleepUntil(double time) const;
};

inline void sleep(double time) {
  il::Timer timer{};
  timer.Start();
  timer.sleepUntil(time);
}

inline Timer::Timer() : point_begin_{} {
  time_ = 0.0;
  launched_ = false;
}

inline void Timer::Start() {
  IL_EXPECT_FAST(!launched_);
  launched_ = true;
  point_begin_ = std::chrono::high_resolution_clock::now();
}

inline void Timer::Stop() {
  std::chrono::time_point<std::chrono::high_resolution_clock> point_end =
      std::chrono::high_resolution_clock::now();
  IL_EXPECT_FAST(launched_);
  launched_ = false;
  time_ += 1.0e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                        point_end - point_begin_)
                        .count();
}

inline void Timer::Reset() {
  time_ = 0.0;
  launched_ = false;
}

inline double Timer::time() const { return time_; }

inline void Timer::sleepUntil(double time) const {
  IL_EXPECT_FAST(launched_);
  const double time_left =
      time -
      1.0e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now() - point_begin_)
                   .count();
  if (time_left > 0.0) {
    std::this_thread::sleep_for(
        std::chrono::nanoseconds{static_cast<std::size_t>(time_left * 1.0e9)});
  }
}

class TimerCycles {
 private:
  std::uint64_t point_begin_;
  std::uint64_t nb_cycles_;

 public:
  TimerCycles();
  void Stop();
  long int cycles() const;
};

inline TimerCycles::TimerCycles() {
#ifdef IL_UNIX
  unsigned int low;
  unsigned int high;
  asm volatile("rdtsc" : "=a"(low), "=d"(high));
  point_begin_ = static_cast<std::uint64_t>(low) |
                 (static_cast<std::uint64_t>(high) << 32);
#endif
}

inline void TimerCycles::Stop() {
#ifdef IL_UNIX
  unsigned int low;
  unsigned int high;
  asm volatile("rdtsc" : "=a"(low), "=d"(high));
  std::uint64_t point_end{static_cast<std::uint64_t>(low) |
                          (static_cast<std::uint64_t>(high) << 32)};
  nb_cycles_ = point_end - point_begin_;
#endif
}

inline long int TimerCycles::cycles() const {
  return static_cast<long int>(nb_cycles_);
}
}  // namespace il

#endif  // IL_TIMER_H
