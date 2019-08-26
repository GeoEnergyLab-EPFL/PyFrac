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

#include <il/Array.h>
#include <il/Dynamic.h>
#include <il/Timer.h>

void integer_boxed();
void floating_point_boxed();

int main() {
  integer_boxed();
  floating_point_boxed();

  return 0;
}

void integer_boxed() {
  const il::int_t n = 1000;
  const il::int_t nb_times = 1000000;
  il::Array<il::int_t> v{n, 0};

  il::Array<il::Dynamic> w{n};
  for (il::int_t i = 0; i < n; ++i) {
    w[i] = 0;
  }

  il::int_t sum_native = 0;
  il::Timer timer{};
  timer.Start();
  for (il::int_t k = 0; k < nb_times; ++k) {
    for (il::int_t i = 0; i < n; ++i) {
      sum_native += v[i];
    }
    sum_native *= 2;
  }
  timer.Stop();
  const double time_native = timer.elapsed();

  timer.Reset();
  timer.Start();
  il::int_t sum_boxed = 0;
  for (il::int_t k = 0; k < nb_times; ++k) {
    for (il::int_t i = 0; i < n; ++i) {
      sum_boxed += w[i].to<il::int_t>();
    }
    sum_boxed *= 2;
  }
  timer.Stop();
  const double time_boxed = timer.elapsed();

  std::cout << "Native: " << sum_native << " , Time: " << time_native
            << std::endl;
  std::cout << " Boxed: " << sum_boxed << " , Time: " << time_boxed
            << std::endl;
}

void floating_point_boxed() {
  const il::int_t n = 1000;
  const il::int_t nb_times = 10000000;
  il::Array<double> v{n};
  for (il::int_t i = 0; i < n; ++i) {
    v[i] = 1.0 / (i + 1);
  }

  il::Array<il::Dynamic> w{n};
  for (il::int_t i = 0; i < n; ++i) {
    w[i] = 1.0 / (i + 1);
  }

  double sum_native = 0.0;
  il::Timer timer{};
  timer.Start();
  for (il::int_t k = 0; k < nb_times; ++k) {
    for (il::int_t i = 0; i < n; ++i) {
      sum_native += v[i];
    }
    sum_native *= 0.5;
  }
  timer.Stop();
  const double time_native = timer.elapsed();

  timer.Reset();
  timer.Start();
  double sum_boxed = 0.0;
  for (il::int_t k = 0; k < nb_times; ++k) {
    for (il::int_t i = 0; i < n; ++i) {
      sum_boxed += w[i].to<double>();
    }
    sum_boxed *= 0.5;
  }
  timer.Stop();
  const double time_boxed = timer.elapsed();

  std::cout << "Native: " << sum_native << " , Time: " << time_native
            << std::endl;
  std::cout << " Boxed: " << sum_boxed << " , Time: " << time_boxed
            << std::endl;
}
