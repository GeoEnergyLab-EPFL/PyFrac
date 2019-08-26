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

////////////////////////////////////////////////////////////////////////////////

// The Mandelbrot set is the set of all complex numbers c such that the sequence
// defined by
//   - z_0 = 0
//   - For all n in N, z_(n+1) = z_n^2 + c
// is bounded. One can prove that if there exists a n in N such that |z_n| > 2,
// the sequence (z_n) is not bounded.
// Therefore, given c in C, we compute z_0, ..., z_n up to the first n such
// that |z_n| > 2 or n >= depth (we use a depth of 50 here) and we store this
// value n. If this value is < depth, we know for sure that c is not in the
// Mandelbrot set. If this value is equal to depth, it is likely that it is
// in the Mandelbrot set.
//
// The following program computes the values n for every z = x + i y with
// x_left <= x <= x_right and y_bottom <= y <= y_top.

////////////////////////////////////////////////////////////////////////////////

// Opportunities for parallelization:
// - Thread level
//   One can assign different chunks of the rectangle to different cores.
//   Usually, the first core computes points for y_top >= y >= y_1, the second
//   core computes points for y_1 >= y >= y_2, etc.
// - Vector level
//   To compute the value n for a given point z, a while loop has to be made.
//   The number of times this loop is executed depends upon z. Therefore, there
//   is no easy vectorization for this kind of loop. However, the Intel compiler
//   can vectorize such a loop: suppose that we have 4 points and we want to
//   compute the values n for each of this point: we iterate the while loop
//   until all the points satisfy the exit condition, but we store the values
//   for which the exit condition has been satistifed for all the points. All
//   this transformation is handled explicitly by the compiler using a
//   #pragma omp simd (in OpenMP) before the x-loop. Note that close points
//   should generally have closed values for n. Therefore, the extra amount of
//   work done by the vectorized loop should ne be a penalty.

////////////////////////////////////////////////////////////////////////////////

// To run, compile with -std=c++11 -Ofast -xHost -openmp -DNDEBUG
//
// - Load imbalance:
//   Without the schedule(dynamic) clause for the OpenMP threads, the fastest
//   is TBB for threads (with OpenMP for vectorization). It is faster
//   than plain OpenMP for threads (and vectorization) without schedule
//   clause because there is load imbalance in the outer for loop: for y
//   close to y_top or y_bottom, the complex number z goes out or the circle
//   of radius 2 very quickly. We get the following timings:
//   - 275 milliseconds for TBB/OpenMP
//   - 351 milliseconds for OpenMP (without schedule clause)/OpenMP
//   But if we change the OpenMP clause to (#pragma omp parallel for
//   schedule(dynamic)), the runtime comes down to 260 milliseconds, a bit
//   faster than TBB/OpenMP.

#ifndef IL_MANDELBROT_H
#define IL_MANDELBROT_H

#include <iostream>

#include <il/Array2D.h>
#include <il/benchmark/tools/memory/memory.h>
#include <il/benchmark/tools/timer/Timer.h>

#ifdef IL_TBB
#include <tbb/tbb.h>
#endif

#ifdef IL_CILK
#include <cilk/cilk.h>
#endif

namespace il {

// const float x_left = -1.0;
// const float x_right = 2.0;
// const float y_bottom = -1.5;
// const float y_top = 1.5;
//
// const il::int_t nx = 10000;
// const il::int_t ny = 10000;
//
// const int depth = 50;

// Mandelbrot set: No threads, no vectorization
//
double time_mandelbrot_serial_serial(float x_left, float x_right,
                                     float y_bottom, float y_top,
                                     il::int_t depth, il::int_t nx,
                                     il::int_t ny, bool warm_cache) {
  const float dx{(x_right - x_left) / nx};
  const float dy{(y_top - y_bottom) / ny};
  il::Array2D<int> v{nx, ny};
  il::commit_memory(il::io, v);
  if (warm_cache) {
    il::warm_cache(il::io, v);
  }

  il::Timer timer{};
  for (il::int_t ky = 0; ky < ny; ++ky) {
    float y{y_top - ky * dy};
    for (il::int_t kx = 0; kx < nx; ++kx) {
      float x{x_left + kx * dx};
      float z_re = 0.0;
      float z_im = 0.0;
      int count = 0;
      while (count < depth) {
        if (z_re * z_re + z_im * z_im > 4.0) {
          break;
        }
        float old_z_re{z_re};
        z_re = z_re * z_re - z_im * z_im + x;
        z_im = 2 * old_z_re * z_im + y;
        ++count;
      }
      v(kx, ky) = count;
    }
  }
  timer.Stop();
  il::escape(v.data());

  return timer.elapsed();
}

// Mandelbrot set: OpenMP for threads, no vectorisation
//
double time_mandelbrot_openmp_serial(float x_left, float x_right,
                                     float y_bottom, float y_top,
                                     il::int_t depth, il::int_t nx,
                                     il::int_t ny, bool warm_cache) {
  const float dx{(x_right - x_left) / nx};
  const float dy{(y_top - y_bottom) / ny};
  il::Array2D<int> v{nx, ny};
  il::commit_memory(il::io, v);
  if (warm_cache) {
    il::warm_cache(il::io, v);
  }

  il::Timer timer{};
#pragma omp parallel for schedule(dynamic)
  for (il::int_t ky = 0; ky < ny; ++ky) {
    auto y = float{y_top - ky * dy};
    for (il::int_t kx = 0; kx < nx; ++kx) {
      float x{x_left + kx * dx};
      float z_re = 0.0;
      float z_im = 0.0;
      int count = 0;
      while (count < depth) {
        if (z_re * z_re + z_im * z_im > 4.0) {
          break;
        }
        float old_z_re{z_re};
        z_re = z_re * z_re - z_im * z_im + x;
        z_im = 2 * old_z_re * z_im + y;
        ++count;
      }
      v(kx, ky) = count;
    }
  }
  timer.Stop();
  il::escape(v.data());

  return timer.elapsed();
}

// Mandelbrot set: OpenMP for threads, OpenMP for vectorisation
//
double time_mandelbrot_openmp_openmp(float x_left, float x_right,
                                     float y_bottom, float y_top,
                                     il::int_t depth, il::int_t nx,
                                     il::int_t ny, bool warm_cache) {
  const float dx{(x_right - x_left) / nx};
  const float dy{(y_top - y_bottom) / ny};
  il::Array2D<int> v{nx, ny};
  il::commit_memory(il::io, v);
  if (warm_cache) {
    il::warm_cache(il::io, v);
  }

  il::Timer timer{};
#pragma omp parallel for schedule(dynamic)
  for (il::int_t ky = 0; ky < ny; ++ky) {
    float y{y_top - ky * dy};
#pragma omp simd
    for (il::int_t kx = 0; kx < nx; ++kx) {
      float x{x_left + kx * dx};
      float z_re = 0.0;
      float z_im = 0.0;
      int count = 0;
      while (count < depth) {
        if (z_re * z_re + z_im * z_im > 4.0) {
          break;
        }
        float old_z_re{z_re};
        z_re = z_re * z_re - z_im * z_im + x;
        z_im = 2 * old_z_re * z_im + y;
        ++count;
      }
      v(kx, ky) = count;
    }
  }
  timer.Stop();
  il::escape(v.data());

  return timer.elapsed();
}

// Mandelbrot set: TBB for threads, no vectorisation
//
#ifdef IL_TBB
double time_mandelbrot_tbb_serial(float x_left, float x_right, float y_bottom,
                                  float y_top, il::int_t depth, il::int_t nx,
                                  il::int_t ny, bool warm_cache) {
  const float dx{(x_right - x_left) / nx};
  const float dy{(y_top - y_bottom) / ny};
  il::Array2D<int> v{nx, ny};
  il::commit_memory(il::io, v);
  if (warm_cache) {
    il::warm_cache(il::io, v);
  }

  il::SimpleTimer timer{};
  tbb::parallel_for(tbb::blocked_range<il::int_t>(0, ny),
                    [=, &v](const tbb::blocked_range<il::int_t>& range) {
                      for (il::int_t ky{range.begin()}; ky < range.end();
                           ++ky) {
                        float y{y_top - ky * dy};
                        for (il::int_t kx = 0; kx < nx; ++kx) {
                          float x{x_left + kx * dx};
                          float z_re = 0.0;
                          float z_im = 0.0;
                          int count = 0;
                          while (count < depth) {
                            if (z_re * z_re + z_im * z_im > 4.0) {
                              break;
                            }
                            float old_z_re{z_re};
                            z_re = z_re * z_re - z_im * z_im + x;
                            z_im = 2 * old_z_re * z_im + y;
                            ++count;
                          }
                          v(kx, ky) = count;
                        }
                      }
                    });
  timer.Stop();
  il::escape(v.data());

  return timer.elapsed();
}
#endif

// Mandelbrot set: TBB for threads, OpenMP for vectorisation
//
#ifdef IL_TBB
double time_mandelbrot_tbb_openmp(float x_left, float x_right, float y_bottom,
                                  float y_top, il::int_t depth, il::int_t nx,
                                  il::int_t ny, bool warm_cache) {
  const float dx{(x_right - x_left) / nx};
  const float dy{(y_top - y_bottom) / ny};
  il::Array2D<int> v{nx, ny};
  il::commit_memory(il::io, v);
  if (warm_cache) {
    il::warm_cache(il::io, v);
  }

  il::SimpleTimer timer{};
  tbb::parallel_for(tbb::blocked_range<il::int_t>(0, ny),
                    [=, &v](const tbb::blocked_range<il::int_t>& range) {
                      for (il::int_t ky{range.begin()}; ky < range.end();
                           ++ky) {
                        float y{y_top - ky * dy};
#pragma omp simd
                        for (il::int_t kx = 0; kx < nx; ++kx) {
                          float x{x_left + kx * dx};
                          float z_re = 0.0;
                          float z_im = 0.0;
                          int count = 0;
                          while (count < depth) {
                            if (z_re * z_re + z_im * z_im > 4.0) {
                              break;
                            }
                            float old_z_re{z_re};
                            z_re = z_re * z_re - z_im * z_im + x;
                            z_im = 2 * old_z_re * z_im + y;
                            ++count;
                          }
                          v(kx, ky) = count;
                        }
                      }
                    });
  timer.Stop();
  il::escape(v.data());

  return timer.elapsed();
}
#endif

// Mandelbrot set: Cilk for threads, no vectorisation
//
#ifdef IL_CILK
double time_mandelbrot_cilk_serial(float x_left, float x_right, float y_bottom,
                                   float y_top, il::int_t depth, il::int_t nx,
                                   il::int_t ny, bool warm_cache) {
  const float dx{(x_right - x_left) / nx};
  const float dy{(y_top - y_bottom) / ny};
  il::Array2D<int> v{nx, ny};
  il::commit_memory(il::io, v);
  if (warm_cache) {
    il::warm_cache(il::io, v);
  }

  il::SimpleTimer timer{};
  cilk_for(il::int_t ky = 0; ky < ny; ++ky) {
    float y{y_top - ky * dy};
    for (il::int_t kx = 0; kx < nx; ++kx) {
      float x{x_left + kx * dx};
      float z_re = 0.0;
      float z_im = 0.0;
      int count = 0;
      while (count < depth) {
        if (z_re * z_re + z_im * z_im > 4.0) {
          break;
        }
        float old_z_re{z_re};
        z_re = z_re * z_re - z_im * z_im + x;
        z_im = 2 * old_z_re * z_im + y;
        ++count;
      }
      v(kx, ky) = count;
    }
  }
  timer.Stop();
  il::escape(v.data());

  return timer.elapsed();
}
#endif

// Mandelbrot set: Cilk for threads, OpenMP for vectorisation
//
#ifdef IL_CILK
double time_mandelbrot_cilk_openmp(float x_left, float x_right, float y_bottom,
                                   float y_top, il::int_t depth, il::int_t nx,
                                   il::int_t ny, bool warm_cache) {
  const float dx{(x_right - x_left) / nx};
  const float dy{(y_top - y_bottom) / ny};
  il::Array2D<int> v{nx, ny};
  il::commit_memory(il::io, v);
  if (warm_cache) {
    il::warm_cache(il::io, v);
  }

  il::SimpleTimer timer{};
  cilk_for(il::int_t ky = 0; ky < ny; ++ky) {
    float y{y_top - ky * dy};
#pragma omp simd
    for (il::int_t kx = 0; kx < nx; ++kx) {
      float x{x_left + kx * dx};
      float z_re = 0.0;
      float z_im = 0.0;
      int count = 0;
      while (count < depth) {
        if (z_re * z_re + z_im * z_im > 4.0) {
          break;
        }
        float old_z_re{z_re};
        z_re = z_re * z_re - z_im * z_im + x;
        z_im = 2 * old_z_re * z_im + y;
        ++count;
      }
      v(kx, ky) = count;
    }
  }
  timer.Stop();
  il::escape(v.data());

  return timer.elapsed();
}
#endif

// Mandelbrot set: Cilk for threads, Cilk for vectorisation
//
#ifdef IL_CILK
double time_mandelbrot_cilk_cilk(float x_left, float x_right, float y_bottom,
                                 float y_top, il::int_t depth, il::int_t nx,
                                 il::int_t ny, bool warm_cache) {
  const float dx{(x_right - x_left) / nx};
  const float dy{(y_top - y_bottom) / ny};
  il::Array2D<int> v{nx, ny};
  il::commit_memory(il::io, v);
  if (warm_cache) {
    il::warm_cache(il::io, v);
  }

  il::SimpleTimer timer{};
  cilk_for(il::int_t ky = 0; ky < ny; ++ky) {
    float y{y_top - ky * dy};
#pragma simd
    for (il::int_t kx = 0; kx < nx; ++kx) {
      float x{x_left + kx * dx};
      float z_re = 0.0;
      float z_im = 0.0;
      int count = 0;
      while (count < depth) {
        if (z_re * z_re + z_im * z_im > 4.0) {
          break;
        }
        float old_z_re{z_re};
        z_re = z_re * z_re - z_im * z_im + x;
        z_im = 2 * old_z_re * z_im + y;
        ++count;
      }
      v(kx, ky) = count;
      x += dx;
    }
    y -= dy;
  }
  timer.Stop();
  il::escape(v.data());

  return timer.elapsed();
}
#endif

#endif  // IL_MANDELBROT_H
