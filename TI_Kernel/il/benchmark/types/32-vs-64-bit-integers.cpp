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

// This benchmark was designed to figure out if there is a difference in terms
// of performance between 32-bit integers and 64-bit integers.
//
//                         int          long
//
// Intel compiler 17.0     37           39
// Clang 3.9               37           37
// Gcc 4.7                 44           38
// Gcc 6.2                 45           38
//
// So it seems that gcc is the only one to struggle with int where the
// performance goes down.

#define INT long

#include <cstdio>
#include <limits>
#include <random>

const INT n = 10;
const INT x_finish = n - 2;
const INT y_finish = n - 2;
char maze[n * n];

const char free_cell = 0;
const char barrier_cell = 1;
const char traversed_cell = 2;

INT minimum(INT a, INT b) { return a < b ? a : b; }

INT find_min_path_32(INT x, INT y, INT best_path_length,
                     INT current_path_length) {
  maze[y * n + x] = traversed_cell;
  ++current_path_length;
  if (current_path_length >= best_path_length) {
    maze[y * n + x] = free_cell;
    return std::numeric_limits<INT>::max();
  }
  if (x == x_finish && y == y_finish) {
    maze[y * n + x] = free_cell;
    return current_path_length;
  }
  INT length = std::numeric_limits<INT>::max();
  if (x > 0 && maze[y * n + (x - 1)] == free_cell) {
    INT rest_length =
        find_min_path_32(x - 1, y, best_path_length, current_path_length);
    length = minimum(rest_length, length);
  }
  if (x < n - 1 && maze[y * n + (x + 1)] == free_cell) {
    INT rest_length =
        find_min_path_32(x + 1, y, best_path_length, current_path_length);
    length = minimum(rest_length, length);
  }
  if (y > 0 && maze[(y - 1) * n + x] == free_cell) {
    INT rest_length =
        find_min_path_32(x, y - 1, best_path_length, current_path_length);
    length = minimum(rest_length, length);
  }
  if (y < n - 1 && maze[(y + 1) * n + x] == free_cell) {
    INT rest_length =
        find_min_path_32(x, y + 1, best_path_length, current_path_length);
    length = minimum(rest_length, length);
  }
  if (length >= best_path_length) {
    maze[y * n + x] = free_cell;
    return std::numeric_limits<INT>::max();
  } else {
    maze[y * n + x] = free_cell;
    return length;
  }
}

void maze() {
  const INT x_start = 1;
  const INT y_start = 1;

  std::mt19937 generator{};
  std::uniform_int_distribution<INT> distribution{0, 4};

  for (INT i = 0; i < n * n; ++i) {
    if (distribution(generator) == 0) {
      maze[i] = barrier_cell;
    } else {
      maze[i] = free_cell;
    }
  }
  maze[y_start * n + x_start] = free_cell;
  maze[y_finish * n + x_finish] = free_cell;
  for (INT y = n - 1; y >= 0; --y) {
    for (INT x = 0; x < n; ++x) {
      if (maze[y * n + x] == free_cell) {
        std::printf(".");
      } else {
        std::printf("O");
      }
    }
    std::printf("\n");
  }

  INT length =
      find_min_path_32(x_start, y_start, std::numeric_limits<INT>::max(), 0);

  std::printf("Best path length: %d\n", static_cast<int>(length));
}
