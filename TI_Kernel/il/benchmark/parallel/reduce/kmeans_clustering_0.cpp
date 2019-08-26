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

#include <il/benchmark/parallel/reduce/kmeans_clustering_0.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <il/Array.h>
#include <il/Array2D.h>

#include <omp.h>

namespace il {

struct Pixel {
  float r;
  float g;
  float b;
};

double kmeans_clustering_0(std::size_t nb_point, std::size_t nb_cluster,
                           std::size_t nb_iteration) {
  std::vector<Pixel> point(nb_point);
  std::vector<std::size_t> cluster(nb_point);

  std::vector<Pixel> centroid(nb_cluster);
  std::vector<std::size_t> point_per_cluster(nb_cluster);

  std::default_random_engine engine{};
  std::uniform_real_distribution<float> r_dist{0.0f, 1.0f};
  std::uniform_int_distribution<std::size_t> i_dist{0, nb_cluster - 1};
  for (std::size_t k = 0; k < nb_point; ++k) {
    point[k].r = r_dist(engine);
    point[k].g = r_dist(engine);
    point[k].b = r_dist(engine);
    cluster[k] = i_dist(engine);
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::size_t iteration = 0;
  while (true) {
    // Compute the centroid of the clusters
    for (std::size_t i = 0; i < nb_cluster; ++i) {
      centroid[i].r = 0.0f;
      centroid[i].g = 0.0f;
      centroid[i].b = 0.0f;
      point_per_cluster[i] = 0;
    }
    for (std::size_t k = 0; k < nb_point; ++k) {
      std::size_t i = cluster[k];
      centroid[i].r += point[k].r;
      centroid[i].g += point[k].g;
      centroid[i].b += point[k].b;
      ++point_per_cluster[i];
    }
    for (std::size_t i = 0; i < nb_cluster; ++i) {
      std::size_t nb_point_cluster = point_per_cluster[i];
      centroid[i].r /= nb_point_cluster;
      centroid[i].g /= nb_point_cluster;
      centroid[i].b /= nb_point_cluster;
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration > nb_iteration) {
      break;
    }

    // Reassign points to clusters
    for (std::size_t k = 0; k < nb_point; ++k) {
      float best_distance = std::numeric_limits<float>::max();
      std::size_t best_centroid = -1;
      for (std::size_t i = 0; i < nb_cluster; ++i) {
        float x = point[k].r - centroid[i].r;
        float y = point[k].g - centroid[i].g;
        float z = point[k].b - centroid[i].b;
        float distance = std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2);
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

double kmeans_clustering_1(std::size_t nb_point, std::size_t nb_cluster,
                           std::size_t nb_iteration) {
  std::vector<Pixel> point(nb_point);
  std::vector<std::size_t> cluster(nb_point);

  std::vector<Pixel> centroid(nb_cluster);
  std::vector<std::size_t> point_per_cluster(nb_cluster);

  std::default_random_engine engine{};
  std::uniform_real_distribution<float> r_dist{0.0f, 1.0f};
  std::uniform_int_distribution<std::size_t> i_dist{0, nb_cluster - 1};
  for (std::size_t k = 0; k < nb_point; ++k) {
    point[k].r = r_dist(engine);
    point[k].g = r_dist(engine);
    point[k].b = r_dist(engine);
    cluster[k] = i_dist(engine);
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::size_t iteration = 0;
  while (true) {
    // Compute the centroid of the clusters
    for (std::size_t i = 0; i < nb_cluster; ++i) {
      centroid[i].r = 0.0f;
      centroid[i].g = 0.0f;
      centroid[i].b = 0.0f;
      point_per_cluster[i] = 0;
    }
    for (std::size_t k = 0; k < nb_point; ++k) {
      std::size_t i = cluster[k];
      centroid[i].r += point[k].r;
      centroid[i].g += point[k].g;
      centroid[i].b += point[k].b;
      ++point_per_cluster[i];
    }
    for (std::size_t i = 0; i < nb_cluster; ++i) {
      std::size_t nb_point_cluster = point_per_cluster[i];
      centroid[i].r /= nb_point_cluster;
      centroid[i].g /= nb_point_cluster;
      centroid[i].b /= nb_point_cluster;
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration > nb_iteration) {
      break;
    }

    // Reassign points to clusters
    for (int k = 0; k < static_cast<int>(nb_point); ++k) {
      float best_distance = std::numeric_limits<float>::max();
      std::size_t best_centroid = -1;
      for (std::size_t i = 0; i < nb_cluster; ++i) {
        float x = point[k].r - centroid[i].r;
        float y = point[k].g - centroid[i].g;
        float z = point[k].b - centroid[i].b;
        float distance = x * x + y * y + z * z;
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

double kmeans_clustering_2(std::ptrdiff_t nb_point, std::ptrdiff_t nb_cluster,
                           std::ptrdiff_t nb_iteration) {
  std::vector<Pixel> point(nb_point);
  std::vector<std::ptrdiff_t> cluster(nb_point);

  std::vector<Pixel> centroid(nb_cluster);
  std::vector<std::ptrdiff_t> point_per_cluster(nb_cluster);

  std::default_random_engine engine{};
  std::uniform_real_distribution<float> r_dist{0.0f, 1.0f};
  std::uniform_int_distribution<std::ptrdiff_t> i_dist{0, nb_cluster - 1};
  for (std::ptrdiff_t k = 0; k < nb_point; ++k) {
    point[k].r = r_dist(engine);
    point[k].g = r_dist(engine);
    point[k].b = r_dist(engine);
    cluster[k] = i_dist(engine);
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::ptrdiff_t iteration = 0;
  while (true) {
    // Compute the centroid of the clusters
    for (std::ptrdiff_t i = 0; i < nb_cluster; ++i) {
      centroid[i].r = 0.0f;
      centroid[i].g = 0.0f;
      centroid[i].b = 0.0f;
      point_per_cluster[i] = 0;
    }
    for (std::ptrdiff_t k = 0; k < nb_point; ++k) {
      std::ptrdiff_t i = cluster[k];
      centroid[i].r += point[k].r;
      centroid[i].g += point[k].g;
      centroid[i].b += point[k].b;
      ++point_per_cluster[i];
    }
    for (std::ptrdiff_t i = 0; i < nb_cluster; ++i) {
      std::ptrdiff_t nb_point_cluster = point_per_cluster[i];
      centroid[i].r /= nb_point_cluster;
      centroid[i].g /= nb_point_cluster;
      centroid[i].b /= nb_point_cluster;
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration > nb_iteration) {
      break;
    }

    // Reassign points to clusters
    for (std::ptrdiff_t k = 0; k < nb_point; ++k) {
      float best_distance = std::numeric_limits<float>::max();
      std::ptrdiff_t best_centroid = -1;
      for (std::ptrdiff_t i = 0; i < nb_cluster; ++i) {
        float x = point[k].r - centroid[i].r;
        float y = point[k].g - centroid[i].g;
        float z = point[k].b - centroid[i].b;
        float distance = x * x + y * y + z * z;
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

struct PixelVector {
  std::vector<float> red;
  std::vector<float> green;
  std::vector<float> blue;
  PixelVector(std::ptrdiff_t n) : red(n), green(n), blue(n) {}
};

double kmeans_clustering_3(std::ptrdiff_t nb_point, std::ptrdiff_t nb_cluster,
                           std::ptrdiff_t nb_iteration) {
  PixelVector point(nb_point);
  std::vector<std::ptrdiff_t> cluster(nb_point);

  PixelVector centroid(nb_cluster);
  std::vector<std::ptrdiff_t> point_per_cluster(nb_cluster);

  std::default_random_engine engine{};
  std::uniform_real_distribution<float> r_dist{0.0f, 1.0f};
  std::uniform_int_distribution<std::ptrdiff_t> i_dist{0, nb_cluster - 1};
  for (std::ptrdiff_t k = 0; k < nb_point; ++k) {
    point.red[k] = r_dist(engine);
    point.green[k] = r_dist(engine);
    point.blue[k] = r_dist(engine);
    cluster[k] = i_dist(engine);
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::ptrdiff_t iteration = 0;
  while (true) {
    // Compute the centroid of the clusters
    for (std::ptrdiff_t i = 0; i < nb_cluster; ++i) {
      centroid.red[i] = 0.0f;
      centroid.green[i] = 0.0f;
      centroid.blue[i] = 0.0f;
      point_per_cluster[i] = 0;
    }
    for (std::ptrdiff_t k = 0; k < nb_point; ++k) {
      std::ptrdiff_t i = cluster[k];
      centroid.red[i] += point.red[k];
      centroid.green[i] += point.green[k];
      centroid.blue[i] += point.blue[k];
      ++point_per_cluster[i];
    }
    for (std::ptrdiff_t i = 0; i < nb_cluster; ++i) {
      std::ptrdiff_t nb_point_cluster = point_per_cluster[i];
      centroid.red[i] /= nb_point_cluster;
      centroid.green[i] /= nb_point_cluster;
      centroid.blue[i] /= nb_point_cluster;
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration > nb_iteration) {
      break;
    }

    // Reassign points to clusters
    for (std::ptrdiff_t k = 0; k < nb_point; ++k) {
      float best_distance = std::numeric_limits<float>::max();
      std::ptrdiff_t best_centroid = -1;
      for (std::ptrdiff_t i = 0; i < nb_cluster; ++i) {
        float x = point.red[k] - centroid.red[i];
        float y = point.green[k] - centroid.green[i];
        float z = point.blue[k] - centroid.blue[i];
        float distance = x * x + y * y + z * z;
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

double kmeans_clustering_4(int nb_point, int nb_cluster, int nb_iteration) {
  PixelVector point(nb_point);
  std::vector<int> cluster(nb_point);

  PixelVector centroid(nb_cluster);
  std::vector<int> point_per_cluster(nb_cluster);

  std::default_random_engine engine{};
  std::uniform_real_distribution<float> r_dist{0.0f, 1.0f};
  std::uniform_int_distribution<int> i_dist{0, nb_cluster - 1};
  for (int k = 0; k < nb_point; ++k) {
    point.red[k] = r_dist(engine);
    point.green[k] = r_dist(engine);
    point.blue[k] = r_dist(engine);
    cluster[k] = i_dist(engine);
  }

  auto start = std::chrono::high_resolution_clock::now();
  int iteration = 0;
  while (true) {
    // Compute the centroid of the clusters
    for (int i = 0; i < nb_cluster; ++i) {
      centroid.red[i] = 0.0f;
      centroid.green[i] = 0.0f;
      centroid.blue[i] = 0.0f;
      point_per_cluster[i] = 0;
    }
    for (int k = 0; k < nb_point; ++k) {
      int i = cluster[k];
      centroid.red[i] += point.red[k];
      centroid.green[i] += point.green[k];
      centroid.blue[i] += point.blue[k];
      ++point_per_cluster[i];
    }
    for (int i = 0; i < nb_cluster; ++i) {
      int nb_point_cluster = point_per_cluster[i];
      centroid.red[i] /= nb_point_cluster;
      centroid.green[i] /= nb_point_cluster;
      centroid.blue[i] /= nb_point_cluster;
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration > nb_iteration) {
      break;
    }

    // Reassign points to clusters
    for (int k = 0; k < nb_point; ++k) {
      float best_distance = std::numeric_limits<float>::max();
      int best_centroid = -1;
      for (int i = 0; i < nb_cluster; ++i) {
        float x = point.red[k] - centroid.red[i];
        float y = point.green[k] - centroid.green[i];
        float z = point.blue[k] - centroid.blue[i];
        float distance = x * x + y * y + z * z;
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

double kmeans_clustering_5(int nb_point, int nb_cluster, int nb_iteration) {
  PixelVector point(nb_point);
  std::vector<int> cluster(nb_point);

  PixelVector centroid(nb_cluster);
  std::vector<int> point_per_cluster(nb_cluster);

  std::default_random_engine engine{};
  std::uniform_real_distribution<float> r_dist{0.0f, 1.0f};
  std::uniform_int_distribution<int> i_dist{0, nb_cluster - 1};
  for (int k = 0; k < nb_point; ++k) {
    point.red[k] = r_dist(engine);
    point.green[k] = r_dist(engine);
    point.blue[k] = r_dist(engine);
    cluster[k] = i_dist(engine);
  }

  auto start = std::chrono::high_resolution_clock::now();
  int iteration = 0;
  while (true) {
    // Compute the centroid of the clusters
    for (int i = 0; i < nb_cluster; ++i) {
      centroid.red[i] = 0.0f;
      centroid.green[i] = 0.0f;
      centroid.blue[i] = 0.0f;
      point_per_cluster[i] = 0;
    }
    for (int k = 0; k < nb_point; ++k) {
      int i = cluster[k];
      centroid.red[i] += point.red[k];
      centroid.green[i] += point.green[k];
      centroid.blue[i] += point.blue[k];
      ++point_per_cluster[i];
    }
    for (int i = 0; i < nb_cluster; ++i) {
      int nb_point_cluster = point_per_cluster[i];
      centroid.red[i] /= nb_point_cluster;
      centroid.green[i] /= nb_point_cluster;
      centroid.blue[i] /= nb_point_cluster;
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration > nb_iteration) {
      break;
    }

// Reassign points to clusters
#pragma omp parallel for
    for (int k = 0; k < nb_point; ++k) {
      float best_distance = std::numeric_limits<float>::max();
      int best_centroid = -1;
      for (int i = 0; i < nb_cluster; ++i) {
        float x = point.red[k] - centroid.red[i];
        float y = point.green[k] - centroid.green[i];
        float z = point.blue[k] - centroid.blue[i];
        float distance = x * x + y * y + z * z;
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

double kmeans_clustering_6(int nb_point, int nb_cluster, int nb_iteration) {
  PixelVector point(nb_point);
  std::vector<int> cluster(nb_point);

  PixelVector centroid(nb_cluster);
  std::vector<int> point_per_cluster(nb_cluster);

  int nb_thread = omp_get_max_threads();
  PixelVector local_centroid(nb_cluster * nb_thread);
  std::vector<int> local_point_per_cluster(nb_cluster * nb_thread);

  // std::default_random_engine engine{};
  // std::uniform_real_distribution<float> r_dist{0.0f, 1.0f};
  // std::uniform_int_distribution<int> i_dist{0, nb_cluster - 1};
  float x = 0.123456789f;
  for (int k = 0; k < nb_point; ++k) {
    point.red[k] = x;
    x = 4 * x * (1 - x);
    point.green[k] = x;
    x = 4 * x * (1 - x);
    point.blue[k] = x;
    x = 4 * x * (1 - x);
    cluster[k] = k % nb_cluster;
  }

  auto start = std::chrono::high_resolution_clock::now();
  int iteration = 0;
  while (true) {
    // Compute the centroid of the clusters
    for (int id = 0; id < nb_thread; ++id) {
      for (int i = 0; i < nb_cluster; ++i) {
        local_centroid.red[nb_cluster * id + i] = 0.0f;
        local_centroid.green[nb_cluster * id + i] = 0.0f;
        local_centroid.blue[nb_cluster * id + i] = 0.0f;
        local_point_per_cluster[nb_cluster * id + i] = 0;
      }
    }
#pragma omp parallel for
    for (int k = 0; k < nb_point; ++k) {
      int id = omp_get_thread_num();
      int i = cluster[k];
      local_centroid.red[nb_cluster * id + i] += point.red[k];
      local_centroid.green[nb_cluster * id + i] += point.green[k];
      local_centroid.blue[nb_cluster * id + i] += point.blue[k];
      ++local_point_per_cluster[nb_cluster * id + i];
    }
    for (int i = 0; i < nb_cluster; ++i) {
      centroid.red[i] = 0.0f;
      centroid.green[i] = 0.0f;
      centroid.blue[i] = 0.0f;
      point_per_cluster[i] = 0;
    }
    for (int id = 0; id < nb_thread; ++id) {
      for (int i = 0; i < nb_cluster; ++i) {
        centroid.red[i] += local_centroid.red[nb_cluster * id + i];
        centroid.green[i] += local_centroid.green[nb_cluster * id + i];
        centroid.blue[i] += local_centroid.blue[nb_cluster * id + i];
        point_per_cluster[i] += local_point_per_cluster[nb_cluster * id + i];
      }
    }
    for (int i = 0; i < nb_cluster; ++i) {
      int nb_point_cluster = point_per_cluster[i];
      centroid.red[i] /= nb_point_cluster;
      centroid.green[i] /= nb_point_cluster;
      centroid.blue[i] /= nb_point_cluster;
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration > nb_iteration) {
      break;
    }

// Reassign points to clusters
#pragma omp parallel for
    for (int k = 0; k < nb_point; ++k) {
      float best_distance = std::numeric_limits<float>::max();
      int best_centroid = -1;
      for (int i = 0; i < nb_cluster; ++i) {
        float x = point.red[k] - centroid.red[i];
        float y = point.green[k] - centroid.green[i];
        float z = point.blue[k] - centroid.blue[i];
        float distance = x * x + y * y + z * z;
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

struct Group {
  il::Array2D<float> centroid;
  il::Array<int> point_per_centroid;
  Group(int nb_cluster = 0)
      : centroid{nb_cluster, 3}, point_per_centroid{nb_cluster} {}
};

double kmeans_clustering_il(int nb_point, int nb_cluster, int nb_iteration) {
  const int nb_thread = omp_get_max_threads();

  Group group{nb_cluster};
  il::Array<Group> local_group{nb_thread, il::emplace, nb_cluster};
  il::Array<int> cluster{nb_point};

  return 0.0;
}
}  // namespace il
