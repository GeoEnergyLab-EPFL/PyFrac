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

#include <il/benchmark/parallel/reduce/kmeans_clustering.h>

#include <random>

#include <omp.h>

#include <il/Array.h>

namespace il {

void fix_clusters(const il::Array2C<float>& point, il::io_t,
                  il::Array<int>& cluster, il::Array2C<float>& centroid,
                  il::Array<int>& point_per_centroid,
                  std::default_random_engine& engine);

void fix_clusters(const il::Array2D<float>& point, il::io_t,
                  il::Array<int>& cluster, il::Array2D<float>& centroid,
                  il::Array<int>& point_per_centroid,
                  std::default_random_engine& engine);

il::Array2C<float> kmeans_clustering_0(const il::Array2C<float>& point,
                                       int nb_cluster, int nb_iteration) {
  IL_EXPECT_FAST(point.size(1) == 3);
  const int nb_point{point.size(0)};
  IL_EXPECT_FAST(nb_point >= nb_cluster);
  IL_EXPECT_FAST(nb_iteration >= 0);

  il::Array2C<float> centroid{nb_cluster, 3};
  il::Array<int> point_per_centroid{nb_cluster};
  il::Array<int> cluster{nb_point};
  for (int k = 0; k < nb_point; ++k) {
    cluster[k] = k % nb_cluster;
  }

  std::default_random_engine engine{};
  int iteration{-1};
  while (true) {
    // Compute the centroid of the clusters
    for (int i = 0; i < nb_cluster; ++i) {
      centroid(i, 0) = 0.0f;
      centroid(i, 1) = 0.0f;
      centroid(i, 2) = 0.0f;
      point_per_centroid[i] = 0;
    }
    for (int k = 0; k < nb_point; ++k) {
      int i{cluster[k]};
      ++point_per_centroid[i];
      centroid(i, 0) += point(k, 0);
      centroid(i, 1) += point(k, 1);
      centroid(i, 2) += point(k, 2);
    }

    // Fix the empty clusters
    fix_clusters(point, il::io, cluster, centroid, point_per_centroid, engine);

    // Finish to compute the centroids of the clusters
    for (int i = 0; i < nb_cluster; ++i) {
      centroid(i, 0) /= point_per_centroid[i];
      centroid(i, 1) /= point_per_centroid[i];
      centroid(i, 2) /= point_per_centroid[i];
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration == nb_iteration) {
      break;
    }

    // Reassign points to clusters
    for (int k = 0; k < nb_point; ++k) {
      float best_distance{std::numeric_limits<float>::max()};
      int best_centroid{-1};
      for (int i = 0; i < nb_cluster; ++i) {
        float x{point(k, 0) - centroid(i, 0)};
        float y{point(k, 1) - centroid(i, 1)};
        float z{point(k, 2) - centroid(i, 2)};
        float distance{x * x + y * y + z * z};
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }
  return centroid;
}

il::Array2D<float> kmeans_clustering_1(const il::Array2D<float>& point,
                                       int nb_cluster, int nb_iteration) {
  IL_EXPECT_FAST(point.size(1) == 3);
  const int nb_point{point.size(0)};
  IL_EXPECT_FAST(nb_point >= nb_cluster);
  IL_EXPECT_FAST(nb_iteration >= 0);

  il::Array<int> cluster{nb_point};
  il::Array2D<float> centroid{nb_cluster, 3};
  il::Array<int> point_per_centroid{nb_cluster};
  for (int k = 0; k < nb_point; ++k) {
    cluster[k] = k % nb_cluster;
  }

  std::default_random_engine engine{};
  int iteration{-1};
  while (true) {
    // Compute the centroid of the clusters
    for (int i = 0; i < nb_cluster; ++i) {
      centroid(i, 0) = 0.0f;
      centroid(i, 1) = 0.0f;
      centroid(i, 2) = 0.0f;
      point_per_centroid[i] = 0;
    }
    for (int k = 0; k < nb_point; ++k) {
      int i{cluster[k]};
      ++point_per_centroid[i];
      centroid(i, 0) += point(k, 0);
      centroid(i, 1) += point(k, 1);
      centroid(i, 2) += point(k, 2);
    }

    // Fix the empty clusters
    fix_clusters(point, il::io, cluster, centroid, point_per_centroid, engine);

    // Finish to compute the centroids of the clusters
    for (int i = 0; i < nb_cluster; ++i) {
      centroid(i, 0) /= point_per_centroid[i];
      centroid(i, 1) /= point_per_centroid[i];
      centroid(i, 2) /= point_per_centroid[i];
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration == nb_iteration) {
      break;
    }

    // Reassign points to clusters
    for (int k = 0; k < nb_point; ++k) {
      float best_distance{std::numeric_limits<float>::max()};
      int best_centroid{-1};
      for (int i = 0; i < nb_cluster; ++i) {
        float x{point(k, 0) - centroid(i, 0)};
        float y{point(k, 1) - centroid(i, 1)};
        float z{point(k, 2) - centroid(i, 2)};
        float distance{x * x + y * y + z * z};
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }

  return centroid;
}

struct Group {
  il::Array2D<float> centroid;
  il::Array<int> point_per_centroid;
  Group(int nb_cluster = 0)
      : centroid{nb_cluster, 3}, point_per_centroid{nb_cluster} {}
};

il::Array2D<float> kmeans_clustering_2(const il::Array2D<float>& point,
                                       int nb_cluster, int nb_iteration) {
  IL_EXPECT_FAST(point.size(1) == 3);
  const int nb_point{point.size(0)};
  IL_EXPECT_FAST(nb_point >= nb_cluster);
  IL_EXPECT_FAST(nb_iteration >= 0);

  const int nb_thread{omp_get_max_threads()};

  Group group{nb_cluster};
  il::Array<Group> local_group{nb_thread, il::emplace, nb_cluster};
  il::Array<int> cluster{nb_point};
#pragma omp parallel for
  for (int k = 0; k < nb_point; ++k) {
    cluster[k] = k % nb_cluster;
  }

  std::default_random_engine engine{};
  int iteration{-1};
  while (true) {
// Compute the centroid of the clusters
#pragma omp parallel for
    for (int id = 0; id < nb_thread; ++id) {
      for (int i = 0; i < nb_cluster; ++i) {
        local_group[id].centroid(i, 0) = 0.0f;
        local_group[id].centroid(i, 1) = 0.0f;
        local_group[id].centroid(i, 2) = 0.0f;
        local_group[id].point_per_centroid[i] = 0;
      }
    }
#pragma omp parallel for
    for (int i = 0; i < nb_cluster; ++i) {
      group.centroid(i, 0) = 0.0f;
      group.centroid(i, 1) = 0.0f;
      group.centroid(i, 2) = 0.0f;
      group.point_per_centroid[i] = 0;
    }
#pragma omp parallel for
    for (int k = 0; k < nb_point; ++k) {
      int id{omp_get_thread_num()};
      int i{cluster[k]};
      local_group[id].centroid(i, 0) += point(k, 0);
      local_group[id].centroid(i, 1) += point(k, 1);
      local_group[id].centroid(i, 2) += point(k, 2);
      local_group[id].point_per_centroid[i] += 1;
    }
    for (int id = 0; id < nb_thread; ++id) {
#pragma omp parallel for
      for (int i = 0; i < nb_cluster; ++i) {
        group.centroid(i, 0) += local_group[id].centroid(i, 0);
        group.centroid(i, 1) += local_group[id].centroid(i, 1);
        group.centroid(i, 2) += local_group[id].centroid(i, 2);
        group.point_per_centroid[i] += local_group[id].point_per_centroid[i];
      }
    }
    // Fix the empty clusters
    fix_clusters(point, il::io, cluster, group.centroid,
                 group.point_per_centroid, engine);

// Finish to compute the centroids of the clusters
#pragma omp parallel for
    for (int i = 0; i < nb_cluster; ++i) {
      group.centroid(i, 0) /= group.point_per_centroid[i];
      group.centroid(i, 1) /= group.point_per_centroid[i];
      group.centroid(i, 2) /= group.point_per_centroid[i];
    }

    // Exit once convergence is reached
    ++iteration;
    if (iteration == nb_iteration) {
      break;
    }

// Reassign points to clusters
#pragma omp parallel for
    for (int k = 0; k < nb_point; ++k) {
      float best_distance{std::numeric_limits<float>::max()};
      int best_centroid{-1};
      for (int i = 0; i < nb_cluster; ++i) {
        float x{point(k, 0) - group.centroid(i, 0)};
        float y{point(k, 1) - group.centroid(i, 1)};
        float z{point(k, 2) - group.centroid(i, 2)};
        float distance{x * x + y * y + z * z};
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
    }
  }

  return group.centroid;
}

il::Array2D<float> kmeans_clustering_3(const il::Array2D<float>& point,
                                       int nb_cluster, int nb_iteration) {
  IL_EXPECT_FAST(point.size(1) == 3);
  const int nb_point{point.size(0)};
  IL_EXPECT_FAST(nb_point >= nb_cluster);
  IL_EXPECT_FAST(nb_iteration >= 0);

  const int nb_thread{omp_get_max_threads()};

  Group group{nb_cluster};
  il::Array<Group> local_group{nb_thread, il::emplace, nb_cluster};
  il::Array<int> cluster{nb_point};

#pragma omp parallel for
  for (int id = 0; id < nb_thread; ++id) {
    for (int i = 0; i < nb_cluster; ++i) {
      local_group[id].centroid(i, 0) = 0.0f;
      local_group[id].centroid(i, 1) = 0.0f;
      local_group[id].centroid(i, 2) = 0.0f;
      local_group[id].point_per_centroid[i] = 0;
    }
  }
#pragma omp parallel for
  for (int i = 0; i < nb_cluster; ++i) {
    group.centroid(i, 0) = 0.0f;
    group.centroid(i, 1) = 0.0f;
    group.centroid(i, 2) = 0.0f;
    group.point_per_centroid[i] = 0;
  }

#pragma omp parallel for
  for (int k = 0; k < nb_point; ++k) {
    int id{omp_get_thread_num()};
    int i{k % nb_cluster};
    local_group[id].centroid(i, 0) += point(k, 0);
    local_group[id].centroid(i, 1) += point(k, 1);
    local_group[id].centroid(i, 2) += point(k, 2);
    local_group[id].point_per_centroid[i] += 1;
    cluster[k] = i;
  }

  std::default_random_engine engine{};
  int iteration{-1};
  while (true) {
// Reduce local sums to global sum
#pragma omp parallel for
    for (int i = 0; i < nb_cluster; ++i) {
      group.centroid(i, 0) = 0.0f;
      group.centroid(i, 1) = 0.0f;
      group.centroid(i, 2) = 0.0f;
      group.point_per_centroid[i] = 0;
    }
    for (int id = 0; id < nb_thread; ++id) {
#pragma omp parallel for
      for (int i = 0; i < nb_cluster; ++i) {
        group.centroid(i, 0) += local_group[id].centroid(i, 0);
        group.centroid(i, 1) += local_group[id].centroid(i, 1);
        group.centroid(i, 2) += local_group[id].centroid(i, 2);
        group.point_per_centroid[i] += local_group[id].point_per_centroid[i];
        local_group[id].centroid(i, 0) = 0.0f;
        local_group[id].centroid(i, 1) = 0.0f;
        local_group[id].centroid(i, 2) = 0.0f;
        local_group[id].point_per_centroid[i] = 0;
      }
    }

    // We might have to fix the clusters if some don't have any point
    fix_clusters(point, il::io, cluster, group.centroid,
                 group.point_per_centroid, engine);

// Compute centroids for global sums
#pragma omp parallel for
    for (int i = 0; i < nb_cluster; ++i) {
      float coeff{1.0f / group.point_per_centroid[i]};
      group.centroid(i, 0) *= coeff;
      group.centroid(i, 1) *= coeff;
      group.centroid(i, 2) *= coeff;
    }

    ++iteration;
    if (iteration == nb_iteration) {
      break;
    }

// Compute the new clusters and their local sums
#pragma omp parallel for
    for (int k = 0; k < nb_point; ++k) {
      int id{omp_get_thread_num()};
      float best_distance{std::numeric_limits<float>::max()};
      int best_centroid{-1};
#pragma omp simd
      for (int i = 0; i < nb_cluster; ++i) {
        float x{point(k, 0) - group.centroid(i, 0)};
        float y{point(k, 1) - group.centroid(i, 1)};
        float z{point(k, 2) - group.centroid(i, 2)};
        float distance{x * x + y * y + z * z};
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      cluster[k] = best_centroid;
      local_group[id].centroid(best_centroid, 0) += point(k, 0);
      local_group[id].centroid(best_centroid, 1) += point(k, 1);
      local_group[id].centroid(best_centroid, 2) += point(k, 2);
      local_group[id].point_per_centroid[best_centroid] += 1;
    }
  }

  return group.centroid;
}

void fix_clusters(const il::Array2C<float>& point, il::io_t,
                  il::Array<int>& cluster, il::Array2C<float>& centroid,
                  il::Array<int>& point_per_centroid,
                  std::default_random_engine& engine) {
  const int nb_point{point.size(0)};
  const int nb_cluster{centroid.size(0)};
  std::uniform_int_distribution<int> distribution{0, nb_point - 1};

  while (true) {
    il::Array<int> cluster_to_fix{};
    for (int i = 0; i < nb_cluster; ++i) {
      if (point_per_centroid[i] == 0) {
        cluster_to_fix.Append(i);
      }
    }
    if (cluster_to_fix.size() == 0) {
      break;
    }
    for (int i : cluster_to_fix) {
      int k_candidate;
      int i_candidate;
      do {
        k_candidate = distribution(engine);
        i_candidate = cluster[k_candidate];
      } while (point_per_centroid[i_candidate] <= 1);
      centroid(i_candidate, 0) -= point(k_candidate, 0);
      centroid(i_candidate, 1) -= point(k_candidate, 1);
      centroid(i_candidate, 2) -= point(k_candidate, 2);
      --point_per_centroid[i_candidate];
      centroid(i, 0) += point(k_candidate, 0);
      centroid(i, 1) += point(k_candidate, 1);
      centroid(i, 2) += point(k_candidate, 2);
      ++point_per_centroid[i];
      cluster[k_candidate] = i;
    }
  }
}

void fix_clusters(const il::Array2D<float>& point, il::io_t,
                  il::Array<int>& cluster, il::Array2D<float>& centroid,
                  il::Array<int>& point_per_centroid,
                  std::default_random_engine& engine) {
  const int nb_point{point.size(0)};
  const int nb_cluster{centroid.size(0)};
  std::uniform_int_distribution<int> distribution{0, nb_point - 1};

  while (true) {
    il::Array<int> cluster_to_fix{};
    for (int i = 0; i < nb_cluster; ++i) {
      if (point_per_centroid[i] == 0) {
        cluster_to_fix.Append(i);
      }
    }
    if (cluster_to_fix.size() == 0) {
      break;
    }
    for (int i : cluster_to_fix) {
      int k_candidate;
      int i_candidate;
      do {
        k_candidate = distribution(engine);
        i_candidate = cluster[k_candidate];
      } while (point_per_centroid[i_candidate] <= 1);
      centroid(i_candidate, 0) -= point(k_candidate, 0);
      centroid(i_candidate, 1) -= point(k_candidate, 1);
      centroid(i_candidate, 2) -= point(k_candidate, 2);
      --point_per_centroid[i_candidate];
      centroid(i, 0) += point(k_candidate, 0);
      centroid(i, 1) += point(k_candidate, 1);
      centroid(i, 2) += point(k_candidate, 2);
      ++point_per_centroid[i];
      cluster[k_candidate] = i;
    }
  }
}
}  // namespace il
