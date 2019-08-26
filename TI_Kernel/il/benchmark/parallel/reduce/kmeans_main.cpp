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

#include <il/benchmark/parallel/reduce/kmeans_main.h>

#include <chrono>
#include <cstdio>
#include <string>

#include <il/Array2D.h>
#include <il/benchmark/parallel/reduce/kmeans_clustering.h>
#include <il/io/png.h>

void kmeans_main() {
  std::string directory{"/Users/fayard/Desktop/"};

  il::Status status{};
  il::Array3D<unsigned char> image{
      il::load(directory + std::string{"lotus.png"}, il::png, il::io, status)};
  status.AbortOnError();
  const int width{image.size(0)};
  const int height{image.size(1)};

  const int nb_point{width * height};
  const int nb_cluster = 8;
  const int nb_iteration = 200;

  il::Array2D<float> point{nb_point, 3};
  for (int ky = 0; ky < image.size(1); ++ky) {
    for (int kx = 0; kx < image.size(0); ++kx) {
      point(ky * width + kx, 0) = static_cast<float>(image(kx, ky, 0)) / 255;
      point(ky * width + kx, 1) = static_cast<float>(image(kx, ky, 1)) / 255;
      point(ky * width + kx, 2) = static_cast<float>(image(kx, ky, 2)) / 255;
    }
  }

  //  auto start = std::chrono::high_resolution_clock::now();
  //  il::Array2C<float> point_bis{nb_point, 3};
  //  for (int k = 0; k < nb_point; ++k) {
  //    point_bis(k, 0) = point(k, 0);
  //    point_bis(k, 1) = point(k, 1);
  //    point_bis(k, 2) = point(k, 2);
  //  }
  //  il::Array2C<float> centroid_bis{
  //      il::kmeans_clustering_0(point_bis, nb_cluster, nb_iteration)};
  //  auto end = std::chrono::high_resolution_clock::now();
  //  double time{1.0e-9 *
  //              std::chrono::duration_cast<std::chrono::nanoseconds>(end -
  //              start)
  //                  .count()};
  //  std::printf("Naive method: %7.3f s\n", time);

  //  start = std::chrono::high_resolution_clock::now();
  //  il::Array2D<float> centroid_1{
  //      il::kmeans_clustering_1(point, nb_cluster, nb_iteration)};
  //  end = std::chrono::high_resolution_clock::now();
  //  time =
  //      1.0e-9 *
  //      std::chrono::duration_cast<std::chrono::nanoseconds>(end -
  //      start).count();
  //  std::printf("  SOA method: %7.3f s\n", time);

  auto start = std::chrono::high_resolution_clock::now();
  il::Array2D<float> centroid_2{
      il::kmeans_clustering_2(point, nb_cluster, nb_iteration)};
  auto end = std::chrono::high_resolution_clock::now();
  double time{1.0e-9 *
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count()};
  std::printf(" SOAp method: %7.3f s\n", time);

  start = std::chrono::high_resolution_clock::now();
  il::Array2D<float> centroid{
      il::kmeans_clustering_3(point, nb_cluster, nb_iteration)};
  end = std::chrono::high_resolution_clock::now();
  time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf(" Best method: %7.3f s\n", time);

  start = std::chrono::high_resolution_clock::now();
  il::Array2D<float> centroid_3{
      il::kmeans_clustering_2(point, nb_cluster, nb_iteration)};
  end = std::chrono::high_resolution_clock::now();
  time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf(" SOAp method: %7.3f s\n", time);

  start = std::chrono::high_resolution_clock::now();
  il::Array2D<float> centroid_4{
      il::kmeans_clustering_3(point, nb_cluster, nb_iteration)};
  end = std::chrono::high_resolution_clock::now();
  time =
      1.0e-9 *
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf(" Best method: %7.3f s\n", time);

  il::Array3D<unsigned char> out{width, height, 3};
  for (int ky = 0; ky < height; ++ky) {
    for (int kx = 0; kx < width; ++kx) {
      float best_distance{std::numeric_limits<float>::max()};
      int best_centroid{-1};
      for (int i = 0; i < nb_cluster; ++i) {
        float x{point(ky * width + kx, 0) - centroid(i, 0)};
        float y{point(ky * width + kx, 1) - centroid(i, 1)};
        float z{point(ky * width + kx, 2) - centroid(i, 2)};
        float distance{x * x + y * y + z * z};
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = i;
        }
      }
      out(kx, ky, 0) =
          static_cast<unsigned char>(centroid(best_centroid, 0) * 255);
      out(kx, ky, 1) =
          static_cast<unsigned char>(centroid(best_centroid, 1) * 255);
      out(kx, ky, 2) =
          static_cast<unsigned char>(centroid(best_centroid, 2) * 255);
    }
  }

  il::save(out, directory + std::string{"lotus-saved.png"}, il::png, il::io,
           error);
  status.AbortOnError();
}
