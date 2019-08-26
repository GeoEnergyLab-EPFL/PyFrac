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

#ifndef IL_KMEANS_CLUSTERING_H
#define IL_KMEANS_CLUSTERING_H

#include <il/Array2C.h>
#include <il/Array2D.h>

namespace il {

il::Array2C<float> kmeans_clustering_0(const il::Array2C<float>& point,
                                       int nb_cluster, int nb_iteration);
il::Array2D<float> kmeans_clustering_1(const il::Array2D<float>& point,
                                       int nb_cluster, int nb_iteration);
il::Array2D<float> kmeans_clustering_2(const il::Array2D<float>& point,
                                       int nb_cluster, int nb_iteration);
il::Array2D<float> kmeans_clustering_3(const il::Array2D<float>& point,
                                       int nb_cluster, int nb_iteration);
}  // namespace il

#endif  // IL_KMEANS_CLUSTERING_H
