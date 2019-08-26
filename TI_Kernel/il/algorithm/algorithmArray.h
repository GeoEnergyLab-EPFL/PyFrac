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

#ifndef IL_ALGORITHM_ARRAY_H
#define IL_ALGORITHM_ARRAY_H

#include <algorithm>

#include <il/Array.h>
#include <il/StaticArray.h>
#include <il/math.h>

namespace il {

template <typename T>
T min(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t min_value = v[0];
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (v[i] < min_value) {
      min_value = v[i];
    }
  }
  return min_value;
}

template <typename T>
T min(const il::Array<T>& v, il::Range range) {
  IL_EXPECT_FAST(range.begin < range.end);

  il::int_t min_value = v[range.begin];
  for (il::int_t i = range.begin + 1; i < range.end; ++i) {
    if (v[i] < min_value) {
      min_value = v[i];
    }
  }
  return min_value;
}

template <typename T>
T max(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t max_value = v[0];
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (v[i] > max_value) {
      max_value = v[i];
    }
  }
  return max_value;
}

template <typename T>
T max(const il::Array<T>& v, il::Range range) {
  IL_EXPECT_FAST(range.begin < range.end);

  il::int_t max_value = v[range.begin];
  for (il::int_t i = range.begin + 1; i < range.end; ++i) {
    if (v[i] > max_value) {
      max_value = v[i];
    }
  }
  return max_value;
}

template <typename T>
T maxAbs(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t max_value = il::abs(v[0]);
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (il::abs(v[i]) > max_value) {
      max_value = il::abs(v[i]);
    }
  }
  return max_value;
}

template <typename T>
T maxAbs(const il::Array<T>& v, il::Range range) {
  IL_EXPECT_FAST(range.begin < range.end);

  il::int_t max_value = il::abs(v[range.begin]);
  for (il::int_t i = range.begin + 1; i < range.end; ++i) {
    if (v[i] > max_value) {
      max_value = il::abs(v[i]);
    }
  }
  return max_value;
}

template <typename T>
il::int_t indexMin(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t min_index = 0;
  T min_value = v[0];
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (v[i] < min_value) {
      min_index = i;
      min_value = v[i];
    }
  }
  return min_index;
}

template <typename T>
il::int_t indexMin(const il::Array<T>& v, il::Range range) {
  IL_EXPECT_FAST(range.begin < range.end);

  il::int_t min_index = range.begin;
  T min_value = v[range.begin];
  for (il::int_t i = range.begin + 1; i < range.end; ++i) {
    if (v[i] < min_value) {
      min_index = i;
      min_value = v[i];
    }
  }
  return min_index;
}

template <typename T>
il::int_t indexMax(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t max_index = 0;
  T max_value = v[0];
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (v[i] > max_value) {
      max_index = i;
      max_value = v[i];
    }
  }
  return max_index;
}

template <typename T>
il::int_t indexMax(const il::Array<T>& v, il::Range range) {
  IL_EXPECT_FAST(range.begin < range.end);

  il::int_t max_index = range.begin;
  T max_value = v[range.begin];
  for (il::int_t i = range.begin + 1; i < range.end; ++i) {
    if (v[i] > max_value) {
      max_index = i;
      max_value = v[i];
    }
  }
  return max_index;
}

template <typename T>
il::int_t indexMaxAbs(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t max_index = 0;
  T max_value = il::abs(v[0]);
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (il::abs(v[i]) > max_value) {
      max_index = i;
      max_value = il::abs(v[i]);
    }
  }
  return max_index;
}

template <typename T>
il::int_t indexMaxAbs(const il::Array<T>& v, il::Range range) {
  IL_EXPECT_FAST(range.begin < range.end);

  il::int_t max_index = range.begin;
  T max_value = il::abs(v[range.begin]);
  for (il::int_t i = range.begin + 1; i < range.end; ++i) {
    if (il::abs(v[i]) > max_value) {
      max_index = i;
      max_value = il::abs(v[i]);
    }
  }
  return max_index;
}

template <typename T>
struct MinMax {
  T min;
  T max;
};

template <typename T>
MinMax<T> minMax(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  MinMax<T> ans{};
  ans.min = v[0];
  ans.max = v[0];
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (v[i] < ans.min) {
      ans.min = v[i];
    }
    if (v[i] > ans.max) {
      ans.max = v[i];
    }
  }
  return ans;
}

template <typename T>
MinMax<T> minMax(const il::Array<T>& v, il::Range range) {
  IL_EXPECT_FAST(v.size() > 0);

  MinMax<T> ans{};
  ans.min = v[range.begin];
  ans.max = v[range.end];
  for (il::int_t i = range.begin + 1; i < range.end; ++i) {
    if (v[i] < ans.min) {
      ans.min = v[i];
    }
    if (v[i] > ans.max) {
      ans.max = v[i];
    }
  }
  return ans;
}

// template <typename T>
// void sort(il::io_t, il::Array<T>& v) {
//  std::sort(v.Data(), v.end());
//}

template <typename T>
void sort_aux(il::Range range, il::io_t, il::Array<T>& v) {
  if (range.end - range.begin <= 1) {
    return;
  } else if (range.end - range.begin <= 15) {
    for (il::int_t i = range.begin + 1; i < range.end; ++i) {
      il::int_t j = i;
      while (j > 0 && v[j - 1] > v[j]) {
        const T aux = v[j];
        v[j] = v[j - 1];
        v[j - 1] = aux;
        --j;
      }
    }
    return;
  }
  // choose the pivot using the median rule
  const il::int_t i_begin = range.begin;
  const il::int_t i_end = range.end - 1;
  const il::int_t i_middle = range.begin + (range.end - range.begin) / 2;
  il::int_t pivot;
  if (v[i_begin] < v[i_end]) {
    if (v[i_middle] < v[i_begin]) {
      pivot = i_begin;
    } else {
      pivot = (v[i_middle] > v[i_end]) ? i_end : i_middle;
    }
  } else {
    if (v[i_middle] < v[i_end]) {
      pivot = i_end;
    } else {
      pivot = (v[i_begin] < v[i_middle]) ? i_begin : i_middle;
    }
  }
  // swap the pivot and the last element so that the pivot is at the end
  {
    const T aux = v[i_end];
    v[i_end] = v[pivot];
    v[pivot] = aux;
    pivot = i_end;
  }
  // Partition into 2 parts
  const T value_pivot = v[pivot];
  il::int_t j = range.begin;
  for (il::int_t i = range.begin; i < range.end - 1; ++i) {
    if (v[i] < value_pivot) {
      // Swap v[j] and v[i]
      const T aux = v[i];
      v[i] = v[j];
      v[j] = aux;
      ++j;
    }
  }
  // Swap the element j and the last element so the pivot is in place
  {
    const T aux = v[i_end];
    v[i_end] = v[j];
    v[j] = aux;
  }
  // Recursively sort the 2 other parts
  sort_aux(il::Range{range.begin, j}, il::io, v);
  sort_aux(il::Range{j + 1, range.end}, il::io, v);
}

template <typename T>
void sort(il::io_t, il::Array<T>& v) {
  il::sort_aux(il::Range{0, v.size()}, il::io, v);
}

template <typename T, il::int_t n>
void sort(il::io_t, il::StaticArray<T, n>& v) {
  std::sort(v.Data(), v.Data() + n);
}

template <typename T>
il::Array<T> sort(il::Array<T>& v) {
  il::Array<T> w = v;
  il::sort_aux(il::Range{0, w.size()}, il::io, w);
  return w;
}

template <typename T>
il::Array<T> sort(const il::Array<T>& v) {
  il::Array<T> ans = v;
  std::sort(ans.Data(), ans.end());
  return ans;
}

template <typename T>
il::int_t binarySearch(const il::Array<T>& v, const T& x) {
  il::int_t i_begin = 0;
  il::int_t i_end = v.size();
  while (i_end > i_begin + 1) {
    const il::int_t i = i_begin + (i_end - i_begin) / 2;
    if (v[i] <= x) {
      i_begin = i;
    } else {
      i_end = i;
    }
  }
  return (i_end == i_begin + 1 && v[i_begin] == x) ? i_begin : -1;
}

template <typename T>
T mean(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  T ans = 0;
  for (il::int_t i = 0; i < v.size(); ++i) {
    ans += v[i];
  }
  ans /= v.size();
  return ans;
}

enum class VarianceKind { Population, Sample };

template <typename T>
T variance(const il::Array<T>& v, il::VarianceKind kind) {
  IL_EXPECT_FAST(v.size() > ((kind == il::VarianceKind::Population) ? 0 : 1));

  T mean = 0;
  for (il::int_t i = 0; i < v.size(); ++i) {
    mean += v[i];
  }
  mean /= v.size();
  T variance = 0;
  for (il::int_t i = 0; i < v.size(); ++i) {
    variance += (v[i] - mean) * (v[i] - mean);
  }
  const il::int_t degrees_of_freedom =
      (kind == il::VarianceKind::Population) ? v.size() : (v.size() - 1);
  variance /= degrees_of_freedom;
  return variance;
}

template <typename T>
struct MeanVariance {
  T mean;
  T variance;
};

template <typename T>
MeanVariance<T> meanVariance(const il::Array<T>& v, il::VarianceKind kind) {
  IL_EXPECT_FAST(v.size() > ((kind == il::VarianceKind::Population) ? 0 : 1));

  T mean = 0;
  for (il::int_t i = 0; i < v.size(); ++i) {
    mean += v[i];
  }
  mean /= v.size();
  T variance = 0;
  for (il::int_t i = 0; i < v.size(); ++i) {
    variance += (v[i] - mean) * (v[i] - mean);
  }
  const il::int_t degrees_of_freedom =
      (kind == il::VarianceKind::Population) ? v.size() : (v.size() - 1);
  variance /= degrees_of_freedom;

  return MeanVariance<T>{mean, variance};
}

}  // namespace il

#endif  // IL_ALGORITHM_ARRAY_H
