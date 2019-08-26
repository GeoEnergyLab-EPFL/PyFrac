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

#ifndef IL_TIME_H
#define IL_TIME_H

#include <string>

#include <il/core.h>

namespace il {

const double nanosecond{1.0e-9};
const double microsecond{1.0e-6};
const double millisecond{1.0e-3};
const double second = 1.0;
const double minute = 60.0;
const double hour{60.0 * minute};
const double day{24.0 * hour};
const double year{365.0 * day};

inline std::string time_toString(double time) {
  IL_EXPECT_FAST(time >= 0);

  if (time == 0.0) {
    return std::string{"0 second"};
  } else if (time < il::nanosecond) {
    return std::to_string(time) + std::string{" seconds"};
  } else if (time < il::microsecond) {
    return std::to_string(time / il::nanosecond) + std::string{" nanoseconds"};
  } else if (time < il::millisecond) {
    return std::to_string(time / il::microsecond) +
           std::string{" microseconds"};
  } else if (time < il::second) {
    return std::to_string(time / il::millisecond) +
           std::string{" milliseconds"};
  } else if (time < il::minute) {
    return std::to_string(time) + std::string{" seconds"};
  } else if (time < il::hour) {
    return std::to_string(time / il::minute) + std::string{" minutes"};
  } else if (time < il::day) {
    return std::to_string(time / il::hour) + std::string{" hours"};
  } else if (time < il::year) {
    return std::to_string(time / il::day) + std::string{" days"};
  } else {
    return std::to_string(time / il::year) + std::string{" years"};
  }
}
}  // namespace il

#endif  // IL_TIME_H
