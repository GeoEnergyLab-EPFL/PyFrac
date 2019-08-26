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

#include <gtest/gtest.h>

#include <il/container/info/Status.h>

//// You are not allowed to asked if it is ok until the status has been set
// TEST(Status, default_constructor) {
//  il::Status status{};
//
//  bool success = false;
//  try {
//    status.Ok();
//  } catch (...) {
//    success = true;
//  }
//
//  ASSERT_TRUE(success);
//}
//
// TEST(Status, check_error) {
//  il::Status status{};
//  status.SetOk();
//
//  ASSERT_TRUE(status.Ok());
//}
//
//// You don't have to check is the status has not been set
// TEST(Status, destructor) {
//  {
//    il::Status  status{};
//  }
//  const bool success = true;
//
//  ASSERT_TRUE(success);
//}
//
//// We can't check if the status has to be verified because we cannot throw an
//// exception in the destructor
