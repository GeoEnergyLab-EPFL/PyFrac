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

#ifndef IL_DUMMY_TEST_H
#define IL_DUMMY_TEST_H

#include <il/Array.h>

class Dummy {
 public:
  static il::int_t current;
  static il::Array<il::int_t> destroyed;

 private:
  il::int_t id_;

 public:
  static void reset() {
    current = 0;
    destroyed.Resize(0);
  }

 public:
  Dummy() {
    id_ = current;
    ++current;
  }
  Dummy(const Dummy& other) { id_ = other.id_; }
  Dummy(Dummy&& other) {
    id_ = other.id_;
    other.id_ = -(1 + other.id_);
  }
  Dummy& operator=(const Dummy& other) {
    id_ = other.id_;
    return *this;
  }
  Dummy& operator=(Dummy&& other) {
    id_ = other.id_;
    other.id_ = -(1001 + other.id_);
    return *this;
  }
  ~Dummy() { destroyed.Append(id_); }
  il::int_t id() { return id_; };
};

#endif  // IL_DUMMY_TEST_H
