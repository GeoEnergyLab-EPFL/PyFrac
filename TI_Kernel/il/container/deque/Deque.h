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

#ifndef IL_QUEUE_H
#define IL_QUEUE_H

#include <il/Array.h>

namespace il {

template <typename T>
class Deque {
 private:
  il::Array<T> data_;
  // Index of the next element to be served
  il::int_t front_;
  // Index of the last element in the deque
  il::int_t back_;

 public:
  Deque();
  Deque(il::int_t n);
  il::int_t size() const;
  il::int_t capacity() const;
  void Reserve(il::int_t n);

  void PushFront(const T& x);
  template <typename... Args>
  void PushFront(il::emplace_t, Args&&... args);
  void PushBack(const T& x);
  template <typename... Args>
  void PushBack(il::emplace_t, Args&&... args);

  const T& front() const;
  T& Front();
  const T& back() const;
  T& Back();

  void PopFront();
  void PopBack();
};

template <typename T>
Deque<T>::Deque() : data_{} {
  front_ = -1;
  back_ = -1;
}

template <typename T>
Deque<T>::Deque(il::int_t n) : data_{n} {
  front_ = -1;
  back_ = -1;
}

template <typename T>
il::int_t Deque<T>::size() const {
  if (front_ == -1) {
    return 0;
  } else if (front_ <= back_) {
    return back_ - front_ + 1;
  } else {
    return back_ + 1 + data_.size() - front_;
  }
}

template <typename T>
il::int_t Deque<T>::capacity() const {
  return data_.size();
}

template <typename T>
void Deque<T>::Reserve(il::int_t n) {
  IL_ASSERT(data_.size() == 0 && front_ == -1 && back_ == -1);

  data_.Resize(n);
  front_ = -1;
  back_ = -1;
}

template <typename T>
void Deque<T>::PushFront(const T& x) {
  if (back_ == -1 && front_ == -1) {
    data_[0] = x;
    front_ = 0;
    back_ = 0;
  } else if (front_ > 0 && back_ + 1 != front_) {
    --front_;
    data_[front_] = x;
  } else if (front_ == 0 && back_ + 1 != data_.size()) {
    front_ = data_.size() - 1;
    data_[front_] = x;
  } else {
    IL_UNREACHABLE;
  }
}

template <typename T>
template <typename... Args>
void Deque<T>::PushFront(il::emplace_t, Args&&... args) {
  if (back_ == -1 && front_ == -1) {
    data_[0] = T(std::forward<Args>(args)...);
    front_ = 0;
    back_ = 0;
  } else if (front_ > 0 && back_ + 1 != front_) {
    --front_;
    data_[front_] = T(std::forward<Args>(args)...);
  } else if (front_ == 0 && back_ + 1 != data_.size()) {
    front_ = data_.size() - 1;
    data_[front_] = T(std::forward<Args>(args)...);
  } else {
    IL_UNREACHABLE;
  }
}

template <typename T>
void Deque<T>::PushBack(const T& x) {
  if (back_ == -1 && front_ == -1) {
    data_[0] = x;
    front_ = 0;
    back_ = 0;
  } else if (back_ + 1 < front_ || back_ + 1 < data_.size()) {
    ++back_;
    data_[back_] = x;
  } else if (back_ + 1 == data_.size()) {
    if (front_ > 0) {
      data_[0] = x;
      back_ = 0;
    } else {
      IL_UNREACHABLE;
    }
  } else {
    IL_UNREACHABLE;
  }
}

template <typename T>
template <typename... Args>
void Deque<T>::PushBack(il::emplace_t, Args&&... args) {
  if (back_ == -1 && front_ == -1) {
    data_[0] = T(std::forward<Args>(args)...);
    front_ = 0;
    back_ = 0;
  } else if (back_ + 1 < front_ || back_ + 1 < data_.size()) {
    ++back_;
    data_[back_] = T(std::forward<Args>(args)...);
  } else if (back_ + 1 == data_.size()) {
    if (front_ > 0) {
      data_[0] = T(std::forward<Args>(args)...);
      back_ = 0;
    } else {
      IL_UNREACHABLE;
    }
  } else {
    IL_UNREACHABLE;
  }
}

template <typename T>
const T& Deque<T>::front() const {
  IL_EXPECT_MEDIUM(front_ >= 0);

  return data_[front_];
}

template <typename T>
T& Deque<T>::Front() {
  IL_EXPECT_MEDIUM(front_ >= 0);

  return data_[front_];
}

template <typename T>
const T& Deque<T>::back() const {
  IL_EXPECT_MEDIUM(back_ >= 0);

  return data_[back_];
}

template <typename T>
T& Deque<T>::Back() {
  IL_EXPECT_MEDIUM(back_ >= 0);

  return data_[back_];
}

template <typename T>
void Deque<T>::PopFront() {
  IL_EXPECT_MEDIUM(size() > 0);

  if (size() == 1) {
    front_ = -1;
    back_ = -1;
  } else if (front_ + 1 < data_.size()) {
    ++front_;
  } else {
    front_ = 0;
  }
};

template <typename T>
void Deque<T>::PopBack() {
  IL_EXPECT_MEDIUM(size() > 0);

  if (back_ > 0) {
    --back_;
  } else {
    back_ = data_.size() - 1;
  }
};

}  // namespace il

#endif  // IL_QUEUE_H
