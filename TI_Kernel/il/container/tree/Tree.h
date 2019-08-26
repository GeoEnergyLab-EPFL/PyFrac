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

#ifndef IL_TREE_H
#define IL_TREE_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <new> is needed for placement new
#include <new>
// <utility> is needed for std::move
#include <utility>

#include <il/Array.h>
#include <il/SmallArray.h>

namespace il {

template <typename T, il::int_t n>
class Tree {
 private:
  struct Node {
   public:
    T value;
    unsigned char flag;

   public:
    Node() : Node{false} {};
    Node(bool valid)
        : value{},
          flag{valid ? static_cast<unsigned char>(1)
                     : static_cast<unsigned char>(0)} {};
    bool isValid() const { return (flag % 2) == 1; };
    bool isEmpty() const { return ((flag / 2) % 2) == 0; };
    bool hasChild() const { return flag >= 4; };
  };
  il::Array<Node> tree_;

 public:
  Tree();
  Tree(il::int_t depth);
  il::int_t depth() const;
  il::spot_t root() const;
  il::spot_t parent(il::spot_t s) const;
  il::spot_t child(il::spot_t s, il::int_t i) const;
  bool isEmpty(il::spot_t s) const;
  bool hasChild(il::spot_t s) const;
  bool hasChild(il::spot_t s, il::int_t i) const;
  il::SmallArray<il::int_t, n> children(il::spot_t s) const;
  void Set(il::spot_t s, T value);
  void AddChild(il::spot_t s, il::int_t i);
  const T& value(il::spot_t) const;
};

template <typename T, il::int_t n>
Tree<T, n>::Tree() : tree_{1, il::emplace, true} {}

template <typename T, il::int_t n>
Tree<T, n>::Tree(il::int_t depth) {
  il::int_t m = n;
  for (il::int_t k = 0; k < depth; ++k) {
    m *= n;
  }
  m = (m - 1) / (n - 1);
  tree_.Reserve(m);
  tree_.Resize(1, il::emplace, true);
};

template <typename T, il::int_t n>
il::int_t Tree<T, n>::depth() const {
  il::int_t m = tree_.size();
  il::int_t fac = 1;
  il::int_t depth = -1;
  while (m >= 0) {
    m -= fac;
    fac *= n;
    ++depth;
  }
  return depth;
}

template <typename T, il::int_t n>
il::spot_t Tree<T, n>::root() const {
  return il::spot_t{0};
};

template <typename T, il::int_t n>
il::spot_t Tree<T, n>::parent(il::spot_t s) const {
  IL_EXPECT_MEDIUM(s.index > 0);

  return il::spot_t{(s.index - 1) / n};
};

template <typename T, il::int_t n>
il::spot_t Tree<T, n>::child(il::spot_t s, il::int_t i) const {
  IL_EXPECT_MEDIUM(tree_[s.index].hasChild());

  return il::spot_t{n * s.index + 1 + i};
};

template <typename T, il::int_t n>
void Tree<T, n>::Set(il::spot_t s, T x) {
  tree_[s.index].value = x;
  if (tree_[s.index].flag % 2 == 0) {
    tree_[s.index].flag += 1;
  }
}

template <typename T, il::int_t n>
bool Tree<T, n>::isEmpty(il::spot_t s) const {
  return tree_[s.index].isEmpty();
};

template <typename T, il::int_t n>
bool Tree<T, n>::hasChild(il::spot_t s) const {
  return tree_[s.index].hasChild();
};

template <typename T, il::int_t n>
bool Tree<T, n>::hasChild(il::spot_t s, il::int_t i) const {
  unsigned char flag = tree_[s.index].flag;
  if (i == 0) {
    return (flag / 4) % 2 == 1;
  } else if (i == 1) {
    return (flag / 8) % 2 == 1;
  } else if (i == 2) {
    return (flag / 16) % 2 == 1;
  } else if (i == 3) {
    return (flag / 32) % 2 == 1;
  } else {
    IL_UNREACHABLE;
  }
  IL_UNREACHABLE;
  return true;
};

template <typename T, il::int_t n>
il::SmallArray<il::int_t, n> Tree<T, n>::children(il::spot_t s) const {
  il::int_t flag = tree_[s.index].flag / 4;
  il::SmallArray<il::int_t, n> ans{};
  il::int_t i = 0;
  while (flag != 0) {
    if (flag % 2 == 1) {
      ans.Append(i);
    }
    ++i;
    flag /= 2;
  }
  return ans;
};

template <typename T, il::int_t n>
void Tree<T, n>::AddChild(il::spot_t s, il::int_t i) {
  IL_EXPECT_MEDIUM(i >= 0 && i < n);

  unsigned char flag = 1 << (2 + i);
  tree_[s.index].flag = tree_[s.index].flag | flag;
  if (n * s.index + 1 + i >= tree_.size()) {
    tree_.Resize(n * s.index + 2 + i);
  }
};

template <typename T, il::int_t n>
const T& Tree<T, n>::value(il::spot_t s) const {
  return tree_[s.index].value;
};

}  // namespace il

#endif  // IL_TREE_H
