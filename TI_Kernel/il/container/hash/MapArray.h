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

#ifndef IL_MAPARRAY_H
#define IL_MAPARRAY_H

#include <il/Array.h>
#include <il/Map.h>

namespace il {

template <typename K, typename V, typename F = HashFunction<K>>
class MapArray {
 private:
  il::Array<il::KeyValue<K, V>> array_;
  il::Map<K, il::int_t, F> map_;
#ifdef IL_DEBUG_CLASS
  std::size_t hash_;
#endif

 public:
  MapArray();
  MapArray(il::int_t n);
  MapArray(il::value_t, std::initializer_list<il::KeyValue<K, V>> list);
  void Set(const K& key, const V& value);
  void Set(const K& key, V&& value);
  void Set(K&& key, const V& value);
  void Set(K&& key, V&& value);
  void Set(const K& key, const V& value, il::io_t, il::spot_t& i);
  void Set(const K& key, V&& value, il::io_t, il::spot_t& i);
  void Set(K&& key, const V& value, il::io_t, il::spot_t& i);
  void Set(K&& key, V&& value, il::io_t, il::spot_t& i);
  il::int_t size() const;
  il::int_t capacity() const;
  il::spot_t search(const K& key) const;
  template <il::int_t m>
  il::spot_t searchCString(const char (&key)[m]) const;
  bool found(il::spot_t i) const;
  const K& key(il::spot_t i) const;
  const V& value(il::spot_t i) const;
  V& Value(il::spot_t i);
  il::spot_t next(il::spot_t i) const;
  il::spot_t spotBegin() const;
  il::spot_t spotEnd() const;
  il::int_t nbElements() const;
};

template <typename K, typename V, typename F>
MapArray<K, V, F>::MapArray() : array_{}, map_{} {
#ifdef IL_DEBUG_CLASS
  hash_ = 0;
#endif
}

template <typename K, typename V, typename F>
MapArray<K, V, F>::MapArray(il::int_t n) : array_{}, map_{n} {
  array_.Reserve(n);
#ifdef IL_DEBUG_CLASS
  hash_ = 0;
#endif
}

template <typename K, typename V, typename F>
MapArray<K, V, F>::MapArray(il::value_t,
                            std::initializer_list<il::KeyValue<K, V>> list) {
  const il::int_t n = static_cast<il::int_t>(list.size());
  if (n > 0) {
    array_.Reserve(n);
    map_.Reserve(n);
    for (il::int_t i = 0; i < n; ++i) {
      array_.Append(il::emplace, (list.begin() + i)->key,
                    (list.begin() + i)->value);
      map_.Set((list.begin() + i)->key, i);
    }
  }
#ifdef IL_DEBUG_CLASS
  hash_ = static_cast<std::size_t>(n);
#endif
};

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(const K& key, const V& value) {
  const il::int_t i = array_.size();
  array_.Append(il::emplace, key, value);
  map_.Set(key, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(const K& key, V&& value) {
  const il::int_t i = array_.size();
  array_.Append(il::emplace, key, std::move(value));
  map_.Set(key, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(K&& key, V&& value) {
  const il::int_t i = array_.size();
  array_.Append(il::emplace, key, std::move(value));
  map_.Set(std::move(key), i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(const K& key, const V& value, il::io_t,
                            il::spot_t& i) {
  const il::int_t j = array_.size();
  array_.Append(il::emplace, key, value);
  map_.Set(key, j, il::io, i);

  i.index = j;
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
  i.signature = hash_;
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(const K& key, V&& value, il::io_t, il::spot_t& i) {
  const il::int_t j = array_.size();
  array_.Append(il::emplace, key, std::move(value));
  map_.Set(key, j, il::io, i);
  i.index = j;
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
  i.signature = hash_;
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(K&& key, V&& value, il::io_t, il::spot_t& i) {
  const il::int_t j = array_.size();
  array_.Append(il::emplace, key, std::move(value));
  map_.Set(std::move(key), j, il::io, i);
  i.index = j;
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
  i.signature = hash_;
#endif
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::size() const {
  return array_.size();
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::capacity() const {
  return array_.capacity();
}

template <typename K, typename V, typename F>
il::spot_t MapArray<K, V, F>::search(const K& key) const {
  const il::spot_t i = map_.search(key);
  il::spot_t s;
  s.index = map_.found(i) ? map_.value(i) : i.index;
#ifdef IL_DEBUG_CLASS
  s.signature = map_.hash();
#endif
  return s;
}

template <typename K, typename V, typename F>
template <il::int_t m>
il::spot_t MapArray<K, V, F>::searchCString(const char (&key)[m]) const {
  const il::spot_t i = map_.search(key);
  il::spot_t s;
  s.index = map_.found(i) ? map_.value(i) : i.index;
#ifdef IL_DEBUG_CLASS
  s.signature = map_.hash();
#endif
  return s;
};

template <typename K, typename V, typename F>
bool MapArray<K, V, F>::found(il::spot_t i) const {
  return i.index >= 0;
}

template <typename K, typename V, typename F>
const K& MapArray<K, V, F>::key(il::spot_t i) const {
  return array_[i.index].key;
}

template <typename K, typename V, typename F>
const V& MapArray<K, V, F>::value(il::spot_t i) const {
  return array_[i.index].value;
}

template <typename K, typename V, typename F>
V& MapArray<K, V, F>::Value(il::spot_t i) {
  return array_[i.index].value;
}

template <typename K, typename V, typename F>
il::spot_t MapArray<K, V, F>::next(il::spot_t i) const {
  il::spot_t s;
  s.index = i.index + 1;
#ifdef IL_DEBUG_CLASS
  s.signature = map_.hash();
#endif
  return s;
}

template <typename K, typename V, typename F>
il::spot_t MapArray<K, V, F>::spotBegin() const {
  il::spot_t s;
  s.index = 0;
#ifdef IL_DEBUG_CLASS
  s.signature = hash_;
#endif
  return s;
}

template <typename K, typename V, typename F>
il::spot_t MapArray<K, V, F>::spotEnd() const {
  il::spot_t s;
  s.index = array_.size();
#ifdef IL_DEBUG_CLASS
  s.signature = map_.hash();
#endif
  return s;
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::nbElements() const {
  return array_.size();
};

}  // namespace il

#endif  // IL_MAPARRAY_H
