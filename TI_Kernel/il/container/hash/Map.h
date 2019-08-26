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

#ifndef IL_MAP_H
#define IL_MAP_H

#include <il/Array.h>
#include <il/Status.h>
#include <il/container/hash/HashFunction.h>
#include <il/math.h>

namespace il {

template <typename K, typename V>
struct KeyValue {
  K key;
  V value;
  KeyValue(const K& the_key, const V& the_value)
      : key{the_key}, value{the_value} {};
  KeyValue(const K& the_key, V&& the_value)
      : key{the_key}, value{std::move(the_value)} {};
  KeyValue(K&& the_key, const V& the_value)
      : key{std::move(the_key)}, value{the_value} {};
  KeyValue(K&& the_key, V&& the_value)
      : key{std::move(the_key)}, value{std::move(the_value)} {};
};

// The number of buckets is: m = 2^p
// We want the number of elements: a <= (3 / 4) m which is 4 a <= 3 m
// We want the number of tombstones: b <= (1 / 8) m which is 8 b <= m
// All those numbers must fit in std::size_t
// As a consequence, if n is the number of bits of std::size_t, p <= n - 2
// Therefore, a should be <= (kMaxSize_ == 3 * 2^(n - 4))

template <typename K, typename V, typename F = HashFunction<K>>
class Map {
 private:
  KeyValue<K, V>* bucket_;
  il::int_t nb_elements_;
  il::int_t nb_tombstones_;
  int p_;
  static constexpr il::int_t kMaxSize_ =
      3 * static_cast<il::int_t>(static_cast<std::size_t>(1)
                                 << (8 * sizeof(std::size_t) - 4));
#ifdef IL_DEBUGGER_HELPERS
  il::int_t size_;
#endif
#ifdef IL_DEBUG_CLASS
  std::size_t hash_;
#endif

 public:
  Map();
  Map(il::int_t n);
  Map(il::value_t, std::initializer_list<il::KeyValue<K, V>> list);
  Map(const Map<K, V, F>& map);
  Map(Map<K, V, F>&& map);
  Map& operator=(const Map<K, V, F>& map);
  Map& operator=(Map<K, V, F>&& map);
  ~Map();

  // All the insertions
  void Set(const K& key, const V& value);
  void Set(const K& key, V&& value);
  void Set(K&& key, const V& value);
  void Set(K&& key, V&& value);
  template <il::int_t m>
  void SetCString(const char (&key)[m], const V& value);
  template <il::int_t m>
  void SetCString(const char (&key)[m], V&& value);

  // Searching for a key
  il::spot_t search(const K& key) const;
  template <il::int_t m>
  il::spot_t searchCString(const char (&key)[m]) const;
  il::spot_t searchCString(const char* key, il::int_t n) const;
  bool found(il::spot_t i) const;

  // Inserting a new (key, value)
  void Set(const K& key, const V& value, il::io_t, il::spot_t& i);
  void Set(const K& key, V&& value, il::io_t, il::spot_t& i);
  void Set(K&& key, const V& value, il::io_t, il::spot_t& i);
  void Set(K&& key, V&& value, il::io_t, il::spot_t& i);
  void SetCString(const char* key, const il::int_t n, const V& value, il::io_t,
                  il::spot_t& i);
  template <il::int_t m>
  void SetCString(const char (&key)[m], const V& value, il::io_t,
                  il::spot_t& i);

  // Getting the key and values for a given slot
  const K& key(il::spot_t i) const;
  const V& value(il::spot_t i) const;
  V& Value(il::spot_t i);
  const V& valueForKey(const K& key) const;
  const V& valueForKey(const K& key, const V& default_value) const;
  template <il::int_t m>
  const V& valueForCString(const char (&key)[m]) const;
  template <il::int_t m>
  const V& valueForCString(const char (&key)[m], const V& default_value) const;

  void erase(il::spot_t i);

  // Changing the size
  void Clear();
  bool isEmpty() const;
  il::int_t nbElements() const;
  il::int_t nbTombstones() const;
  il::int_t nbBuckets() const;
  void Reserve(il::int_t r);
  void Rehash();

  // Looping over the map
  il::spot_t spotBegin() const;
  il::spot_t spotEnd() const;
  il::spot_t next(il::spot_t i) const;

  // To remove: Ugly hack for MapArray
#ifdef IL_DEBUG_CLASS
  std::size_t hash() const;
#endif

 private:
  static int pForSlots(il::int_t n);
  static il::int_t nbBuckets(int p);
  double load() const;
  double displaced() const;
  double displacedTwice() const;
  void ReserveWithP(int p);
};

template <typename K, typename V, typename F>
int Map<K, V, F>::pForSlots(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(n <= kMaxSize_);

  switch (n) {
    case 0:
      return -1;
    case 1:
      return 1;
    default: {
      int ans = 2;
      il::int_t val = 3;
      while (n > val) {
        ans += 1;
        val *= 2;
      }
      return ans;
    }
  }
}

template <typename K, typename V, typename F>
Map<K, V, F>::Map() {
  bucket_ = nullptr;
  nb_elements_ = 0;
  nb_tombstones_ = 0;
  p_ = -1;
#ifdef IL_DEBUGGER_HELPERS
  size_ = 0;
#endif
#ifdef IL_DEBUG_CLASS
  hash_ = 0;
#endif
}

template <typename K, typename V, typename F>
Map<K, V, F>::Map(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(n <= kMaxSize_);

  if (n > kMaxSize_) {
    il::abort();
  } else if (n == 0) {
    bucket_ = nullptr;
    p_ = -1;
#ifdef IL_DEBUGGER_HELPERS
    size_ = 0;
#endif
  } else {
    const int p = pForSlots(n);
    const il::int_t m = nbBuckets(p);
    bucket_ = il::allocateArray<KeyValue<K, V>>(m);
    for (il::int_t i = 0; i < m; ++i) {
      F::constructEmpty(il::io, reinterpret_cast<K*>(bucket_ + i));
    }
    p_ = p;
#ifdef IL_DEBUGGER_HELPERS
    size_ = il::ipow(2, p_);
#endif
  }
  nb_elements_ = 0;
  nb_tombstones_ = 0;
#ifdef IL_DEBUG_CLASS
  hash_ = 0;
#endif
}

template <typename K, typename V, typename F>
Map<K, V, F>::Map(il::value_t, std::initializer_list<il::KeyValue<K, V>> list) {
  if (list.size() > static_cast<std::size_t>(kMaxSize_)) {
    il::abort();
  }
  const il::int_t n = static_cast<il::int_t>(list.size());

  if (n == 0) {
    bucket_ = nullptr;
    p_ = -1;
#ifdef IL_DEBUGGER_HELPERS
    size_ = 0;
#endif
#ifdef IL_DEBUG_CLASS
    hash_ = 0;
#endif
  } else {
    const int p = pForSlots(n);
    const il::int_t m = nbBuckets(p);
    bucket_ = il::allocateArray<KeyValue<K, V>>(m);
    for (il::int_t i = 0; i < m; ++i) {
      F::constructEmpty(il::io, reinterpret_cast<K*>(bucket_ + i));
    }
    p_ = p;
#ifdef IL_DEBUGGER_HELPERS
    size_ = il::ipow(2, p_);
#endif
    nb_elements_ = 0;
    nb_tombstones_ = 0;
#ifdef IL_DEBUG_CLASS
    hash_ = 0;
#endif
    for (il::int_t k = 0; k < n; ++k) {
      il::spot_t i = search((list.begin() + k)->key);
      IL_EXPECT_FAST(!found(i));
      Set((list.begin() + k)->key, (list.begin() + k)->value, il::io, i);
#ifdef IL_DEBUG_CLASS
      hash_ += F::hash((list.begin() + k)->key, p_);
#endif
    }
  }
}

template <typename K, typename V, typename F>
Map<K, V, F>::Map(const Map<K, V, F>& map) {
  p_ = map.p_;
#ifdef IL_DEBUGGER_HELPERS
  size_ = map.size_;
#endif
  nb_elements_ = 0;
  nb_tombstones_ = 0;
#ifdef IL_DEBUG_CLASS
  hash_ = 0;
#endif
  if (p_ >= 0) {
    const il::int_t m = nbBuckets(p_);
    bucket_ = il::allocateArray<KeyValue<K, V>>(m);
    for (il::int_t i = 0; i < m; ++i) {
      F::constructEmpty(il::io, reinterpret_cast<K*>(bucket_ + i));
    }
    for (il::spot_t i = map.spotBegin(); i != map.spotEnd(); i = map.next(i)) {
      Set(map.key(i), map.value(i));
#ifdef IL_DEBUG_CLASS
      hash_ += F::hash(map.key(i), p_);
#endif
    }
  }
}

template <typename K, typename V, typename F>
Map<K, V, F>::Map(Map<K, V, F>&& map) {
  bucket_ = map.bucket_;
  p_ = map.p_;
#ifdef IL_DEBUGGER_HELPERS
  size_ = map.size_;
#endif
#ifdef IL_DEBUG_CLASS
  hash_ = map.hash_;
#endif
  nb_elements_ = map.nb_elements_;
  nb_tombstones_ = map.nb_tombstones_;
  map.bucket_ = nullptr;
  map.p_ = -1;
#ifdef IL_DEBUGGER_HELPERS
  map.size_ = 0;
#endif
#ifdef IL_DEBUG_CLASS
  map.hash_ = 0;
#endif
  map.nb_elements_ = 0;
  map.nb_tombstones_ = 0;
}

template <typename K, typename V, typename F>
Map<K, V, F>& Map<K, V, F>::operator=(const Map<K, V, F>& map) {
  const int p = map.p_;
  const int old_p = p_;
  if (p >= 0) {
    if (old_p >= 0) {
      const il::int_t old_m = nbBuckets(old_p);
      for (il::int_t i = 0; i < old_m; ++i) {
        if (!F::isEmpty(bucket_[i].key) && !F::isTombstone(bucket_[i].key)) {
          (&((bucket_ + i)->value))->~V();
          (&((bucket_ + i)->key))->~K();
        }
      }
    }
    il::deallocate(bucket_);
    const il::int_t m = nbBuckets(p);
    bucket_ = il::allocateArray<KeyValue<K, V>>(m);
    for (il::int_t i = 0; i < m; ++i) {
      if (F::isEmpty(map.bucket_[i].key)) {
        F::constructEmpty(il::io, reinterpret_cast<K*>(bucket_ + i));
      } else if (F::isTombstone(map.bucket_[i].key)) {
        F::constructTombstone(il::io, reinterpret_cast<K*>(bucket_ + i));
      } else {
        new (const_cast<K*>(&((bucket_ + i)->key))) K(map.bucket_[i].key);
        new (&((bucket_ + i)->value)) V(map.bucket_[i].value);
      }
    }
    p_ = p;
#ifdef IL_DEBUGGER_HELPERS
    size_ = il::ipow(2, p_);
#endif
  } else {
    bucket_ = nullptr;
    p_ = -1;
#ifdef IL_DEBUGGER_HELPERS
    size_ = 0;
#endif
  }
  nb_elements_ = map.nb_elements_;
  nb_tombstones_ = map.nb_tombstones_;
#ifdef IL_DEBUG_CLASS
  hash_ = map.hash_;
#endif
  return *this;
}

template <typename K, typename V, typename F>
Map<K, V, F>& Map<K, V, F>::operator=(Map<K, V, F>&& map) {
  if (this != &map) {
    const int old_p = p_;
    if (old_p >= 0) {
      const il::int_t old_m = nbBuckets(old_p);
      for (il::int_t i = 0; i < old_m; ++i) {
        if (!F::is_empy(bucket_ + i) && !F::isTombstone(bucket_ + i)) {
          (&((bucket_ + i)->value))->~V();
          (&((bucket_ + i)->key))->~K();
        }
      }
      il::deallocate(bucket_);
    }
    bucket_ = map.bucket_;
    p_ = map.p_;
#ifdef IL_DEBUGGER_HELPERS
    size_ = map.size_;
#endif
#ifdef IL_DEBUG_CLASS
    hash_ = map.hash_;
#endif
    nb_elements_ = map.nb_elements_;
    nb_tombstones_ = map.nb_tombstones_;
    map.bucket_ = nullptr;
    map.p_ = -1;
    map.nb_elements_ = 0;
    map.nb_tombstones_ = 0;
#ifdef IL_DEBUG_CLASS
    map.hash_ = 0;
#endif
  }
  return *this;
}

template <typename K, typename V, typename F>
Map<K, V, F>::~Map() {
  if (p_ >= 0) {
    const il::int_t m = nbBuckets();
    for (il::int_t i = 0; i < m; ++i) {
      if (!F::isEmpty(bucket_[i].key) && !F::isTombstone(bucket_[i].key)) {
        (&((bucket_ + i)->value))->~V();
        (&((bucket_ + i)->key))->~K();
      }
    }
    il::deallocate(bucket_);
  }
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Set(const K& key, const V& value) {
  il::spot_t i = search(key);
  if (!found(i)) {
    Set(key, value, il::io, i);
  } else {
    (bucket_ + i.index)->value.~V();
    new (&((bucket_ + i.index)->value)) V(value);
  }
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash(this->key(i), p_);
#endif
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Set(const K& key, V&& value) {
  il::spot_t i = search(key);
  if (!found(i)) {
    Set(key, std::move(value), il::io, i);
  } else {
    (bucket_ + i.index)->value.~V();
    new (&((bucket_ + i.index)->value)) V(std::move(value));
  }
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash(this->key(i), p_);
#endif
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Set(K&& key, const V& value) {
  il::spot_t i = search(key);
  if (!found(i)) {
    Set(std::move(key), value, il::io, i);
  } else {
    (bucket_ + i.index)->value.~V();
    new (&((bucket_ + i.index)->value)) V(std::move(value));
  }
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash(this->key(i), p_);
#endif
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Set(K&& key, V&& value) {
  il::spot_t i = search(key);
  if (!found(i)) {
    Set(std::move(key), std::move(value), il::io, i);
  } else {
    (bucket_ + i.index)->value.~V();
    new (&((bucket_ + i.index)->value)) V(std::move(value));
  }
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash(this->key(i), p_);
#endif
}

template <typename K, typename V, typename F>
template <il::int_t m>
void Map<K, V, F>::SetCString(const char (&key)[m], const V& value) {
  il::spot_t i = searchCString(key);
  if (!found(i)) {
    Set(key, value, il::io, i);
  } else {
    (bucket_ + i.index)->value.~V();
    new (&((bucket_ + i.index)->value)) V(value);
  }
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash(this->key(i), p_);
#endif
}

template <typename K, typename V, typename F>
template <il::int_t m>
void Map<K, V, F>::SetCString(const char (&key)[m], V&& value) {
  il::spot_t i = searchCString(key);
  if (!found(i)) {
    Set(key, value, il::io, i);
  } else {
    (bucket_ + i.index)->value.~V();
    new (&((bucket_ + i.index)->value)) V(std::move(value));
  }
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash(this->key(i), p_);
#endif
}

template <typename K, typename V, typename F>
il::spot_t Map<K, V, F>::search(const K& key) const {
  IL_EXPECT_MEDIUM(!F::isEmpty(key));
  IL_EXPECT_MEDIUM(!F::isTombstone(key));

  if (p_ == -1) {
    il::spot_t s{-1};
#ifdef IL_DEBUG_CLASS
    s.signature = hash_;
#endif
    return s;
  }

  const std::size_t mask = (static_cast<std::size_t>(1) << p_) - 1;
  std::size_t i = F::hash(key, p_);
  std::size_t i_tombstone = -1;
  std::size_t delta_i = 1;
  while (true) {
    if (F::isEmpty(bucket_[i].key)) {
      il::spot_t s{(i_tombstone == static_cast<std::size_t>(-1))
                       ? -(1 + static_cast<il::int_t>(i))
                       : -(1 + static_cast<il::int_t>(i_tombstone))};
#ifdef IL_DEBUG_CLASS
      s.signature = hash_;
#endif
      return s;
    } else if (F::isTombstone(bucket_[i].key)) {
      i_tombstone = i;
    } else if (F::isEqual(bucket_[i].key, key)) {
      il::spot_t s{static_cast<il::int_t>(i)};
#ifdef IL_DEBUG_CLASS
      s.signature = hash_;
#endif
      return s;
    }
    i += delta_i;
    i &= mask;
    ++delta_i;
  }
}

template <typename K, typename V, typename F>
template <il::int_t m>
il::spot_t Map<K, V, F>::searchCString(const char (&key)[m]) const {
  IL_EXPECT_MEDIUM(!F::isEmpty(key));
  IL_EXPECT_MEDIUM(!F::isTombstone(key));

  if (p_ == -1) {
    il::spot_t s{-1};
#ifdef IL_DEBUG_CLASS
    s.signature = hash_;
#endif
    return s;
  }

  const std::size_t mask = (static_cast<std::size_t>(1) << p_) - 1;
  std::size_t i = F::hash(key, p_);
  std::size_t i_tombstone = -1;
  std::size_t delta_i = 1;
  while (true) {
    if (F::isEmpty(bucket_[i].key)) {
      il::spot_t s{(i_tombstone == static_cast<std::size_t>(-1))
                       ? -(1 + static_cast<il::int_t>(i))
                       : -(1 + static_cast<il::int_t>(i_tombstone))};
#ifdef IL_DEBUG_CLASS
      s.signature = hash_;
#endif
      return s;
    } else if (F::isTombstone(bucket_[i].key)) {
      i_tombstone = i;
    } else if (F::isEqual(bucket_[i].key, key)) {
      il::spot_t s{static_cast<il::int_t>(i)};
#ifdef IL_DEBUG_CLASS
      s.signature = hash_;
#endif
      return s;
    }
    i += delta_i;
    i &= mask;
    ++delta_i;
  }
}

template <typename K, typename V, typename F>
il::spot_t Map<K, V, F>::searchCString(const char* key, il::int_t n) const {
  if (p_ == -1) {
    il::spot_t s{-1};
#ifdef IL_DEBUG_CLASS
    s.signature = hash_;
#endif
    return s;
  }

  const std::size_t mask = (static_cast<std::size_t>(1) << p_) - 1;
  std::size_t i = F::hash(key, n) & mask;
  std::size_t i_tombstone = -1;
  std::size_t delta_i = 1;
  while (true) {
    if (F::isEmpty(bucket_[i].key)) {
      il::spot_t s{(i_tombstone == static_cast<std::size_t>(-1))
                       ? -(1 + static_cast<il::int_t>(i))
                       : -(1 + static_cast<il::int_t>(i_tombstone))};
#ifdef IL_DEBUG_CLASS
      s.signature = hash_;
#endif
      return s;
    } else if (F::isTombstone(bucket_[i].key)) {
      i_tombstone = i;
    } else if (F::isEqual(bucket_[i].key, key, n)) {
      il::spot_t s{static_cast<il::int_t>(i)};
#ifdef IL_DEBUG_CLASS
      s.signature = hash_;
#endif
      return s;
    }
    i += delta_i;
    i &= mask;
    ++delta_i;
  }
}

template <typename K, typename V, typename F>
bool Map<K, V, F>::found(il::spot_t i) const {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_MEDIUM(i.signature == hash_);
#endif
  return i.index >= 0;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::SetCString(const char* key, const il::int_t n,
                              const V& value, il::io_t, il::spot_t& i) {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_FAST(i.signature == hash_);
#endif
  IL_EXPECT_FAST(!found(i));

  il::int_t i_local = -(1 + i.index);
  const il::int_t nb_buckets = nbBuckets();
  if (4 * (static_cast<std::size_t>(nb_elements_) + 1) >
      3 * static_cast<std::size_t>(nb_buckets)) {
    if (p_ + 1 <= static_cast<int>(sizeof(std::size_t) * 8 - 2)) {
      ReserveWithP(p_ == -1 ? 1 : p_ + 1);
    } else {
      il::abort();
    }
    il::spot_t j = searchCString(key, n);
    i_local = -(1 + j.index);
  }
  new (const_cast<K*>(&((bucket_ + i_local)->key))) K(key, n);
  new (&((bucket_ + i_local)->value)) V(value);
  ++nb_elements_;

  i.index = i_local;
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash((bucket_ + i_local)->key, p_);
  i.signature = hash_;
#endif
}

template <typename K, typename V, typename F>
template <il::int_t m>
void Map<K, V, F>::SetCString(const char (&key)[m], const V& value, il::io_t,
                              il::spot_t& i) {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_FAST(i.signature == hash_);
#endif
  IL_EXPECT_FAST(!found(i));

  il::int_t i_local = -(1 + i.index);
  const il::int_t nb_buckets = nbBuckets();
  if (4 * (static_cast<std::size_t>(nb_elements_) + 1) >
      3 * static_cast<std::size_t>(nb_buckets)) {
    if (p_ + 1 <= static_cast<int>(sizeof(std::size_t) * 8 - 2)) {
      ReserveWithP(p_ == -1 ? 1 : p_ + 1);
    } else {
      il::abort();
    }
    il::spot_t j = searchCString(key);
    i_local = -(1 + j.index);
  }
  new (const_cast<K*>(&((bucket_ + i_local)->key))) K(key);
  new (&((bucket_ + i_local)->value)) V(value);
  ++nb_elements_;
  i.index = i_local;
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash((bucket_ + i_local)->key, p_);
  i.signature = hash_;
#endif
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Set(const K& key, const V& value, il::io_t, il::spot_t& i) {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_FAST(i.signature == hash_);
#endif
  IL_EXPECT_FAST(!found(i));

  il::int_t i_local = -(1 + i.index);
  const il::int_t m = nbBuckets();
  if (4 * (static_cast<std::size_t>(nb_elements_) + 1) >
      3 * static_cast<std::size_t>(m)) {
    if (p_ + 1 <= static_cast<int>(sizeof(std::size_t) * 8 - 2)) {
      ReserveWithP(p_ == -1 ? 1 : p_ + 1);
    } else {
      il::abort();
    }
    il::spot_t j = search(key);
    i_local = -(1 + j.index);
  }
  new (const_cast<K*>(&((bucket_ + i_local)->key))) K(key);
  new (&((bucket_ + i_local)->value)) V(value);
  ++nb_elements_;
  i.index = i_local;
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash((bucket_ + i_local)->key, p_);
  i.signature = hash_;
#endif
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Set(const K& key, V&& value, il::io_t, il::spot_t& i) {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_FAST(i.signature == hash_);
#endif
  IL_EXPECT_FAST(!found(i));

  il::int_t i_local = -(1 + i.index);
  const il::int_t m = nbBuckets();
  if (4 * (static_cast<std::size_t>(nb_elements_) + 1) >
      3 * static_cast<std::size_t>(m)) {
    if (p_ + 1 <= static_cast<int>(sizeof(std::size_t) * 8 - 2)) {
      ReserveWithP(p_ == -1 ? 1 : p_ + 1);
    } else {
      il::abort();
    }
    il::spot_t j = search(key);
    i_local = -(1 + j.index);
  }
  new (const_cast<K*>(&((bucket_ + i_local)->key))) K(key);
  new (&((bucket_ + i_local)->value)) V(value);
  ++nb_elements_;
  i.index = i_local;
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash((bucket_ + i_local)->key, p_);
  i.signature = hash_;
#endif
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Set(K&& key, const V& value, il::io_t, il::spot_t& i) {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_FAST(i.signature == hash_);
#endif
  IL_EXPECT_FAST(!found(i));

  il::int_t i_local = -(1 + i.index);
  const il::int_t m = nbBuckets();
  if (4 * (static_cast<std::size_t>(nb_elements_) + 1) >
      3 * static_cast<std::size_t>(m)) {
    if (p_ + 1 <= static_cast<int>(sizeof(std::size_t) * 8 - 2)) {
      ReserveWithP(p_ == -1 ? 1 : p_ + 1);
    } else {
      il::abort();
    }
    il::spot_t j = search(key);
    i_local = -(1 + j.index);
  }
  new (const_cast<K*>(&((bucket_ + i_local)->key))) K(std::move(key));
  new (&((bucket_ + i_local)->value)) V(value);
  ++nb_elements_;
  i.index = i_local;
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash((bucket_ + i_local)->key, p_);
  i.signature = hash_;
#endif
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Set(K&& key, V&& value, il::io_t, il::spot_t& i) {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_FAST(i.signature == hash_);
#endif
  IL_EXPECT_FAST(!found(i));

  il::int_t i_local = -(1 + i.index);
  const il::int_t m = nbBuckets();
  if (4 * (static_cast<std::size_t>(nb_elements_) + 1) >
      3 * static_cast<std::size_t>(m)) {
    if (p_ + 1 <= static_cast<int>(sizeof(std::size_t) * 8 - 2)) {
      ReserveWithP(p_ == -1 ? 1 : p_ + 1);
    } else {
      il::abort();
    }
    il::spot_t j = search(key);
    i_local = -(1 + j.index);
  }
  new (const_cast<K*>(&((bucket_ + i_local)->key))) K(std::move(key));
  new (&((bucket_ + i_local)->value)) V(std::move(value));
  ++nb_elements_;
  i.index = i_local;
#ifdef IL_DEBUG_CLASS
  hash_ += F::hash((bucket_ + i_local)->key, p_);
  i.signature = hash_;
#endif
}

template <typename K, typename V, typename F>
void Map<K, V, F>::erase(il::spot_t i) {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_FAST(i.signature == hash_);
#endif
  IL_EXPECT_FAST(found(i));

  (&((bucket_ + i.index)->key))->~K();
#ifdef IL_DEBUG_CLASS
  hash_ -= F::hash((bucket_ + i.index)->key, p_);
#endif
  F::constructTombstone(il::io, reinterpret_cast<K*>(bucket_ + i.index));
  (&((bucket_ + i.index)->value))->~V();
  --nb_elements_;
  ++nb_tombstones_;
  const il::int_t m = nbBuckets();
  if (8 * (static_cast<std::size_t>(nb_tombstones_) + 1) >
      static_cast<std::size_t>(m)) {
    Rehash();
  }
  return;
}

template <typename K, typename V, typename F>
const K& Map<K, V, F>::key(il::spot_t i) const {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_MEDIUM(i.signature == hash_);
#endif
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i.index) <
                   static_cast<std::size_t>((p_ >= 0) ? (1 << p_) : 0));

  return bucket_[i.index].key;
}

template <typename K, typename V, typename F>
const V& Map<K, V, F>::value(il::spot_t i) const {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_MEDIUM(i.signature == hash_);
#endif
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i.index) <
                   static_cast<std::size_t>((p_ >= 0) ? (1 << p_) : 0));

  return bucket_[i.index].value;
}

template <typename K, typename V, typename F>
V& Map<K, V, F>::Value(il::spot_t i) {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_MEDIUM(i.signature == hash_);
#endif
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i.index) <
                   static_cast<std::size_t>((p_ >= 0) ? (1 << p_) : 0));

  return bucket_[i.index].value;
}

template <typename K, typename V, typename F>
const V& Map<K, V, F>::valueForKey(const K& key) const {
  const il::spot_t i = search(key);
  if (found(i)) {
    return value(i);
  } else {
    return V{};
  }
}

template <typename K, typename V, typename F>
const V& Map<K, V, F>::valueForKey(const K& key, const V& default_value) const {
  const il::spot_t i = search(key);
  if (found(i)) {
    return value(i);
  } else {
    return default_value;
  }
}

template <typename K, typename V, typename F>
template <il::int_t m>
const V& Map<K, V, F>::valueForCString(const char (&key)[m]) const {
  const il::spot_t i = search(key);
  if (found(i)) {
    return value(i);
  } else {
    return V{};
  }
}

template <typename K, typename V, typename F>
template <il::int_t m>
const V& Map<K, V, F>::valueForCString(const char (&key)[m],
                                       const V& default_value) const {
  const il::spot_t i = search(key);
  if (found(i)) {
    return value(i);
  } else {
    return default_value;
  }
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::nbElements() const {
  return nb_elements_;
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::nbTombstones() const {
  return nb_tombstones_;
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::nbBuckets() const {
  return (p_ >= 0) ? static_cast<il::int_t>(static_cast<std::size_t>(1) << p_)
                   : 0;
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::nbBuckets(int p) {
  return (p >= 0) ? static_cast<il::int_t>(static_cast<std::size_t>(1) << p)
                  : 0;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Reserve(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  ReserveWithP(pForSlots(n));
}

template <typename K, typename V, typename F>
double Map<K, V, F>::load() const {
  IL_EXPECT_MEDIUM(p_ >= 0);

  return static_cast<double>(nb_elements_) / nbBuckets();
}

template <typename K, typename V, typename F>
double Map<K, V, F>::displaced() const {
  const il::int_t m = nbBuckets();
  il::int_t nb_displaced = 0;
  for (il::int_t i = 0; i < m; ++i) {
    if (!F::isEmpty(bucket_[i].key) && !F::isTombstone(bucket_[i].key)) {
      const il::int_t hashed =
          static_cast<il::int_t>(F::hash(bucket_[i].key, p_));
      if (i != hashed) {
        ++nb_displaced;
      }
    }
  }

  return static_cast<double>(nb_displaced) / nb_elements_;
}

template <typename K, typename V, typename F>
double Map<K, V, F>::displacedTwice() const {
  const il::int_t m = nbBuckets();
  const il::int_t mask = (1 << p_) - 1;
  il::int_t nb_displaced_twice = 0;
  for (il::int_t i = 0; i < m; ++i) {
    if (!F::isEmpty(bucket_[i].key) && !F::isTombstone(bucket_[i].key)) {
      const il::int_t hashed =
          static_cast<il::int_t>(F::hash(bucket_[i].key, p_));
      if (i != hashed && (((i - 1) - hashed) & mask) != 0) {
        ++nb_displaced_twice;
      }
    }
  }

  return static_cast<double>(nb_displaced_twice) / nb_elements_;
}

template <typename K, typename V, typename F>
bool Map<K, V, F>::isEmpty() const {
  return nb_elements_ == 0;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Clear() {
  if (p_ >= 0) {
    const il::int_t m = nbBuckets();
    for (il::int_t i = 0; i < m; ++i) {
      if (!F::isEmpty(bucket_[i].key) && !F::isTombstone(bucket_[i].key)) {
        (&((bucket_ + i)->value))->~V();
        (&((bucket_ + i)->key))->~K();
      }
      F::constructEmpty(il::io, reinterpret_cast<K*>(bucket_ + i));
    }
  }
  nb_elements_ = 0;
  nb_tombstones_ = 0;
}

template <typename K, typename V, typename F>
il::spot_t Map<K, V, F>::spotBegin() const {
  const il::int_t m = il::int_t{1} << p_;

  il::int_t i = 0;
  while (i < m &&
         (F::isEmpty(bucket_[i].key) || F::isTombstone(bucket_[i].key))) {
    ++i;
  }
  il::spot_t s;
  s.index = i;
#ifdef IL_DEBUG_CLASS
  s.signature = hash_;
#endif
  return s;
}

template <typename K, typename V, typename F>
il::spot_t Map<K, V, F>::spotEnd() const {
  il::spot_t s;
  s.index = il::int_t{1} << p_;
#ifdef IL_DEBUG_CLASS
  s.signature = hash_;
#endif
  return s;
}

template <typename K, typename V, typename F>
il::spot_t Map<K, V, F>::next(il::spot_t i) const {
#ifdef IL_DEBUG_CLASS
  IL_EXPECT_MEDIUM(i.signature == hash_);
#endif
  const il::int_t m = il::int_t{1} << p_;

  il::int_t i_local = i.index;
  ++i_local;
  while (i_local < m && (F::isEmpty(bucket_[i_local].key) ||
                         F::isTombstone(bucket_[i_local].key))) {
    ++i_local;
  }
  il::spot_t s{i_local};
#ifdef IL_DEBUG_CLASS
  s.signature = hash_;
#endif
  return s;
}

#ifdef IL_DEBUG_CLASS
template <typename K, typename V, typename F>
std::size_t Map<K, V, F>::hash() const {
  return hash_;
};
#endif

template <typename K, typename V, typename F>
void Map<K, V, F>::ReserveWithP(int p) {
  KeyValue<K, V>* old_bucket_ = bucket_;
  const il::int_t old_m = nbBuckets(p_);
  const il::int_t m = nbBuckets(p);

  bucket_ = il::allocateArray<KeyValue<K, V>>(m);
  for (il::int_t i = 0; i < m; ++i) {
    F::constructEmpty(il::io, reinterpret_cast<K*>(bucket_ + i));
  }
  p_ = p;
#ifdef IL_DEBUGGER_HELPERS
  size_ = il::ipow(2, p_);
#endif
  nb_elements_ = 0;
  nb_tombstones_ = 0;

  if (p_ >= 0) {
    for (il::int_t i = 0; i < old_m; ++i) {
      if (!F::isEmpty(old_bucket_[i].key) &&
          !F::isTombstone(old_bucket_[i].key)) {
        il::spot_t new_i = search(old_bucket_[i].key);
        Set(std::move(old_bucket_[i].key), std::move(old_bucket_[i].value),
            il::io, new_i);
        (&((old_bucket_ + i)->key))->~K();
        (&((old_bucket_ + i)->value))->~V();
      }
    }
    il::deallocate(old_bucket_);
  }
}

template <typename K, typename V, typename F>
void Map<K, V, F>::Rehash() {
  if (nb_tombstones_ > 0) {
    const il::int_t m = nbBuckets();
    KeyValue<K, V>* bucket = il::allocateArray<KeyValue<K, V>>(m);
    for (il::int_t i = 0; i < m; ++i) {
      F::constructEmpty(il::io, reinterpret_cast<K*>(bucket + i));
    }
    for (il::int_t i = 0; i < m; ++i) {
      if (!(F::isEmpty(bucket_[i].key) || F::isTombstone(bucket_[i].key))) {
        new (&((bucket + i)->key)) K(std::move(bucket_[i].key));
        new (&((bucket + i)->value)) V(std::move(bucket_[i].value));
        ((bucket_ + i)->value).~V();
        ((bucket_ + i)->key).~K();
      }
    }
    il::deallocate(bucket_);
    bucket_ = bucket;
    nb_tombstones_ = 0;
  }
#ifdef IL_DEBUG_CLASS
  hash_ = 2 * hash_;
#endif
}

}  // namespace il

#endif  // IL_MAP_H
