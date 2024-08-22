//
// Created by Richard Cernansky on 20/08/2024.
//
#include "BiMap.h"
#include <unordered_map>


void
BiMap::insert(const int key, const char& value) {
    key_to_value[key] = value;
    value_to_key[value] = key;
}

// Get value by key
[[nodiscard]] int
BiMap::get_value(const char& key) const {
    auto it = key_to_value.find(key);
    if (it != key_to_value.end()) {
        return it->second;
    }
    throw std::runtime_error("Key not found");
}

// Get key by value
[[nodiscard]] char
BiMap::get_key(const int value) const {
    auto it = value_to_key.find(value);
    if (it != value_to_key.end()) {
        return it->second;
    }
    throw std::runtime_error("Value not found");
}
