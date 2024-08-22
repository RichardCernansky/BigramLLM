//
// Created by Richard Cernansky on 22/08/2024.
//

#ifndef BIMAP_H
#define BIMAP_H

#include <unordered_map>

class BiMap {
public:
    // Insert a key-value pair into the BiMap
    void insert(const int key, const char& value);

    // Get value by key
    [[nodiscard]] int get_value(const char& key) const;

    // Get key by value
    [[nodiscard]] char get_key(const int value) const;

private:
    std::unordered_map<char, int> value_to_key;  // Maps characters to integers
    std::unordered_map<int, char> key_to_value;  // Maps integers to characters
};

#endif // BIMAP_H


