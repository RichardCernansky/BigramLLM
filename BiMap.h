//
// Created by Richard Cernansky on 22/08/2024.
//

#ifndef BIMAP_H
#define BIMAP_H

#include <unordered_map>

class BiMap {
public:
    // Insert a key-value pair into the BiMap
    void insert(const char key, const  int value);

    // Get value by key
    [[nodiscard]] int get_value(const  char key) const;

    // Get key by value
    [[nodiscard]] char get_key(const int value) const;

    std::unordered_map<char, int> key_to_value;  // Maps integers to characters
private:
    std::unordered_map<int, char> value_to_key;  // Maps characters to integers
};

#endif // BIMAP_H


