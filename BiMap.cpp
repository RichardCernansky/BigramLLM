//
// Created by Richard Cernansky on 20/08/2024.
//
#include "BiMap.h"

#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>



void
BiMap::insert(const char key, const int value) {
    key_to_value[key] = value;
    value_to_key[value] = key;
}

// Get value by key
[[nodiscard]] int
BiMap::get_value(const char key) const {
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


Eigen::Matrix<char, BATCH_SIZE, BLOCK_SIZE>
BiMap::decode_matrix(const Eigen::MatrixXi& int_matrix) const {
    // Ensure the input matrix dimensions match the output matrix dimensions
    assert(int_matrix.rows() == BATCH_SIZE && int_matrix.cols() == BLOCK_SIZE);

    // Create a matrix to hold the decoded characters
    Eigen::Matrix<char, BATCH_SIZE, BLOCK_SIZE> char_matrix;

    for (int i = 0; i < int_matrix.rows(); ++i) {
        for (int j = 0; j < int_matrix.cols(); ++j) {
            // Get the corresponding character for each integer in the matrix
            char_matrix(i, j) = this->get_key(int_matrix(i, j));
        }
    }
    // Manually iterate and print the matrix as characters
    for (int i = 0; i < char_matrix.rows(); ++i) {
        for (int j = 0; j < char_matrix.cols(); ++j) {
            std::cout << char_matrix(i, j) << " ";  // Print as character
        }
        std::cout << std::endl;
    }

    return char_matrix;
}
