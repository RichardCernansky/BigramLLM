//
// Created by Richard Cernansky on 22/08/2024.
//
#include "utils.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include "BiMap.h"
#include "config.h"
#include <iostream>
#include <fstream>
#include <set>

Eigen::MatrixXi
map_chars_to_idxs(Eigen::Matrix<char, BATCH_SIZE, BLOCK_SIZE> input_data, const BiMap& charsHashed) {
    Eigen::MatrixXi chars_to_idxs(BATCH_SIZE, BLOCK_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            chars_to_idxs(i, j) = charsHashed.get_value(input_data(i,j));
        }
    }

    return chars_to_idxs;
}

int
generate_random_int(const int a, const int b) {
    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(a, b);

    return dis(gen);
}

std::pair<Eigen::Matrix<char, BATCH_SIZE, BLOCK_SIZE>, Eigen::Matrix<char, BATCH_SIZE, BLOCK_SIZE>>
get_batch(const std::string_view data) {

    // Create matrices to hold the batches
    Eigen::Matrix<char,BATCH_SIZE, BLOCK_SIZE> input_matrix;
    Eigen::Matrix<char, BATCH_SIZE, BLOCK_SIZE> target_matrix;

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - BLOCK_SIZE - 1);

    // Generate batches
    for (std::size_t i = 0; i < BATCH_SIZE; ++i) {
        // Get a random start index
        std::size_t start_index = dis(gen);

        // Copy the data from the string to the matrices
        for (std::size_t j = 0; j < BLOCK_SIZE; ++j) {
            input_matrix(i, j) = data[start_index + j];
            target_matrix(i, j) = data[start_index + j + 1]; // Predict the next character
        }
    }

    return std::make_pair(input_matrix, target_matrix);
}

std::string
loadFileToString(const std::string& file_path) {
    std::ifstream fileHandle(file_path);
    if (!fileHandle) {
        throw std::runtime_error("Failed to open the file: " + file_path);
    }

    // Read the entire file into a string
    std::string fileContents((std::istreambuf_iterator<char>(fileHandle)), std::istreambuf_iterator<char>());

    return fileContents;
}

BiMap
getCharsHashed(const std::set<char> &charSet) {
    BiMap bimap;
    int idx = 0;
    for (char ch: charSet) {
       bimap.insert(ch, idx);
       ++idx;
    }

    return bimap;
}

std::set<char>
getFileCharSet(const std::string &file_path) {
    //open file
    std::ifstream fileHandle(file_path);
    if (!fileHandle) {
        throw std::runtime_error("Failed to open the file: " + file_path);
    }

    std::set<char> charSet;
    char ch;
    // Insert each character into the set
    while (fileHandle.get(ch)) {
        charSet.insert(static_cast<unsigned char>(ch));
    }
    fileHandle.close();

    return charSet;
}