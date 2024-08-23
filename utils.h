//
// Created by Richard Cernansky on 22/08/2024.
//

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <fstream>
#include <set>
#include <random>
#include "BiMap.h"
#include "config.h"

// Function to map characters to indices
Eigen::MatrixXi map_chars_to_idxs(Eigen::Matrix<char, BATCH_SIZE, BLOCK_SIZE> input_data, const BiMap& charsHashed);

// Function to generate a random integer between a and b
int generate_random_int(const int a, const int b);

// Function to get a batch of input and target matrices from the data
std::pair<Eigen::Matrix<char, BATCH_SIZE, BLOCK_SIZE>, Eigen::Matrix<char, BATCH_SIZE, BLOCK_SIZE>>
get_batch(const std::string_view data);

// Function to load a file into a string
std::string loadFileToString(const std::string& file_path);

// Function to create a BiMap from a set of characters
BiMap getCharsHashed(const std::set<char> &charSet);

// Function to get the set of characters from a file
std::set<char> getFileCharSet(const std::string &file_path);

double generate_random_real(double a, double b);

#endif // UTILS_H

