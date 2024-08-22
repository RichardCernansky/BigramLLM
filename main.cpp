#include <iostream>
#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include "BiMap.cpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "config.cpp"
#include "EmbeddingTable.cpp"

int
generate_random_int(const int a, const int b) {
    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(a, b);

    return dis(gen);
}

std::pair<Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic>>
get_batch(const std::string_view data) {

    // Create matrices to hold the batches
    Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> input_matrix(BLOCK_SIZE, BATCH_SIZE);
    Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> target_matrix(BLOCK_SIZE, BATCH_SIZE);

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
            input_matrix(j, i) = data[start_index + j];
            target_matrix(j, i) = data[start_index + j + 1]; // Predict the next character
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
getCharsHashed(const std::set<char>& charSet) {
    BiMap bimap;
    int idx = 0;
    for (char iter : charSet) {
       bimap.insert(idx, iter);
       ++idx;
    }

    return bimap;
}

std::set<char>
getFileCharSet(const std::string& file_path) {
    //open file
    std::ifstream fileHandle(file_path);
    if (!fileHandle) {
        throw std::runtime_error("Failed to open the file: " + file_path);
    }

    std::set<char> charSet;
    char ch;
    // Insert each character into the set
    while (fileHandle.get(ch)) {
        charSet.insert(ch);
    }
    fileHandle.close();

    return charSet;
}

int
main() {
    //TRAINING PHASE
    //get the file character set, std::set maintains the ASCII order, prepare the data
    const auto fileString = loadFileToString(TRAIN_FILE_PATH);
    const auto charSet= getFileCharSet(TRAIN_FILE_PATH);
    const int vocab_size = charSet.size();
    const auto train_size = fileString.size() * TRAIN_SIZE_PERCENTAGE / 100;

    BiMap charsHashed = getCharsHashed(charSet);
    std::string_view train_data(fileString.data(), train_size);
    std::string_view validation_data(fileString.data() + train_size);

    std::cout << train_size;

    return 0;
}