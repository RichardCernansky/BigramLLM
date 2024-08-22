//
// Created by Richard Cernansky on 22/08/2024.
//

#ifndef CONFIG_H
#define CONFIG_H

#include <cmath>   // For pow and M_E
#include <string>  // For std::string

// Model configuration parameters
constexpr int BLOCK_SIZE = 8;
constexpr int BATCH_SIZE = 4;
constexpr int MAX_ITERS = 10000;
constexpr int TRAIN_SIZE_PERCENTAGE = 80;
constexpr int EPOCHS = 100;

// Learning rate
//constexpr double LEARNING_RATE = 3 * std::pow(M_E, -4); == (next line)
constexpr double LEARNING_RATE = 0.055431;

// File path for training data
const std::string TRAIN_FILE_PATH = "resources/wiz_of_oz.txt";

#endif // CONFIG_H
