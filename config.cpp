//
// Created by Richard Cernansky on 20/08/2024.
//
#include <math.h>

constexpr int BLOCK_SIZE = 8;
constexpr int BATCH_SIZE = 4;
constexpr int MAX_ITERS = 10000;
constexpr int TRAIN_SIZE_PERCENTAGE = 80;
constexpr int EPOCHS = 100;

const double LEARNING_RATE = 3 * pow(M_E, -4);

const std::string TRAIN_FILE_PATH = "resources/wiz_of_oz.txt";  // Specify the file path
