//
// Created by Richard Cernansky on 20/08/2024.
//
const int BLOCK_SIZE = 8;
const int BATCH_SIZE = 4;
const int MAX_ITERS = 10000;
const int TRAIN_SIZE_PERCENTAGE = 80;
const double LEARNING_RATE = 3 * pow(M_E, -4);
const std::string TRAIN_FILE_PATH = "resources/wiz_of_oz.txt";  // Specify the file path
