cmake_minimum_required(VERSION 3.28)
project(BigramLLM)

set(CMAKE_CXX_STANDARD 17)

# Include Eigen
include_directories("~/Library/eigen")

add_executable(BigramLLM main.cpp
        main.cpp
        BiMap.cpp
        EmbeddingTable.cpp
        BigramLanguageModel.cpp
        utils.cpp
        utils.h
        BigramLanguageModel.h
        EmbeddingTable.h
        BiMap.h
        config.h)
