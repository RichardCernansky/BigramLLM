//
// Created by Richard Cernansky on 22/08/2024.
//

#ifndef BIGRAMLANGUAGEMODEL_H
#define BIGRAMLANGUAGEMODEL_H

#include <unsupported/Eigen/CXX11/Tensor>
#include "EmbeddingTable.h"
#include "BiMap.h"  // Assuming BiMap is defined in BiMap.h

class BigramLanguageModel {
public:
    // Constructor
    BigramLanguageModel(const int vocab_size, const int embedding_dim, const BiMap&& charsHashed);

    // Forward pass
    [[nodiscard]] Eigen::Tensor<double, 3> forward(const Eigen::MatrixXi& input_indices) const;

    // Cross-entropy loss calculation
    [[nodiscard]] double cross_entropy_loss(const Eigen::Tensor<double, 3>& logits, const Eigen::MatrixXi& target_indices) const;

    // Backward pass for gradient computation
    void backward(const Eigen::MatrixXi& input_indices, const Eigen::MatrixXi& target_indices, const Eigen::Tensor<double, 3>& logits);

    BiMap charsHashed;

private:
    EmbeddingTable embedding_table;
    int vocab_size;
    int embedding_dim;
    Eigen::MatrixXd weights;    // Weight matrix for bigram prediction

    // Softmax function
    [[nodiscard]] static Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits);

    // Get probabilities from logits
    [[nodiscard]] static Eigen::Tensor<double, 3> get_probs(const Eigen::Tensor<double, 3>& logits);
};

#endif // BIGRAMLANGUAGEMODEL_H

