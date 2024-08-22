//
// Created by Richard Cernansky on 22/08/2024.
//

#ifndef EMBEDDINGTABLE_H
#define EMBEDDINGTABLE_H

#include <Eigen/Dense>

class EmbeddingTable {
public:
    // Constructor to initialize the embedding table
    EmbeddingTable(const int vocab_size, const int embedding_dim);

    // Method to get the embedding vector for a specific index
    [[nodiscard]] Eigen::VectorXd get_embedding(const int idx) const;

    // Method to get the entire embedding table
    [[nodiscard]] Eigen::MatrixXd get_et() const;

    // Method to update the embedding vector for a specific index
    void update_embedding(const int idx, const Eigen::VectorXd& new_embedding);

private:
    Eigen::MatrixXd embeddings;  // The matrix storing all embeddings
};

#endif // EMBEDDINGTABLE_H

