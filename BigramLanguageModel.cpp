//
// Created by Richard Cernansky on 20/08/2024.
//
#include "BigramLanguageModel.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "EmbeddingTable.h"
#include "config.h"
#include "BiMap.h"
#include "utils.h"


//__init__
BigramLanguageModel::BigramLanguageModel(const int vocab_size, const int embedding_dim, const BiMap&& charsHashed) :
    vocab_size(vocab_size),
    embedding_dim(embedding_dim),
    charsHashed(charsHashed),
    embedding_table(vocab_size, embedding_dim),
    weights(embedding_dim, vocab_size)
{
    for (int i = 0; i < embedding_dim; ++i) {
        for (int j = 0; j < vocab_size; ++j) {
            weights(i, j) = generate_random_real(-0.01, 0.01);
        }
    }
}

[[nodiscard]] Eigen::Tensor<double, 3>
BigramLanguageModel::forward(const Eigen::MatrixXi& input_indices) const {
    int batch_size = input_indices.rows();
    int block_size = input_indices.cols();

    // Initialize a 3D tensor to hold the embeddings: [batch_size, block_size, embedding_dim]
    Eigen::Tensor<double, 3> input_embeddings(batch_size, block_size, embedding_dim);

    // Fill the tensor with embeddings based on input indices
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            int char_index = input_indices(i, j);
            Eigen::VectorXd embedding = embedding_table.get_embedding(char_index);
            for (int k = 0; k < embedding_dim; ++k) {
                input_embeddings(i, j, k) = embedding(k);
            }
        }
    }

    // Initialize a tensor to hold the logits for each character in each sequence
    Eigen::Tensor<double, 3> logits(batch_size, block_size, vocab_size);

    // Compute logits for every character in each sequence
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            // Get the embedding for the current character in the sequence
            Eigen::VectorXd embedding(embedding_dim);
            for (int k = 0; k < embedding_dim; ++k) {
                embedding(k) = input_embeddings(i, j, k);
            }

            //division of dependency across the multiple wieghts in the nural network - each embedding ha influence on each vocab token
            // Compute logits: [1 x embedding_dim] * [embedding_dim x vocab_size] = [1 x vocab_size]
            Eigen::VectorXd logit_row = embedding.transpose() * weights; //so it fits as emb1 -> emb1 in both matrices

            // Store the computed logits for this (i, j) position
            for (int v = 0; v < vocab_size; ++v) {
                logits(i, j, v) = logit_row(v);
            }
        }
    }

    return logits;
}

[[nodiscard]] double
BigramLanguageModel::cross_entropy_loss(const Eigen::Tensor<double, 3>& logits, const Eigen::MatrixXi& target_indices) const {
    int batch_size = logits.dimension(0);
    int block_size = logits.dimension(1);
    int vocab_size = logits.dimension(2);

    // Initialize a tensor to hold the softmax probabilities
    auto probs = get_probs(logits);

    // Compute the cross-entropy loss
    double loss = 0.0;
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            int target_idx = target_indices(i, j);
            loss -= std::log(probs(i, j, target_idx)); //get only the value only for the correct one
            //-> -sum[y*P(i)] -> inner is 1 only for the correct one and others IGNORE
        }
    }

    // Return the average loss over all samples and positions
    return loss / (batch_size * block_size);
}

void
BigramLanguageModel::backward(const Eigen::MatrixXi& input_indices, const Eigen::MatrixXi& target_indices, const Eigen::Tensor<double, 3>& logits) {
    int batch_size = logits.dimension(0);
    int block_size = logits.dimension(1);
    int vocab_size = logits.dimension(2);

    // Initialize a tensor to hold the softmax probabilities
    Eigen::Tensor<double, 3> probs= get_probs(logits);

    // Initialize matrices to accumulate gradients with respect to the weights and embeddings
    Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(embedding_dim, vocab_size);  // Gradient for weights, set to zero
    Eigen::MatrixXd dE = Eigen::MatrixXd::Zero(vocab_size, embedding_dim);  // Gradient for embeddings

    //Gradient computation
    for(int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            int target_idx = target_indices(i, j);
            //Compute grad_logits (probs - target)
            Eigen::VectorXd grad_logits = Eigen::VectorXd::Zero(vocab_size);
            for (int v = 0; v < vocab_size; ++v) {
                grad_logits(v) = probs(i,j,v) - (v == target_idx ? 1.0 : 0.0);
            }

            //retrieve the embedding for current i,j position
            Eigen::VectorXd orig_embedding = embedding_table.get_embedding(input_indices(i,j));
            //update dW for weights:
            dW += orig_embedding * grad_logits.transpose();

            // Update dE for embeddings: accumulate the gradient for the embedding of the word at input_indices(i, j)
            dE.row(input_indices(i,j)) += grad_logits.transpose() * weights.transpose();
        }
    }

    //Update weights and embeddings
    weights -= LEARNING_RATE * dW;
    for (int i = 0; i < vocab_size; ++i) {
        embedding_table.update_embedding(0, dE.row(0));
    }
}


[[nodiscard]] Eigen::Tensor<double, 3>
BigramLanguageModel::get_probs(const Eigen::Tensor<double, 3>& logits) {
    int batch_size = logits.dimension(0);
    int block_size = logits.dimension(1);
    int vocab_size = logits.dimension(2);

    // Initialize a tensor to hold the softmax probabilities
    Eigen::Tensor<double, 3> probs(batch_size, block_size, vocab_size);

    // Apply softmax to logits to get probabilities
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            // Extract logits vector
            Eigen::VectorXd logits_vector(vocab_size);
            for (int v = 0; v < vocab_size; ++v) {
                logits_vector(v) = logits(i, j, v);
            }

            // Compute softmax probabilities
            Eigen::VectorXd probs_vector = softmax(logits_vector);

            // Store the softmax probabilities
            for (int v = 0; v < vocab_size; ++v) {
                probs(i, j, v) = probs_vector(v);
            }
        }
    }
    return probs;
}

// Softmax function
[[nodiscard]] Eigen::MatrixXd
BigramLanguageModel::softmax(const Eigen::MatrixXd& logits) {
    Eigen::MatrixXd exp_logits = logits.array().exp();
    Eigen::VectorXd sum_exp = exp_logits.rowwise().sum();
    return exp_logits.array().colwise() / sum_exp.array();
}