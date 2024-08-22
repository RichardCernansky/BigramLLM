//
// Created by Richard Cernansky on 21/08/2024.
//
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include "BiMap.cpp"

class BigramLanguageModel {
public:
    BigramLanguageModel(int vocab_size, int embedding_dim, BiMap& char_to_idx)
        : vocab_size(vocab_size),
          embedding_dim(embedding_dim),
          char_to_idx(char_to_idx),
          idx_to_char(char_to_idx.reverseMap()), // Assuming BiMap has a reverseMap() function
          embeddings(Eigen::MatrixXd::Random(vocab_size, embedding_dim) * 0.01), // Random init
          weights(Eigen::MatrixXd::Random(embedding_dim, vocab_size) * 0.01), // Random init
          learning_rate(0.01) // You can parameterize this as well
    {}

    // Forward pass: calculate logits for the next character
    Eigen::MatrixXd forward(const Eigen::MatrixXi& input_indices) {
        // Convert input indices to embeddings
        Eigen::MatrixXd input_embeddings = embeddings(input_indices, Eigen::all);

        // Compute logits for each character in the vocab
        return input_embeddings * weights;
    }

    // Compute the cross-entropy loss
    double compute_loss(const Eigen::MatrixXd& logits, const Eigen::MatrixXi& targets) {
        Eigen::MatrixXd probs = softmax(logits);
        double loss = 0.0;

        for (int i = 0; i < probs.rows(); ++i) {
            for (int j = 0; j < probs.cols(); ++j) {
                int target_idx = targets(i, j);
                loss -= std::log(probs(i, target_idx));
            }
        }

        return loss / (probs.rows() * probs.cols());
    }

    // Backward pass: compute gradients and update embeddings and weights
    void backward(const Eigen::MatrixXi& input_indices, const Eigen::MatrixXi& targets, const Eigen::MatrixXd& logits) {
        Eigen::MatrixXd probs = softmax(logits);

        // Gradients with respect to the logits
        for (int i = 0; i < probs.rows(); ++i) {
            for (int j = 0; j < probs.cols(); ++j) {
                int target_idx = targets(i, j);
                probs(i, target_idx) -= 1.0; // Subtract 1 for the true class
            }
        }

        // Update weights and embeddings
        Eigen::MatrixXd dW = embeddings(input_indices, Eigen::all).transpose() * probs;
        weights -= learning_rate * dW;

        // Calculate and update the gradient for embeddings
        Eigen::MatrixXd dE = probs * weights.transpose();
        embeddings(input_indices, Eigen::all) -= learning_rate * dE;
    }

    // Training loop
    void train(const std::string_view train_data, int epochs, int batch_size) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto [input_matrix, target_matrix] = get_batch(train_data);
            Eigen::MatrixXi input_indices = map_chars_to_indices(input_matrix);
            Eigen::MatrixXi target_indices = map_chars_to_indices(target_matrix);

            Eigen::MatrixXd logits = forward(input_indices);
            double loss = compute_loss(logits, target_indices);
            std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;

            backward(input_indices, target_indices, logits);
        }
    }

    // Generate text using the model
    std::string generate_text(const std::string& seed, int length) {
        std::string generated_text = seed;

        for (int i = 0; i < length; ++i) {
            int char_idx = char_to_idx.at(generated_text.back());
            Eigen::VectorXd input_embedding = embeddings.row(char_idx);
            Eigen::MatrixXd logits = (input_embedding.transpose() * weights).transpose();
            Eigen::VectorXd probs = softmax(logits.row(0)).transpose();
            char next_char = sample_next_char(probs);
            generated_text += next_char;
        }

        return generated_text;
    }

private:
    int vocab_size;
    int embedding_dim;
    double learning_rate;
    BiMap& char_to_idx;
    BiMap& idx_to_char;
    Eigen::MatrixXd embeddings; // Embedding matrix
    Eigen::MatrixXd weights;    // Weight matrix for bigram prediction

    // Softmax function
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits) const {
        Eigen::MatrixXd exp_logits = logits.array().exp();
        Eigen::VectorXd sum_exp = exp_logits.rowwise().sum();
        return exp_logits.array().colwise() / sum_exp.array();
    }

    // Function to sample the next character based on probabilities
    char sample_next_char(const Eigen::VectorXd& probs) const {
        std::discrete_distribution<int> dist(probs.data(), probs.data() + probs.size());
        std::random_device rd;
        std::mt19937 gen(rd());
        return idx_to_char.at(dist(gen));
    }

    // Helper function to convert character matrices to index matrices
    Eigen::MatrixXi map_chars_to_indices(const Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic>& char_matrix) {
        Eigen::MatrixXi indices(char_matrix.rows(), char_matrix.cols());

        for (int i = 0; i < char_matrix.rows(); ++i) {
            for (int j = 0; j < char_matrix.cols(); ++j) {
                indices(i, j) = char_to_idx.at(char_matrix(i, j));
            }
        }

        return indices;
    }
};

//--------------------------


void backward(const Eigen::MatrixXi& input_indices, const Eigen::MatrixXi& targets, const Eigen::Tensor<double, 3>& logits) {
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

    // Initialize matrices to accumulate gradients with respect to the weights and embeddings
    Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(embedding_dim, vocab_size);  // Gradient for weights
    Eigen::Tensor<double, 3> dE = Eigen::Tensor<double, 3>(batch_size, block_size, embedding_dim);  // Gradient for embeddings
    dE.setZero();

    // Compute gradients
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            int target_idx = targets(i, j);

            // Compute grad_logits (probs - target)
            Eigen::VectorXd grad_logits = Eigen::VectorXd::Zero(vocab_size);
            for (int v = 0; v < vocab_size; ++v) {
                grad_logits(v) = probs(i, j, v) - (v == target_idx ? 1.0 : 0.0);
            }

            // Retrieve the embedding for the current (i, j) position
            Eigen::VectorXd embedding(embedding_dim);
            for (int k = 0; k < embedding_dim; ++k) {
                embedding(k) = logits(i, j, k);
            }

            // Update dW for weights: dW += embedding * grad_logits.transpose()
            dW += embedding * grad_logits.transpose();

            // Update dE for embeddings
            dE.chip(j, 1).chip(i, 0) += grad_logits.transpose() * weights.transpose();
        }
    }

    // Update weights using gradient descent
    weights -= learning_rate * dW;

    // Update embeddings
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            int char_index = input_indices(i, j);
            Eigen::VectorXd embedding_update(embedding_dim);
            for (int k = 0; k < embedding_dim; ++k) {
                embedding_update(k) = dE(i, j, k);
            }
            embedding_table.update_embedding(char_index, embedding_table.get_embedding(char_index) - learning_rate * embedding_update);
        }
    }
}
