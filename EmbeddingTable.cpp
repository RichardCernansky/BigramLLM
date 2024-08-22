//
// Created by Richard Cernansky on 20/08/2024.
//
#include <Eigen/Dense>
#include <random>

class EmbeddingTable {
public:
    EmbeddingTable(const int vocab_size, const int embedding_dim) {
        // Random initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.01, 0.01);

        embeddings = Eigen::MatrixXd(vocab_size, embedding_dim);
        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < embedding_dim; ++j) {
                embeddings(i, j) = dis(gen);
            }
        }
    }

    [[nodiscard]] Eigen::VectorXd
    get_embedding(const int idx) const {
        return embeddings.row(idx);
    }

    [[nodiscard]] Eigen::MatrixXd
    get_et() const {
        return embeddings;
    }

    void
    update_embedding(const int idx, const Eigen::VectorXd& new_embedding) {
        embeddings.row(idx) = new_embedding;
    }

private:
    Eigen::MatrixXd embeddings;
};



