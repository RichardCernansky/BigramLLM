#include <iostream>
#include <fstream>
#include <set>
#include <string>
#include "BiMap.cpp"
#include "config.cpp"
#include "utils.cpp"
#include "BigramLanguageModel.cpp"

BigramLanguageModel
train(const std::string_view train_data, BigramLanguageModel&& blm) {
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto [input_data, target_data] = get_batch(train_data);

        auto input_indices = map_chars_to_idxs(input_data, blm.charsHashed);
        auto target_indices = map_chars_to_idxs(target_data, blm.charsHashed);

        auto logits = blm.forward(input_indices);
        auto loss = blm.cross_entropy_loss(logits, target_indices);
        std::cout << 'Epoch: ' << epoch << 'Loss: ' << loss;

        blm.backward(input_indices, target_indices, logits);
    }

    return blm;
}

int
main() {
    //TRAINING PHASE
    //get the file character set, std::set maintains the ASCII order, prepare the data
    const auto fileString = loadFileToString(TRAIN_FILE_PATH);
    const auto charSet= getFileCharSet(TRAIN_FILE_PATH);
    const int vocab_size = charSet.size();
    const int embedding_dim =vocab_size;
    const auto train_size = fileString.size() * TRAIN_SIZE_PERCENTAGE / 100;

    BiMap charsHashed = getCharsHashed(charSet);
    std::string_view train_data(fileString.data(), train_size);
    std::string_view validation_data(fileString.data() + train_size);

    BigramLanguageModel blm(vocab_size, embedding_dim, std::move(charsHashed));
    train(train_data, std::move(blm));


    return 0;
}