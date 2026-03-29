#include "dataset.h"
#include "model.h"
#include "tokenizer.h"
#include "trainer.h"
#include "vocab.h"

#include <iomanip>
#include <iostream>

int main() {
    torch::manual_seed(42);

    auto data = agnews::load_ag_news("../data/raw/train.csv", "../data/raw/test.csv", 0.9, 42);

    agnews::Tokenizer tokenizer;
    std::vector<std::vector<std::string>> train_tokens;
    train_tokens.reserve(data.train.size());

    for (const auto& ex : data.train) {
        train_tokens.push_back(tokenizer.tokenize(ex.text));
    }

    agnews::Vocabulary vocab;
    vocab.build(train_tokens, 3);

    agnews::ExperimentConfig experiment;
    experiment.model.vocab_size = vocab.size();

    const auto encoded_train = agnews::encode_examples(
        data.train,
        tokenizer,
        vocab,
        experiment.training.max_sequence_length);
    const auto encoded_valid = agnews::encode_examples(
        data.valid,
        tokenizer,
        vocab,
        experiment.training.max_sequence_length);
    const auto encoded_test = agnews::encode_examples(
        data.test,
        tokenizer,
        vocab,
        experiment.training.max_sequence_length);

    const torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    agnews::LstmClassifier model(experiment.model);
    model->to(device);

    std::cout << "Vocab size: " << vocab.size() << std::endl;
    std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
    std::cout << experiment.summary() << std::endl;
    std::cout << agnews::LstmClassifierSpec(experiment.model).pooling_explanation() << std::endl;

    const auto history = agnews::fit(
        model,
        encoded_train,
        encoded_valid,
        experiment,
        device);

    if (const auto* best = history.best_validation_epoch(); best != nullptr) {
        std::cout << "Best validation epoch: " << best->epoch
                  << " (val_accuracy=" << best->validation_accuracy
                  << ", val_loss=" << best->validation_loss << ")" << std::endl;
        std::cout << "Restored best validation checkpoint for test evaluation." << std::endl;
    }

    const auto test_report = agnews::evaluate_with_predictions(
        model,
        encoded_test,
        experiment.training,
        device);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Test accuracy: " << test_report.metrics.accuracy
              << " (" << (test_report.metrics.accuracy * 100.0) << "%)" << std::endl;

    constexpr size_t max_misclassified_to_show = 8;
    size_t shown = 0;
    std::cout << "Misclassified test examples:" << std::endl;
    for (size_t i = 0; i < data.test.size() && shown < max_misclassified_to_show; ++i) {
        if (test_report.predicted_labels[i] == data.test[i].label) {
            continue;
        }

        ++shown;
        std::cout << shown << "." << std::endl;
        std::cout << "Input text: " << data.test[i].text << std::endl;
        std::cout << "True label: " << agnews::label_to_string(data.test[i].label) << std::endl;
        std::cout << "Predicted label: "
                  << agnews::label_to_string(static_cast<int>(test_report.predicted_labels[i]))
                  << std::endl;
        std::cout << std::endl;
    }

    if (shown == 0) {
        std::cout << "No misclassified test examples found." << std::endl;
    }

    return 0;
}
