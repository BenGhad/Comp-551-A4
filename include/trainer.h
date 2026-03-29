#ifndef INC_551_A4_TRAINER_H
#define INC_551_A4_TRAINER_H

#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "dataset.h"
#include "model.h"

namespace agnews {

enum class OptimizerType {
    Adam,
    AdamW,
};

struct TrainingConfig {
    size_t max_sequence_length = 128;
    OptimizerType optimizer = OptimizerType::Adam;
    double learning_rate = 1e-3;
    size_t batch_size = 64;
    int epochs = 6;
    double max_gradient_norm = 1.0;

    [[nodiscard]] std::string optimizer_name() const;
    [[nodiscard]] std::string summary() const;
};

struct ExperimentConfig {
    ModelConfig model;
    TrainingConfig training;

    [[nodiscard]] std::string summary() const;
};

struct DatasetMetrics {
    double loss = 0.0;
    double accuracy = 0.0;
};

struct PredictionReport {
    DatasetMetrics metrics;
    std::vector<int64_t> predicted_labels;
};

struct EpochMetrics {
    int epoch = 0;
    double training_loss = 0.0;
    double validation_loss = 0.0;
    double validation_accuracy = 0.0;

    [[nodiscard]] std::string summary() const;
};

class TrainingHistory {
public:
    void add(EpochMetrics metrics);

    [[nodiscard]] bool empty() const;
    [[nodiscard]] const std::vector<EpochMetrics>& epochs() const;
    [[nodiscard]] const EpochMetrics* best_validation_epoch() const;
    [[nodiscard]] std::string summary() const;

private:
    std::vector<EpochMetrics> epochs_;
};

DatasetMetrics train_epoch(LstmClassifier& model,
                           torch::optim::Optimizer& optimizer,
                           const std::vector<EncodedExample>& train_examples,
                           const TrainingConfig& config,
                           torch::Device device,
                           std::mt19937& rng);

DatasetMetrics evaluate(LstmClassifier& model,
                        const std::vector<EncodedExample>& examples,
                        const TrainingConfig& config,
                        torch::Device device);

PredictionReport evaluate_with_predictions(LstmClassifier& model,
                                           const std::vector<EncodedExample>& examples,
                                           const TrainingConfig& config,
                                           torch::Device device);

TrainingHistory fit(LstmClassifier& model,
                    const std::vector<EncodedExample>& train_examples,
                    const std::vector<EncodedExample>& valid_examples,
                    const ExperimentConfig& config,
                    torch::Device device);
} // namespace agnews

#endif //INC_551_A4_TRAINER_H
