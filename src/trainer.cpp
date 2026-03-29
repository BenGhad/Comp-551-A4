#include "trainer.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <torch/nn/functional/loss.h>
#include <torch/nn/utils/clip_grad.h>

namespace agnews {
    namespace {
        struct TensorBatch {
            torch::Tensor input_ids;
            torch::Tensor lengths;
            torch::Tensor labels;
            int64_t size = 0;
        };


        TensorBatch build_tensor_batch(const std::vector<EncodedExample> &dataset,
                                       const std::vector<size_t> &order,
                                       const size_t start,
                                       const size_t batch_size,
                                       const size_t max_sequence_length,
                                       torch::Device device) {
            const size_t end = std::min(order.size(), start + batch_size);
            std::vector batch_indices(order.begin() + start, order.begin() + end);

            std::ranges::sort(batch_indices,
                              [&](size_t a, size_t b) {
                                  return dataset[a].token_ids.size() > dataset[b].token_ids.size();
                              });

            size_t local_max_len = 0;
            for (const size_t idx : batch_indices) {
                local_max_len = std::max(local_max_len,
                                         std::min(dataset[idx].token_ids.size(), max_sequence_length));
            }

            const auto opts = torch::TensorOptions().dtype(torch::kInt64);
            const auto input_ids = torch::full({(int64_t)batch_indices.size(), (int64_t)local_max_len},
                                         Vocabulary::PAD_ID, opts);
            const auto lengths = torch::empty({(int64_t)batch_indices.size()}, opts);
            const auto labels = torch::empty({(int64_t)batch_indices.size()}, opts);

            auto* input_ptr = input_ids.data_ptr<int64_t>();
            auto* lengths_ptr = lengths.data_ptr<int64_t>();
            auto* labels_ptr = labels.data_ptr<int64_t>();

            for (size_t row = 0; row < batch_indices.size(); ++row) {
                const auto&[label, token_ids] = dataset[batch_indices[row]];
                const size_t len = std::min(token_ids.size(), max_sequence_length);

                lengths_ptr[row] = static_cast<int64_t>(len);
                labels_ptr[row] = static_cast<int64_t>(label);

                std::copy_n(token_ids.begin(), len, input_ptr + row * local_max_len);
            }

            return {
                input_ids.to(device),
                lengths.to(device),
                labels.to(device),
                static_cast<int64_t>(batch_indices.size())
            };
        }

        std::unique_ptr<torch::optim::Optimizer> make_optimizer(LstmClassifier &model,
                                                                const TrainingConfig &config) {
            switch (config.optimizer) {
                case OptimizerType::Adam:
                    return std::make_unique<torch::optim::Adam>(
                        model->parameters(),
                        torch::optim::AdamOptions(config.learning_rate));
                case OptimizerType::AdamW:
                    return std::make_unique<torch::optim::AdamW>(
                        model->parameters(),
                        torch::optim::AdamWOptions(config.learning_rate));
            }

            throw std::runtime_error("Unsupported optimizer.");
        }

        DatasetMetrics run_epoch(LstmClassifier &model,
                                 torch::optim::Optimizer *optimizer,
                                 const std::vector<EncodedExample> &examples,
                                 const TrainingConfig &config,
                                 const torch::Device device,
                                 std::vector<size_t> order) {
            if (examples.empty()) {
                throw std::runtime_error("Cannot run training or evaluation on an empty dataset.");
            }

            const bool training = optimizer != nullptr;
            if (training) {
                model->train();
            } else {
                model->eval();
            }

            std::unique_ptr<torch::NoGradGuard> maybe_no_grad;
            if (!training) {
                maybe_no_grad = std::make_unique<torch::NoGradGuard>();
            }

            double total_loss = 0.0;
            int64_t total_correct = 0;
            int64_t total_examples = 0;

            for (size_t start = 0; start < order.size(); start += config.batch_size) {
                auto batch = build_tensor_batch(examples, order, start, config.batch_size, config.max_sequence_length, device);


                if (training) {
                    optimizer->zero_grad();
                }

                const auto logits = model->forward(batch.input_ids, batch.lengths);
                const auto loss = torch::nn::functional::cross_entropy(logits, batch.labels);

                if (training) {
                    loss.backward();
                    torch::nn::utils::clip_grad_norm_(model->parameters(), config.max_gradient_norm);
                    optimizer->step();
                }

                const auto predictions = logits.argmax(1);
                total_correct += predictions.eq(batch.labels).sum().item<int64_t>();
                total_loss += loss.item<double>() * static_cast<double>(batch.size);
                total_examples += batch.size;
            }

            DatasetMetrics metrics;
            metrics.loss = total_loss / static_cast<double>(total_examples);
            metrics.accuracy = static_cast<double>(total_correct) / static_cast<double>(total_examples);
            return metrics;
        }
    } // namespace

    std::string TrainingConfig::optimizer_name() const {
        switch (optimizer) {
            case OptimizerType::Adam:
                return "Adam";
            case OptimizerType::AdamW:
                return "AdamW";
        }

        throw std::runtime_error("Unsupported optimizer.");
    }


    std::string TrainingConfig::summary() const {
        std::ostringstream out;
        out << "Training(max_seq_len=" << max_sequence_length
                << ", optimizer=" << optimizer_name()
                << ", learning_rate=" << learning_rate
                << ", batch_size=" << batch_size
                << ", epochs=" << epochs
                << ", gradient_clip=" << max_gradient_norm << ")";
        return out.str();
    }


    std::string ExperimentConfig::summary() const {
        std::ostringstream out;
        out << LstmClassifierSpec(model).summary() << "\n"
                << training.summary();
        return out.str();
    }

    std::string EpochMetrics::summary() const {
        std::ostringstream out;
        out << "Epoch " << epoch
                << ": train_loss=" << training_loss
                << ", val_loss=" << validation_loss
                << ", val_accuracy=" << validation_accuracy;
        return out.str();
    }

    void TrainingHistory::add(EpochMetrics metrics) {
        epochs_.push_back(std::move(metrics));
    }

    bool TrainingHistory::empty() const {
        return epochs_.empty();
    }

    const std::vector<EpochMetrics> &TrainingHistory::epochs() const {
        return epochs_;
    }

    const EpochMetrics *TrainingHistory::best_validation_epoch() const {
        if (epochs_.empty()) {
            return nullptr;
        }

        const EpochMetrics *best = &epochs_.front();
        for (const auto &metrics: epochs_) {
            if (metrics.validation_accuracy > best->validation_accuracy) {
                best = &metrics;
                continue;
            }

            if (metrics.validation_accuracy == best->validation_accuracy &&
                metrics.validation_loss < best->validation_loss) {
                best = &metrics;
            }
        }

        return best;
    }

    std::string TrainingHistory::summary() const {
        if (epochs_.empty()) {
            return "Training history is empty.";
        }

        std::ostringstream out;
        for (size_t i = 0; i < epochs_.size(); ++i) {
            if (i > 0) {
                out << "\n";
            }
            out << epochs_[i].summary();
        }

        if (const auto *best = best_validation_epoch(); best != nullptr) {
            out << "\nBest validation epoch: " << best->epoch
                    << " (val_accuracy=" << best->validation_accuracy
                    << ", val_loss=" << best->validation_loss << ")";
        }

        return out.str();
    }

    DatasetMetrics train_epoch(LstmClassifier &model,
                               torch::optim::Optimizer &optimizer,
                               const std::vector<EncodedExample> &train_examples,
                               const TrainingConfig &config,
                               const torch::Device device,
                               std::mt19937 &rng) {
        std::vector<size_t> order(train_examples.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        return run_epoch(model, &optimizer, train_examples, config, device, std::move(order));
    }

    DatasetMetrics evaluate(LstmClassifier &model,
                            const std::vector<EncodedExample> &examples,
                            const TrainingConfig &config,
                            const torch::Device device) {
        std::vector<size_t> order(examples.size());
        std::iota(order.begin(), order.end(), 0);

        return run_epoch(model, nullptr, examples, config, device, std::move(order));
    }

    TrainingHistory fit(LstmClassifier &model,
                        const std::vector<EncodedExample> &train_examples,
                        const std::vector<EncodedExample> &valid_examples,
                        const ExperimentConfig &config,
                        const torch::Device device) {
        if (train_examples.empty()) {
            throw std::runtime_error("Training set is empty.");
        }
        if (valid_examples.empty()) {
            throw std::runtime_error("Validation set is empty.");
        }

        auto optimizer = make_optimizer(model, config.training);
        std::mt19937 rng(42);

        TrainingHistory history;
        for (int epoch = 1; epoch <= config.training.epochs; ++epoch) {
            const auto train_metrics = train_epoch(
                model,
                *optimizer,
                train_examples,
                config.training,
                device,
                rng);

            const auto valid_metrics = evaluate(
                model,
                valid_examples,
                config.training,
                device);

            EpochMetrics epoch_metrics;
            epoch_metrics.epoch = epoch;
            epoch_metrics.training_loss = train_metrics.loss;
            epoch_metrics.validation_loss = valid_metrics.loss;
            epoch_metrics.validation_accuracy = valid_metrics.accuracy;

            std::cout << epoch_metrics.summary() << std::endl;
            history.add(std::move(epoch_metrics));
        }

        return history;
    }
} // namespace agnews
