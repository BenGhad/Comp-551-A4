#include "trainer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <torch/nn/functional/loss.h>
#include <torch/nn/utils/clip_grad.h>

namespace agnews {
    namespace {
        bool tensor_is_finite(const torch::Tensor& tensor) {
            return torch::isfinite(tensor).all().item<bool>();
        }

        void require_finite_tensor(const torch::Tensor& tensor,
                                   const std::string& tensor_name,
                                   const bool training,
                                   const size_t batch_start) {
            if (tensor_is_finite(tensor)) {
                return;
            }

            std::ostringstream message;
            message << "Non-finite values detected in " << tensor_name
                    << " during " << (training ? "training" : "evaluation")
                    << " at batch starting index " << batch_start << ".";
            throw std::runtime_error(message.str());
        }

        void require_finite_scalar(const double value,
                                   const std::string& value_name,
                                   const bool training,
                                   const size_t batch_start) {
            if (std::isfinite(value)) {
                return;
            }

            std::ostringstream message;
            message << "Non-finite value detected in " << value_name
                    << " during " << (training ? "training" : "evaluation")
                    << " at batch starting index " << batch_start << ".";
            throw std::runtime_error(message.str());
        }

        struct TensorBatch {
            torch::Tensor input_ids;
            torch::Tensor lengths;
            torch::Tensor labels;
            std::vector<size_t> example_indices;
        };


        TensorBatch build_tensor_batch(const std::vector<EncodedExample> &dataset,
                                       const std::vector<size_t> &order,
                                       const size_t start,
                                       const size_t batch_size,
                                       const size_t max_sequence_length,
                                       torch::Device device) {
            if (batch_size == 0) {
                throw std::runtime_error("batch_size must be positive.");
            }
            if (start >= order.size()) {
                throw std::runtime_error("Batch start index is outside the dataset order.");
            }

            const size_t end = std::min(order.size(), start + batch_size);
            std::vector batch_indices(order.begin() + start, order.begin() + end);
            if (batch_indices.empty()) {
                std::ostringstream message;
                message << "build_tensor_batch produced no examples "
                        << "(start=" << start
                        << ", end=" << end
                        << ", order_size=" << order.size()
                        << ", batch_size=" << batch_size << ").";
                throw std::runtime_error(message.str());
            }

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
                std::move(batch_indices)
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
                                 std::vector<size_t> order,
                                 std::vector<int64_t> *predicted_labels = nullptr) {
            if (examples.empty()) {
                throw std::runtime_error("Cannot run training or evaluation on an empty dataset.");
            }
            if (config.batch_size == 0) {
                throw std::runtime_error("batch_size must be positive.");
            }
            if (config.max_sequence_length == 0) {
                throw std::runtime_error("max_sequence_length must be positive.");
            }

            const bool training = optimizer != nullptr;
            if (training && predicted_labels != nullptr) {
                throw std::runtime_error("Cannot collect predictions during training.");
            }

            if (predicted_labels != nullptr) {
                predicted_labels->assign(examples.size(), -1);
            }

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
                const auto batch_examples = static_cast<int64_t>(batch.example_indices.size());
                if (batch_examples <= 0) {
                    throw std::runtime_error("Encountered an empty tensor batch.");
                }
                if (batch.input_ids.size(0) != batch_examples ||
                    batch.lengths.size(0) != batch_examples ||
                    batch.labels.size(0) != batch_examples) {
                    throw std::runtime_error("Tensor batch components have inconsistent batch sizes.");
                }
                if (batch.input_ids.size(1) <= 0) {
                    throw std::runtime_error("Encountered a batch with zero padded sequence length.");
                }
                const auto batch_labels = batch.labels.to(torch::kCPU);
                const auto min_label = batch_labels.min().item<int64_t>();
                const auto max_label = batch_labels.max().item<int64_t>();
                const auto num_classes = static_cast<int64_t>(model->config().num_classes);
                if (min_label < 0 || max_label >= num_classes) {
                    std::ostringstream message;
                    message << "Label out of range for cross-entropy: expected [0, "
                            << (num_classes - 1) << "], got [" << min_label
                            << ", " << max_label << "].";
                    throw std::runtime_error(message.str());
                }


                if (training) {
                    optimizer->zero_grad();
                }

                const auto logits = model->forward(batch.input_ids, batch.lengths);
                require_finite_tensor(logits, "logits", training, start);
                const auto loss = torch::nn::functional::cross_entropy(logits, batch.labels);
                require_finite_tensor(loss, "loss", training, start);

                if (training) {
                    loss.backward();
                    const auto gradient_norm = torch::nn::utils::clip_grad_norm_(
                        model->parameters(),
                        config.max_gradient_norm);
                    require_finite_scalar(gradient_norm, "gradient norm", training, start);
                    optimizer->step();
                }

                const auto predictions = logits.argmax(1);
                if (predicted_labels != nullptr) {
                    const auto predictions_cpu = predictions.to(torch::kCPU).contiguous();
                    const auto *prediction_ptr = predictions_cpu.data_ptr<int64_t>();
                    for (size_t row = 0; row < batch.example_indices.size(); ++row) {
                        (*predicted_labels)[batch.example_indices[row]] = prediction_ptr[row];
                    }
                }
                total_correct += predictions.eq(batch.labels).sum().item<int64_t>();
                total_loss += loss.item<double>() * static_cast<double>(batch_examples);
                total_examples += batch_examples;
            }
            if (total_examples <= 0) {
                throw std::runtime_error("No examples were processed in the epoch.");
            }

            DatasetMetrics metrics;
            metrics.loss = total_loss / static_cast<double>(total_examples);
            metrics.accuracy = static_cast<double>(total_correct) / static_cast<double>(total_examples);
            require_finite_scalar(metrics.loss, "dataset loss", training, order.size());
            require_finite_scalar(metrics.accuracy, "dataset accuracy", training, order.size());
            return metrics;
        }

        bool is_better_validation_epoch(const EpochMetrics &candidate,
                                        const EpochMetrics *best) {
            if (best == nullptr) {
                return true;
            }

            if (candidate.validation_accuracy > best->validation_accuracy) {
                return true;
            }

            return candidate.validation_accuracy == best->validation_accuracy &&
                   candidate.validation_loss < best->validation_loss;
        }

        std::string serialize_model_state(LstmClassifier &model) {
            torch::serialize::OutputArchive archive;
            model->save(archive);

            std::ostringstream buffer(std::ios::out | std::ios::binary);
            archive.save_to(buffer);
            return buffer.str();
        }

        void restore_model_state(LstmClassifier &model,
                                 const std::string &serialized_state,
                                 const torch::Device device) {
            std::istringstream buffer(serialized_state, std::ios::in | std::ios::binary);
            torch::serialize::InputArchive archive;
            archive.load_from(buffer);
            model->load(archive);
            model->to(device);
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

    PredictionReport evaluate_with_predictions(LstmClassifier &model,
                                               const std::vector<EncodedExample> &examples,
                                               const TrainingConfig &config,
                                               const torch::Device device) {
        std::vector<size_t> order(examples.size());
        std::iota(order.begin(), order.end(), 0);

        PredictionReport report;
        report.metrics = run_epoch(model, nullptr, examples, config, device, std::move(order), &report.predicted_labels);
        return report;
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
        std::string best_model_state;
        std::optional<EpochMetrics> best_epoch_metrics;
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
            if (is_better_validation_epoch(epoch_metrics,
                                           best_epoch_metrics.has_value() ? &best_epoch_metrics.value() : nullptr)) {
                best_model_state = serialize_model_state(model);
                best_epoch_metrics = epoch_metrics;
            }
            history.add(std::move(epoch_metrics));
        }

        if (!best_model_state.empty()) {
            restore_model_state(model, best_model_state, device);
        }

        return history;
    }
} // namespace agnews
