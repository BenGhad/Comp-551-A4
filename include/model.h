#ifndef INC_551_A4_MODEL_H
#define INC_551_A4_MODEL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace agnews {

enum class PoolingStrategy {
    LastHiddenState,
};

struct ModelConfig {
    size_t vocab_size = 0;
    size_t embedding_dim = 128;
    size_t hidden_size = 256;
    int num_layers = 1;
    double dropout = 0.2;
    bool bidirectional = false;
    size_t num_classes = 4;
    PoolingStrategy pooling = PoolingStrategy::LastHiddenState;

};

class LstmClassifierSpec {
public:
    explicit LstmClassifierSpec(ModelConfig config);

    [[nodiscard]] const ModelConfig& config() const;
    [[nodiscard]] size_t sequence_representation_dim() const;
    [[nodiscard]] std::string pooling_name() const;
    [[nodiscard]] std::string pooling_explanation() const;
    [[nodiscard]] std::string summary() const;

private:
    ModelConfig config_;
};

class LstmClassifierImpl : public torch::nn::Module {
public:
    explicit LstmClassifierImpl(const ModelConfig& config);

    [[nodiscard]] const ModelConfig& config() const;
    [[nodiscard]] size_t sequence_representation_dim() const;

    torch::Tensor forward(const torch::Tensor& input_ids,
                          const torch::Tensor& lengths);

private:
    torch::Tensor pool_last_hidden(const torch::Tensor& h_n) const;

    ModelConfig config_;
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Linear classifier_{nullptr};
};

TORCH_MODULE(LstmClassifier);

std::vector<size_t> last_real_token_positions(const std::vector<int64_t>& lengths);
} // namespace agnews

#endif //INC_551_A4_MODEL_H
