#include "model.h"

#include <sstream>
#include <stdexcept>
#include <utility>

#include "vocab.h"

namespace agnews {



LstmClassifierSpec::LstmClassifierSpec(ModelConfig config) : config_(std::move(config)) {}

const ModelConfig& LstmClassifierSpec::config() const {
    return config_;
}

size_t LstmClassifierSpec::sequence_representation_dim() const {
    return config_.hidden_size * (config_.bidirectional ? 2u : 1u);
}

std::string LstmClassifierSpec::pooling_name() const {
    switch (config_.pooling) {
        case PoolingStrategy::LastHiddenState:
            return "Last hidden state";
    }

    throw std::runtime_error("Unsupported pooling strategy.");
}

std::string LstmClassifierSpec::pooling_explanation() const {
    switch (config_.pooling) {
        case PoolingStrategy::LastHiddenState:
            return "Use the hidden state from the final real token in each sequence. "
                   "The model packs padded batches with the true lengths so PAD tokens "
                   "are skipped by the LSTM, then it uses the final hidden state from "
                   "the top LSTM layer as the sequence representation.";
    }

    throw std::runtime_error("Unsupported pooling strategy.");
}

std::string LstmClassifierSpec::summary() const {
    std::ostringstream out;
    out << "Embedding(vocab_size=" << config_.vocab_size
        << ", d_emb=" << config_.embedding_dim << ") -> "
        << "LSTM(input_size=" << config_.embedding_dim
        << ", hidden_size=" << config_.hidden_size
        << ", num_layers=" << config_.num_layers
        << ", dropout=" << config_.dropout
        << ", bidirectional=" << (config_.bidirectional ? "true" : "false") << ") -> "
        << pooling_name()
        << " -> Linear(in_features=" << sequence_representation_dim()
        << ", out_features=" << config_.num_classes << ")";
    return out.str();
}

LstmClassifierImpl::LstmClassifierImpl(const ModelConfig& config) : config_(config) {

    embedding_ = register_module(
        "embedding",
        torch::nn::Embedding(
            torch::nn::EmbeddingOptions(
                static_cast<int64_t>(config_.vocab_size),
                static_cast<int64_t>(config_.embedding_dim))
                .padding_idx(Vocabulary::PAD_ID)));

    lstm_ = register_module(
        "lstm",
        torch::nn::LSTM(
            torch::nn::LSTMOptions(
                static_cast<int64_t>(config_.embedding_dim),
                static_cast<int64_t>(config_.hidden_size))
                .num_layers(config_.num_layers)
                .dropout(config_.num_layers > 1 ? config_.dropout : 0.0)
                .bidirectional(config_.bidirectional)
                .batch_first(true)));

    classifier_ = register_module(
        "classifier",
        torch::nn::Linear(
            static_cast<int64_t>(sequence_representation_dim()),
            static_cast<int64_t>(config_.num_classes)));
}

const ModelConfig& LstmClassifierImpl::config() const {
    return config_;
}

size_t LstmClassifierImpl::sequence_representation_dim() const {
    return LstmClassifierSpec(config_).sequence_representation_dim();
}

torch::Tensor LstmClassifierImpl::forward(const torch::Tensor& input_ids,
                                          const torch::Tensor& lengths) {
    if (input_ids.dim() != 2) {
        throw std::runtime_error("input_ids must have shape [batch_size, seq_len].");
    }
    if (lengths.dim() != 1) {
        throw std::runtime_error("lengths must have shape [batch_size].");
    }
    if (input_ids.size(0) != lengths.size(0)) {
        throw std::runtime_error("Batch size mismatch between input_ids and lengths.");
    }

    const auto embedded = embedding_->forward(input_ids);
    const auto packed = torch::nn::utils::rnn::pack_padded_sequence(
        embedded,
        lengths.to(torch::kCPU),
        true,
        false);

    auto packed_result = lstm_->forward_with_packed_input(packed);
    auto hidden = std::get<1>(packed_result);
    const auto& h_n = std::get<0>(hidden);

    torch::Tensor pooled;
    switch (config_.pooling) {
        case PoolingStrategy::LastHiddenState:
            pooled = pool_last_hidden(h_n);
            break;
        default:
            throw std::runtime_error("Unsupported pooling strategy.");
    }

    return classifier_->forward(pooled);
}

torch::Tensor LstmClassifierImpl::pool_last_hidden(const torch::Tensor& h_n) const {
    const int64_t num_directions = config_.bidirectional ? 2 : 1;
    const int64_t layer_offset = static_cast<int64_t>(config_.num_layers - 1) * num_directions;

    if (!config_.bidirectional) {
        return h_n.select(0, layer_offset);
    }

    const auto forward_hidden = h_n.select(0, layer_offset);
    const auto backward_hidden = h_n.select(0, layer_offset + 1);
    return torch::cat({forward_hidden, backward_hidden}, 1);
}

std::vector<size_t> last_real_token_positions(const std::vector<int64_t>& lengths) {
    std::vector<size_t> positions;
    positions.reserve(lengths.size());

    for (const int64_t length : lengths) {
        if (length <= 0) {
            throw std::runtime_error("Sequence lengths must be positive.");
        }
        positions.push_back(static_cast<size_t>(length - 1));
    }

    return positions;
}

} // namespace agnews
