#include "dataset.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace agnews {
namespace {
    namespace fs = std::filesystem;

    fs::path resolve_csv_path(const std::string& csv_path) {
        const fs::path requested(csv_path);
        if (requested.is_absolute() && fs::exists(requested)) {
            return requested;
        }

        const fs::path cwd_candidate = fs::current_path() / requested;
        if (fs::exists(cwd_candidate)) {
            return cwd_candidate;
        }

        const fs::path source_candidate = fs::path(CMAKE_SOURCE_DIR) / requested;
        if (fs::exists(source_candidate)) {
            return source_candidate;
        }

        return requested;
    }

    // label+1,"title","description"

    std::vector<std::string> parse_csv_line(const std::string& line) {
        std::vector<std::string> fields;
        std::string cur;
        bool in_quotes = false;

        for (size_t i = 0; i < line.size(); i++) {
            char c = line[i];

            if (c == '"') {
                // Handle escaped quote: ""
                if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    i++;
                } else {
                    in_quotes = !in_quotes;
                }
            } else if (c == ',' && !in_quotes) {
                fields.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }

        fields.push_back(cur);
        return fields;
    }

    int normalize_label(int raw_label) {
        return raw_label - 1;
    }

    std::string join_text(const std::string& title, const std::string& description) {
        if (title.empty()) return description;
        if (description.empty()) return title;
        return title + " " + description;
    }
} // namespace

size_t PaddedBatch::batch_size() const {
    return input_ids.size();
}

size_t PaddedBatch::max_seq_len() const {
    return input_ids.empty() ? 0 : input_ids.front().size();
}

std::vector<Example> read_ag_news_csv(const std::string& csv_path) {
    const fs::path resolved_path = resolve_csv_path(csv_path);
    std::ifstream fin(resolved_path);
    if (!fin.is_open()) {
        std::ostringstream message;
        message << "Failed to open file: " << csv_path
                << " (cwd: " << fs::current_path().string()
                << ", tried: " << resolved_path.string() << ")";
        throw std::runtime_error(message.str());
    }

    std::vector<Example> examples;
    std::string line;

    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }

        auto fields = parse_csv_line(line);
        if (fields.size() != 3) {
            throw std::runtime_error("Malformed CSV line in " + resolved_path.string() + ": " + line);
        }

        int raw_label = std::stoi(fields[0]);

        Example ex;
        ex.label = normalize_label(raw_label);
        ex.title = fields[1];
        ex.description = fields[2];
        ex.text = join_text(ex.title, ex.description);

        examples.push_back(ex);
    }

    return examples;
}

    // no scikit
void split_train_valid(const std::vector<Example>& full_train,
                       double train_ratio,
                       unsigned int seed,
                       std::vector<Example>& train_out,
                       std::vector<Example>& valid_out) {
    std::vector<size_t> indices(full_train.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    auto train_size = static_cast<size_t>(full_train.size() * train_ratio);

    train_out.clear();
    valid_out.clear();
    train_out.reserve(train_size);
    valid_out.reserve(full_train.size() - train_size);

    for (size_t i = 0; i < indices.size(); ++i) {
        const Example& ex = full_train[indices[i]];
        if (i < train_size) {
            train_out.push_back(ex);
        } else {
            valid_out.push_back(ex);
        }
    }
}

DatasetSplit load_ag_news(const std::string& train_csv_path,
                          const std::string& test_csv_path,
                          double train_ratio,
                          int seed) {
    DatasetSplit split;

    auto full_train = read_ag_news_csv(train_csv_path);
    split.test = read_ag_news_csv(test_csv_path);

    split_train_valid(full_train, train_ratio, seed, split.train, split.valid);

    return split;
}

std::vector<EncodedExample> encode_examples(const std::vector<Example>& examples,
                                            const Tokenizer& tokenizer,
                                            const Vocabulary& vocab,
                                            const size_t max_len) {
    std::vector<EncodedExample> encoded;
    encoded.reserve(examples.size());

    for (const auto& example : examples) {
        auto tokens = tokenizer.tokenize(example.text);

        EncodedExample encoded_example;
        encoded_example.label = example.label;
        encoded_example.token_ids = vocab.encode(tokens, max_len);
        if (encoded_example.token_ids.empty()) {
            encoded_example.token_ids.push_back(Vocabulary::UNK_ID);
        }

        encoded.push_back(std::move(encoded_example));
    }

    return encoded;
}

PaddedBatch make_padded_batch(const std::vector<EncodedExample>& examples,
                              const size_t start_index,
                              const size_t batch_size,
                              const size_t max_len) {
    if (start_index >= examples.size()) {
        throw std::out_of_range("start_index is outside the encoded dataset.");
    }

    const size_t end_index = std::min(examples.size(), start_index + batch_size);
    const size_t actual_batch_size = end_index - start_index;

    PaddedBatch batch;
    batch.input_ids.reserve(actual_batch_size);
    batch.lengths.reserve(actual_batch_size);
    batch.labels.reserve(actual_batch_size);

    size_t padded_seq_len = 0;
    for (size_t i = start_index; i < end_index; ++i) {
        const size_t true_len = std::min(examples[i].token_ids.size(), max_len);
        batch.lengths.push_back(static_cast<int64_t>(true_len));
        batch.labels.push_back(static_cast<int64_t>(examples[i].label));
        padded_seq_len = std::max(padded_seq_len, true_len);
    }

    // Pad only to the longest truncated sequence in this batch, not to a global max.
    for (size_t i = start_index; i < end_index; ++i) {
        const size_t true_len = static_cast<size_t>(batch.lengths[i - start_index]);
        std::vector<int64_t> padded(padded_seq_len, Vocabulary::PAD_ID);
        std::copy_n(examples[i].token_ids.begin(), true_len, padded.begin());
        batch.input_ids.push_back(std::move(padded));
    }

    return batch;
}

std::string label_to_string(int label) {
    switch (label) {
        case 0: return "World";
        case 1: return "Sports";
        case 2: return "Business";
        case 3: return "Sci/Tech";
        default: return "Unknown";
    }
}

} // namespace agnews
