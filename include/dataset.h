#ifndef INC_551_A4_DATASET_H
#define INC_551_A4_DATASET_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tokenizer.h"
#include "vocab.h"

namespace agnews {
struct Example {
    int label;
    std::string title;
    std::string description;
    std::string text;
};

struct DatasetSplit {
    std::vector<Example> train;
    std::vector<Example> valid;
    std::vector<Example> test;
};

struct EncodedExample {
    int label;
    std::vector<int64_t> token_ids;
};

struct PaddedBatch {
    std::vector<std::vector<int64_t>> input_ids;
    std::vector<int64_t> lengths;
    std::vector<int64_t> labels;

    [[nodiscard]] size_t batch_size() const;
    [[nodiscard]] size_t max_seq_len() const;
};

std::vector<Example> read_ag_news_csv(const std::string& path);

void split_train_valid(const std::vector<Example>& train,
                       double train_ratio,
                       unsigned int seed,
                       std::vector<Example>& train_out,
                       std::vector<Example>& valid_out);

DatasetSplit load_ag_news(const std::string& train_path,
                          const std::string& test_path,
                          double train_ratio,
                          int seed);

std::vector<EncodedExample> encode_examples(const std::vector<Example>& examples,
                                            const Tokenizer& tokenizer,
                                            const Vocabulary& vocab,
                                            size_t max_len = 128);

PaddedBatch make_padded_batch(const std::vector<EncodedExample>& examples,
                              size_t start_index,
                              size_t batch_size,
                              size_t max_len = 128);

std::string label_to_string(int label);

} // namespace agnews

#endif //INC_551_A4_DATASET_H
