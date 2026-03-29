//
// Created by been on 3/29/26.
//
#include "vocab.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace agnews {

    void Vocabulary::build(const std::vector<std::vector<std::string>>& tokenized_texts,
                           int min_freq) {
        stoi_.clear();
        itos_.clear();

        // Reserve special tokens first.
        stoi_["<PAD>"] = PAD_ID;
        stoi_["<UNK>"] = UNK_ID;

        itos_.emplace_back("<PAD>");
        itos_.emplace_back("<UNK>");

        std::unordered_map<std::string, int> freq;
        for (const auto& tokens : tokenized_texts) {
            for (const auto& token : tokens) {
                freq[token]++;
            }
        }

        std::vector<std::pair<std::string, int>> items(freq.begin(), freq.end());

        // Deterministic order
        std::ranges::sort(items,
                          [](const auto& a, const auto& b) {
                              if (a.second != b.second) {
                                  return a.second > b.second;   // higher frequency first
                              }
                              return a.first < b.first;         // tie-break alphabetically
                          });

        for (const auto& [token, count] : items) {
            if (count >= min_freq) {
                auto id = static_cast<int64_t>(itos_.size());
                stoi_[token] = id;
                itos_.push_back(token);
            }
        }
    }

    int64_t Vocabulary::token_to_id(const std::string& token) const {
        auto it = stoi_.find(token);
        if (it == stoi_.end()) {
            return UNK_ID;
        }
        return it->second;
    }

    std::vector<int64_t> Vocabulary::encode(const std::vector<std::string>& tokens,
                                            const size_t max_len) const {
        std::vector<int64_t> ids;
        ids.reserve(std::min(tokens.size(), max_len));

        for (size_t i = 0; i < tokens.size() && i < max_len; ++i) {
            ids.push_back(token_to_id(tokens[i]));
        }

        return ids;
    }

    size_t Vocabulary::size() const {
        return itos_.size();
    }

} // namespace agnews