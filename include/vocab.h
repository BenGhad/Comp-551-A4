//
// Created by been on 3/29/26.
//

#ifndef INC_551_A4_VOCAB_H
#define INC_551_A4_VOCAB_H

#include <string>
#include <unordered_map>
#include <vector>


namespace agnews {
    class Vocabulary {
    public:
        static constexpr int64_t PAD_ID = 0;
        static constexpr int64_t UNK_ID = 1;

        void build(const std::vector<std::vector<std::string> > &tokenized_texts,
                   int min_freq = 3);

        int64_t token_to_id(const std::string &token) const;

        std::vector<int64_t> encode(const std::vector<std::string> &tokens,
                                    size_t max_len = 128) const;

        size_t size() const;

    private:
        std::unordered_map<std::string, int64_t> stoi_;
        std::vector<std::string> itos_;
    };
}

#endif //INC_551_A4_VOCAB_H
