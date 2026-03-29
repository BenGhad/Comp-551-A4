//
// Created by been on 3/29/26.
//

#ifndef INC_551_A4_TOKENIZER_H
#define INC_551_A4_TOKENIZER_H

#include <string>
#include <vector>


namespace agnews {
    struct TokenizerConfig {
        bool lowercase = true;
    };

    class Tokenizer {
    public:
        explicit Tokenizer(TokenizerConfig config = {});

        std::string normalize(const std::string& text) const;
        std::vector<std::string> tokenize(const std::string& text) const;

    private:
        TokenizerConfig config_;
    };

} // namespace agnews

#endif // INC_551_A4_TOKENIZER_H
