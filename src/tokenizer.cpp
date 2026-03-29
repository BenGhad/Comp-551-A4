//
// Created by been on 3/29/26.
//
#include "tokenizer.h"

#include <sstream>

namespace agnews {
    Tokenizer::Tokenizer(TokenizerConfig config) : config_(config) {}

    std::string Tokenizer::normalize(const std::string& text) const {
        std::string out;
        out.reserve(text.length());

        for (unsigned char c : text) {
            if (std::isalnum(c)) {
                if (config_.lowercase) {
                    out.push_back(static_cast<char>(std::tolower(c)));
                } else {
                    out.push_back(static_cast<char>(c));
                }
            } else {
                // ignore non alnums
                out.push_back(' ');
            }
        }
        return out;
    }

    std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
        std::string clean = normalize(text);

        std::vector<std::string> tokens;
        std::istringstream iss(clean);
        std::string token;

        while (iss >> token) {
            tokens.push_back(token);
        }

        return tokens;
        
    }
}
