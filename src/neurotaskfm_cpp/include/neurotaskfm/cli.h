#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace neurotaskfm {

class Arguments {
 public:
  Arguments(int argc, char** argv, int start = 1);
  [[nodiscard]] bool has(const std::string& name) const;
  [[nodiscard]] std::string require(const std::string& name) const;
  [[nodiscard]] std::string get(const std::string& name, std::string fallback = {}) const;
  [[nodiscard]] std::int64_t integer(const std::string& name, std::int64_t fallback) const;
  [[nodiscard]] double number(const std::string& name, double fallback) const;
  [[nodiscard]] std::vector<std::string> all(const std::string& name) const;
  [[nodiscard]] const std::vector<std::string>& positional() const { return positional_; }

 private:
  std::unordered_map<std::string, std::vector<std::string>> options_;
  std::vector<std::string> positional_;
};

}  // namespace neurotaskfm
