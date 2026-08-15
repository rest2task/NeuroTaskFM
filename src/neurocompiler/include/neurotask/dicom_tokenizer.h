#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ntfm {
struct DicomTokenConfig { size_t max_tokens = 4096; size_t max_pixel_bytes = 262144; bool include_pixel_bytes = true; };
class DicomTokenizer {
 public:
  explicit DicomTokenizer(DicomTokenConfig config = {}): config_(config) {}
  std::vector<uint32_t> tokenize_tree(const std::string& root) const;
  std::vector<uint32_t> tokenize_file(const std::string& path) const;
 private:
  DicomTokenConfig config_;
};
}
