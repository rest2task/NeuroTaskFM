#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace neurotaskfm {

struct ModelConfig {
  std::string name;
  std::string variant;
  std::string parameter_label;
  std::int64_t hidden_size{};
  std::int64_t num_layers{};
  std::int64_t num_attention_heads{};
  std::int64_t num_kv_heads{};
  std::int64_t head_dim{};
  std::int64_t ffn_hidden_size{};
  std::vector<std::int64_t> moe_layers;
  std::int64_t num_experts{};
  std::int64_t top_k{};
  bool shared_expert{};
  std::int64_t max_sequence_length{};
  std::vector<std::string> attention_pattern;
  double dropout{};
  double drop_path{};
  double rope_theta{};
  std::int64_t observation_latents{};
  std::int64_t modality_latents{};

  std::unordered_map<std::string, bool> modalities;
  std::int64_t dicom_vocab_size{};
  std::int64_t dicom_max_tokens{};
  std::int64_t slice_size{};
  std::int64_t slice_patch_size{};
  std::int64_t tokens_per_slice{};
  std::int64_t max_slices{};
  std::int64_t compiled_channels{};
  std::int64_t max_spatial_tokens{};
  std::int64_t max_temporal_tokens{};
  std::vector<std::int64_t> volume_patch_size{8, 8, 8};
  std::int64_t volume_max_tokens{1024};
  std::int64_t biomarker_features{64};
  std::int64_t clinical_features{64};

  std::int64_t signature_tokens{};
  std::vector<std::int64_t> signature_partitions;
  std::vector<std::int64_t> map_shape;
  std::int64_t task_states{};
  std::vector<std::int64_t> future_offsets;
  std::int64_t behavior_targets{};
  std::int64_t quality_targets{};
  std::int64_t clinical_targets{};
  std::vector<std::string> behavior_names;
  std::vector<std::string> quality_names;
  std::vector<std::string> clinical_names;

  static ModelConfig load(const std::filesystem::path& path);
  void validate() const;
  [[nodiscard]] std::unordered_map<std::string, double> estimate_parameters() const;
};

YAML::Node load_yaml(const std::filesystem::path& path);

}  // namespace neurotaskfm
