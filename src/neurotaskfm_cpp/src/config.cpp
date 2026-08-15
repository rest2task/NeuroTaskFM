#include "neurotaskfm/config.h"

#include <algorithm>
#include <stdexcept>

namespace neurotaskfm {
namespace {

template <typename T>
T optional(const YAML::Node& node, const char* key, T fallback) {
  return node[key] ? node[key].as<T>() : std::move(fallback);
}

std::vector<std::string> strings(const YAML::Node& node) {
  return node ? node.as<std::vector<std::string>>() : std::vector<std::string>{};
}

}  // namespace

YAML::Node load_yaml(const std::filesystem::path& path) {
  return YAML::LoadFile(path.string());
}

ModelConfig ModelConfig::load(const std::filesystem::path& path) {
  const auto raw = load_yaml(path);
  ModelConfig config;
  config.name = raw["name"].as<std::string>();
  config.variant = raw["variant"].as<std::string>();
  config.parameter_label = raw["parameter_label"].as<std::string>();
  config.hidden_size = raw["hidden_size"].as<std::int64_t>();
  config.num_layers = raw["num_layers"].as<std::int64_t>();
  config.num_attention_heads = raw["num_attention_heads"].as<std::int64_t>();
  config.num_kv_heads = raw["num_kv_heads"].as<std::int64_t>();
  config.head_dim = raw["head_dim"].as<std::int64_t>();
  config.ffn_hidden_size = raw["ffn_hidden_size"].as<std::int64_t>();
  config.moe_layers = raw["moe_layers"].as<std::vector<std::int64_t>>();
  config.num_experts = raw["num_experts"].as<std::int64_t>();
  config.top_k = raw["top_k"].as<std::int64_t>();
  config.shared_expert = raw["shared_expert"].as<bool>();
  config.max_sequence_length = raw["max_sequence_length"].as<std::int64_t>();
  config.attention_pattern = strings(raw["attention_pattern"]);
  config.dropout = raw["dropout"].as<double>();
  config.drop_path = raw["drop_path"].as<double>();
  config.rope_theta = raw["rope_theta"].as<double>();
  config.observation_latents = raw["observation_latents"].as<std::int64_t>();
  config.modality_latents = raw["modality_latents"].as<std::int64_t>();

  for (const auto& item : raw["modalities"]) {
    config.modalities.emplace(item.first.as<std::string>(), item.second.as<bool>());
  }
  const auto dicom = raw["dicom"];
  config.dicom_vocab_size = dicom["vocab_size"].as<std::int64_t>();
  config.dicom_max_tokens = dicom["max_tokens"].as<std::int64_t>();
  const auto slices = raw["slices"];
  config.slice_size = slices["size"].as<std::int64_t>();
  config.slice_patch_size = slices["patch_size"].as<std::int64_t>();
  config.tokens_per_slice = slices["tokens_per_slice"].as<std::int64_t>();
  config.max_slices = slices["max_slices"].as<std::int64_t>();
  const auto compiled = raw["compiled"];
  config.compiled_channels = compiled["channels"].as<std::int64_t>();
  config.max_spatial_tokens = compiled["max_spatial_tokens"].as<std::int64_t>();
  config.max_temporal_tokens = compiled["max_temporal_tokens"].as<std::int64_t>();
  if (raw["volumes"]) {
    config.volume_patch_size = optional(raw["volumes"], "patch_size", config.volume_patch_size);
    config.volume_max_tokens = optional(raw["volumes"], "max_tokens", config.volume_max_tokens);
  }
  if (raw["tabular"]) {
    config.biomarker_features = optional(raw["tabular"], "biomarker_features", config.biomarker_features);
    config.clinical_features = optional(raw["tabular"], "clinical_features", config.clinical_features);
  }

  const auto signature = raw["signature"];
  config.signature_tokens = signature["tokens"].as<std::int64_t>();
  for (const auto* key : {"trait_tokens", "state_tokens", "task_tokens", "longitudinal_tokens",
                          "provenance_tokens", "uncertainty_tokens"}) {
    config.signature_partitions.push_back(signature[key].as<std::int64_t>());
  }
  const auto outputs = raw["outputs"];
  config.map_shape = outputs["map_shape"].as<std::vector<std::int64_t>>();
  config.task_states = outputs["task_states"].as<std::int64_t>();
  config.future_offsets = outputs["future_offsets"].as<std::vector<std::int64_t>>();
  config.behavior_targets = outputs["behavior_targets"].as<std::int64_t>();
  config.quality_targets = outputs["quality_targets"].as<std::int64_t>();
  config.clinical_targets = outputs["clinical_targets"].as<std::int64_t>();
  config.behavior_names = strings(outputs["behavior_names"]);
  config.quality_names = strings(outputs["quality_names"]);
  config.clinical_names = strings(outputs["clinical_names"]);
  config.validate();
  return config;
}

void ModelConfig::validate() const {
  if (hidden_size != num_attention_heads * head_dim) {
    throw std::invalid_argument("hidden_size must equal attention_heads * head_dim");
  }
  if (num_attention_heads % num_kv_heads != 0) {
    throw std::invalid_argument("num_attention_heads must be divisible by num_kv_heads");
  }
  if (num_experts < top_k) {
    throw std::invalid_argument("num_experts must be >= top_k");
  }
  for (const auto index : moe_layers) {
    if (index < 0 || index >= num_layers) throw std::invalid_argument("moe layer index out of range");
  }
  std::int64_t partition_total = 0;
  for (const auto value : signature_partitions) partition_total += value;
  if (partition_total != signature_tokens) {
    throw std::invalid_argument("signature token partitions do not sum to tokens");
  }
  if (map_shape.size() != 3 || attention_pattern.empty()) {
    throw std::invalid_argument("map_shape must be 3D and attention_pattern cannot be empty");
  }
}

std::unordered_map<std::string, double> ModelConfig::estimate_parameters() const {
  const auto h = static_cast<double>(hidden_size);
  const auto f = static_cast<double>(ffn_hidden_size);
  const auto layers = static_cast<double>(num_layers);
  const auto moe = static_cast<double>(moe_layers.size());
  const auto q = h * num_attention_heads * head_dim;
  const auto kv = 2.0 * h * num_kv_heads * head_dim;
  const auto out = h * num_attention_heads * head_dim;
  const auto attention = layers * (q + kv + out);
  const auto dense_mlp = (layers - moe) * 3.0 * h * f;
  const auto experts = moe * num_experts * 3.0 * h * f;
  const auto shared = shared_expert ? moe * 3.0 * h * f : 0.0;
  const auto active_experts = moe * top_k * 3.0 * h * f;
  const auto embeddings = dicom_vocab_size * h;
  const auto heads = h >= 4096.0 ? 0.9e9 : 0.7e9;
  const auto total = attention + dense_mlp + experts + shared + embeddings + heads;
  const auto active = attention + dense_mlp + active_experts + shared + embeddings + heads;
  return {{"total", total}, {"active", active}, {"total_b", total / 1e9}, {"active_b", active / 1e9}};
}

}  // namespace neurotaskfm
