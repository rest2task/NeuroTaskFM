#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "neurotaskfm/model.h"

namespace neurotaskfm {

struct ManifestRow {
  std::string sample_id;
  std::string subject_id;
  std::string visit_id;
  std::string dataset;
  std::string split;
  std::string t1_nifti;
  std::string fmri_nifti;
  std::string output_pack;
  std::string family_id;
  std::string site_id;
  std::string task;
  std::string contrast;
  std::string t1_dicom;
  std::string fmri_dicom;
  std::string atlas_nifti;
  double tr_seconds{1.0};
  nlohmann::json metadata = nlohmann::json::object();
  nlohmann::json mr_resources = nlohmann::json::array();
  nlohmann::json source;

  static ManifestRow from_json(nlohmann::json value);
};

std::vector<ManifestRow> load_manifest(const std::filesystem::path& path,
                                       const std::optional<std::string>& split = std::nullopt);
void assert_no_subject_leakage(const std::vector<std::vector<ManifestRow>>& manifests);
torch::Tensor pair_tokens(const std::string& text, std::size_t limit = 256);
std::int64_t stable_id(const std::string& text, std::int64_t modulo = 32768);
std::int64_t modality_id(const std::string& name);
std::int64_t view_id(const std::string& name);

struct FeaturePack {
  TensorMap values;
  TensorMap targets;
  nlohmann::json metadata = nlohmann::json::object();
  std::string sample_id;
  std::string subject_id;
  std::string visit_id;
  std::string dataset;
};

FeaturePack load_feature_pack(const ManifestRow& row, bool load_physics = false,
                              const std::filesystem::path& root = {}, bool load_resources = true);
TensorMap collate_feature_packs(const std::vector<FeaturePack>& packs,
                                TensorMap* targets = nullptr);
TensorMap apply_context_limits(TensorMap batch, const YAML::Node& limits);
TensorMap apply_training_augmentation(TensorMap batch, const YAML::Node& config);

class FeaturePackDataset : public torch::data::datasets::Dataset<FeaturePackDataset, FeaturePack> {
 public:
  explicit FeaturePackDataset(std::filesystem::path manifest,
                              std::optional<std::string> split = std::nullopt,
                              bool load_physics = false, bool load_resources = true);
  FeaturePack get(std::size_t index) override;
  std::optional<std::size_t> size() const override;
  const std::vector<ManifestRow>& rows() const { return rows_; }

 private:
  std::filesystem::path root_;
  std::vector<ManifestRow> rows_;
  bool load_physics_;
  bool load_resources_;
};

class HierarchicalDistributedSampler {
 public:
  HierarchicalDistributedSampler(const std::vector<ManifestRow>& rows, std::int64_t replicas,
                                 std::int64_t rank, std::uint64_t seed,
                                 double dataset_temperature = 0.5);
  void set_epoch(std::int64_t epoch);
  [[nodiscard]] std::vector<std::size_t> indices() const;

 private:
  const std::vector<ManifestRow>* rows_;
  std::int64_t replicas_, rank_, epoch_{};
  std::uint64_t seed_;
  double temperature_;
};

}  // namespace neurotaskfm
