#include "neurotaskfm/tools.h"

#include <H5Cpp.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include "neurotask/nifti_reader.h"
#include "neurotaskfm/config.h"
#include "neurotaskfm/data.h"
#include "neurotaskfm/runtime.h"

namespace neurotaskfm {
namespace {

torch::Tensor nifti(const std::filesystem::path& path) {
  const auto image = ntfm::read_nifti(path.string());
  return torch::from_blob(const_cast<float*>(image.data.data()),
      {static_cast<std::int64_t>(image.data.size())}, torch::kFloat32).clone();
}

torch::Tensor text_values(const std::filesystem::path& path) {
  std::ifstream stream(path);
  std::vector<double> values;
  double value;
  while (stream >> value) values.push_back(value);
  return torch::tensor(values, torch::kFloat64);
}

double correlation(const torch::Tensor& first, const torch::Tensor& second) {
  auto x = first.flatten().to(torch::kFloat64);
  auto y = second.flatten().to(torch::kFloat64);
  const auto count = std::min(x.numel(), y.numel());
  x = x.narrow(0, 0, count); y = y.narrow(0, 0, count);
  const auto valid = torch::isfinite(x) & torch::isfinite(y);
  x = x.masked_select(valid); y = y.masked_select(valid);
  if (x.numel() < 3) return std::numeric_limits<double>::quiet_NaN();
  x -= x.mean(); y -= y.mean();
  return ((x * y).sum() / (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1e-12)).item<double>();
}

double cka(torch::Tensor x, torch::Tensor y) {
  x = x.reshape({-1, x.size(-1)}).to(torch::kFloat64);
  y = y.reshape({-1, y.size(-1)}).to(torch::kFloat64);
  const auto count = std::min(x.size(0), y.size(0));
  x = x.narrow(0, 0, count); y = y.narrow(0, 0, count);
  x -= x.mean(0, true); y -= y.mean(0, true);
  const auto cross = torch::linalg_matrix_norm(x.transpose(0, 1).matmul(y)).square();
  return (cross / (torch::linalg_matrix_norm(x.transpose(0, 1).matmul(x)) *
                   torch::linalg_matrix_norm(y.transpose(0, 1).matmul(y)) + 1e-12)).item<double>();
}

torch::Tensor h5(H5::H5File& file, const std::string& path) {
  auto dataset = file.openDataSet(path); auto space = dataset.getSpace();
  std::vector<hsize_t> raw(static_cast<std::size_t>(space.getSimpleExtentNdims()));
  space.getSimpleExtentDims(raw.data());
  std::vector<std::int64_t> shape; std::size_t count = 1;
  for (auto value : raw) { shape.push_back(static_cast<std::int64_t>(value)); count *= value; }
  std::vector<float> values(count); dataset.read(values.data(), H5::PredType::NATIVE_FLOAT);
  return torch::from_blob(values.data(), shape, torch::kFloat32).clone();
}

TensorMap device_map(const TensorMap& input, const torch::Device& device) {
  TensorMap output; for (const auto& [key, value] : input) output[key] = value.to(device); return output;
}

ManifestRow row_for_pack(const std::filesystem::path& path, const std::string& task,
                         const std::string& contrast) {
  return ManifestRow::from_json({{"sample_id", path.stem().string()}, {"subject_id", "probe"},
      {"visit_id", "visit"}, {"dataset", "probe"}, {"split", "test"},
      {"output_pack", path.string()}, {"task", task}, {"contrast", contrast}});
}

}  // namespace

int run_evaluate(const Arguments& arguments) {
  nlohmann::json result;
  if (arguments.has("reference-map") && arguments.has("predicted-map")) {
    auto reference = nifti(arguments.require("reference-map"));
    auto prediction = nifti(arguments.require("predicted-map"));
    const auto count = std::min(reference.numel(), prediction.numel());
    reference = reference.narrow(0, 0, count); prediction = prediction.narrow(0, 0, count);
    const auto threshold_reference = torch::quantile(reference.abs(), 0.95);
    const auto threshold_prediction = torch::quantile(prediction.abs(), 0.95);
    const auto selected_reference = reference.abs() >= threshold_reference;
    const auto selected_prediction = prediction.abs() >= threshold_prediction;
    const auto overlap = selected_reference & selected_prediction;
    result["map"] = {{"voxel_pearson", correlation(reference, prediction)},
        {"top5_dice", (2.0 * overlap.sum().item<double>()) /
                      std::max(1.0, (selected_reference.sum() + selected_prediction.sum()).item<double>())},
        {"top5_signed_agreement", overlap.any().item<bool>()
            ? (torch::sign(reference.masked_select(overlap)) == torch::sign(prediction.masked_select(overlap)))
                  .to(torch::kFloat32).mean().item<double>() : 0.0},
        {"mae", (reference - prediction).abs().mean().item<double>()}};
  }
  if (arguments.has("reference-scores") && arguments.has("predicted-scores")) {
    const auto reference = text_values(arguments.require("reference-scores"));
    const auto prediction = text_values(arguments.require("predicted-scores"));
    const auto residual = reference - prediction;
    const auto total = (reference - reference.mean()).square().sum();
    result["scores"] = {{"pearson", correlation(reference, prediction)},
        {"mae", residual.abs().mean().item<double>()},
        {"r2", (1.0 - residual.square().sum() / total.clamp_min(1e-12)).item<double>()}};
  }
  std::ofstream output(arguments.require("output")); output << std::setw(2) << result << '\n';
  return 0;
}

int run_probe(const Arguments& arguments) {
  const auto deployment = load_yaml(arguments.require("config"));
  auto context = init_distributed(deployment["expert_parallel_size"].as<std::int64_t>());
  const auto config = ModelConfig::load(deployment["model_config"].as<std::string>());
  auto model = NeuroTaskFM(config); model->to(context.device);
  load_checkpoint(model, nullptr, deployment["checkpoint"].as<std::string>(), context); model->eval();
  auto pack = load_feature_pack(row_for_pack(arguments.require("pack"), arguments.get("task", "unknown"),
                                               arguments.get("contrast", "unspecified")));
  auto batch = device_map(collate_feature_packs({pack}), context.device);
  torch::InferenceMode guard;
  const auto baseline = model->forward(batch, {}, true);
  nlohmann::json result;
  for (const auto* key : {"distill_embedding", "behavior_mean", "quality_mean", "clinical_mean", "trajectory_mean"}) {
    const auto value = baseline.values.at(key).select(0, 0).to(torch::kCPU, torch::kFloat32).contiguous();
    result["outputs"][key] = std::vector<float>(value.data_ptr<float>(), value.data_ptr<float>() + value.numel());
  }
  const auto map = baseline.values.at("map_mean").to(torch::kFloat32);
  result["outputs"]["map"] = {{"mean", map.mean().item<double>()}, {"std", map.std().item<double>()},
                                 {"l2", map.square().mean().sqrt().item<double>()}};
  for (const auto& perturbation : {"dicom_mask", "temporal_reverse", "spatial_roll", "quality_zero"}) {
    auto changed = batch;
    if (perturbation == std::string("dicom_mask") && changed.count("dicom_mask")) changed["dicom_mask"] = torch::zeros_like(changed["dicom_mask"]);
    if (perturbation == std::string("temporal_reverse") && changed.count("compiled_temporal")) changed["compiled_temporal"] = changed["compiled_temporal"].flip(1);
    if (perturbation == std::string("spatial_roll") && changed.count("compiled_spatial")) changed["compiled_spatial"] = changed["compiled_spatial"].roll({changed["compiled_spatial"].size(1) / 7}, {1});
    if (perturbation == std::string("quality_zero")) changed["quality"] = torch::zeros_like(changed["quality"]);
    const auto output = model->forward(changed, {}, false);
    result["perturbations"][perturbation]["hidden_cka"] = cka(baseline.values.at("hidden"), output.values.at("hidden"));
  }
  if (context.primary()) { std::ofstream stream(arguments.require("output")); stream << std::setw(2) << result << '\n'; }
  context.barrier();
  return 0;
}

int run_cross_state(const Arguments& arguments) {
  const auto deployment = load_yaml(arguments.require("config"));
  auto context = init_distributed(deployment["expert_parallel_size"].as<std::int64_t>());
  const auto config = ModelConfig::load(deployment["model_config"].as<std::string>());
  auto model = NeuroTaskFM(config); model->to(context.device);
  load_checkpoint(model, nullptr, deployment["checkpoint"].as<std::string>(), context); model->eval();
  const auto rows = load_manifest(arguments.require("manifest"));
  const auto source_values = arguments.all("source-task");
  const std::set<std::string> sources(source_values.begin(), source_values.end());
  const auto target_task = arguments.require("target-task");
  nlohmann::json records = nlohmann::json::array();
  torch::InferenceMode guard;
  for (const auto& target : rows) {
    if (target.task != target_task) continue;
    if (arguments.has("target-contrast") && target.contrast != arguments.require("target-contrast")) continue;
    torch::Tensor signature;
    std::vector<std::string> source_ids;
    for (const auto& source : rows) {
      if (source.subject_id != target.subject_id || !sources.count(source.task) || source.sample_id == target.sample_id) continue;
      if (arguments.get("observation-policy", "same_visit") == "same_visit" && source.visit_id != target.visit_id) continue;
      auto pack = load_feature_pack(source);
      auto batch = device_map(collate_feature_packs({pack}), context.device);
      signature = model->forward(batch, signature, false).values.at("updated_signature");
      source_ids.push_back(source.sample_id);
    }
    if (!signature.defined()) continue;
    auto tokens = pair_tokens("task=" + target.task + ";contrast=" +
                              (target.contrast.empty() ? "unspecified" : target.contrast), 192)
                      .unsqueeze(0).to(context.device);
    auto prediction = model->decode_signature(signature, tokens, torch::ones_like(tokens, torch::kBool), true);
    auto truth = load_feature_pack(target).targets;
    nlohmann::json record{{"sample_id", target.sample_id}, {"subject_id", target.subject_id},
                          {"source_samples", source_ids}};
    if (truth.count("map")) {
      record["map_pearson"] = correlation(truth.at("map"), prediction.values.at("map_mean").select(0, 0).cpu());
      record["map_mae"] = (truth.at("map") - prediction.values.at("map_mean").select(0, 0).cpu()).abs().nanmean().item<double>();
    }
    records.push_back(std::move(record));
    if (arguments.integer("max-targets", 0) > 0 && static_cast<std::int64_t>(records.size()) >= arguments.integer("max-targets", 0)) break;
  }
  if (context.primary()) {
    std::vector<double> correlations;
    for (const auto& row : records) if (row.contains("map_pearson")) correlations.push_back(row["map_pearson"].get<double>());
    nlohmann::json result{{"targets_evaluated", records.size()}, {"records", records}};
    if (!correlations.empty()) {
      std::sort(correlations.begin(), correlations.end());
      result["map"]["median_pearson"] = correlations[correlations.size() / 2];
    }
    std::ofstream stream(arguments.require("output")); stream << std::setw(2) << result << '\n';
  }
  context.barrier();
  return 0;
}

int compiler_metrics(const Arguments& arguments) {
  H5::H5File reference(arguments.require("reference"), H5F_ACC_RDONLY);
  H5::H5File candidate(arguments.require("candidate"), H5F_ACC_RDONLY);
  const auto spatial = cka(h5(reference, "/compiled/spatial"), h5(candidate, "/compiled/spatial"));
  const auto temporal = cka(h5(reference, "/compiled/temporal"), h5(candidate, "/compiled/temporal"));
  const auto epi = correlation(h5(reference, "/images/epi_slices"), h5(candidate, "/images/epi_slices"));
  const auto motion = (h5(reference, "/motion/rigid") - h5(candidate, "/motion/rigid")).abs().mean().item<double>();
  const auto quality = (h5(reference, "/quality/metrics") - h5(candidate, "/quality/metrics")).abs().mean().item<double>();
  const auto runtime = arguments.number("runtime-seconds", 0.0);
  nlohmann::json result{{"spatial_cka", spatial}, {"temporal_cka", temporal}, {"epi_slice_ncc", epi},
      {"motion_mae", motion}, {"quality_mae", quality}, {"runtime_seconds", runtime},
      {"pareto_score", spatial + temporal + 0.5 * epi - 0.01 * motion - 0.002 * runtime}};
  std::ofstream stream(arguments.require("output")); stream << std::setw(2) << result << '\n';
  return 0;
}

}  // namespace neurotaskfm
