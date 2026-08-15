#include "neurotaskfm/runtime.h"

#include <nifti1_io.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace neurotaskfm {
namespace {

TensorMap cuda(const TensorMap& values, const torch::Device& device) {
  TensorMap output;
  for (const auto& [key, value] : values) output[key] = value.to(device, true);
  return output;
}

void write_nifti(const torch::Tensor& source, const std::filesystem::path& path) {
  const auto tensor = source.detach().to(torch::kCPU, torch::kFloat32).contiguous();
  if (tensor.dim() != 3) throw std::invalid_argument("NIfTI output must be a three-dimensional tensor");
  int dimensions[8]{3, static_cast<int>(tensor.size(0)), static_cast<int>(tensor.size(1)),
                    static_cast<int>(tensor.size(2)), 1, 1, 1, 1};
  auto* image = nifti_make_new_nim(dimensions, NIFTI_TYPE_FLOAT32, 1);
  if (image == nullptr) throw std::runtime_error("failed to allocate NIfTI image");
  image->dx = -2.0F;
  image->dy = 2.0F;
  image->dz = 2.0F;
  image->sform_code = NIFTI_XFORM_MNI_152;
  image->sto_xyz.m[0][0] = -2.0F; image->sto_xyz.m[0][3] = 90.0F;
  image->sto_xyz.m[1][1] = 2.0F; image->sto_xyz.m[1][3] = -126.0F;
  image->sto_xyz.m[2][2] = 2.0F; image->sto_xyz.m[2][3] = -72.0F;
  image->sto_xyz.m[3][3] = 1.0F;
  std::memcpy(image->data, tensor.data_ptr<float>(), static_cast<std::size_t>(tensor.numel()) * sizeof(float));
  nifti_set_filenames(image, path.string().c_str(), 0, 1);
  nifti_image_write(image);
  nifti_image_free(image);
}

nlohmann::json named(const torch::Tensor& source, const std::vector<std::string>& names) {
  const auto tensor = source.detach().flatten().to(torch::kCPU, torch::kFloat32).contiguous();
  const auto* values = tensor.data_ptr<float>();
  nlohmann::json output = nlohmann::json::object();
  const auto count = std::min<std::int64_t>(tensor.numel(), static_cast<std::int64_t>(names.size()));
  for (std::int64_t index = 0; index < count; ++index) output[names[static_cast<std::size_t>(index)]] = values[index];
  return output;
}

ManifestRow inference_row(const std::filesystem::path& pack, const nlohmann::json& query) {
  nlohmann::json row{{"sample_id", pack.stem().string()}, {"subject_id", "subject"},
                     {"visit_id", query.value("visit", "visit")}, {"dataset", "inference"},
                     {"split", "test"}, {"output_pack", pack.string()},
                     {"task", query.value("task", "unknown")},
                     {"contrast", query.value("contrast", "unspecified")}};
  return ManifestRow::from_json(std::move(row));
}

void save_signature(const torch::Tensor& signature, const std::filesystem::path& path,
                    const torch::Tensor& initial = {}) {
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  torch::serialize::OutputArchive archive;
  archive.write("signature", signature.detach().to(torch::kCPU, torch::kFloat32));
  if (initial.defined()) archive.write("initial_signature", initial.detach().to(torch::kCPU, torch::kFloat32));
  archive.save_to(path.string());
}

}  // namespace

nlohmann::json infer_request(const std::filesystem::path& deployment_config,
                             const std::filesystem::path& request_path,
                             const std::optional<std::filesystem::path>& output_directory) {
  const auto deployment = load_yaml(deployment_config);
  auto context = init_distributed(deployment["expert_parallel_size"].as<std::int64_t>());
  seed_model(20260814);
  const auto config = ModelConfig::load(deployment["model_config"].as<std::string>());
  auto model = NeuroTaskFM(config);
  model->to(context.device);
  load_checkpoint(model, nullptr, deployment["checkpoint"].as<std::string>(), context);
  model->eval();
  std::ifstream request_stream(request_path);
  if (!request_stream) throw std::runtime_error("cannot open inference request");
  nlohmann::json request;
  request_stream >> request;
  const auto output = output_directory.value_or(request.at("output_dir").get<std::string>());
  if (context.primary()) std::filesystem::create_directories(output);
  torch::InferenceMode inference_guard;
  torch::Tensor signature;
  if (request.contains("signature") && !request["signature"].is_null()) {
    torch::serialize::InputArchive archive;
    archive.load_from(request["signature"].get<std::string>(), context.device);
    archive.read("signature", signature);
    if (signature.dim() == 2) signature = signature.unsqueeze(0);
  }
  auto source_observations = request.value("observations", nlohmann::json::array());
  if (source_observations.empty() && request.contains("pack")) {
    source_observations.push_back({{"pack", request["pack"]},
        {"task", request.value("source_task", "unknown")},
        {"contrast", request.value("source_contrast", "unspecified")}});
  }
  if (source_observations.empty()) throw std::invalid_argument("inference request has no observations");
  nlohmann::json used = nlohmann::json::array();
  for (const auto& source : source_observations) {
    auto pack = load_feature_pack(inference_row(source.at("pack").get<std::string>(), source));
    auto batch = cuda(collate_feature_packs({pack}), context.device);
    signature = model->forward(batch, signature, false).values.at("updated_signature");
    used.push_back(source);
  }
  const auto query = request.value("query", nlohmann::json{{"task", "unknown"}, {"contrast", "unspecified"}});
  auto query_tokens = pair_tokens("task=" + query.value("task", "unknown") +
                                  ";contrast=" + query.value("contrast", "unspecified"), 192)
                          .unsqueeze(0).to(context.device);
  auto query_mask = torch::ones_like(query_tokens, torch::kBool);
  auto prediction = model->decode_signature(signature, query_tokens, query_mask, true);
  Calibration calibration = Calibration::load(deployment["calibration"].as<std::string>());
  calibration.apply(prediction);
  nlohmann::json result;
  if (context.primary()) {
    write_nifti(prediction.values.at("map_mean").select(0, 0), output / "task_effect_mean.nii.gz");
    write_nifti(torch::exp(0.5 * prediction.values.at("map_logvar").select(0, 0)), output / "task_effect_std.nii.gz");
    write_nifti(prediction.values.at("map_template").select(0, 0), output / "task_effect_template.nii.gz");
    write_nifti(prediction.values.at("map_residual").select(0, 0), output / "task_effect_individual_residual.nii.gz");
    save_signature(prediction.values.at("updated_signature").select(0, 0), output / "neurosignature.pt");
    result = {{"request_id", request.value("request_id", "")}, {"query", query}, {"observations", used},
              {"map_mean", "task_effect_mean.nii.gz"}, {"map_std", "task_effect_std.nii.gz"},
              {"map_template", "task_effect_template.nii.gz"},
              {"map_individual_residual", "task_effect_individual_residual.nii.gz"},
              {"behavior_mean", named(prediction.values.at("behavior_mean").select(0, 0), config.behavior_names)},
              {"behavior_std", named(torch::exp(0.5 * prediction.values.at("behavior_logvar").select(0, 0)), config.behavior_names)},
              {"quality_mean", named(prediction.values.at("quality_mean").select(0, 0), config.quality_names)},
              {"clinical_mean", named(prediction.values.at("clinical_mean").select(0, 0), config.clinical_names)},
              {"trajectory_mean", named(prediction.values.at("trajectory_mean").select(0, 0), config.clinical_names)},
              {"decision", calibration.abstain(prediction) ? "abstain" : "report"},
              {"representation", "inferred_task_effect_not_observed_task_dynamics"}};
    std::ofstream stream(output / "result.json");
    stream << std::setw(2) << result << '\n';
  }
  context.barrier();
  return result;
}

void personalize(const std::filesystem::path& deployment_config,
                 const std::vector<std::filesystem::path>& pack_paths,
                 const std::vector<std::string>& queries,
                 const std::filesystem::path& output_path,
                 const std::int64_t steps, const double rate) {
  const auto deployment = load_yaml(deployment_config);
  auto context = init_distributed(deployment["expert_parallel_size"].as<std::int64_t>());
  const auto config = ModelConfig::load(deployment["model_config"].as<std::string>());
  auto model = NeuroTaskFM(config);
  model->to(context.device);
  load_checkpoint(model, nullptr, deployment["checkpoint"].as<std::string>(), context);
  model->eval();
  for (auto& parameter : model->parameters()) parameter.set_requires_grad(false);
  std::vector<TensorMap> batches;
  std::vector<ModelOutput> anchors;
  torch::Tensor signature;
  {
    torch::InferenceMode guard;
    for (std::size_t index = 0; index < pack_paths.size(); ++index) {
      const auto text = index < queries.size() ? queries[index] : "unknown:unspecified";
      const auto separator = text.find(':');
      nlohmann::json query{{"task", text.substr(0, separator)},
                           {"contrast", separator == std::string::npos ? "unspecified" : text.substr(separator + 1)}};
      auto pack = load_feature_pack(inference_row(pack_paths[index], query));
      batches.push_back(cuda(collate_feature_packs({pack}), context.device));
      anchors.push_back(model->forward(batches.back(), signature, false));
      signature = anchors.back().values.at("updated_signature");
    }
  }
  const auto initial = signature.detach();
  signature = initial.clone().set_requires_grad(true);
  torch::optim::AdamW optimizer(std::vector<torch::Tensor>{signature},
                                torch::optim::AdamWOptions(rate).weight_decay(0.01));
  for (std::int64_t step = 0; step < steps; ++step) {
    optimizer.zero_grad();
    auto total = torch::zeros({}, signature.options());
    for (std::size_t index = 0; index < batches.size(); ++index) {
      auto prediction = model->decode_signature(signature, batches[index].at("query_tokens"),
                                                 batches[index].at("query_mask"), false);
      total = total + 1.0 - torch::cosine_similarity(
          prediction.values.at("distill_embedding"), anchors[index].values.at("distill_embedding"), -1).mean();
      total = total + 0.05 * torch::mse_loss(prediction.values.at("behavior_mean"),
                                             anchors[index].values.at("behavior_mean"));
      total = total + 0.02 * torch::mse_loss(prediction.values.at("quality_mean"),
                                             anchors[index].values.at("quality_mean"));
    }
    total = total / std::max<std::size_t>(batches.size(), 1) +
            5e-4 * (signature - initial).to(torch::kFloat32).square().mean();
    total.backward();
    if (context.process_group) {
      std::vector<torch::Tensor> gradient{signature.grad()};
      context.process_group->allreduce(gradient)->wait();
      signature.grad().div_(context.world_size);
    }
    optimizer.step();
  }
  if (context.primary()) save_signature(signature.select(0, 0), output_path, initial.select(0, 0));
  context.barrier();
}

}  // namespace neurotaskfm
