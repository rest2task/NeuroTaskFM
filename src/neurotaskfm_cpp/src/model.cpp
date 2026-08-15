#include "neurotaskfm/model.h"

#include <algorithm>
#include <stdexcept>

using torch::indexing::Slice;

namespace neurotaskfm {
namespace {

bool enabled(const ModelConfig& config, const std::string& name) {
  const auto item = config.modalities.find(name);
  return item != config.modalities.end() && item->second;
}

bool contains(const TensorMap& values, const std::string& key) {
  const auto item = values.find(key);
  return item != values.end() && item->second.defined();
}

torch::Tensor required(const TensorMap& values, const std::string& key) {
  const auto item = values.find(key);
  if (item == values.end() || !item->second.defined()) {
    throw std::invalid_argument("batch is missing tensor '" + key + "'");
  }
  return item->second;
}

bool is_moe_layer(const ModelConfig& config, const std::int64_t index) {
  return std::find(config.moe_layers.begin(), config.moe_layers.end(), index) != config.moe_layers.end();
}

}  // namespace

NeuroTaskFMImpl::NeuroTaskFMImpl(ModelConfig model_config, const bool physics_enabled,
                                 std::int64_t physics_regions, const bool resource_encoders)
    : config(std::move(model_config)) {
  const auto hidden_size = config.hidden_size;
  dicom = register_module("dicom", DicomEncoder(config.dicom_vocab_size, hidden_size, config.dicom_max_tokens));
  slices = register_module("slices", SliceEncoder(hidden_size, config.slice_patch_size,
                                                   config.tokens_per_slice, config.max_slices));
  spatial = register_module("spatial", CompiledEncoder(config.compiled_channels, hidden_size,
                                                        config.max_spatial_tokens));
  temporal = register_module("temporal", CompiledEncoder(config.compiled_channels, hidden_size,
                                                          config.max_temporal_tokens));
  if (enabled(config, "t1_volume")) {
    t1_volume = register_module("t1_volume", VolumeEncoder(hidden_size, config.volume_patch_size,
                                                            config.volume_max_tokens));
  }
  if (enabled(config, "fmri_volume")) {
    fmri_volume = register_module("fmri_volume", VolumeEncoder(hidden_size, config.volume_patch_size,
                                                                config.volume_max_tokens));
  }
  if (resource_encoders) {
    mr_images = register_module("mr_images", FrameEncoder(hidden_size, config.slice_patch_size, 4, 48));
    mr_video = register_module("mr_video", FrameEncoder(hidden_size, config.slice_patch_size, 4, 32));
    mr_volumes = register_module("mr_volumes", VolumeEncoder(hidden_size, config.volume_patch_size, 64));
  }
  if (enabled(config, "biomarkers")) {
    biomarkers = register_module("biomarkers", TabularEncoder(hidden_size, config.biomarker_features));
  }
  if (enabled(config, "clinical")) {
    clinical_inputs = register_module("clinical_inputs", TabularEncoder(hidden_size, config.clinical_features));
  }

  const auto resample_heads = std::max<std::int64_t>(1, std::min<std::int64_t>(config.num_attention_heads, 16));
  const auto resample_dim = hidden_size / resample_heads;
  observation_resampler = register_module("observation_resampler",
      PerceiverResampler(hidden_size, config.observation_latents, resample_heads, resample_dim, 2));
  query = register_module("query", QueryEncoder(config.dicom_vocab_size, hidden_size, 512,
                                                 config.modality_latents, resample_heads, resample_dim));
  signature_updater = register_module("signature_updater",
      NeuroSignatureUpdater(hidden_size, config.signature_partitions, resample_heads, resample_dim, 16));
  cls = register_parameter("cls", torch::randn({1, 1, hidden_size}) * 0.02);
  null_observation = register_parameter("null_observation",
      torch::randn({1, config.observation_latents, hidden_size}) * 0.02);

  blocks.reserve(static_cast<std::size_t>(config.num_layers));
  for (std::int64_t index = 0; index < config.num_layers; ++index) {
    const auto rate = config.drop_path * index / std::max<std::int64_t>(config.num_layers - 1, 1);
    auto block = NeuroBlock(hidden_size, config.num_attention_heads, config.num_kv_heads,
                            config.head_dim, config.ffn_hidden_size,
                            config.attention_pattern[static_cast<std::size_t>(index) % config.attention_pattern.size()],
                            is_moe_layer(config, index), config.num_experts, config.top_k,
                            config.shared_expert, config.rope_theta, config.dropout, rate);
    blocks.push_back(register_module("block_" + std::to_string(index), block));
  }
  norm = register_module("norm", RMSNorm(hidden_size));
  fuse_norm = register_module("fuse_norm", RMSNorm(hidden_size * 3));
  fuse_projection = register_module("fuse_projection", torch::nn::Linear(hidden_size * 3, hidden_size));
  map_decoder = register_module("map_decoder", MapDecoder(hidden_size, config.map_shape));
  behavior = register_module("behavior", DistributionHead(hidden_size, config.behavior_targets));
  quality_head = register_module("quality_head", DistributionHead(hidden_size, config.quality_targets));
  clinical = register_module("clinical", DistributionHead(hidden_size, config.clinical_targets));
  trajectory = register_module("trajectory", DistributionHead(hidden_size, config.clinical_targets));
  state = register_module("state", DynamicStateHead(hidden_size, config.task_states, resample_heads, resample_dim));
  future = register_module("future", FutureWindowHead(hidden_size, config.compiled_channels, config.future_offsets));
  if (physics_enabled) {
    if (physics_regions <= 0) physics_regions = config.compiled_channels;
    physics_head = register_module("physics_head", PhysicsHead(hidden_size, physics_regions));
  }
  latent_predictor = register_module("latent_predictor", torch::nn::Sequential(
      RMSNorm(hidden_size), torch::nn::Linear(hidden_size, hidden_size), torch::nn::SiLU(),
      torch::nn::Linear(hidden_size, hidden_size)));
  distill_head = register_module("distill_head", torch::nn::Sequential(
      RMSNorm(hidden_size), torch::nn::Linear(torch::nn::LinearOptions(hidden_size, 1024).bias(false))));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, TensorMap>
NeuroTaskFMImpl::encode_observations(const TensorMap& batch) {
  std::vector<torch::Tensor> tokens;
  std::vector<torch::Tensor> masks;
  TensorMap summaries;
  torch::Tensor temporal_tokens;

  const auto add = [&](const std::string& name, const torch::Tensor& value, const torch::Tensor& mask) {
    tokens.push_back(value);
    masks.push_back(mask);
    const auto weight = mask.to(value.scalar_type()).unsqueeze(-1);
    summaries[name] = (value * weight).sum(1) / weight.sum(1).clamp_min(1.0);
    summaries[name + "__available"] = mask.any(1);
  };

  if (enabled(config, "dicom_raw") && contains(batch, "dicom_tokens")) {
    const auto value = dicom(required(batch, "dicom_tokens"));
    auto mask = contains(batch, "dicom_mask")
                    ? required(batch, "dicom_mask")
                    : torch::ones_like(required(batch, "dicom_tokens"), torch::kBool);
    add("dicom", value, mask.index({Slice(), Slice(0, value.size(1))}));
  }
  if (enabled(config, "image_slices") && contains(batch, "slices")) {
    const auto encoded = slices->forward(required(batch, "slices"), required(batch, "slice_types"),
                                         required(batch, "slice_mask"));
    add("slices", encoded.first, encoded.second);
  }
  if (enabled(config, "compiled_features") && contains(batch, "compiled_spatial")) {
    const auto value = spatial(required(batch, "compiled_spatial"));
    add("compiled_spatial", value, required(batch, "spatial_mask").index({Slice(), Slice(0, value.size(1))}));
  }
  if (enabled(config, "compiled_features") && contains(batch, "compiled_temporal")) {
    auto mask = required(batch, "temporal_mask");
    const auto input = required(batch, "compiled_temporal");
    mask = mask.index({Slice(), Slice(0, input.size(1))});
    temporal_tokens = temporal(input * mask.unsqueeze(-1).to(input.scalar_type()));
    add("compiled_temporal", temporal_tokens,
        mask.index({Slice(), Slice(0, temporal_tokens.size(1))}));
  }
  if (t1_volume && contains(batch, "t1_volume")) {
    const auto value = t1_volume->forward(required(batch, "t1_volume"),
                                          contains(batch, "t1_volume_mask") ? required(batch, "t1_volume_mask") : torch::Tensor{});
    add("t1_volume", value.first, value.second);
  }
  if (fmri_volume && contains(batch, "fmri_volume")) {
    const auto value = fmri_volume->forward(required(batch, "fmri_volume"),
                                            contains(batch, "fmri_volume_mask") ? required(batch, "fmri_volume_mask") : torch::Tensor{});
    add("fmri_volume", value.first, value.second);
  }
  if (mr_images && contains(batch, "mr_images")) {
    const auto value = mr_images->forward(required(batch, "mr_images"), required(batch, "mr_image_types"),
                                          required(batch, "mr_image_views"), required(batch, "mr_image_positions"),
                                          required(batch, "mr_image_mask"));
    add("mr_images", value.first, value.second);
  }
  if (mr_video && contains(batch, "mr_video")) {
    const auto value = mr_video->forward(required(batch, "mr_video"), required(batch, "mr_video_types"),
                                         required(batch, "mr_video_views"), required(batch, "mr_video_positions"),
                                         required(batch, "mr_video_mask"));
    add("mr_video", value.first, value.second);
  }
  if (mr_volumes && contains(batch, "mr_volumes")) {
    const auto value = mr_volumes->forward(required(batch, "mr_volumes"), required(batch, "mr_volume_mask"));
    add("mr_volumes", value.first, value.second);
  }
  if (biomarkers && contains(batch, "biomarkers")) {
    const auto value = biomarkers->forward(required(batch, "biomarkers"), required(batch, "biomarker_mask"));
    add("biomarkers", value.first, value.second);
  }
  if (clinical_inputs && contains(batch, "clinical_inputs")) {
    const auto value = clinical_inputs->forward(required(batch, "clinical_inputs"),
                                                required(batch, "clinical_input_mask"));
    add("clinical", value.first, value.second);
  }
  if (tokens.empty()) throw std::invalid_argument("no enabled imaging modality was present in the batch");
  return {torch::cat(tokens, 1), torch::cat(masks, 1), temporal_tokens, std::move(summaries)};
}

ModelOutput NeuroTaskFMImpl::decode(const torch::Tensor& signature,
                                    const torch::Tensor& observation_latents,
                                    const torch::Tensor& query_latents,
                                    const torch::Tensor& temporal_tokens,
                                    const bool should_decode_map,
                                    const torch::Tensor& shifted_temporal_tokens) {
  ModelOutput output;
  const auto class_token = cls.expand({observation_latents.size(0), -1, -1});
  auto hidden = torch::cat({class_token, signature, observation_latents, query_latents}, 1);
  const auto mask = torch::ones({hidden.size(0), hidden.size(1)}, hidden.options().dtype(torch::kBool));
  auto balance = torch::zeros({}, hidden.options());
  auto router_z = torch::zeros({}, hidden.options());
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    torch::Tensor local_balance, local_router_z;
    std::tie(hidden, local_balance, local_router_z) = blocks[index]->forward(hidden, mask);
    const auto adapter = adapters.find(static_cast<std::int64_t>(index));
    if (adapter != adapters.end()) hidden = hidden + adapter->second->forward(hidden);
    balance = balance + local_balance;
    router_z = router_z + local_router_z;
  }
  hidden = norm(hidden);
  const auto pooled = hidden.select(1, 0);
  const auto signature_hidden = hidden.index({Slice(), Slice(1, 1 + signature.size(1))});
  const auto query_hidden = hidden.index({Slice(), Slice(hidden.size(1) - query_latents.size(1), hidden.size(1))});
  const auto fused = torch::silu(fuse_projection(fuse_norm(torch::cat(
      {pooled, signature_hidden.mean(1), query_hidden.mean(1)}, -1))));
  auto behavior_values = behavior->forward(fused);
  auto quality_values = quality_head->forward(fused);
  auto clinical_values = clinical->forward(fused);
  auto trajectory_values = trajectory->forward(fused);
  auto& values = output.values;
  values["pooled"] = pooled;
  values["predicted_latent"] = latent_predictor->forward(pooled);
  values["distill_embedding"] = distill_head->forward(pooled);
  values["hidden"] = hidden;
  values["updated_signature"] = signature;
  values["signature_hidden"] = signature_hidden;
  values["behavior_mean"] = behavior_values.first;
  values["behavior_logvar"] = behavior_values.second;
  values["quality_mean"] = quality_values.first;
  values["quality_logvar"] = quality_values.second;
  values["clinical_mean"] = clinical_values.first;
  values["clinical_logvar"] = clinical_values.second;
  values["trajectory_mean"] = trajectory_values.first;
  values["trajectory_logvar"] = trajectory_values.second;
  const auto moe_denominator = std::max<std::size_t>(config.moe_layers.size(), 1);
  values["moe_balance"] = balance / static_cast<double>(moe_denominator);
  values["router_z"] = router_z / static_cast<double>(moe_denominator);

  if (should_decode_map) {
    std::tie(values["map_mean"], values["map_logvar"], values["map_template"], values["map_residual"]) =
        map_decoder->forward(fused, query_latents.mean(1));
  }
  if (temporal_tokens.defined()) {
    values["state_logits"] = state->forward(temporal_tokens, hidden, mask);
    values["future_temporal"] = future->forward(temporal_tokens);
    values["future_offsets"] = torch::tensor(config.future_offsets,
        torch::TensorOptions().dtype(torch::kLong).device(hidden.device()));
    if (physics_head) {
      output.physics = physics_head->forward(temporal_tokens, fused);
      if (shifted_temporal_tokens.defined()) {
        output.shifted_physics = physics_head->forward(shifted_temporal_tokens, fused);
      }
    }
  }
  if (concept_head) values["concepts"] = concept_head->forward(fused);
  return output;
}

ModelOutput NeuroTaskFMImpl::forward(const TensorMap& batch, const torch::Tensor& previous_signature,
                                     const bool should_decode_map) {
  torch::Tensor observations, observation_mask, temporal_tokens;
  TensorMap summaries;
  std::tie(observations, observation_mask, temporal_tokens, summaries) = encode_observations(batch);
  const auto observation_latents = observation_resampler(observations, observation_mask);
  torch::Tensor quality;
  if (contains(batch, "quality")) {
    quality = required(batch, "quality");
    if (quality.size(-1) < 16) {
      quality = torch::cat({quality, torch::zeros({quality.size(0), 16 - quality.size(-1)}, quality.options())}, -1);
    }
    quality = quality.index({Slice(), Slice(0, 16)});
  }
  const auto updated_signature = signature_updater->forward(
      observation_latents, {}, quality, previous_signature);
  const auto query_latents = query->forward(required(batch, "query_tokens"), required(batch, "query_mask"));
  torch::Tensor shifted_temporal;
  if (physics_head && temporal_tokens.defined() && required(batch, "compiled_temporal").size(1) > 1) {
    const auto shifted = required(batch, "compiled_temporal").index({Slice(), Slice(1, torch::indexing::None)});
    const auto shifted_mask = required(batch, "temporal_mask").index({Slice(), Slice(1, 1 + shifted.size(1))});
    shifted_temporal = temporal(shifted * shifted_mask.unsqueeze(-1).to(shifted.scalar_type()));
  }
  auto output = decode(updated_signature, observation_latents, query_latents, temporal_tokens,
                       should_decode_map, shifted_temporal);
  output.modality_summaries = std::move(summaries);
  return output;
}

ModelOutput NeuroTaskFMImpl::decode_signature(const torch::Tensor& signature,
                                              const torch::Tensor& query_tokens,
                                              const torch::Tensor& query_mask,
                                              const bool should_decode_map) {
  const auto query_latents = query->forward(query_tokens, query_mask);
  const auto null = null_observation.expand({signature.size(0), -1, -1});
  return decode(signature, null, query_latents, {}, should_decode_map);
}

void NeuroTaskFMImpl::enable_clinical_adapters(const std::int64_t rank,
                                               const std::int64_t upper_layers,
                                               const std::int64_t concepts) {
  const auto start = std::max<std::int64_t>(0, config.num_layers - upper_layers);
  for (auto index = start; index < config.num_layers; ++index) {
    auto adapter = torch::nn::Sequential(RMSNorm(config.hidden_size),
        torch::nn::Linear(torch::nn::LinearOptions(config.hidden_size, rank).bias(false)),
        torch::nn::SiLU(),
        torch::nn::Linear(torch::nn::LinearOptions(rank, config.hidden_size).bias(false)));
    adapters.emplace(index, register_module("adapter_" + std::to_string(index), adapter));
  }
  if (concepts > 0) {
    concept_head = register_module("concept_head", torch::nn::Sequential(
        RMSNorm(config.hidden_size), torch::nn::Linear(config.hidden_size, concepts)));
  }
}

void NeuroTaskFMImpl::freeze_shared_backbone() {
  for (auto& item : named_parameters()) {
    const auto& name = item.key();
    const auto trainable = name.rfind("signature_updater.", 0) == 0 ||
                           name.rfind("behavior.", 0) == 0 ||
                           name.rfind("quality_head.", 0) == 0 ||
                           name.rfind("clinical.", 0) == 0 ||
                           name.rfind("trajectory.", 0) == 0 ||
                           name.rfind("state.", 0) == 0 ||
                           name.rfind("adapter_", 0) == 0 ||
                           name.rfind("concept_head.", 0) == 0;
    item.value().set_requires_grad(trainable);
  }
}

}  // namespace neurotaskfm
