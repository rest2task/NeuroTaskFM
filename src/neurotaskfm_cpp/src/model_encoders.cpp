#include "neurotaskfm/model.h"

#include <cmath>
#include <stdexcept>

using torch::indexing::Slice;

namespace neurotaskfm {

DicomEncoderImpl::DicomEncoderImpl(const std::int64_t vocabulary, const std::int64_t dim,
                                   const std::int64_t max_tokens)
    : max_tokens(max_tokens) {
  embedding = register_module("embedding", torch::nn::Embedding(vocabulary, dim));
  position = register_module("position", torch::nn::Embedding(max_tokens, dim));
  norm = register_module("norm", RMSNorm(dim));
}

torch::Tensor DicomEncoderImpl::forward(const torch::Tensor& input_tokens) {
  auto tokens = input_tokens.index({Slice(), Slice(0, max_tokens)});
  auto positions = torch::arange(tokens.size(1), tokens.options());
  const auto bounded = tokens.clamp(0, embedding->options.num_embeddings() - 1);
  return norm(embedding(bounded) + position(positions).unsqueeze(0));
}

SliceEncoderImpl::SliceEncoderImpl(const std::int64_t dim, const std::int64_t patch_size,
                                   const std::int64_t tokens_per_slice,
                                   const std::int64_t max_slices)
    : tokens_per_slice(tokens_per_slice),
      side(static_cast<std::int64_t>(std::sqrt(tokens_per_slice))),
      max_slices(max_slices) {
  if (side * side != tokens_per_slice) throw std::invalid_argument("tokens_per_slice must be a square");
  patch = register_module("patch", torch::nn::Conv2d(
      torch::nn::Conv2dOptions(1, dim, patch_size).stride(patch_size).bias(false)));
  pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(
      torch::nn::AdaptiveAvgPool2dOptions({side, side})));
  type_embedding = register_module("type_embedding", torch::nn::Embedding(4, dim));
  index_embedding = register_module("index_embedding", torch::nn::Embedding(max_slices, dim));
  norm = register_module("norm", RMSNorm(dim));
}

std::pair<torch::Tensor, torch::Tensor> SliceEncoderImpl::forward(
    const torch::Tensor& input_images, const torch::Tensor& input_types,
    const torch::Tensor& input_mask) {
  const auto images = input_images.index({Slice(), Slice(0, max_slices)});
  const auto types = input_types.index({Slice(), Slice(0, max_slices)});
  const auto mask = input_mask.index({Slice(), Slice(0, max_slices)});
  const auto batch = images.size(0);
  const auto count = images.size(1);
  auto encoded = pool(patch(images.reshape({batch * count, images.size(2), images.size(3), images.size(4)})))
                     .flatten(2).transpose(1, 2)
                     .reshape({batch, count * tokens_per_slice, -1});
  auto type_tokens = type_embedding(types.clamp(0, type_embedding->options.num_embeddings() - 1))
                         .unsqueeze(2).expand({-1, -1, tokens_per_slice, -1}).reshape_as(encoded);
  auto indices = torch::arange(count, types.options());
  auto index_tokens = index_embedding(indices).unsqueeze(0).unsqueeze(2)
                          .expand({batch, -1, tokens_per_slice, -1}).reshape_as(encoded);
  auto output_mask = mask.unsqueeze(-1).expand({-1, -1, tokens_per_slice}).reshape({batch, -1});
  return {norm(encoded + type_tokens + index_tokens), output_mask};
}

FrameEncoderImpl::FrameEncoderImpl(const std::int64_t dim, const std::int64_t patch_size,
                                   const std::int64_t tokens_per_frame,
                                   const std::int64_t max_frames)
    : tokens_per_frame(tokens_per_frame),
      side(static_cast<std::int64_t>(std::sqrt(tokens_per_frame))),
      max_frames(max_frames) {
  if (side * side != tokens_per_frame) throw std::invalid_argument("tokens_per_frame must be a square");
  patch = register_module("patch", torch::nn::Conv2d(
      torch::nn::Conv2dOptions(1, dim, patch_size).stride(patch_size).bias(false)));
  pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(
      torch::nn::AdaptiveAvgPool2dOptions({side, side})));
  modality_embedding = register_module("modality_embedding", torch::nn::Embedding(16, dim));
  view_embedding = register_module("view_embedding", torch::nn::Embedding(8, dim));
  position_embedding = register_module("position_embedding", torch::nn::Embedding(max_frames, dim));
  norm = register_module("norm", RMSNorm(dim));
}

std::pair<torch::Tensor, torch::Tensor> FrameEncoderImpl::forward(
    const torch::Tensor& input_frames, const torch::Tensor& input_modalities,
    const torch::Tensor& input_views, const torch::Tensor& input_positions,
    const torch::Tensor& input_mask) {
  const auto frames = input_frames.index({Slice(), Slice(0, max_frames)});
  const auto modalities = input_modalities.index({Slice(), Slice(0, max_frames)});
  const auto views = input_views.index({Slice(), Slice(0, max_frames)});
  const auto positions = input_positions.index({Slice(), Slice(0, max_frames)});
  const auto mask = input_mask.index({Slice(), Slice(0, max_frames)});
  const auto batch = frames.size(0);
  const auto count = frames.size(1);
  auto encoded = pool(patch(frames.reshape({batch * count, frames.size(2), frames.size(3), frames.size(4)})))
                     .flatten(2).transpose(1, 2)
                     .reshape({batch, count, tokens_per_frame, -1});
  auto modality = modality_embedding(modalities.clamp(0, 15)).unsqueeze(2);
  auto view = view_embedding(views.clamp(0, 7)).unsqueeze(2);
  auto position = position_embedding(positions.clamp(0, max_frames - 1)).unsqueeze(2);
  encoded = norm(encoded + modality + view + position).reshape({batch, count * tokens_per_frame, -1});
  auto output_mask = mask.unsqueeze(-1).expand({-1, -1, tokens_per_frame}).reshape({batch, -1});
  return {encoded, output_mask};
}

CompiledEncoderImpl::CompiledEncoderImpl(const std::int64_t channels, const std::int64_t dim,
                                         const std::int64_t max_tokens)
    : max_tokens(max_tokens) {
  projection = register_module("projection", torch::nn::Linear(torch::nn::LinearOptions(channels, dim).bias(false)));
  position = register_module("position", torch::nn::Embedding(max_tokens, dim));
  type_embedding = register_parameter("type_embedding", torch::randn({1, 1, dim}) * 0.02);
  norm = register_module("norm", RMSNorm(dim));
}

torch::Tensor CompiledEncoderImpl::forward(const torch::Tensor& input) {
  const auto bounded = input.index({Slice(), Slice(0, max_tokens)});
  const auto positions = torch::arange(bounded.size(1), input.options().dtype(torch::kLong));
  return norm(projection(bounded) + position(positions).unsqueeze(0) + type_embedding);
}

VolumeEncoderImpl::VolumeEncoderImpl(const std::int64_t dim,
                                     const std::vector<std::int64_t>& patch,
                                     const std::int64_t max_tokens)
    : max_tokens(max_tokens) {
  if (patch.size() != 3) throw std::invalid_argument("volume patch must have three dimensions");
  convolution = register_module("convolution", torch::nn::Conv3d(
      torch::nn::Conv3dOptions(1, dim, patch).stride(patch).bias(false)));
  norm = register_module("norm", RMSNorm(dim));
}

std::pair<torch::Tensor, torch::Tensor> VolumeEncoderImpl::forward(
    const torch::Tensor& input_volume, const torch::Tensor& input_mask) {
  auto volume = input_volume.dim() == 5 ? input_volume.unsqueeze(1) : input_volume;
  if (volume.dim() != 6 || volume.size(2) != 1) {
    throw std::invalid_argument("volume must have shape BxNx1xDxHxW or Bx1xDxHxW");
  }
  const auto batch = volume.size(0);
  const auto count = volume.size(1);
  auto encoded = convolution(volume.reshape({batch * count, 1, volume.size(3), volume.size(4), volume.size(5)}))
                     .flatten(2).transpose(1, 2)
                     .index({Slice(), Slice(0, max_tokens)});
  const auto tokens = encoded.size(1);
  encoded = norm(encoded).reshape({batch, count * tokens, -1});
  auto mask = input_mask.defined()
                  ? input_mask.index({Slice(), Slice(0, count)}).to(encoded.device(), torch::kBool)
                  : torch::ones({batch, count}, encoded.options().dtype(torch::kBool));
  return {encoded, mask.unsqueeze(-1).expand({-1, -1, tokens}).reshape({batch, -1})};
}

PerceiverResamplerImpl::PerceiverResamplerImpl(const std::int64_t dim,
                                               const std::int64_t latent_count,
                                               const std::int64_t heads,
                                               const std::int64_t head_dim,
                                               const std::int64_t depth) {
  latents = register_parameter("latents", torch::randn({1, latent_count, dim}) * 0.02);
  cross.reserve(static_cast<std::size_t>(depth));
  norms.reserve(static_cast<std::size_t>(depth));
  for (std::int64_t index = 0; index < depth; ++index) {
    cross.push_back(register_module("cross_" + std::to_string(index), CrossAttention(dim, heads, head_dim)));
    norms.push_back(register_module("norm_" + std::to_string(index), RMSNorm(dim)));
  }
}

torch::Tensor PerceiverResamplerImpl::forward(const torch::Tensor& context,
                                              const torch::Tensor& mask) {
  auto output = latents.expand({context.size(0), -1, -1});
  for (std::size_t index = 0; index < cross.size(); ++index) {
    output = output + cross[index]->forward(norms[index](output), context, mask);
  }
  return output;
}

QueryEncoderImpl::QueryEncoderImpl(const std::int64_t vocabulary, const std::int64_t dim,
                                   const std::int64_t max_tokens, const std::int64_t latents,
                                   const std::int64_t heads, const std::int64_t head_dim)
    : max_tokens(max_tokens) {
  token_encoder = register_module("token_encoder", DicomEncoder(vocabulary, dim, max_tokens));
  resampler = register_module("resampler", PerceiverResampler(dim, latents, heads, head_dim, 2));
}

torch::Tensor QueryEncoderImpl::forward(const torch::Tensor& tokens, const torch::Tensor& mask) {
  return resampler(token_encoder(tokens.index({Slice(), Slice(0, max_tokens)})),
                   mask.index({Slice(), Slice(0, max_tokens)}));
}

TabularEncoderImpl::TabularEncoderImpl(const std::int64_t dim, const std::int64_t features)
    : max_features(features) {
  feature_embedding = register_parameter("feature_embedding", torch::randn({features, dim}) * 0.02);
  value_projection = register_module("value_projection", torch::nn::Sequential(
      torch::nn::Linear(2, dim), torch::nn::SiLU(), torch::nn::Linear(dim, dim)));
  norm = register_module("norm", RMSNorm(dim));
}

std::pair<torch::Tensor, torch::Tensor> TabularEncoderImpl::forward(
    const torch::Tensor& input_values, const torch::Tensor& input_mask) {
  const auto count = std::min<std::int64_t>(input_values.size(1), max_features);
  const auto values = input_values.index({Slice(), Slice(0, count)});
  const auto mask = input_mask.index({Slice(), Slice(0, count)});
  const auto pairs = torch::stack({values, mask.to(values.scalar_type())}, -1);
  return {norm(value_projection->forward(pairs) + feature_embedding.index({Slice(0, count)}).unsqueeze(0)), mask};
}

NeuroSignatureUpdaterImpl::NeuroSignatureUpdaterImpl(
    const std::int64_t dim, const std::vector<std::int64_t>& partitions,
    const std::int64_t heads, const std::int64_t head_dim, const std::int64_t quality_dim) {
  token_count = 0;
  std::vector<std::int64_t> identifiers;
  for (std::size_t type = 0; type < partitions.size(); ++type) {
    token_count += partitions[type];
    identifiers.insert(identifiers.end(), static_cast<std::size_t>(partitions[type]), static_cast<std::int64_t>(type));
  }
  base = register_parameter("base", torch::randn({1, token_count, dim}) * 0.02);
  type_ids = register_buffer("type_ids", torch::tensor(identifiers, torch::TensorOptions().dtype(torch::kLong)));
  type_embedding = register_module("type_embedding", torch::nn::Embedding(6, dim));
  cross = register_module("cross", CrossAttention(dim, heads, head_dim));
  norm = register_module("norm", RMSNorm(dim));
  quality_projection = register_module("quality_projection", torch::nn::Sequential(
      torch::nn::Linear(quality_dim, dim), torch::nn::SiLU(), torch::nn::Linear(dim, dim)));
  gate = register_module("gate", torch::nn::Linear(dim * 3, dim));
}

torch::Tensor NeuroSignatureUpdaterImpl::forward(
    const torch::Tensor& observations, const torch::Tensor& observation_mask,
    const torch::Tensor& quality, const torch::Tensor& previous) {
  const auto initial = base.expand({observations.size(0), -1, -1}) + type_embedding(type_ids).unsqueeze(0);
  const auto prior = previous.defined() ? previous : initial;
  const auto candidate = prior + cross(norm(prior), observations, observation_mask);
  const auto quality_embedding = quality.defined()
      ? quality_projection->forward(quality).unsqueeze(1).expand_as(prior)
      : torch::zeros_like(prior);
  const auto update_gate = torch::sigmoid(gate(torch::cat({prior, candidate, quality_embedding}, -1)));
  return norm(update_gate * candidate + (1.0 - update_gate) * prior);
}

}  // namespace neurotaskfm
