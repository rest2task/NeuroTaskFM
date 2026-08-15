#include "neurotaskfm/model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

using torch::indexing::Slice;

namespace neurotaskfm {
namespace {

double inverse_bound(const double value, const double low, const double high) {
  const auto fraction = std::clamp((value - low) / (high - low), 1e-5, 1.0 - 1e-5);
  return std::log(fraction / (1.0 - fraction));
}

torch::Tensor bounded(const torch::Tensor& raw, const double low, const double high) {
  return low + (high - low) * torch::sigmoid(raw);
}

constexpr std::array<const char*, 8> kParameterNames{
    "neural_tau", "coupling", "signal_decay", "flow_feedback", "transit_time",
    "grubb_alpha", "oxygen_extraction", "resting_blood_volume"};
constexpr std::array<std::pair<double, double>, 8> kParameterBounds{{
    {0.05, 2.00}, {0.00, 1.50}, {0.40, 1.00}, {0.10, 0.50},
    {0.50, 3.00}, {0.20, 0.50}, {0.20, 0.60}, {0.01, 0.08}}};

}  // namespace

DistributionHeadImpl::DistributionHeadImpl(const std::int64_t dim, const std::int64_t outputs) {
  norm = register_module("norm", RMSNorm(dim));
  hidden = register_module("hidden", torch::nn::Linear(dim, dim / 2));
  mean = register_module("mean", torch::nn::Linear(dim / 2, outputs));
  log_variance = register_module("log_variance", torch::nn::Linear(dim / 2, outputs));
}

std::pair<torch::Tensor, torch::Tensor> DistributionHeadImpl::forward(const torch::Tensor& input) {
  const auto encoded = torch::silu(hidden(norm(input)));
  return {mean(encoded), log_variance(encoded).clamp(-8.0, 6.0)};
}

MapDecoderImpl::MapDecoderImpl(const std::int64_t dim, const std::vector<std::int64_t>& shape)
    : shape(shape) {
  if (shape.size() != 3) throw std::invalid_argument("map shape must contain three dimensions");
  constexpr std::int64_t base = 256 * 6 * 7 * 6;
  template_norm = register_module("template_norm", RMSNorm(dim));
  residual_norm = register_module("residual_norm", RMSNorm(dim));
  template_projection = register_module("template_projection", torch::nn::Linear(dim, base));
  residual_projection = register_module("residual_projection", torch::nn::Linear(dim, base));
  up = register_module("up", torch::nn::Sequential());
  const std::array<std::int64_t, 5> channels{256, 192, 128, 64, 32};
  for (std::size_t index = 0; index + 1 < channels.size(); ++index) {
    up->push_back(torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(
        channels[index], channels[index + 1], 4).stride(2).padding(1)));
    up->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(8, channels[index + 1])));
    up->push_back(torch::nn::SiLU());
  }
  template_output = register_module("template_output", torch::nn::Conv3d(
      torch::nn::Conv3dOptions(32, 1, 3).padding(1)));
  residual_output = register_module("residual_output", torch::nn::Conv3d(
      torch::nn::Conv3dOptions(32, 1, 3).padding(1)));
  log_variance_output = register_module("log_variance_output", torch::nn::Conv3d(
      torch::nn::Conv3dOptions(32, 1, 3).padding(1)));
  residual_scale = register_parameter("residual_scale", torch::tensor(-1.5));
}

torch::Tensor MapDecoderImpl::resize(const torch::Tensor& input) const {
  auto result = input.index({Slice(), Slice(), Slice(0, shape[0]), Slice(0, shape[1]), Slice(0, shape[2])});
  if (result.size(2) != shape[0] || result.size(3) != shape[1] || result.size(4) != shape[2]) {
    result = torch::nn::functional::interpolate(result,
        torch::nn::functional::InterpolateFuncOptions().size(shape).mode(torch::kTrilinear).align_corners(false));
  }
  return result.select(1, 0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MapDecoderImpl::forward(
    const torch::Tensor& subject, const torch::Tensor& task_query) {
  auto template_features = up->forward(torch::silu(template_projection(template_norm(task_query)))
                                           .view({task_query.size(0), 256, 6, 7, 6}));
  auto residual_features = up->forward(torch::silu(residual_projection(residual_norm(subject)))
                                           .view({subject.size(0), 256, 6, 7, 6}));
  auto template_map = resize(template_output(template_features));
  auto residual_map = resize(residual_output(residual_features)) * torch::softplus(residual_scale);
  auto log_variance = resize(log_variance_output(residual_features)).clamp(-8.0, 6.0);
  return {template_map + residual_map, log_variance, template_map, residual_map};
}

DynamicStateHeadImpl::DynamicStateHeadImpl(const std::int64_t dim, const std::int64_t states,
                                           const std::int64_t heads, const std::int64_t head_dim) {
  cross = register_module("cross", CrossAttention(dim, heads, head_dim));
  norm = register_module("norm", RMSNorm(dim));
  output = register_module("output", torch::nn::Linear(dim, states));
}

torch::Tensor DynamicStateHeadImpl::forward(const torch::Tensor& temporal,
                                            const torch::Tensor& context,
                                            const torch::Tensor& context_mask) {
  return output(temporal + cross(norm(temporal), context, context_mask));
}

FutureWindowHeadImpl::FutureWindowHeadImpl(const std::int64_t dim, const std::int64_t channels,
                                           std::vector<std::int64_t> offsets)
    : channels(channels), offsets(std::move(offsets)) {
  norm = register_module("norm", RMSNorm(dim));
  hidden = register_module("hidden", torch::nn::Linear(dim, dim / 2));
  output = register_module("output", torch::nn::Linear(dim / 2, channels * this->offsets.size()));
}

torch::Tensor FutureWindowHeadImpl::forward(const torch::Tensor& temporal) {
  return output(torch::silu(hidden(norm(temporal))))
      .view({temporal.size(0), temporal.size(1), static_cast<std::int64_t>(offsets.size()), channels});
}

PhysicsHeadImpl::PhysicsHeadImpl(const std::int64_t dim, const std::int64_t regions)
    : regions(regions) {
  if (regions <= 0) throw std::invalid_argument("physics regions must be positive");
  const auto bottleneck = std::max<std::int64_t>(64, std::min<std::int64_t>(dim / 4, 512));
  state_projection = register_module("state_projection", torch::nn::Sequential(
      RMSNorm(dim), torch::nn::Linear(dim, bottleneck), torch::nn::SiLU(),
      torch::nn::Linear(bottleneck, regions * 5)));
  parameter_projection = register_module("parameter_projection", torch::nn::Sequential(
      RMSNorm(dim), torch::nn::Linear(dim, bottleneck), torch::nn::SiLU(),
      torch::nn::Linear(bottleneck, regions * 8)));

  torch::NoGradGuard guard;
  auto state_last = state_projection->ptr<torch::nn::LinearImpl>(3);
  torch::nn::init::normal_(state_last->weight, 0.0, 1e-3);
  const std::array<double, 5> state_bias{
      0.0, 0.0, inverse_bound(1.0, 0.2, 3.0), inverse_bound(1.0, 0.2, 3.0),
      inverse_bound(1.0, 0.2, 3.0)};
  state_last->bias.copy_(torch::tensor(std::vector<double>(state_bias.begin(), state_bias.end())).repeat({regions}));
  auto parameter_last = parameter_projection->ptr<torch::nn::LinearImpl>(3);
  torch::nn::init::normal_(parameter_last->weight, 0.0, 1e-3);
  const std::array<double, 8> typical{0.80, 0.20, 0.65, 0.41, 0.98, 0.32, 0.34, 0.02};
  std::vector<double> parameter_bias;
  parameter_bias.reserve(8);
  for (std::size_t index = 0; index < typical.size(); ++index) {
    parameter_bias.push_back(inverse_bound(typical[index], kParameterBounds[index].first,
                                          kParameterBounds[index].second));
  }
  parameter_last->bias.copy_(torch::tensor(parameter_bias).repeat({regions}));
}

TensorMap PhysicsHeadImpl::forward(const torch::Tensor& temporal, const torch::Tensor& subject) {
  auto states = state_projection->forward(temporal)
                    .view({temporal.size(0), temporal.size(1), regions, 5});
  auto parameters = parameter_projection->forward(subject).view({subject.size(0), regions, 8});
  TensorMap result;
  result["neural"] = torch::tanh(states.select(-1, 0));
  result["vasoactive_signal"] = 1.5 * torch::tanh(states.select(-1, 1));
  result["flow"] = bounded(states.select(-1, 2), 0.20, 3.00);
  result["volume"] = bounded(states.select(-1, 3), 0.20, 3.00);
  result["deoxyhemoglobin"] = bounded(states.select(-1, 4), 0.20, 3.00);
  for (std::size_t index = 0; index < kParameterNames.size(); ++index) {
    result[kParameterNames[index]] = bounded(parameters.select(-1, static_cast<std::int64_t>(index)),
                                             kParameterBounds[index].first,
                                             kParameterBounds[index].second);
  }
  return result;
}

}  // namespace neurotaskfm
