#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "neurotaskfm/model.h"

namespace neurotaskfm {

struct LossResult {
  torch::Tensor total;
  TensorMap metrics;
};

torch::Tensor gaussian_nll(const torch::Tensor& mean, const torch::Tensor& log_variance,
                           const torch::Tensor& target, const torch::Tensor& mask = {});
torch::Tensor correlation_loss(const torch::Tensor& prediction, const torch::Tensor& target);
torch::Tensor temporal_future_loss(const torch::Tensor& prediction, const torch::Tensor& target,
                                   const torch::Tensor& mask, const std::vector<std::int64_t>& offsets);
bool physics_enabled(const YAML::Node& weights);
LossResult physics_informed_loss(const ModelOutput& output, const TensorMap& targets,
                                 const YAML::Node& physics_config, const TensorMap& context = {});
LossResult compute_losses(const ModelOutput& output, const TensorMap& targets,
                          const YAML::Node& weights, const TensorMap& context = {});

}  // namespace neurotaskfm
