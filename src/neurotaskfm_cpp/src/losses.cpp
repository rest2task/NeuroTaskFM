#include "neurotaskfm/losses.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

using torch::indexing::Slice;

namespace neurotaskfm {
namespace {

constexpr double kDynamicsWeight = 0.15;
constexpr double kBoldWeight = 0.08;
constexpr double kStabilityWeight = 0.05;
constexpr double kEquivarianceWeight = 0.04;

const torch::Tensor* find(const TensorMap& values, const std::string& key) {
  const auto item = values.find(key);
  return item == values.end() || !item->second.defined() ? nullptr : &item->second;
}

const torch::Tensor& require(const TensorMap& values, const std::string& key,
                             const std::string& label) {
  const auto* value = find(values, key);
  if (value == nullptr) throw std::invalid_argument(label + " requires tensor '" + key + "'");
  return *value;
}

double weight(const YAML::Node& weights, const std::string& name, const double fallback = 1.0) {
  return weights && weights[name] ? weights[name].as<double>() : fallback;
}

bool term_enabled(const YAML::Node& config, const std::string& name) {
  return !config["terms"] || !config["terms"][name] || config["terms"][name].as<bool>();
}

torch::Tensor masked_mean(const torch::Tensor& input, const torch::Tensor& mask) {
  auto valid = mask.to(input.device(), torch::kBool);
  if (valid.sizes() != input.sizes()) valid = torch::broadcast_to(valid, input.sizes());
  const auto safe = torch::where(valid, input, torch::zeros_like(input));
  return safe.sum() / valid.sum().to(input.scalar_type()).clamp_min(1.0);
}

torch::Tensor masked_smooth_l1(const torch::Tensor& residual, const torch::Tensor& mask) {
  const auto elementwise = torch::nn::functional::smooth_l1_loss(
      residual, torch::zeros_like(residual),
      torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone).beta(0.1));
  return masked_mean(elementwise, mask);
}

torch::Tensor normalized_adjacency(const torch::Tensor& input, const std::int64_t batch,
                                   const std::int64_t regions, const torch::Device& device) {
  auto graph = input.to(device, torch::kFloat32);
  if (graph.dim() == 2) graph = graph.unsqueeze(0).expand({batch, -1, -1});
  if (graph.size(0) != batch || graph.size(1) != regions || graph.size(2) != regions) {
    throw std::invalid_argument("physics anatomy graph has an incompatible shape");
  }
  graph = 0.5 * (graph + graph.transpose(-1, -2));
  const auto eye = torch::eye(regions, graph.options().dtype(torch::kBool)).unsqueeze(0);
  graph = graph.masked_fill(eye, 0.0);
  const auto inverse_sqrt = graph.sum(-1).clamp_min(1e-8).rsqrt();
  return graph * inverse_sqrt.unsqueeze(-1) * inverse_sqrt.unsqueeze(-2);
}

torch::Tensor centered(const torch::Tensor& input, const torch::Tensor& mask) {
  const auto weights = mask.to(input.scalar_type());
  const auto mean = (torch::where(mask, input, torch::zeros_like(input)) * weights).sum(1, true) /
                    weights.sum(1, true).clamp_min(1.0);
  return input - mean;
}

void validate_physics_weights(const YAML::Node& config) {
  const std::array<std::pair<const char*, double>, 4> locked{{
      {"lambda_dyn", kDynamicsWeight}, {"lambda_bold", kBoldWeight},
      {"lambda_stab", kStabilityWeight}, {"lambda_equiv", kEquivarianceWeight}}};
  for (const auto& [name, expected] : locked) {
    if (config[name] && std::abs(config[name].as<double>() - expected) > 1e-12) {
      throw std::invalid_argument(std::string("loss.physics.") + name + " is locked");
    }
  }
}

}  // namespace

torch::Tensor gaussian_nll(const torch::Tensor& mean, const torch::Tensor& log_variance,
                           const torch::Tensor& target, const torch::Tensor& mask) {
  const auto finite = torch::isfinite(target);
  const auto valid = mask.defined() ? finite & mask.to(torch::kBool) : finite;
  const auto safe_target = torch::where(valid, target, torch::zeros_like(target));
  const auto loss = 0.5 * (log_variance + (safe_target - mean).square() * torch::exp(-log_variance));
  return masked_mean(loss, valid);
}

torch::Tensor correlation_loss(const torch::Tensor& prediction, const torch::Tensor& target) {
  const auto flat_prediction = prediction.flatten(1).to(torch::kFloat32);
  const auto flat_target = target.flatten(1).to(torch::kFloat32);
  const auto centered_prediction = flat_prediction - flat_prediction.mean(1, true);
  const auto centered_target = flat_target - flat_target.mean(1, true);
  const auto numerator = (centered_prediction * centered_target).sum(1);
  const auto denominator = centered_prediction.square().sum(1).sqrt() *
                           centered_target.square().sum(1).sqrt();
  return 1.0 - (numerator / denominator.clamp_min(1e-8)).mean();
}

torch::Tensor temporal_future_loss(const torch::Tensor& prediction, const torch::Tensor& target,
                                   const torch::Tensor& mask,
                                   const std::vector<std::int64_t>& offsets) {
  std::vector<torch::Tensor> losses;
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    const auto offset = offsets[index];
    if (offset >= target.size(1)) continue;
    const auto available = std::min(prediction.size(1), target.size(1) - offset);
    const auto predicted = prediction.index({Slice(), Slice(0, available), static_cast<std::int64_t>(index)});
    const auto expected = target.index({Slice(), Slice(offset, offset + available)}).to(predicted.scalar_type());
    const auto valid = mask.index({Slice(), Slice(offset, offset + available)}).unsqueeze(-1);
    losses.push_back(masked_smooth_l1(predicted - expected, valid));
  }
  return losses.empty() ? torch::zeros({}, prediction.options()) : torch::stack(losses).mean();
}

bool physics_enabled(const YAML::Node& weights) {
  if (!weights || !weights["physics"]) return false;
  const auto physics = weights["physics"];
  if (physics.IsScalar()) return physics.as<bool>();
  validate_physics_weights(physics);
  return !physics["enabled"] || physics["enabled"].as<bool>();
}

LossResult physics_informed_loss(const ModelOutput& output, const TensorMap& targets,
                                 const YAML::Node& config, const TensorMap& context) {
  validate_physics_weights(config);
  if (output.physics.empty()) throw std::invalid_argument("physics output is missing");
  const auto neural = require(output.physics, "neural", "physics loss").to(torch::kFloat32);
  if (neural.dim() != 3 || neural.size(1) < 2) {
    throw std::invalid_argument("physics neural state must be BxTxR with at least two time points");
  }
  const auto batch = neural.size(0);
  const auto length = neural.size(1);
  const auto regions = neural.size(2);
  auto temporal_mask = find(targets, "physics_temporal_mask");
  if (temporal_mask == nullptr) temporal_mask = find(context, "temporal_mask");
  auto mask = temporal_mask == nullptr
      ? torch::ones({batch, length, regions}, neural.options().dtype(torch::kBool))
      : temporal_mask->index({Slice(), Slice(0, length)}).to(neural.device(), torch::kBool)
            .unsqueeze(-1).expand({batch, length, regions});
  if (const auto* node_mask = find(targets, "physics_mask")) {
    auto local = node_mask->to(neural.device(), torch::kBool);
    if (local.dim() == 2 && local.size(1) == length) local = local.unsqueeze(-1);
    if (local.dim() == 2 && local.size(1) == regions) local = local.unsqueeze(1);
    mask = mask & local.expand({batch, length, regions});
  }
  const auto pair_mask = mask.index({Slice(), Slice(0, length - 1)}) &
                         mask.index({Slice(), Slice(1, length)});
  auto dt = find(targets, "physics_dt");
  auto step = dt == nullptr ? torch::ones({batch, 1, 1}, neural.options())
                            : dt->to(neural.device(), torch::kFloat32).reshape({-1, 1, 1});
  if (step.size(0) == 1) step = step.expand({batch, 1, 1});
  const auto* anatomy = find(targets, "physics_anatomy");
  if (anatomy == nullptr) anatomy = find(context, "anatomy_graph");
  const auto use_anatomy = term_enabled(config, "anatomy");
  if (use_anatomy && anatomy == nullptr) throw std::invalid_argument("physics anatomy graph is missing");
  const auto adjacency = use_anatomy
      ? normalized_adjacency(*anatomy, batch, regions, neural.device())
      : torch::zeros({batch, regions, regions}, neural.options());
  const auto graph_drive = use_anatomy
      ? torch::einsum("bij,btj->bti", {adjacency, torch::tanh(neural)})
      : torch::zeros_like(neural);
  auto drive = find(targets, "physics_drive");
  const auto external_drive = drive == nullptr ? torch::zeros_like(neural)
                                                : drive->index({Slice(), Slice(0, length)}).to(neural.device(), torch::kFloat32);
  const auto tau = require(output.physics, "neural_tau", "physics dynamics").to(torch::kFloat32).unsqueeze(1);
  const auto coupling = require(output.physics, "coupling", "physics dynamics").to(torch::kFloat32).unsqueeze(1);
  const auto neural_rhs = -neural / tau + external_drive + coupling * graph_drive;
  const auto derivative = (neural.index({Slice(), Slice(1, length)}) -
                           neural.index({Slice(), Slice(0, length - 1)})) / step;
  const auto dynamics_residual = derivative - neural_rhs.index({Slice(), Slice(0, length - 1)});
  const auto zero = torch::zeros({}, neural.options());
  const auto dynamics = term_enabled(config, "dynamics")
      ? masked_smooth_l1(dynamics_residual, pair_mask) : zero;

  torch::Tensor bold = zero;
  torch::Tensor hemodynamic = zero;
  torch::Tensor observation = zero;
  if (term_enabled(config, "hemodynamics")) {
    const auto signal = require(output.physics, "vasoactive_signal", "hemodynamics").to(torch::kFloat32);
    const auto flow = require(output.physics, "flow", "hemodynamics").to(torch::kFloat32).clamp_min(1e-4);
    const auto volume = require(output.physics, "volume", "hemodynamics").to(torch::kFloat32).clamp_min(1e-4);
    const auto deoxy = require(output.physics, "deoxyhemoglobin", "hemodynamics").to(torch::kFloat32).clamp_min(1e-4);
    const auto decay = require(output.physics, "signal_decay", "hemodynamics").to(torch::kFloat32).unsqueeze(1);
    const auto feedback = require(output.physics, "flow_feedback", "hemodynamics").to(torch::kFloat32).unsqueeze(1);
    const auto transit = require(output.physics, "transit_time", "hemodynamics").to(torch::kFloat32).unsqueeze(1);
    const auto alpha = require(output.physics, "grubb_alpha", "hemodynamics").to(torch::kFloat32).unsqueeze(1);
    const auto extraction0 = require(output.physics, "oxygen_extraction", "hemodynamics").to(torch::kFloat32).unsqueeze(1);
    const auto resting_volume = require(output.physics, "resting_blood_volume", "hemodynamics").to(torch::kFloat32).unsqueeze(1);
    const auto outflow = torch::pow(volume, 1.0 / alpha);
    const auto extraction = 1.0 - torch::pow(1.0 - extraction0, 1.0 / flow);
    std::vector<torch::Tensor> expected{
        neural - decay * signal - feedback * (flow - 1.0), signal,
        (flow - outflow) / transit,
        (flow * extraction / extraction0 - deoxy * torch::pow(volume, 1.0 / alpha - 1.0)) / transit};
    std::vector<torch::Tensor> states{signal, flow, volume, deoxy};
    std::vector<torch::Tensor> residuals;
    for (std::size_t index = 0; index < states.size(); ++index) {
      residuals.push_back((states[index].index({Slice(), Slice(1, length)}) -
                           states[index].index({Slice(), Slice(0, length - 1)})) / step -
                          expected[index].index({Slice(), Slice(0, length - 1)}));
    }
    hemodynamic = masked_smooth_l1(torch::stack(residuals, -1), pair_mask.unsqueeze(-1));
    auto observed = find(targets, "physics_bold");
    if (observed == nullptr) observed = find(context, "bold_observation");
    if (observed == nullptr) throw std::invalid_argument("physics BOLD observation is missing");
    auto measured = observed->index({Slice(), Slice(0, length)}).to(neural.device(), torch::kFloat32);
    auto predicted = resting_volume * (7.0 * extraction0 * (1.0 - deoxy) +
        2.0 * (1.0 - deoxy / volume) + (2.0 * extraction0 - 0.2) * (1.0 - volume));
    const auto observation_mask = mask & torch::isfinite(measured);
    measured = torch::nan_to_num(measured);
    if (!config["demean_bold"] || config["demean_bold"].as<bool>()) {
      measured = centered(measured, observation_mask);
      predicted = centered(predicted, observation_mask);
    }
    observation = masked_smooth_l1(predicted - measured, observation_mask);
    bold = hemodynamic + observation;
  }

  auto stability = zero;
  if (term_enabled(config, "stability")) {
    const auto margin = config["stability_margin"] ? config["stability_margin"].as<double>() : 0.02;
    const auto violation = torch::relu(coupling.squeeze(1) - tau.squeeze(1).reciprocal() + margin).square();
    stability = masked_mean(violation, mask.any(1));
  }
  auto equivariance = zero;
  if (term_enabled(config, "equivariance")) {
    if (output.shifted_physics.empty()) throw std::invalid_argument("shifted physics output is missing");
    const auto shift = config["equivariance_shift_steps"] ? config["equivariance_shift_steps"].as<std::int64_t>() : 1;
    const auto available = std::min(length - shift, require(output.shifted_physics, "neural", "equivariance").size(1));
    std::vector<torch::Tensor> residuals;
    const std::array<std::pair<const char*, double>, 5> states{{
        {"neural", 1.0}, {"vasoactive_signal", 1.5}, {"flow", 2.8},
        {"volume", 2.8}, {"deoxyhemoglobin", 2.8}}};
    for (const auto& [name, scale] : states) {
      residuals.push_back((require(output.physics, name, "equivariance").index(
          {Slice(), Slice(shift, shift + available)}) -
          require(output.shifted_physics, name, "equivariance").index(
          {Slice(), Slice(0, available)})) / scale);
    }
    equivariance = masked_smooth_l1(torch::stack(residuals, -1),
        mask.index({Slice(), Slice(shift, shift + available)}).unsqueeze(-1));
  }

  LossResult result;
  result.total = kDynamicsWeight * dynamics + kBoldWeight * bold +
                 kStabilityWeight * stability + kEquivarianceWeight * equivariance;
  result.metrics = {{"physics_dynamics", dynamics}, {"physics_bold", bold},
                    {"physics_hemodynamic_residual", hemodynamic},
                    {"physics_bold_observation", observation},
                    {"physics_stability", stability}, {"physics_equivariance", equivariance},
                    {"physics_total", result.total}};
  return result;
}

LossResult compute_losses(const ModelOutput& output, const TensorMap& targets,
                          const YAML::Node& weights, const TensorMap& context) {
  LossResult result;
  const auto& values = output.values;
  const auto pooled = require(values, "pooled", "loss");
  result.total = torch::zeros({}, pooled.options());
  auto add = [&](const std::string& name, const torch::Tensor& value,
                 const std::string& weight_name = "") {
    result.metrics[name] = value;
    result.total = result.total + weight(weights, weight_name.empty() ? name : weight_name) * value;
  };
  if (find(targets, "map") && find(values, "map_mean")) {
    const auto target = require(targets, "map", "map loss").to(require(values, "map_mean", "map loss").scalar_type());
    add("map_nll", gaussian_nll(require(values, "map_mean", "map loss"),
                                require(values, "map_logvar", "map loss"), target), "map");
    add("map_corr", correlation_loss(require(values, "map_mean", "map loss"), target), "map");
  }
  if (find(targets, "map_template") && find(values, "map_template")) {
    add("map_template", torch::nn::functional::smooth_l1_loss(
        require(values, "map_template", "template loss"),
        require(targets, "map_template", "template loss").to(
            require(values, "map_template", "template loss").scalar_type())), "map");
  }
  if (find(targets, "map_residual") && find(values, "map_residual")) {
    add("map_residual", torch::nn::functional::smooth_l1_loss(
        require(values, "map_residual", "residual loss"),
        require(targets, "map_residual", "residual loss").to(
            require(values, "map_residual", "residual loss").scalar_type())), "map");
  }
  if (find(targets, "behavior")) add("behavior", gaussian_nll(
      require(values, "behavior_mean", "behavior loss"), require(values, "behavior_logvar", "behavior loss"),
      require(targets, "behavior", "behavior loss").to(require(values, "behavior_mean", "behavior loss").scalar_type())));
  if (find(targets, "clinical")) add("clinical", gaussian_nll(
      require(values, "clinical_mean", "clinical loss"), require(values, "clinical_logvar", "clinical loss"),
      require(targets, "clinical", "clinical loss").to(require(values, "clinical_mean", "clinical loss").scalar_type())));
  if (find(targets, "state") && find(values, "state_logits")) {
    const auto logits = require(values, "state_logits", "state loss");
    const auto target = require(targets, "state", "state loss").to(torch::kLong);
    const auto length = std::min(logits.size(1), target.size(1));
    add("dynamic_state", torch::nn::functional::cross_entropy(
        logits.index({Slice(), Slice(0, length)}).reshape({-1, logits.size(-1)}),
        target.index({Slice(), Slice(0, length)}).reshape({-1}),
        torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-1)));
  }
  if (find(targets, "teacher_hidden")) add("hidden_distillation", torch::mse_loss(
      require(values, "distill_embedding", "distillation"),
      require(targets, "teacher_hidden", "distillation").to(require(values, "distill_embedding", "distillation").scalar_type())));
  if (find(targets, "teacher_map") && find(values, "map_mean")) add("output_distillation", gaussian_nll(
      require(values, "map_mean", "output distillation"), require(values, "map_logvar", "output distillation"),
      require(targets, "teacher_map", "output distillation").to(require(values, "map_mean", "output distillation").scalar_type())));
  if (find(targets, "teacher_behavior")) add("behavior_distillation", torch::mse_loss(
      require(values, "behavior_mean", "behavior distillation"),
      require(targets, "teacher_behavior", "behavior distillation").to(
          require(values, "behavior_mean", "behavior distillation").scalar_type())));
  if (find(targets, "future")) add("longitudinal", gaussian_nll(
      require(values, "trajectory_mean", "longitudinal"), require(values, "trajectory_logvar", "longitudinal"),
      require(targets, "future", "longitudinal").to(require(values, "trajectory_mean", "longitudinal").scalar_type())));
  add("moe_balance", require(values, "moe_balance", "moe loss"));
  add("router_z", require(values, "router_z", "router loss"));
  if (physics_enabled(weights)) {
    auto physics = physics_informed_loss(output, targets, weights["physics"], context);
    result.total = result.total + physics.total;
    result.metrics.insert(physics.metrics.begin(), physics.metrics.end());
  }
  return result;
}

}  // namespace neurotaskfm
