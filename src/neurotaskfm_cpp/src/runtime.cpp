#include "neurotaskfm/runtime.h"

#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace neurotaskfm {
namespace {

std::int64_t environment_integer(const char* name, const std::int64_t fallback) {
  const auto* value = std::getenv(name);
  return value == nullptr ? fallback : std::stoll(value);
}

std::string environment_string(const char* name, std::string fallback) {
  const auto* value = std::getenv(name);
  return value == nullptr ? std::move(fallback) : value;
}

TensorMap move_to_device(const TensorMap& values, const torch::Device& device) {
  TensorMap output;
  for (const auto& [key, value] : values) output[key] = value.to(device, true);
  return output;
}

std::filesystem::path resolve_checkpoint(const std::filesystem::path& path, const std::int64_t rank) {
  if (std::filesystem::is_regular_file(path)) return path;
  const auto rank_file = path / ("rank-" + std::to_string(rank) + ".pt");
  if (std::filesystem::exists(rank_file)) return rank_file;
  std::ostringstream padded_name;
  padded_name << "rank-" << std::setw(2) << std::setfill('0') << rank << ".pt";
  const auto padded_rank_file = path / padded_name.str();
  if (std::filesystem::exists(padded_rank_file)) return padded_rank_file;
  const auto final_file = path / "model.pt";
  if (std::filesystem::exists(final_file)) return final_file;
  throw std::runtime_error("checkpoint not found: " + path.string());
}

}  // namespace

DistributedContext init_distributed(const std::int64_t expert_parallel_size) {
  DistributedContext context;
  context.rank = environment_integer("RANK", environment_integer("SLURM_PROCID", environment_integer("OMPI_COMM_WORLD_RANK", 0)));
  context.local_rank = environment_integer("LOCAL_RANK", environment_integer("SLURM_LOCALID", environment_integer("OMPI_COMM_WORLD_LOCAL_RANK", 0)));
  context.world_size = environment_integer("WORLD_SIZE", environment_integer("SLURM_NTASKS", environment_integer("OMPI_COMM_WORLD_SIZE", 1)));
  context.expert_parallel_size = expert_parallel_size;
  if (!torch::cuda::is_available()) throw std::runtime_error("NeuroTaskFM requires CUDA");
  c10::cuda::set_device(static_cast<c10::DeviceIndex>(context.local_rank));
  context.device = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(context.local_rank));
  if (context.world_size > 1) {
    c10d::TCPStoreOptions options;
    options.port = static_cast<std::uint16_t>(environment_integer("MASTER_PORT", 29500));
    options.isServer = context.rank == 0;
    options.numWorkers = static_cast<std::size_t>(context.world_size);
    options.waitWorkers = true;
    auto store = c10::make_intrusive<c10d::TCPStore>(environment_string("MASTER_ADDR", "127.0.0.1"), options);
    auto nccl_options = c10d::ProcessGroupNCCL::Options::create();
    nccl_options->is_high_priority_stream = true;
    context.process_group = c10::make_intrusive<c10d::ProcessGroupNCCL>(store,
        static_cast<int>(context.rank), static_cast<int>(context.world_size), nccl_options);
    if (context.world_size % expert_parallel_size != 0) {
      throw std::invalid_argument("world size must be divisible by expert parallel size");
    }
    const auto expert_rank = context.rank % expert_parallel_size;
    const auto expert_group_id = context.rank / expert_parallel_size;
    auto expert_store = c10::make_intrusive<c10d::PrefixStore>(
        "expert_" + std::to_string(expert_group_id), store);
    context.expert_process_group = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        expert_store, static_cast<int>(expert_rank), static_cast<int>(expert_parallel_size), nccl_options);
    const auto data_size = context.world_size / expert_parallel_size;
    auto data_store = c10::make_intrusive<c10d::PrefixStore>(
        "data_" + std::to_string(expert_rank), store);
    context.data_process_group = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        data_store, static_cast<int>(expert_group_id), static_cast<int>(data_size), nccl_options);
  }
  const auto effective_expert_size = context.world_size > 1 ? expert_parallel_size : 1;
  configure_expert_parallel(context.expert_process_group, effective_expert_size,
                            context.rank % effective_expert_size);
  return context;
}

void DistributedContext::all_reduce_gradients(torch::nn::Module& model) const {
  const auto group = data_process_group ? data_process_group : process_group;
  if (!group || world_size == 1) return;
  std::vector<torch::Tensor> gradients;
  for (const auto& parameter : model.parameters()) {
    if (parameter.grad().defined()) gradients.push_back(parameter.grad());
  }
  constexpr std::size_t bucket_size = 64;
  for (std::size_t start = 0; start < gradients.size(); start += bucket_size) {
    const auto stop = std::min(start + bucket_size, gradients.size());
    std::vector<torch::Tensor> bucket(gradients.begin() + static_cast<std::ptrdiff_t>(start),
                                      gradients.begin() + static_cast<std::ptrdiff_t>(stop));
    auto work = group->allreduce(bucket);
    work->wait();
    const auto data_size = world_size / expert_parallel_size;
    for (auto& gradient : bucket) gradient.div_(data_size);
  }
}

void DistributedContext::barrier() const {
  if (process_group && world_size > 1) process_group->barrier()->wait();
}

void seed_model(const std::uint64_t seed) {
  torch::manual_seed(seed);
  torch::cuda::manual_seed_all(seed);
}

void seed_data(const std::uint64_t seed, const std::int64_t data_rank) {
  torch::manual_seed(seed + static_cast<std::uint64_t>(data_rank) * 100003ULL);
}

double learning_rate(const std::int64_t step, const std::int64_t total,
                     const std::int64_t warmup, const double maximum, const double minimum) {
  if (step < warmup) return maximum * static_cast<double>(step + 1) / std::max<std::int64_t>(warmup, 1);
  const auto progress = std::clamp(static_cast<double>(step - warmup) /
                                   std::max<std::int64_t>(total - warmup, 1), 0.0, 1.0);
  return minimum + 0.5 * (maximum - minimum) * (1.0 + std::cos(std::numbers::pi * progress));
}

void set_learning_rate(torch::optim::Optimizer& optimizer, const double value) {
  for (auto& group : optimizer.param_groups()) {
    static_cast<torch::optim::AdamWOptions&>(group.options()).lr(value);
  }
}

torch::optim::AdamW build_optimizer(NeuroTaskFM& model, const YAML::Node& config) {
  const auto beta_values = config["betas"] ? config["betas"].as<std::vector<double>>()
                                            : std::vector<double>{0.9, 0.95};
  auto options = torch::optim::AdamWOptions(config["learning_rate"].as<double>())
      .betas({beta_values.at(0), beta_values.at(1)})
      .eps(config["eps"] ? config["eps"].as<double>() : 1e-8)
      .weight_decay(config["weight_decay"].as<double>());
  return torch::optim::AdamW(model->parameters(), options);
}

std::filesystem::path save_checkpoint(NeuroTaskFM& model, torch::optim::Optimizer& optimizer,
                                      const std::int64_t step, const std::filesystem::path& output,
                                      const DistributedContext& context) {
  const auto directory = output / ("step-" + std::to_string(step));
  std::filesystem::create_directories(directory);
  const auto path = directory / ("rank-" + std::to_string(context.rank) + ".pt");
  torch::serialize::OutputArchive archive;
  model->save(archive);
  torch::serialize::OutputArchive optimizer_archive;
  optimizer.save(optimizer_archive);
  archive.write("optimizer", optimizer_archive);
  archive.write("step", torch::tensor(step, torch::kLong));
  archive.save_to(path.string());
  context.barrier();
  return path;
}

std::int64_t load_checkpoint(NeuroTaskFM& model, torch::optim::Optimizer* optimizer,
                             const std::filesystem::path& checkpoint,
                             const DistributedContext& context) {
  torch::serialize::InputArchive archive;
  archive.load_from(resolve_checkpoint(checkpoint, context.rank).string(), context.device);
  model->load(archive);
  if (optimizer != nullptr) {
    torch::serialize::InputArchive optimizer_archive;
    archive.read("optimizer", optimizer_archive);
    optimizer->load(optimizer_archive);
  }
  torch::Tensor step;
  archive.read("step", step);
  return step.item<std::int64_t>();
}

Calibration Calibration::load(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open calibration: " + path.string());
  nlohmann::json value;
  stream >> value;
  Calibration calibration;
  calibration.map_temperature = value.value("map_temperature", 1.0);
  calibration.behavior_temperature = value.value("behavior_temperature", 1.0);
  const auto thresholds = value.value("thresholds", nlohmann::json::object());
  calibration.map_minimum_confidence = thresholds.value("minimum_map_confidence", 0.70);
  calibration.behavior_minimum_confidence = thresholds.value("minimum_behavior_confidence", 0.65);
  calibration.maximum_ood_score = thresholds.value("maximum_ood_score", 0.20);
  return calibration;
}

void Calibration::apply(ModelOutput& output) const {
  if (output.values.count("map_logvar")) output.values["map_logvar"] += 2.0 * std::log(map_temperature);
  if (output.values.count("behavior_logvar")) output.values["behavior_logvar"] += 2.0 * std::log(behavior_temperature);
}

bool Calibration::abstain(const ModelOutput& output) const {
  const auto map_confidence = output.values.count("map_logvar")
      ? torch::exp(-0.5 * output.values.at("map_logvar")).mean().item<double>() : 1.0;
  const auto behavior_confidence = output.values.count("behavior_logvar")
      ? torch::exp(-0.5 * output.values.at("behavior_logvar")).mean().item<double>() : 1.0;
  const auto ood = output.values.count("quality_mean")
      ? torch::sigmoid(output.values.at("quality_mean").select(-1, -1)).mean().item<double>() : 0.0;
  return map_confidence < map_minimum_confidence || behavior_confidence < behavior_minimum_confidence ||
         ood > maximum_ood_score;
}

Trainer::Trainer(YAML::Node config, DistributedContext context)
    : config_(std::move(config)), context_(std::move(context)),
      model_config_(ModelConfig::load(config_["model"].as<std::string>())) {
  const auto loss = config_["loss"];
  const auto physics = physics_enabled(loss);
  const auto regions = physics && loss["physics"]["regions"]
      ? loss["physics"]["regions"].as<std::int64_t>() : 0;
  const auto resources = config_["resources"] && config_["resources"]["enabled"] &&
                         config_["resources"]["enabled"].as<bool>();
  model_ = NeuroTaskFM(model_config_, physics, regions, resources);
  model_->to(context_.device);
  if (config_["checkpoint"] && std::filesystem::exists(config_["checkpoint"].as<std::string>())) {
    load_checkpoint(model_, nullptr, config_["checkpoint"].as<std::string>(), context_);
  }
  const auto stage = config_["stage"].as<std::string>();
  if (stage == "clinical_posttraining") {
    const auto rank = config_["trainable"]["disease_adapter_rank"].as<std::int64_t>();
    const auto upper = config_["trainable"]["upper_layers"].as<std::int64_t>();
    const auto concepts = config_["concepts"] ? static_cast<std::int64_t>(config_["concepts"].size()) : 0;
    model_->enable_clinical_adapters(rank, upper, concepts);
    model_->freeze_shared_backbone();
  } else if (stage == "digital_twin_signature") {
    model_->freeze_shared_backbone();
  }
  optimizer_ = std::make_unique<torch::optim::AdamW>(build_optimizer(model_, config_["optimizer"]));
}

void Trainer::run() {
  const auto seed = config_["seed"].as<std::uint64_t>();
  seed_model(seed);
  seed_data(seed, context_.rank);
  const auto manifest = config_["manifest"] ? config_["manifest"].as<std::string>()
                                              : config_["train_manifest"].as<std::string>();
  const auto load_resources = config_["resources"] && config_["resources"]["enabled"] &&
                              config_["resources"]["enabled"].as<bool>();
  FeaturePackDataset dataset(manifest, std::string("train"),
                             physics_enabled(config_["loss"]), load_resources);
  HierarchicalDistributedSampler sampler(dataset.rows(), context_.world_size, context_.rank, seed);
  const auto maximum_steps = config_["max_steps"].as<std::int64_t>();
  const auto accumulation = config_["gradient_accumulation_steps"].as<std::int64_t>();
  const auto checkpoint_node = config_["checkpointing"] ? config_["checkpointing"] : config_["checkpoint"];
  const auto checkpoint_interval = checkpoint_node["interval_steps"].as<std::int64_t>();
  const auto maximum_rate = config_["optimizer"]["learning_rate"].as<double>();
  const auto minimum_rate = config_["optimizer"]["minimum_learning_rate"]
      ? config_["optimizer"]["minimum_learning_rate"].as<double>() : maximum_rate * 0.05;
  const auto warmup = config_["optimizer"]["warmup_steps"].as<std::int64_t>();
  const auto clip = config_["optimizer"]["gradient_clip"].as<double>();
  model_->train();
  optimizer_->zero_grad();
  std::int64_t step = 0;
  while (step < maximum_steps) {
    sampler.set_epoch(step);
    for (const auto index : sampler.indices()) {
      TensorMap targets;
      auto batch = collate_feature_packs({dataset.get(index)}, &targets);
      batch = apply_training_augmentation(std::move(batch), config_["augmentation"]);
      batch = move_to_device(batch, context_.device);
      targets = move_to_device(targets, context_.device);
      at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      auto output = model_->forward(batch);
      auto losses = compute_losses(output, targets, config_["loss"], batch);
      at::autocast::set_autocast_enabled(at::kCUDA, false);
      (losses.total / accumulation).backward();
      if ((step + 1) % accumulation == 0) {
        context_.all_reduce_gradients(*model_);
        torch::nn::utils::clip_grad_norm_(model_->parameters(), clip);
        set_learning_rate(*optimizer_, learning_rate(step, maximum_steps, warmup, maximum_rate, minimum_rate));
        optimizer_->step();
        optimizer_->zero_grad();
      }
      ++step;
      if (context_.primary() && step % 20 == 0) {
        std::cout << "step=" << step << " loss=" << losses.total.detach().to(torch::kFloat32).item<double>() << '\n';
      }
      if (step % checkpoint_interval == 0) {
        save_checkpoint(model_, *optimizer_, step, config_["output_dir"].as<std::string>(), context_);
      }
      if (step >= maximum_steps) break;
    }
  }
}

}  // namespace neurotaskfm
