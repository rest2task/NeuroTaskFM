#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <yaml-cpp/yaml.h>

#include "neurotaskfm/data.h"
#include "neurotaskfm/losses.h"
#include "neurotaskfm/model.h"

namespace neurotaskfm {

struct DistributedContext {
  std::int64_t rank{};
  std::int64_t local_rank{};
  std::int64_t world_size{1};
  std::int64_t expert_parallel_size{1};
  torch::Device device{torch::kCUDA, 0};
  c10::intrusive_ptr<c10d::ProcessGroup> process_group;
  c10::intrusive_ptr<c10d::ProcessGroup> data_process_group;
  c10::intrusive_ptr<c10d::ProcessGroup> expert_process_group;

  [[nodiscard]] bool primary() const { return rank == 0; }
  void all_reduce_gradients(torch::nn::Module& model) const;
  void barrier() const;
};

DistributedContext init_distributed(std::int64_t expert_parallel_size);
void seed_model(std::uint64_t seed);
void seed_data(std::uint64_t seed, std::int64_t data_rank);
double learning_rate(std::int64_t step, std::int64_t total, std::int64_t warmup,
                     double maximum, double minimum);
void set_learning_rate(torch::optim::Optimizer& optimizer, double value);
torch::optim::AdamW build_optimizer(NeuroTaskFM& model, const YAML::Node& config);

std::filesystem::path save_checkpoint(NeuroTaskFM& model, torch::optim::Optimizer& optimizer,
                                      std::int64_t step, const std::filesystem::path& output,
                                      const DistributedContext& context);
std::int64_t load_checkpoint(NeuroTaskFM& model, torch::optim::Optimizer* optimizer,
                             const std::filesystem::path& checkpoint,
                             const DistributedContext& context);

struct Calibration {
  double map_temperature{1.0};
  double behavior_temperature{1.0};
  double map_minimum_confidence{0.0};
  double behavior_minimum_confidence{0.0};
  double maximum_ood_score{1.0};

  static Calibration load(const std::filesystem::path& path);
  void apply(ModelOutput& output) const;
  [[nodiscard]] bool abstain(const ModelOutput& output) const;
};

class Trainer {
 public:
  Trainer(YAML::Node config, DistributedContext context);
  void run();

 private:
  YAML::Node config_;
  DistributedContext context_;
  ModelConfig model_config_;
  NeuroTaskFM model_{nullptr};
  std::unique_ptr<torch::optim::AdamW> optimizer_;
};

nlohmann::json infer_request(const std::filesystem::path& deployment_config,
                             const std::filesystem::path& request_path,
                             const std::optional<std::filesystem::path>& output_directory = std::nullopt);
void personalize(const std::filesystem::path& deployment_config,
                 const std::vector<std::filesystem::path>& packs,
                 const std::vector<std::string>& queries,
                 const std::filesystem::path& output,
                 std::int64_t steps = 160, double rate = 0.015);

}  // namespace neurotaskfm
