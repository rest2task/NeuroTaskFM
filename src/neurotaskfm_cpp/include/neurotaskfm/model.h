#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "neurotaskfm/config.h"

namespace neurotaskfm {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

void configure_expert_parallel(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                               std::int64_t size, std::int64_t rank);

struct ModelOutput {
  TensorMap values;
  TensorMap physics;
  TensorMap shifted_physics;
  TensorMap modality_summaries;
};

struct RMSNormImpl : torch::nn::Module {
  explicit RMSNormImpl(std::int64_t dim, double epsilon = 1e-6);
  torch::Tensor forward(const torch::Tensor& input);
  torch::Tensor weight;
  double epsilon;
};
TORCH_MODULE(RMSNorm);

torch::Tensor rotate_half(const torch::Tensor& input);
torch::Tensor drop_path(const torch::Tensor& input, double probability, bool training);

struct RotaryEmbeddingImpl : torch::nn::Module {
  RotaryEmbeddingImpl(std::int64_t dim, double theta);
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& query, const torch::Tensor& key);
  torch::Tensor inverse_frequency;
};
TORCH_MODULE(RotaryEmbedding);

struct SelfAttentionImpl : torch::nn::Module {
  SelfAttentionImpl(std::int64_t dim, std::int64_t heads, std::int64_t kv_heads,
                    std::int64_t head_dim, double theta, double dropout);
  torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& mask = {});
  std::int64_t heads, kv_heads, head_dim;
  double dropout;
  torch::nn::Linear query{nullptr}, key{nullptr}, value{nullptr}, output{nullptr};
  RotaryEmbedding rotary{nullptr};
};
TORCH_MODULE(SelfAttention);

struct CrossAttentionImpl : torch::nn::Module {
  CrossAttentionImpl(std::int64_t dim, std::int64_t heads, std::int64_t head_dim, double dropout = 0.0);
  torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& context,
                        const torch::Tensor& context_mask = {});
  std::int64_t heads, head_dim;
  double dropout;
  torch::nn::Linear query_projection{nullptr}, key_projection{nullptr}, value_projection{nullptr}, output{nullptr};
};
TORCH_MODULE(CrossAttention);

struct SwiGLUImpl : torch::nn::Module {
  SwiGLUImpl(std::int64_t dim, std::int64_t hidden);
  torch::Tensor forward(const torch::Tensor& input);
  torch::nn::Linear gate{nullptr}, up{nullptr}, down{nullptr};
};
TORCH_MODULE(SwiGLU);

struct DiagonalSSMImpl : torch::nn::Module {
  explicit DiagonalSSMImpl(std::int64_t dim);
  torch::Tensor forward(const torch::Tensor& input);
  torch::nn::Linear input_projection{nullptr}, output_projection{nullptr};
  torch::Tensor log_decay, gain, skip;
};
TORCH_MODULE(DiagonalSSM);

struct MoEOutput {
  torch::Tensor hidden;
  torch::Tensor balance_loss;
  torch::Tensor router_z_loss;
};

struct ExpertParallelMoEImpl : torch::nn::Module {
  ExpertParallelMoEImpl(std::int64_t dim, std::int64_t hidden, std::int64_t experts,
                        std::int64_t top_k, bool shared_expert);
  MoEOutput forward(const torch::Tensor& input);
  std::int64_t dim, expert_count, top_k;
  std::int64_t local_count, expert_parallel_size, expert_parallel_rank;
  c10::intrusive_ptr<c10d::ProcessGroup> process_group;
  torch::nn::Linear router{nullptr};
  std::vector<SwiGLU> experts;
  SwiGLU shared{nullptr};
};
TORCH_MODULE(ExpertParallelMoE);

struct NeuroBlockImpl : torch::nn::Module {
  NeuroBlockImpl(std::int64_t dim, std::int64_t heads, std::int64_t kv_heads,
                 std::int64_t head_dim, std::int64_t ffn, const std::string& mixer,
                 bool is_moe, std::int64_t experts, std::int64_t top_k,
                 bool shared_expert, double theta, double dropout, double drop_path_rate);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& input, const torch::Tensor& mask = {});
  bool uses_attention, uses_moe;
  double drop_path_rate;
  RMSNorm norm1{nullptr}, norm2{nullptr};
  SelfAttention attention{nullptr};
  DiagonalSSM ssm{nullptr};
  SwiGLU dense_ffn{nullptr};
  ExpertParallelMoE moe_ffn{nullptr};
};
TORCH_MODULE(NeuroBlock);

struct DicomEncoderImpl : torch::nn::Module {
  DicomEncoderImpl(std::int64_t vocabulary, std::int64_t dim, std::int64_t max_tokens);
  torch::Tensor forward(const torch::Tensor& tokens);
  std::int64_t max_tokens;
  torch::nn::Embedding embedding{nullptr}, position{nullptr};
  RMSNorm norm{nullptr};
};
TORCH_MODULE(DicomEncoder);

struct SliceEncoderImpl : torch::nn::Module {
  SliceEncoderImpl(std::int64_t dim, std::int64_t patch, std::int64_t tokens_per_slice,
                   std::int64_t max_slices);
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& images,
                                                  const torch::Tensor& types,
                                                  const torch::Tensor& mask);
  std::int64_t tokens_per_slice, side, max_slices;
  torch::nn::Conv2d patch{nullptr};
  torch::nn::AdaptiveAvgPool2d pool{nullptr};
  torch::nn::Embedding type_embedding{nullptr}, index_embedding{nullptr};
  RMSNorm norm{nullptr};
};
TORCH_MODULE(SliceEncoder);

struct FrameEncoderImpl : torch::nn::Module {
  FrameEncoderImpl(std::int64_t dim, std::int64_t patch, std::int64_t tokens_per_frame,
                   std::int64_t max_frames);
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& frames,
                                                  const torch::Tensor& modalities,
                                                  const torch::Tensor& views,
                                                  const torch::Tensor& positions,
                                                  const torch::Tensor& mask);
  std::int64_t tokens_per_frame, side, max_frames;
  torch::nn::Conv2d patch{nullptr};
  torch::nn::AdaptiveAvgPool2d pool{nullptr};
  torch::nn::Embedding modality_embedding{nullptr}, view_embedding{nullptr}, position_embedding{nullptr};
  RMSNorm norm{nullptr};
};
TORCH_MODULE(FrameEncoder);

struct CompiledEncoderImpl : torch::nn::Module {
  CompiledEncoderImpl(std::int64_t channels, std::int64_t dim, std::int64_t max_tokens);
  torch::Tensor forward(const torch::Tensor& input);
  std::int64_t max_tokens;
  torch::nn::Linear projection{nullptr};
  torch::nn::Embedding position{nullptr};
  torch::Tensor type_embedding;
  RMSNorm norm{nullptr};
};
TORCH_MODULE(CompiledEncoder);

struct VolumeEncoderImpl : torch::nn::Module {
  VolumeEncoderImpl(std::int64_t dim, const std::vector<std::int64_t>& patch, std::int64_t max_tokens);
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& volume,
                                                  const torch::Tensor& volume_mask = {});
  std::int64_t max_tokens;
  torch::nn::Conv3d convolution{nullptr};
  RMSNorm norm{nullptr};
};
TORCH_MODULE(VolumeEncoder);

struct PerceiverResamplerImpl : torch::nn::Module {
  PerceiverResamplerImpl(std::int64_t dim, std::int64_t latents, std::int64_t heads,
                         std::int64_t head_dim, std::int64_t depth = 2);
  torch::Tensor forward(const torch::Tensor& context, const torch::Tensor& mask = {});
  torch::Tensor latents;
  std::vector<CrossAttention> cross;
  std::vector<RMSNorm> norms;
};
TORCH_MODULE(PerceiverResampler);

struct QueryEncoderImpl : torch::nn::Module {
  QueryEncoderImpl(std::int64_t vocabulary, std::int64_t dim, std::int64_t max_tokens,
                   std::int64_t latents, std::int64_t heads, std::int64_t head_dim);
  torch::Tensor forward(const torch::Tensor& tokens, const torch::Tensor& mask);
  std::int64_t max_tokens;
  DicomEncoder token_encoder{nullptr};
  PerceiverResampler resampler{nullptr};
};
TORCH_MODULE(QueryEncoder);

struct TabularEncoderImpl : torch::nn::Module {
  TabularEncoderImpl(std::int64_t dim, std::int64_t features);
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& values,
                                                  const torch::Tensor& mask);
  std::int64_t max_features;
  torch::Tensor feature_embedding;
  torch::nn::Sequential value_projection;
  RMSNorm norm{nullptr};
};
TORCH_MODULE(TabularEncoder);

struct NeuroSignatureUpdaterImpl : torch::nn::Module {
  NeuroSignatureUpdaterImpl(std::int64_t dim, const std::vector<std::int64_t>& partitions,
                            std::int64_t heads, std::int64_t head_dim, std::int64_t quality_dim = 16);
  torch::Tensor forward(const torch::Tensor& observations, const torch::Tensor& observation_mask = {},
                        const torch::Tensor& quality = {}, const torch::Tensor& previous = {});
  std::int64_t token_count;
  torch::Tensor base, type_ids;
  torch::nn::Embedding type_embedding{nullptr};
  CrossAttention cross{nullptr};
  RMSNorm norm{nullptr};
  torch::nn::Sequential quality_projection;
  torch::nn::Linear gate{nullptr};
};
TORCH_MODULE(NeuroSignatureUpdater);

struct DistributionHeadImpl : torch::nn::Module {
  DistributionHeadImpl(std::int64_t dim, std::int64_t outputs);
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input);
  RMSNorm norm{nullptr};
  torch::nn::Linear hidden{nullptr}, mean{nullptr}, log_variance{nullptr};
};
TORCH_MODULE(DistributionHead);

struct MapDecoderImpl : torch::nn::Module {
  MapDecoderImpl(std::int64_t dim, const std::vector<std::int64_t>& shape);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& subject, const torch::Tensor& task_query);
  torch::Tensor resize(const torch::Tensor& input) const;
  std::vector<std::int64_t> shape;
  RMSNorm template_norm{nullptr}, residual_norm{nullptr};
  torch::nn::Linear template_projection{nullptr}, residual_projection{nullptr};
  torch::nn::Sequential up;
  torch::nn::Conv3d template_output{nullptr}, residual_output{nullptr}, log_variance_output{nullptr};
  torch::Tensor residual_scale;
};
TORCH_MODULE(MapDecoder);

struct DynamicStateHeadImpl : torch::nn::Module {
  DynamicStateHeadImpl(std::int64_t dim, std::int64_t states, std::int64_t heads, std::int64_t head_dim);
  torch::Tensor forward(const torch::Tensor& temporal, const torch::Tensor& context,
                        const torch::Tensor& context_mask = {});
  CrossAttention cross{nullptr};
  RMSNorm norm{nullptr};
  torch::nn::Linear output{nullptr};
};
TORCH_MODULE(DynamicStateHead);

struct FutureWindowHeadImpl : torch::nn::Module {
  FutureWindowHeadImpl(std::int64_t dim, std::int64_t channels, std::vector<std::int64_t> offsets);
  torch::Tensor forward(const torch::Tensor& temporal);
  std::int64_t channels;
  std::vector<std::int64_t> offsets;
  RMSNorm norm{nullptr};
  torch::nn::Linear hidden{nullptr}, output{nullptr};
};
TORCH_MODULE(FutureWindowHead);

struct PhysicsHeadImpl : torch::nn::Module {
  PhysicsHeadImpl(std::int64_t dim, std::int64_t regions);
  TensorMap forward(const torch::Tensor& temporal, const torch::Tensor& subject);
  std::int64_t regions;
  torch::nn::Sequential state_projection, parameter_projection;
};
TORCH_MODULE(PhysicsHead);

struct NeuroTaskFMImpl : torch::nn::Module {
  explicit NeuroTaskFMImpl(ModelConfig config, bool physics_enabled = false,
                           std::int64_t physics_regions = 0, bool resource_encoders = false);
  ModelOutput forward(const TensorMap& batch, const torch::Tensor& previous_signature = {},
                      bool decode_map = true);
  ModelOutput decode_signature(const torch::Tensor& signature, const torch::Tensor& query_tokens,
                               const torch::Tensor& query_mask, bool decode_map = true);
  void enable_clinical_adapters(std::int64_t rank, std::int64_t upper_layers, std::int64_t concepts = 0);
  void freeze_shared_backbone();

  ModelConfig config;
  DicomEncoder dicom{nullptr};
  SliceEncoder slices{nullptr};
  CompiledEncoder spatial{nullptr}, temporal{nullptr};
  VolumeEncoder t1_volume{nullptr}, fmri_volume{nullptr};
  VolumeEncoder mr_volumes{nullptr};
  FrameEncoder mr_images{nullptr}, mr_video{nullptr};
  TabularEncoder biomarkers{nullptr}, clinical_inputs{nullptr};
  PerceiverResampler observation_resampler{nullptr};
  QueryEncoder query{nullptr};
  NeuroSignatureUpdater signature_updater{nullptr};
  torch::Tensor cls, null_observation;
  std::vector<NeuroBlock> blocks;
  RMSNorm norm{nullptr}, fuse_norm{nullptr};
  torch::nn::Linear fuse_projection{nullptr};
  MapDecoder map_decoder{nullptr};
  DistributionHead behavior{nullptr}, quality_head{nullptr}, clinical{nullptr}, trajectory{nullptr};
  DynamicStateHead state{nullptr};
  FutureWindowHead future{nullptr};
  PhysicsHead physics_head{nullptr};
  torch::nn::Sequential latent_predictor, distill_head;
  std::unordered_map<std::int64_t, torch::nn::Sequential> adapters;
  torch::nn::Sequential concept_head;

 private:
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, TensorMap> encode_observations(
      const TensorMap& batch);
  ModelOutput decode(const torch::Tensor& signature, const torch::Tensor& observation_latents,
                     const torch::Tensor& query_latents, const torch::Tensor& temporal_tokens,
                     bool decode_map, const torch::Tensor& shifted_temporal_tokens = {});
};
TORCH_MODULE(NeuroTaskFM);

}  // namespace neurotaskfm
