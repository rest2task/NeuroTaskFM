#include "neurotaskfm/model.h"

#include <ATen/ops/scaled_dot_product_attention.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

using torch::indexing::Slice;

namespace neurotaskfm {
namespace {

c10::intrusive_ptr<c10d::ProcessGroup> g_expert_process_group;
std::int64_t g_expert_parallel_size = 1;
std::int64_t g_expert_parallel_rank = 0;

std::vector<std::int64_t> cpu_counts(const torch::Tensor& input);

class DifferentiableAllToAll : public torch::autograd::Function<DifferentiableAllToAll> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
                               const torch::Tensor& input,
                               const torch::Tensor& output_split_tensor,
                               const torch::Tensor& input_split_tensor) {
    auto output_splits = cpu_counts(output_split_tensor);
    auto input_splits = cpu_counts(input_split_tensor);
    context->saved_data["output_splits"] = output_splits;
    context->saved_data["input_splits"] = input_splits;
    auto output = torch::empty({std::accumulate(output_splits.begin(), output_splits.end(), std::int64_t{0}),
                                input.size(1)}, input.options());
    auto contiguous_input = input.contiguous();
    g_expert_process_group->alltoall_base(output, contiguous_input,
                                          output_splits, input_splits)->wait();
    return output;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* context,
                                                  torch::autograd::variable_list gradients) {
    auto output_splits = context->saved_data["output_splits"].toIntVector();
    auto input_splits = context->saved_data["input_splits"].toIntVector();
    auto gradient = gradients.at(0).contiguous();
    auto returned = torch::empty({std::accumulate(input_splits.begin(), input_splits.end(), std::int64_t{0}),
                                  gradient.size(1)}, gradient.options());
    g_expert_process_group->alltoall_base(returned, gradient, input_splits, output_splits)->wait();
    return {returned, torch::Tensor{}, torch::Tensor{}};
  }
};

torch::Tensor all_to_all(const torch::Tensor& input, std::vector<std::int64_t> output_splits,
                         std::vector<std::int64_t> input_splits) {
  return DifferentiableAllToAll::apply(input, torch::tensor(output_splits, torch::kLong),
                                       torch::tensor(input_splits, torch::kLong));
}

std::vector<std::int64_t> cpu_counts(const torch::Tensor& input) {
  const auto values = input.to(torch::kCPU, torch::kLong).contiguous();
  return std::vector<std::int64_t>(values.data_ptr<std::int64_t>(),
                                   values.data_ptr<std::int64_t>() + values.numel());
}

}  // namespace

void configure_expert_parallel(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                               const std::int64_t size, const std::int64_t rank) {
  g_expert_process_group = std::move(process_group);
  g_expert_parallel_size = size;
  g_expert_parallel_rank = rank;
}

RMSNormImpl::RMSNormImpl(const std::int64_t dim, const double epsilon)
    : weight(register_parameter("weight", torch::ones({dim}))), epsilon(epsilon) {}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& input) {
  const auto scale = torch::rsqrt(input.to(torch::kFloat32).square().mean(-1, true) + epsilon)
                         .to(input.scalar_type());
  return input * scale * weight;
}

torch::Tensor rotate_half(const torch::Tensor& input) {
  const auto chunks = input.chunk(2, -1);
  return torch::cat({-chunks[1], chunks[0]}, -1);
}

torch::Tensor drop_path(const torch::Tensor& input, const double probability, const bool training) {
  if (!training || probability == 0.0) return input;
  const auto keep = 1.0 - probability;
  std::vector<std::int64_t> shape(static_cast<std::size_t>(input.dim()), 1);
  shape[0] = input.size(0);
  const auto random = torch::rand(shape, input.options().dtype(torch::kFloat32)).lt(keep).to(input.scalar_type());
  return input * random / keep;
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(const std::int64_t dim, const double theta) {
  auto indices = torch::arange(0, dim, 2, torch::TensorOptions().dtype(torch::kFloat32));
  inverse_frequency = register_buffer("inverse_frequency", 1.0 / torch::pow(theta, indices / dim));
}

std::pair<torch::Tensor, torch::Tensor> RotaryEmbeddingImpl::forward(
    const torch::Tensor& query, const torch::Tensor& key) {
  const auto positions = torch::arange(query.size(-2), inverse_frequency.options());
  const auto frequencies = torch::outer(positions, inverse_frequency);
  const auto embedding = torch::cat({frequencies, frequencies}, -1)
                             .to(query.scalar_type()).unsqueeze(0).unsqueeze(0);
  const auto cosine = embedding.cos();
  const auto sine = embedding.sin();
  return {query * cosine + rotate_half(query) * sine,
          key * cosine + rotate_half(key) * sine};
}

SelfAttentionImpl::SelfAttentionImpl(const std::int64_t dim, const std::int64_t heads,
                                     const std::int64_t kv_heads, const std::int64_t head_dim,
                                     const double theta, const double dropout)
    : heads(heads), kv_heads(kv_heads), head_dim(head_dim), dropout(dropout) {
  query = register_module("query", torch::nn::Linear(torch::nn::LinearOptions(dim, heads * head_dim).bias(false)));
  key = register_module("key", torch::nn::Linear(torch::nn::LinearOptions(dim, kv_heads * head_dim).bias(false)));
  value = register_module("value", torch::nn::Linear(torch::nn::LinearOptions(dim, kv_heads * head_dim).bias(false)));
  output = register_module("output", torch::nn::Linear(torch::nn::LinearOptions(heads * head_dim, dim).bias(false)));
  rotary = register_module("rotary", RotaryEmbedding(head_dim, theta));
}

torch::Tensor SelfAttentionImpl::forward(const torch::Tensor& input, const torch::Tensor& mask) {
  const auto batch = input.size(0);
  const auto sequence = input.size(1);
  auto q = query(input).view({batch, sequence, heads, head_dim}).transpose(1, 2);
  auto k = key(input).view({batch, sequence, kv_heads, head_dim}).transpose(1, 2);
  auto v = value(input).view({batch, sequence, kv_heads, head_dim}).transpose(1, 2);
  std::tie(q, k) = rotary->forward(q, k);
  std::optional<torch::Tensor> attention_mask;
  if (mask.defined()) attention_mask = mask.unsqueeze(1).unsqueeze(1);
  const auto y = at::scaled_dot_product_attention(
      q, k, v, attention_mask, is_training() ? dropout : 0.0, false, std::nullopt,
      kv_heads != heads);
  return output(y.transpose(1, 2).reshape({batch, sequence, heads * head_dim}));
}

CrossAttentionImpl::CrossAttentionImpl(const std::int64_t dim, const std::int64_t heads,
                                       const std::int64_t head_dim, const double dropout)
    : heads(heads), head_dim(head_dim), dropout(dropout) {
  const auto inner = heads * head_dim;
  query_projection = register_module("query", torch::nn::Linear(torch::nn::LinearOptions(dim, inner).bias(false)));
  key_projection = register_module("key", torch::nn::Linear(torch::nn::LinearOptions(dim, inner).bias(false)));
  value_projection = register_module("value", torch::nn::Linear(torch::nn::LinearOptions(dim, inner).bias(false)));
  output = register_module("output", torch::nn::Linear(torch::nn::LinearOptions(inner, dim).bias(false)));
}

torch::Tensor CrossAttentionImpl::forward(const torch::Tensor& query_tensor,
                                          const torch::Tensor& context,
                                          const torch::Tensor& context_mask) {
  const auto batch = query_tensor.size(0);
  const auto query_count = query_tensor.size(1);
  const auto key_count = context.size(1);
  auto q = query_projection(query_tensor).view({batch, query_count, heads, head_dim}).transpose(1, 2);
  auto k = key_projection(context).view({batch, key_count, heads, head_dim}).transpose(1, 2);
  auto v = value_projection(context).view({batch, key_count, heads, head_dim}).transpose(1, 2);
  std::optional<torch::Tensor> attention_mask;
  if (context_mask.defined()) attention_mask = context_mask.unsqueeze(1).unsqueeze(1);
  auto y = at::scaled_dot_product_attention(q, k, v, attention_mask,
                                             is_training() ? dropout : 0.0, false);
  return output(y.transpose(1, 2).reshape({batch, query_count, heads * head_dim}));
}

SwiGLUImpl::SwiGLUImpl(const std::int64_t dim, const std::int64_t hidden) {
  gate = register_module("gate", torch::nn::Linear(torch::nn::LinearOptions(dim, hidden).bias(false)));
  up = register_module("up", torch::nn::Linear(torch::nn::LinearOptions(dim, hidden).bias(false)));
  down = register_module("down", torch::nn::Linear(torch::nn::LinearOptions(hidden, dim).bias(false)));
}

torch::Tensor SwiGLUImpl::forward(const torch::Tensor& input) {
  return down(torch::silu(gate(input)) * up(input));
}

DiagonalSSMImpl::DiagonalSSMImpl(const std::int64_t dim) {
  input_projection = register_module("input_projection",
      torch::nn::Linear(torch::nn::LinearOptions(dim, dim * 2).bias(false)));
  output_projection = register_module("output_projection",
      torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  log_decay = register_parameter("log_decay", torch::linspace(-5.0, -1.0, dim));
  gain = register_parameter("gain", torch::ones({dim}));
  skip = register_parameter("skip", torch::ones({dim}));
}

torch::Tensor DiagonalSSMImpl::forward(const torch::Tensor& input) {
  const auto projected = input_projection(input).chunk(2, -1);
  const auto values = projected[0];
  const auto gate_values = projected[1];
  const auto length = std::min<std::int64_t>(input.size(1), 2048);
  const auto time = torch::arange(length, input.options().dtype(torch::kFloat32));
  const auto decay = -torch::softplus(log_decay.to(torch::kFloat32));
  auto kernel = gain.to(torch::kFloat32).unsqueeze(1) * torch::exp(decay.unsqueeze(1) * time.unsqueeze(0));
  kernel = kernel.to(values.scalar_type()).unsqueeze(1).flip(-1);
  auto convolved = torch::nn::functional::conv1d(
      values.transpose(1, 2), kernel,
      torch::nn::functional::Conv1dFuncOptions().padding(length - 1).groups(values.size(-1)));
  convolved = convolved.index({Slice(), Slice(), Slice(0, input.size(1))}).transpose(1, 2);
  return output_projection(torch::silu(gate_values) * (convolved + skip * values));
}

ExpertParallelMoEImpl::ExpertParallelMoEImpl(const std::int64_t dim, const std::int64_t hidden,
                                             const std::int64_t experts,
                                             const std::int64_t top_k,
                                             const bool shared_expert)
    : dim(dim), expert_count(experts), top_k(top_k),
      local_count(experts / g_expert_parallel_size),
      expert_parallel_size(g_expert_parallel_size), expert_parallel_rank(g_expert_parallel_rank),
      process_group(g_expert_process_group) {
  if (experts % expert_parallel_size != 0) throw std::invalid_argument("experts must divide expert parallel size");
  router = register_module("router", torch::nn::Linear(torch::nn::LinearOptions(dim, experts).bias(false)));
  this->experts.reserve(static_cast<std::size_t>(local_count));
  for (std::int64_t index = 0; index < local_count; ++index) {
    auto expert = SwiGLU(dim, hidden);
    this->experts.push_back(register_module("expert_" + std::to_string(index), expert));
  }
  if (shared_expert) shared = register_module("shared", SwiGLU(dim, hidden));
}

MoEOutput ExpertParallelMoEImpl::forward(const torch::Tensor& input) {
  const auto shape = input.sizes().vec();
  const auto flat = input.reshape({-1, dim});
  const auto logits = router(flat).to(torch::kFloat32);
  const auto probabilities = torch::softmax(logits, -1);
  const auto selected = torch::topk(probabilities, top_k, -1);
  const auto weights = std::get<0>(selected);
  const auto expert_ids = std::get<1>(selected);
  auto hidden = torch::zeros_like(flat);
  const auto normalized_weights = weights / weights.sum(-1, true);
  const auto token_ids = torch::arange(flat.size(0), expert_ids.options()).repeat_interleave(top_k);
  const auto routed_experts = expert_ids.flatten();
  const auto route_weights = normalized_weights.flatten().to(flat.scalar_type());
  if (expert_parallel_size == 1) {
    for (std::int64_t expert = 0; expert < local_count; ++expert) {
      const auto routes = torch::nonzero(routed_experts == expert).flatten();
      if (routes.numel() == 0) continue;
      const auto tokens = token_ids.index_select(0, routes);
      hidden.index_add_(0, tokens, experts[static_cast<std::size_t>(expert)]->forward(flat.index_select(0, tokens)) *
                        route_weights.index_select(0, routes).unsqueeze(1));
    }
  } else {
    const auto destination = torch::floor_divide(routed_experts, local_count);
    const auto local_expert = routed_experts.remainder(local_count);
    const auto order = torch::argsort(destination);
    const auto send_x = flat.index_select(0, token_ids.index_select(0, order)).contiguous();
    const auto send_local = local_expert.index_select(0, order).contiguous();
    const auto send_token = token_ids.index_select(0, order);
    const auto send_weight = route_weights.index_select(0, order);
    const auto send_counts_tensor = torch::bincount(destination, {}, expert_parallel_size).to(torch::kLong);
    auto receive_counts_tensor = torch::empty_like(send_counts_tensor);
    std::vector<std::int64_t> ones(static_cast<std::size_t>(expert_parallel_size), 1);
    auto send_counts_buffer = send_counts_tensor.contiguous();
    process_group->alltoall_base(receive_counts_tensor, send_counts_buffer, ones, ones)->wait();
    auto send_counts = cpu_counts(send_counts_tensor);
    auto receive_counts = cpu_counts(receive_counts_tensor);
    const auto received_x = all_to_all(send_x, receive_counts, send_counts);
    auto received_local = torch::empty({std::accumulate(receive_counts.begin(), receive_counts.end(), std::int64_t{0})}, send_local.options());
    auto send_local_buffer = send_local.contiguous();
    process_group->alltoall_base(received_local, send_local_buffer, receive_counts, send_counts)->wait();
    auto received_y = torch::zeros_like(received_x);
    for (std::int64_t expert = 0; expert < local_count; ++expert) {
      const auto routes = torch::nonzero(received_local == expert).flatten();
      if (routes.numel()) received_y.index_put_({routes}, experts[static_cast<std::size_t>(expert)]->forward(received_x.index_select(0, routes)));
    }
    const auto returned = all_to_all(received_y, send_counts, receive_counts);
    hidden.index_add_(0, send_token, returned * send_weight.unsqueeze(1));
  }
  if (shared) hidden = hidden + shared->forward(flat);
  const auto density = torch::bincount(expert_ids.flatten(), {}, expert_count).to(torch::kFloat32) /
                       std::max<std::int64_t>(expert_ids.numel(), 1);
  const auto importance = probabilities.mean(0);
  const auto balance = expert_count * torch::sum(density * importance);
  const auto router_z = torch::logsumexp(logits, -1).square().mean();
  return {hidden.reshape(shape), balance, router_z};
}

NeuroBlockImpl::NeuroBlockImpl(const std::int64_t dim, const std::int64_t heads,
                               const std::int64_t kv_heads, const std::int64_t head_dim,
                               const std::int64_t ffn, const std::string& mixer,
                               const bool is_moe, const std::int64_t experts,
                               const std::int64_t top_k, const bool shared_expert,
                               const double theta, const double dropout,
                               const double drop_path_rate)
    : uses_attention(mixer == "attention"), uses_moe(is_moe), drop_path_rate(drop_path_rate) {
  norm1 = register_module("norm1", RMSNorm(dim));
  norm2 = register_module("norm2", RMSNorm(dim));
  if (uses_attention) attention = register_module("attention", SelfAttention(dim, heads, kv_heads, head_dim, theta, dropout));
  else ssm = register_module("ssm", DiagonalSSM(dim));
  if (uses_moe) moe_ffn = register_module("moe_ffn", ExpertParallelMoE(dim, ffn, experts, top_k, shared_expert));
  else dense_ffn = register_module("dense_ffn", SwiGLU(dim, ffn));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> NeuroBlockImpl::forward(
    const torch::Tensor& input, const torch::Tensor& mask) {
  auto mixed = uses_attention ? attention->forward(norm1(input), mask) : ssm->forward(norm1(input));
  auto output = input + drop_path(mixed, drop_path_rate, is_training());
  if (uses_moe) {
    const auto result = moe_ffn->forward(norm2(output));
    output = output + drop_path(result.hidden, drop_path_rate, is_training());
    return {output, result.balance_loss, result.router_z_loss};
  }
  output = output + drop_path(dense_ffn->forward(norm2(output)), drop_path_rate, is_training());
  const auto zero = torch::zeros({}, output.options());
  return {output, zero, zero};
}

}  // namespace neurotaskfm
