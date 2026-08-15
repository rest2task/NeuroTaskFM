#include "neurotask/operators.h"
#include "neurotask/cuda_check.h"
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ntfm {
namespace {
__global__ void histogram_kernel(const float* x, size_t n, float lo, float hi, unsigned int* hist, int bins) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n || !isfinite(x[i])) return;
  const float u = fminf(fmaxf((x[i] - lo) / fmaxf(hi - lo, 1e-12f), 0.0f), 0.999999f);
  atomicAdd(hist + static_cast<int>(u * bins), 1U);
}

__global__ void normalize_kernel(float* x, size_t n, float low, float high) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float center = 0.5f * (low + high), scale = fmaxf(0.25f * (high - low), 1e-6f);
  const float v = fminf(fmaxf(x[i], low), high);
  x[i] = (v - center) / scale;
}
}

void robust_normalize(DeviceBuffer<float>& data, size_t n, float lower, float upper, int bins, cudaStream_t stream) {
  if (!n) return;
  DeviceBuffer<float> d_min(1), d_max(1);
  size_t temp_bytes = 0;
  cub::DeviceReduce::Min(nullptr, temp_bytes, data.data(), d_min.data(), n, stream);
  DeviceBuffer<unsigned char> temp(temp_bytes);
  cub::DeviceReduce::Min(temp.data(), temp_bytes, data.data(), d_min.data(), n, stream);
  cub::DeviceReduce::Max(nullptr, temp_bytes, data.data(), d_max.data(), n, stream);
  temp.resize(temp_bytes);
  cub::DeviceReduce::Max(temp.data(), temp_bytes, data.data(), d_max.data(), n, stream);
  float lo = 0.0f, hi = 1.0f;
  NTFM_CUDA(cudaMemcpyAsync(&lo, d_min.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
  NTFM_CUDA(cudaMemcpyAsync(&hi, d_max.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
  NTFM_CUDA(cudaStreamSynchronize(stream));
  if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo) return;
  DeviceBuffer<unsigned int> hist(bins);
  NTFM_CUDA(cudaMemsetAsync(hist.data(), 0, hist.bytes(), stream));
  histogram_kernel<<<(n + 255) / 256, 256, 0, stream>>>(data.data(), n, lo, hi, hist.data(), bins);
  NTFM_CUDA(cudaGetLastError());
  std::vector<unsigned int> h(bins);
  hist.copy_to_host(h.data(), h.size());
  const uint64_t total = std::accumulate(h.begin(), h.end(), uint64_t{0});
  const uint64_t lower_count = static_cast<uint64_t>(total * lower / 100.0f);
  const uint64_t upper_count = static_cast<uint64_t>(total * upper / 100.0f);
  uint64_t acc = 0;
  int ilow = 0, ihigh = bins - 1;
  for (int i = 0; i < bins; ++i) { acc += h[i]; if (acc >= lower_count) { ilow = i; break; } }
  acc = 0;
  for (int i = 0; i < bins; ++i) { acc += h[i]; if (acc >= upper_count) { ihigh = i; break; } }
  const float low_value = lo + (hi - lo) * (static_cast<float>(ilow) / bins);
  const float high_value = lo + (hi - lo) * (static_cast<float>(ihigh + 1) / bins);
  normalize_kernel<<<(n + 255) / 256, 256, 0, stream>>>(data.data(), n, low_value, high_value);
  NTFM_CUDA(cudaGetLastError());
}
}
