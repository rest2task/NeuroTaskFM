#include "neurotask/operators.h"
#include "neurotask/cuda_check.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ntfm {
namespace {
__global__ void parcel_count_kernel(const float* atlas, size_t vox, int parcels, int* counts) {
  const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= vox) return;
  const int p = static_cast<int>(lrintf(atlas[v])) - 1;
  if (p >= 0 && p < parcels) atomicAdd(counts + p, 1);
}

__global__ void parcel_sum_kernel(const float* fmri, const float* atlas, size_t vox, int nt, int parcels, float* output) {
  const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = vox * nt;
  if (linear >= total) return;
  const int t = linear / vox;
  const size_t v = linear - static_cast<size_t>(t) * vox;
  const int p = static_cast<int>(lrintf(atlas[v])) - 1;
  if (p >= 0 && p < parcels) atomicAdd(output + static_cast<size_t>(t) * parcels + p, fmri[linear]);
}

__global__ void parcel_scale_kernel(float* output, int nt, int parcels, const int* counts) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(nt) * parcels;
  if (i >= total) return;
  const int p = i % parcels;
  output[i] /= max(counts[p], 1);
}

__global__ void spatial_features_kernel(const float* series, int nt, int parcels, int channels, float* out) {
  const int p = blockIdx.x;
  const int c = threadIdx.x;
  if (p >= parcels || c >= channels) return;
  float value = 0.0f;
  if (c == 0 || c == 1 || c == 2 || c == 3) {
    float mean = 0.0f, sq = 0.0f, absmean = 0.0f, diff = 0.0f;
    for (int t = 0; t < nt; ++t) {
      const float x = series[static_cast<size_t>(t) * parcels + p];
      mean += x; sq += x * x; absmean += fabsf(x);
      if (t) { const float prev = series[static_cast<size_t>(t - 1) * parcels + p]; const float d = x - prev; diff += d * d; }
    }
    mean /= max(nt, 1); sq = sqrtf(fmaxf(sq / max(nt, 1) - mean * mean, 0.0f)); absmean /= max(nt, 1); diff = sqrtf(diff / max(nt - 1, 1));
    value = c == 0 ? mean : c == 1 ? sq : c == 2 ? absmean : diff;
  } else {
    const int k = c - 3;
    float s = 0.0f, q = 0.0f;
    for (int t = 0; t < nt; ++t) {
      const float angle = 6.28318530718f * k * t / max(nt, 1);
      const float x = series[static_cast<size_t>(t) * parcels + p];
      s += x * cosf(angle); q += x * sinf(angle);
    }
    value = sqrtf(s * s + q * q) / max(nt, 1);
  }
  out[static_cast<size_t>(p) * channels + c] = value;
}

__global__ void temporal_features_kernel(const float* series, int nt, int parcels, int channels, const float* projection, float* out) {
  const int t = blockIdx.x;
  const int c = threadIdx.x;
  if (t >= nt || c >= channels) return;
  float sum = 0.0f;
  for (int p = 0; p < parcels; ++p) {
    float x = series[static_cast<size_t>(t) * parcels + p];
    if (c >= channels / 2 && t) x -= series[static_cast<size_t>(t - 1) * parcels + p];
    sum += projection[static_cast<size_t>(p) * channels + c] * x;
  }
  out[static_cast<size_t>(t) * channels + c] = sum;
}

__global__ void tsnr_kernel(const float* x, size_t vox, int nt, float* stats) {
  const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= vox) return;
  float mean = 0.0f, sq = 0.0f;
  for (int t = 0; t < nt; ++t) { const float a = x[static_cast<size_t>(t) * vox + v]; mean += a; sq += a * a; }
  mean /= max(nt, 1); const float sd = sqrtf(fmaxf(sq / max(nt, 1) - mean * mean, 1e-8f));
  atomicAdd(stats + 0, fabsf(mean) / sd); atomicAdd(stats + 1, 1.0f);
}

__global__ void dvars_kernel(const float* x, size_t vox, int nt, float* stats) {
  const int t = blockIdx.x + 1;
  if (t >= nt) return;
  float local = 0.0f;
  for (size_t v = threadIdx.x; v < vox; v += blockDim.x) { const float d = x[static_cast<size_t>(t) * vox + v] - x[static_cast<size_t>(t - 1) * vox + v]; local += d * d; }
  __shared__ float buf[256]; buf[threadIdx.x] = local; __syncthreads();
  for (int s = 128; s; s >>= 1) { if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s]; __syncthreads(); }
  if (!threadIdx.x) { atomicAdd(stats + 2, sqrtf(buf[0] / max(static_cast<float>(vox), 1.0f))); atomicAdd(stats + 3, 1.0f); }
}
}

void parcel_project(const DeviceImage& fmri, const DeviceImage& atlas, int parcels, DeviceBuffer<float>& series, cudaStream_t stream) {
  if (atlas.shape.voxels() != fmri.shape.voxels()) throw std::runtime_error("Atlas and fMRI grid must match before parcel projection");
  const size_t vox = fmri.shape.voxels();
  DeviceBuffer<int> counts(parcels);
  series.resize(static_cast<size_t>(fmri.shape.t) * parcels);
  NTFM_CUDA(cudaMemsetAsync(counts.data(), 0, counts.bytes(), stream));
  NTFM_CUDA(cudaMemsetAsync(series.data(), 0, series.bytes(), stream));
  parcel_count_kernel<<<(vox + 255) / 256, 256, 0, stream>>>(atlas.data.data(), vox, parcels, counts.data());
  parcel_sum_kernel<<<(fmri.shape.elements() + 255) / 256, 256, 0, stream>>>(fmri.data.data(), atlas.data.data(), vox, fmri.shape.t, parcels, series.data());
  parcel_scale_kernel<<<(series.size() + 255) / 256, 256, 0, stream>>>(series.data(), fmri.shape.t, parcels, counts.data());
  NTFM_CUDA(cudaGetLastError());
}

void compile_spatial_features(const float* parcel_series, int timepoints, int parcels, int channels, DeviceBuffer<float>& output, cudaStream_t stream) {
  output.resize(static_cast<size_t>(parcels) * channels);
  spatial_features_kernel<<<parcels, channels, 0, stream>>>(parcel_series, timepoints, parcels, channels, output.data());
  NTFM_CUDA(cudaGetLastError());
}

void compile_temporal_features(const float* parcel_series, int timepoints, int parcels, int channels, const float* projection, DeviceBuffer<float>& output, cudaStream_t stream) {
  output.resize(static_cast<size_t>(timepoints) * channels);
  temporal_features_kernel<<<timepoints, channels, 0, stream>>>(parcel_series, timepoints, parcels, channels, projection, output.data());
  NTFM_CUDA(cudaGetLastError());
}

std::vector<float> quality_metrics(const DeviceImage& raw, const DeviceImage& clean, const std::vector<float>& motion, cudaStream_t stream) {
  DeviceBuffer<float> stats(4);
  NTFM_CUDA(cudaMemsetAsync(stats.data(), 0, stats.bytes(), stream));
  const size_t vox = clean.shape.voxels();
  tsnr_kernel<<<(vox + 255) / 256, 256, 0, stream>>>(clean.data.data(), vox, clean.shape.t, stats.data());
  if (clean.shape.t > 1) dvars_kernel<<<clean.shape.t - 1, 256, 0, stream>>>(clean.data.data(), vox, clean.shape.t, stats.data());
  std::vector<float> h(4); stats.copy_to_host(h.data(), h.size());
  float mean_fd = 0.0f, max_fd = 0.0f;
  for (int t = 1; t < clean.shape.t; ++t) {
    float fd = 0.0f;
    for (int j = 0; j < 6; ++j) fd += fabsf(motion[static_cast<size_t>(t) * 6 + j] - motion[static_cast<size_t>(t - 1) * 6 + j]);
    mean_fd += fd; max_fd = std::max(max_fd, fd);
  }
  mean_fd /= std::max(clean.shape.t - 1, 1);
  return {h[0] / std::max(h[1], 1.0f), h[2] / std::max(h[3], 1.0f), mean_fd, max_fd, static_cast<float>(clean.shape.t), static_cast<float>(vox), static_cast<float>(raw.shape.t), 1.0f};
}
}
