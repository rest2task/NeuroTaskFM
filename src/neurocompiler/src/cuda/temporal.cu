#include "neurotask/operators.h"
#include "neurotask/cuda_check.h"
#include <cublas_v2.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ntfm {
namespace {
void cublas_check(cublasStatus_t status) { if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure in nuisance regression"); }

__global__ void add_ridge(float* a, int n, float ridge) { const int i = threadIdx.x; if (i < n) a[i + i * n] += ridge; }

__global__ void invert_small(float* a, float* inv, int n) {
  if (blockIdx.x || threadIdx.x) return;
  for (int i = 0; i < n * n; ++i) inv[i] = 0.0f;
  for (int i = 0; i < n; ++i) inv[i + i * n] = 1.0f;
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    float best = fabsf(a[col + col * n]);
    for (int row = col + 1; row < n; ++row) { const float v = fabsf(a[row + col * n]); if (v > best) { best = v; pivot = row; } }
    if (pivot != col) for (int j = 0; j < n; ++j) { const int ia = col + j * n, ib = pivot + j * n; const float ta = a[ia], ti = inv[ia]; a[ia] = a[ib]; inv[ia] = inv[ib]; a[ib] = ta; inv[ib] = ti; }
    const float diag = fabsf(a[col + col * n]) < 1e-8f ? 1e-8f : a[col + col * n];
    for (int j = 0; j < n; ++j) { a[col + j * n] /= diag; inv[col + j * n] /= diag; }
    for (int row = 0; row < n; ++row) if (row != col) {
      const float factor = a[row + col * n];
      for (int j = 0; j < n; ++j) { a[row + j * n] -= factor * a[col + j * n]; inv[row + j * n] -= factor * inv[col + j * n]; }
    }
  }
}

__global__ void subtract_kernel(float* y, const float* fit, size_t n) { const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; if (i < n) y[i] -= fit[i]; }
}

std::vector<float> build_design(int t, const std::vector<float>& motion, int& k) {
  k = 15;
  std::vector<float> x(static_cast<size_t>(t) * k, 0.0f);
  for (int i = 0; i < t; ++i) {
    const float u = t > 1 ? (2.0f * i / (t - 1) - 1.0f) : 0.0f;
    x[i + 0 * t] = 1.0f; x[i + 1 * t] = u; x[i + 2 * t] = u * u;
    for (int j = 0; j < 6; ++j) {
      const float v = motion[static_cast<size_t>(i) * 6 + j];
      const float prev = i > 0 ? motion[static_cast<size_t>(i - 1) * 6 + j] : v;
      x[i + (3 + j) * t] = v;
      x[i + (9 + j) * t] = v - prev;
    }
  }
  return x;
}
}

void nuisance_regression(DeviceImage& fmri, const std::vector<float>& motion, cudaStream_t stream) {
  const int t = fmri.shape.t;
  const int v = static_cast<int>(fmri.shape.voxels());
  int k = 0;
  const auto design = build_design(t, motion, k);
  DeviceBuffer<float> x(design.size()), xtx(static_cast<size_t>(k) * k), inv(static_cast<size_t>(k) * k), b1(static_cast<size_t>(v) * k), beta(static_cast<size_t>(v) * k), fit(fmri.shape.elements());
  x.copy_from_host(design.data(), design.size());
  cublasHandle_t handle;
  cublas_check(cublasCreate(&handle)); cublas_check(cublasSetStream(handle, stream));
  const float one = 1.0f, zero = 0.0f;
  cublas_check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, k, t, &one, x.data(), t, x.data(), t, &zero, xtx.data(), k));
  add_ridge<<<1, 32, 0, stream>>>(xtx.data(), k, 1e-3f);
  invert_small<<<1, 1, 0, stream>>>(xtx.data(), inv.data(), k);
  cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, v, k, t, &one, fmri.data.data(), v, x.data(), t, &zero, b1.data(), v));
  cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, v, k, k, &one, b1.data(), v, inv.data(), k, &zero, beta.data(), v));
  cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, v, t, k, &one, beta.data(), v, x.data(), t, &zero, fit.data(), v));
  subtract_kernel<<<(fmri.shape.elements() + 255) / 256, 256, 0, stream>>>(fmri.data.data(), fit.data(), fmri.shape.elements());
  NTFM_CUDA(cudaGetLastError());
  cublasDestroy(handle);
}
}
