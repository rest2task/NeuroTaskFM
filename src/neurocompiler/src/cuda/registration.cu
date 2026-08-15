#include "neurotask/operators.h"
#include "neurotask/cuda_check.h"
#include <array>
#include <cmath>
#include <vector>

namespace ntfm {
namespace {
__device__ inline float reg_at3(const float* x, int nx, int ny, int x0, int y0, int z0) { return x[(static_cast<size_t>(z0) * ny + y0) * nx + x0]; }

__device__ float sample3d(const float* x, int nx, int ny, int nz, float px, float py, float pz) {
  if (px < 0 || py < 0 || pz < 0 || px > nx - 1 || py > ny - 1 || pz > nz - 1) return 0.0f;
  const int x0 = static_cast<int>(floorf(px)), y0 = static_cast<int>(floorf(py)), z0 = static_cast<int>(floorf(pz));
  const int x1 = min(x0 + 1, nx - 1), y1 = min(y0 + 1, ny - 1), z1 = min(z0 + 1, nz - 1);
  const float dx = px - x0, dy = py - y0, dz = pz - z0;
  const float c00 = reg_at3(x, nx, ny, x0, y0, z0) * (1 - dx) + reg_at3(x, nx, ny, x1, y0, z0) * dx;
  const float c01 = reg_at3(x, nx, ny, x0, y0, z1) * (1 - dx) + reg_at3(x, nx, ny, x1, y0, z1) * dx;
  const float c10 = reg_at3(x, nx, ny, x0, y1, z0) * (1 - dx) + reg_at3(x, nx, ny, x1, y1, z0) * dx;
  const float c11 = reg_at3(x, nx, ny, x0, y1, z1) * (1 - dx) + reg_at3(x, nx, ny, x1, y1, z1) * dx;
  const float c0 = c00 * (1 - dy) + c10 * dy, c1 = c01 * (1 - dy) + c11 * dy;
  return c0 * (1 - dz) + c1 * dz;
}

__global__ void ncc_accumulate(const float* moving, int mx, int my, int mz, const float* fixed, int fx, int fy, int fz, const float* mat, int stride, float* stats) {
  const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t sx = (fx + stride - 1) / stride, sy = (fy + stride - 1) / stride, sz = (fz + stride - 1) / stride;
  const size_t n = sx * sy * sz;
  if (linear >= n) return;
  const int ix = static_cast<int>(linear % sx) * stride;
  const int iy = static_cast<int>((linear / sx) % sy) * stride;
  const int iz = static_cast<int>(linear / (sx * sy)) * stride;
  if (ix >= fx || iy >= fy || iz >= fz) return;
  const float cx = 0.5f * (fx - 1), cy = 0.5f * (fy - 1), cz = 0.5f * (fz - 1);
  const float x = ix - cx, y = iy - cy, z = iz - cz;
  const float px = mat[0] * x + mat[1] * y + mat[2] * z + mat[3] + 0.5f * (mx - 1);
  const float py = mat[4] * x + mat[5] * y + mat[6] * z + mat[7] + 0.5f * (my - 1);
  const float pz = mat[8] * x + mat[9] * y + mat[10] * z + mat[11] + 0.5f * (mz - 1);
  const float a = sample3d(moving, mx, my, mz, px, py, pz), b = fixed[(static_cast<size_t>(iz) * fy + iy) * fx + ix];
  atomicAdd(stats + 0, a); atomicAdd(stats + 1, b); atomicAdd(stats + 2, a * a); atomicAdd(stats + 3, b * b); atomicAdd(stats + 4, a * b); atomicAdd(stats + 5, 1.0f);
}

__global__ void moment_kernel(const float* x, int nx, int ny, int nz, int nt, float* out) {
  const int t = blockIdx.y;
  const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t vox = static_cast<size_t>(nx) * ny * nz;
  if (v >= vox || t >= nt) return;
  const int ix = v % nx, iy = (v / nx) % ny, iz = v / (static_cast<size_t>(nx) * ny);
  const float w = fabsf(x[static_cast<size_t>(t) * vox + v]);
  atomicAdd(out + t * 4 + 0, w); atomicAdd(out + t * 4 + 1, w * ix); atomicAdd(out + t * 4 + 2, w * iy); atomicAdd(out + t * 4 + 3, w * iz);
}

__global__ void sobel_kernel(const float* x, float* y, int nx, int ny, int nz) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t n = static_cast<size_t>(nx) * ny * nz;
  if (i >= n) return;
  const int px = i % nx, py = (i / nx) % ny, pz = i / (static_cast<size_t>(nx) * ny);
  if (px == 0 || py == 0 || pz == 0 || px == nx - 1 || py == ny - 1 || pz == nz - 1) { y[i] = 0; return; }
  const float gx = 0.5f * (reg_at3(x, nx, ny, px + 1, py, pz) - reg_at3(x, nx, ny, px - 1, py, pz));
  const float gy = 0.5f * (reg_at3(x, nx, ny, px, py + 1, pz) - reg_at3(x, nx, ny, px, py - 1, pz));
  const float gz = 0.5f * (reg_at3(x, nx, ny, px, py, pz + 1) - reg_at3(x, nx, ny, px, py, pz - 1));
  y[i] = sqrtf(gx * gx + gy * gy + gz * gz);
}

std::array<float, 16> matrix_from_params(const std::array<float, 6>& p) {
  const float cx = cosf(p[0]), sx = sinf(p[0]), cy = cosf(p[1]), sy = sinf(p[1]), cz = cosf(p[2]), sz = sinf(p[2]);
  return {cy * cz, -cy * sz, sy, p[3], sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy, p[4], -cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy, p[5], 0, 0, 0, 1};
}

float score(const float* moving, Shape3D ms, const float* fixed, Shape3D fs, const std::array<float, 6>& p, int stride, cudaStream_t stream) {
  const auto m = matrix_from_params(p);
  DeviceBuffer<float> d_mat(16), d_stats(6);
  d_mat.copy_from_host(m.data(), 16);
  NTFM_CUDA(cudaMemsetAsync(d_stats.data(), 0, d_stats.bytes(), stream));
  const size_t n = static_cast<size_t>((fs.x + stride - 1) / stride) * ((fs.y + stride - 1) / stride) * ((fs.z + stride - 1) / stride);
  ncc_accumulate<<<(n + 255) / 256, 256, 0, stream>>>(moving, ms.x, ms.y, ms.z, fixed, fs.x, fs.y, fs.z, d_mat.data(), stride, d_stats.data());
  NTFM_CUDA(cudaGetLastError());
  std::array<float, 6> s{};
  d_stats.copy_to_host(s.data(), s.size());
  const float count = fmaxf(s[5], 1.0f), ma = s[0] / count, mb = s[1] / count;
  const float va = fmaxf(s[2] / count - ma * ma, 1e-8f), vb = fmaxf(s[3] / count - mb * mb, 1e-8f);
  return (s[4] / count - ma * mb) / sqrtf(va * vb);
}
}

void sobel3d(const float* input, float* output, Shape3D shape, cudaStream_t stream) {
  const size_t n = shape.voxels();
  sobel_kernel<<<(n + 255) / 256, 256, 0, stream>>>(input, output, shape.x, shape.y, shape.z);
  NTFM_CUDA(cudaGetLastError());
}

RigidTransform rigid_ncc(const float* moving, Shape3D moving_shape, const float* fixed, Shape3D fixed_shape, const std::vector<int>& iterations, float rotation_step, float translation_step, const std::array<float, 6>& initial, cudaStream_t stream) {
  std::array<float, 6> p = initial;
  float rstep = rotation_step * 4.0f, tstep = translation_step * 4.0f;
  for (size_t level = 0; level < iterations.size(); ++level) {
    const int stride = 1 << static_cast<int>(iterations.size() - level - 1);
    float best = score(moving, moving_shape, fixed, fixed_shape, p, stride, stream);
    for (int it = 0; it < iterations[level]; ++it) {
      bool improved = false;
      for (int d = 0; d < 6; ++d) {
        const float step = d < 3 ? rstep : tstep;
        auto plus = p, minus = p;
        plus[d] += step; minus[d] -= step;
        const float sp = score(moving, moving_shape, fixed, fixed_shape, plus, stride, stream);
        const float sm = score(moving, moving_shape, fixed, fixed_shape, minus, stride, stream);
        if (sp > best || sm > best) { p = sp >= sm ? plus : minus; best = fmaxf(sp, sm); improved = true; }
      }
      if (!improved) { rstep *= 0.7f; tstep *= 0.7f; }
    }
    rstep *= 0.5f; tstep *= 0.5f;
  }
  RigidTransform out; out.matrix = matrix_from_params(p); return out;
}

std::vector<float> estimate_series_motion(const DeviceImage& fmri, const float* reference, cudaStream_t stream) {
  const size_t vox = fmri.shape.voxels();
  DeviceBuffer<float> moments(static_cast<size_t>(fmri.shape.t) * 4), ref_moments(4);
  NTFM_CUDA(cudaMemsetAsync(moments.data(), 0, moments.bytes(), stream));
  NTFM_CUDA(cudaMemsetAsync(ref_moments.data(), 0, ref_moments.bytes(), stream));
  moment_kernel<<<dim3((vox + 255) / 256, fmri.shape.t), 256, 0, stream>>>(fmri.data.data(), fmri.shape.x, fmri.shape.y, fmri.shape.z, fmri.shape.t, moments.data());
  moment_kernel<<<dim3((vox + 255) / 256, 1), 256, 0, stream>>>(reference, fmri.shape.x, fmri.shape.y, fmri.shape.z, 1, ref_moments.data());
  std::vector<float> host(static_cast<size_t>(fmri.shape.t) * 4), ref_host(4), motion(static_cast<size_t>(fmri.shape.t) * 6, 0.0f);
  moments.copy_to_host(host.data(), host.size()); ref_moments.copy_to_host(ref_host.data(), 4);
  const float rw = fmaxf(ref_host[0], 1e-6f), rx = ref_host[1] / rw, ry = ref_host[2] / rw, rz = ref_host[3] / rw;
  for (int t = 0; t < fmri.shape.t; ++t) {
    const float w = fmaxf(host[t * 4], 1e-6f);
    motion[t * 6 + 3] = host[t * 4 + 1] / w - rx; motion[t * 6 + 4] = host[t * 4 + 2] / w - ry; motion[t * 6 + 5] = host[t * 4 + 3] / w - rz;
  }
  return motion;
}
}
