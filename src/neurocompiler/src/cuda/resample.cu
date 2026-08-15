#include "neurotask/operators.h"
#include "neurotask/cuda_check.h"
#include <array>
#include <stdexcept>

namespace ntfm {
namespace {
__device__ inline float at3(const float* x, int nx, int ny, int x0, int y0, int z0) { return x[(static_cast<size_t>(z0) * ny + y0) * nx + x0]; }

__device__ float trilinear(const float* x, int nx, int ny, int nz, float px, float py, float pz) {
  if (px < 0 || py < 0 || pz < 0 || px > nx - 1 || py > ny - 1 || pz > nz - 1) return 0.0f;
  const int x0 = static_cast<int>(floorf(px)), y0 = static_cast<int>(floorf(py)), z0 = static_cast<int>(floorf(pz));
  const int x1 = min(x0 + 1, nx - 1), y1 = min(y0 + 1, ny - 1), z1 = min(z0 + 1, nz - 1);
  const float dx = px - x0, dy = py - y0, dz = pz - z0;
  const float c00 = at3(x, nx, ny, x0, y0, z0) * (1 - dx) + at3(x, nx, ny, x1, y0, z0) * dx;
  const float c01 = at3(x, nx, ny, x0, y0, z1) * (1 - dx) + at3(x, nx, ny, x1, y0, z1) * dx;
  const float c10 = at3(x, nx, ny, x0, y1, z0) * (1 - dx) + at3(x, nx, ny, x1, y1, z0) * dx;
  const float c11 = at3(x, nx, ny, x0, y1, z1) * (1 - dx) + at3(x, nx, ny, x1, y1, z1) * dx;
  const float c0 = c00 * (1 - dy) + c10 * dy, c1 = c01 * (1 - dy) + c11 * dy;
  return c0 * (1 - dz) + c1 * dz;
}

__global__ void affine_volume_kernel(const float* input, int in_x, int in_y, int in_z, float* output, int out_x, int out_y, int out_z, const float* matrix, bool nearest) {
  const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(out_x) * out_y * out_z;
  if (linear >= total) return;
  const int ox = linear % out_x, oy = (linear / out_x) % out_y, oz = linear / (static_cast<size_t>(out_x) * out_y);
  const float px = matrix[0] * ox + matrix[1] * oy + matrix[2] * oz + matrix[3];
  const float py = matrix[4] * ox + matrix[5] * oy + matrix[6] * oz + matrix[7];
  const float pz = matrix[8] * ox + matrix[9] * oy + matrix[10] * oz + matrix[11];
  if (nearest) {
    const int ix = static_cast<int>(lrintf(px)), iy = static_cast<int>(lrintf(py)), iz = static_cast<int>(lrintf(pz));
    output[linear] = ix >= 0 && iy >= 0 && iz >= 0 && ix < in_x && iy < in_y && iz < in_z ? at3(input, in_x, in_y, ix, iy, iz) : 0.0f;
  } else output[linear] = trilinear(input, in_x, in_y, in_z, px, py, pz);
}

__global__ void reference_kernel(const float* x, float* ref, size_t vox, int nt) {
  const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= vox) return;
  float sum = 0.0f, weight = 0.0f;
  for (int t = 0; t < nt; ++t) { const float a = x[static_cast<size_t>(t) * vox + v]; if (isfinite(a)) { sum += a; weight += 1.0f; } }
  ref[v] = weight > 0 ? sum / weight : 0.0f;
}


__global__ void selected_reference_kernel(const float* x, const int* frames, int selected, float* ref, size_t vox) {
  const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= vox) return;
  float sum = 0.0f, weight = 0.0f;
  for (int i = 0; i < selected; ++i) { const float a = x[static_cast<size_t>(frames[i]) * vox + v]; if (isfinite(a)) { sum += a; weight += 1.0f; } }
  ref[v] = weight > 0 ? sum / weight : 0.0f;
}
__global__ void resample_kernel(const float* input, float* output, int nx, int ny, int nz, int nt, const float* base, const float* motion) {
  const size_t vox = static_cast<size_t>(nx) * ny * nz;
  const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = vox * nt;
  if (linear >= total) return;
  const int t = linear / vox;
  const size_t v = linear - static_cast<size_t>(t) * vox;
  const int ox = v % nx, oy = (v / nx) % ny, oz = v / (static_cast<size_t>(nx) * ny);
  const float cx = 0.5f * (nx - 1), cy = 0.5f * (ny - 1), cz = 0.5f * (nz - 1);
  const float x = ox - cx, y = oy - cy, z = oz - cz;
  const float tx = motion[t * 6 + 3], ty = motion[t * 6 + 4], tz = motion[t * 6 + 5];
  const float px = base[0] * x + base[1] * y + base[2] * z + base[3] + cx + tx;
  const float py = base[4] * x + base[5] * y + base[6] * z + base[7] + cy + ty;
  const float pz = base[8] * x + base[9] * y + base[10] * z + base[11] + cz + tz;
  output[linear] = trilinear(input + static_cast<size_t>(t) * vox, nx, ny, nz, px, py, pz);
}
}

void resample_volume_to_grid(const DeviceImage& input, Shape3D output_shape, const std::array<float, 16>& output_to_input, bool nearest, DeviceBuffer<float>& output, cudaStream_t stream) {
  if (input.shape.t != 1) throw std::runtime_error("resample_volume_to_grid requires one 3D input volume");
  const size_t total = output_shape.voxels(); output.resize(total); DeviceBuffer<float> matrix(16); matrix.copy_from_host(output_to_input.data(), 16);
  affine_volume_kernel<<<(total + 255) / 256, 256, 0, stream>>>(input.data.data(), input.shape.x, input.shape.y, input.shape.z, output.data(), output_shape.x, output_shape.y, output_shape.z, matrix.data(), nearest);
  NTFM_CUDA(cudaGetLastError());
}

void temporal_reference(const DeviceImage& fmri, DeviceBuffer<float>& reference, cudaStream_t stream) {
  const size_t vox = fmri.shape.voxels();
  reference.resize(vox);
  reference_kernel<<<(vox + 255) / 256, 256, 0, stream>>>(fmri.data.data(), reference.data(), vox, fmri.shape.t);
  NTFM_CUDA(cudaGetLastError());
}

void temporal_reference_selected(const DeviceImage& fmri, const std::vector<int>& frames, DeviceBuffer<float>& reference, cudaStream_t stream) {
  if (frames.empty()) { temporal_reference(fmri, reference, stream); return; }
  const size_t vox = fmri.shape.voxels(); DeviceBuffer<int> selected(frames.size()); selected.copy_from_host(frames.data(), frames.size()); reference.resize(vox);
  selected_reference_kernel<<<(vox + 255) / 256, 256, 0, stream>>>(fmri.data.data(), selected.data(), static_cast<int>(frames.size()), reference.data(), vox);
  NTFM_CUDA(cudaGetLastError());
}

void resample_series(const DeviceImage& fmri, DeviceImage& output, const RigidTransform& epi_to_t1, const std::vector<float>& motion, cudaStream_t stream) {
  output.shape = fmri.shape; output.spacing = fmri.spacing; output.affine = fmri.affine; output.data.resize(fmri.shape.elements());
  DeviceBuffer<float> d_base(16), d_motion(motion.size());
  d_base.copy_from_host(epi_to_t1.matrix.data(), 16); d_motion.copy_from_host(motion.data(), motion.size());
  const size_t total = fmri.shape.elements();
  resample_kernel<<<(total + 255) / 256, 256, 0, stream>>>(fmri.data.data(), output.data.data(), fmri.shape.x, fmri.shape.y, fmri.shape.z, fmri.shape.t, d_base.data(), d_motion.data());
  NTFM_CUDA(cudaGetLastError());
}
}
