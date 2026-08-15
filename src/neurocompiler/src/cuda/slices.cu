#include "neurotask/operators.h"
#include "neurotask/cuda_check.h"

namespace ntfm {
namespace {
__device__ inline float nearest(const float* x, int nx, int ny, int nz, float px, float py, float pz) {
  const int ix = min(max(static_cast<int>(lrintf(px)), 0), nx - 1);
  const int iy = min(max(static_cast<int>(lrintf(py)), 0), ny - 1);
  const int iz = min(max(static_cast<int>(lrintf(pz)), 0), nz - 1);
  return x[(static_cast<size_t>(iz) * ny + iy) * nx + ix];
}

__global__ void slice_kernel(const float* x, int nx, int ny, int nz, int count, int size, float* out) {
  const size_t pixels = static_cast<size_t>(3) * count * size * size;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pixels) return;
  const int u = i % size, v = (i / size) % size, s = (i / (static_cast<size_t>(size) * size)) % count, orientation = i / (static_cast<size_t>(count) * size * size);
  const float f = count > 1 ? static_cast<float>(s) / (count - 1) : 0.5f;
  const float a = size > 1 ? static_cast<float>(u) / (size - 1) : 0.5f;
  const float b = size > 1 ? static_cast<float>(v) / (size - 1) : 0.5f;
  float px = 0, py = 0, pz = 0;
  if (orientation == 0) { px = a * (nx - 1); py = b * (ny - 1); pz = f * (nz - 1); }
  else if (orientation == 1) { px = a * (nx - 1); py = f * (ny - 1); pz = b * (nz - 1); }
  else { px = f * (nx - 1); py = a * (ny - 1); pz = b * (nz - 1); }
  out[i] = nearest(x, nx, ny, nz, px, py, pz);
}
}

void extract_slices(const float* volume, Shape3D shape, int count, int size, DeviceBuffer<float>& output, cudaStream_t stream) {
  const size_t n = static_cast<size_t>(3) * count * size * size;
  output.resize(n);
  slice_kernel<<<(n + 255) / 256, 256, 0, stream>>>(volume, shape.x, shape.y, shape.z, count, size, output.data());
  NTFM_CUDA(cudaGetLastError());
}
}
