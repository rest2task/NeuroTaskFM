#include "neurotaskfm/imaging_tools.h"

#include <cufft.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "neurotask/cuda_check.h"
#include "neurotask/device_buffer.h"

namespace neurotaskfm {
namespace {

void cufft_check(const cufftResult status, const char* operation) {
  if (status == CUFFT_SUCCESS) return;
  throw std::runtime_error(std::string("cuFFT failure in ") + operation + ": code " +
                           std::to_string(static_cast<int>(status)));
}

class CufftPlan {
 public:
  CufftPlan() = default;
  ~CufftPlan() { if (handle_ != 0) cufftDestroy(handle_); }
  CufftPlan(const CufftPlan&) = delete;
  CufftPlan& operator=(const CufftPlan&) = delete;
  cufftHandle* address() { return &handle_; }
  cufftHandle get() const { return handle_; }

 private:
  cufftHandle handle_ = 0;
};

__global__ void shift_complex(const cufftComplex* source, cufftComplex* destination,
                              const std::int64_t batches, const std::int64_t depth,
                              const std::int64_t height, const std::int64_t width,
                              const int fft_dimensions, const bool inverse) {
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto spatial = depth * height * width;
  const auto count = batches * spatial;
  if (index >= count) return;
  const auto batch = index / spatial;
  const auto voxel = index % spatial;
  const auto x = voxel % width;
  const auto y = (voxel / width) % height;
  const auto z = voxel / (width * height);
  const auto sx = (x + (inverse ? width / 2 : (width + 1) / 2)) % width;
  const auto sy = (y + (inverse ? height / 2 : (height + 1) / 2)) % height;
  const auto sz = fft_dimensions == 3
      ? (z + (inverse ? depth / 2 : (depth + 1) / 2)) % depth : z;
  const auto source_index = batch * spatial + (sz * height + sy) * width + sx;
  destination[index] = source[source_index];
}

__global__ void root_sum_squares(const cufftComplex* values, float* magnitude,
                                 const std::int64_t frames, const std::int64_t coils,
                                 const std::int64_t spatial, const float scale) {
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= frames * spatial) return;
  const auto frame = index / spatial;
  const auto voxel = index % spatial;
  float sum = 0.0F;
  for (std::int64_t coil = 0; coil < coils; ++coil) {
    const auto value = values[(frame * coils + coil) * spatial + voxel];
    sum = fmaf(value.x, value.x, sum);
    sum = fmaf(value.y, value.y, sum);
  }
  magnitude[index] = sqrtf(sum) * scale;
}

std::size_t checked_product(const CartesianKSpace& input) {
  const std::array<std::int64_t, 5> dimensions{
      input.frames, input.coils, input.depth, input.height, input.width};
  std::size_t product = 1;
  for (const auto dimension : dimensions) {
    if (dimension <= 0) throw std::invalid_argument("k-space dimensions must be positive");
    const auto value = static_cast<std::size_t>(dimension);
    if (product > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error("k-space allocation exceeds addressable memory");
    }
    product *= value;
  }
  return product;
}

}  // namespace

ReconstructedImage reconstruct_cartesian_cuda(const CartesianKSpace& input) {
  if (input.fft_dimensions != 2 && input.fft_dimensions != 3) {
    throw std::invalid_argument("fft_dimensions must be 2 or 3");
  }
  if (input.fft_dimensions == 3 && input.depth == 1) {
    throw std::invalid_argument("3D reconstruction requires a depth greater than one");
  }
  ntfm::require_b200();
  const auto element_count = checked_product(input);
  if (input.samples.size() != element_count) {
    throw std::invalid_argument("k-space sample count does not match its declared shape");
  }
  const auto batches64 = input.frames * input.coils * (input.fft_dimensions == 2 ? input.depth : 1);
  if (batches64 > std::numeric_limits<int>::max()) throw std::overflow_error("cuFFT batch count exceeds int range");

  std::vector<cufftComplex> host(element_count);
  for (std::size_t index = 0; index < element_count; ++index) {
    host[index].x = input.samples[index].real;
    host[index].y = input.samples[index].imag;
  }
  ntfm::DeviceBuffer<cufftComplex> source(element_count);
  ntfm::DeviceBuffer<cufftComplex> workspace(element_count);
  source.copy_from_host(host.data(), host.size());

  constexpr int threads = 256;
  const auto blocks = static_cast<unsigned int>((element_count + threads - 1) / threads);
  shift_complex<<<blocks, threads>>>(source.data(), workspace.data(), batches64, input.depth,
                                     input.height, input.width, input.fft_dimensions, true);
  NTFM_CUDA(cudaGetLastError());

  CufftPlan plan;
  if (input.fft_dimensions == 2) {
    int dimensions[2]{static_cast<int>(input.height), static_cast<int>(input.width)};
    cufft_check(cufftPlanMany(plan.address(), 2, dimensions, nullptr, 1, 0, nullptr, 1, 0,
                              CUFFT_C2C, static_cast<int>(batches64)), "2D plan");
  } else {
    int dimensions[3]{static_cast<int>(input.depth), static_cast<int>(input.height),
                      static_cast<int>(input.width)};
    cufft_check(cufftPlanMany(plan.address(), 3, dimensions, nullptr, 1, 0, nullptr, 1, 0,
                              CUFFT_C2C, static_cast<int>(batches64)), "3D plan");
  }
  cufft_check(cufftExecC2C(plan.get(), workspace.data(), workspace.data(), CUFFT_INVERSE), "inverse transform");

  shift_complex<<<blocks, threads>>>(workspace.data(), source.data(), batches64, input.depth,
                                     input.height, input.width, input.fft_dimensions, false);
  NTFM_CUDA(cudaGetLastError());
  const auto spatial = input.depth * input.height * input.width;
  const auto output_count = input.frames * spatial;
  ntfm::DeviceBuffer<float> magnitude(static_cast<std::size_t>(output_count));
  const auto output_blocks = static_cast<unsigned int>((output_count + threads - 1) / threads);
  const auto transform_size = input.fft_dimensions == 2
      ? input.height * input.width : input.depth * input.height * input.width;
  root_sum_squares<<<output_blocks, threads>>>(source.data(), magnitude.data(), input.frames,
                                               input.coils, spatial,
                                               1.0F / static_cast<float>(transform_size));
  NTFM_CUDA(cudaGetLastError());
  NTFM_CUDA(cudaDeviceSynchronize());

  ReconstructedImage output;
  output.frames = input.frames;
  output.depth = input.depth;
  output.height = input.height;
  output.width = input.width;
  output.magnitude.resize(static_cast<std::size_t>(output_count));
  magnitude.copy_to_host(output.magnitude.data(), output.magnitude.size());
  return output;
}

}  // namespace neurotaskfm
