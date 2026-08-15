#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ntfm {
inline void cuda_check(cudaError_t code, const char* file, int line) {
  if (code == cudaSuccess) return;
  throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) + ": " + cudaGetErrorString(code));
}

inline void require_b200(int device = 0) {
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, device), __FILE__, __LINE__);
  if (prop.major != 10 || prop.minor != 0) throw std::runtime_error("NeuroTaskFM native runtime requires NVIDIA B200/SM100");
}
}

#define NTFM_CUDA(expr) ::ntfm::cuda_check((expr), __FILE__, __LINE__)
