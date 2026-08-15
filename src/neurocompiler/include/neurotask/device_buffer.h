#pragma once
#include "neurotask/cuda_check.h"
#include <cstddef>
#include <utility>

namespace ntfm {
template <class T> class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { resize(n); }
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept { if (this != &other) { reset(); swap(other); } return *this; }
  void resize(size_t n) { if (n == size_) return; reset(); if (n) NTFM_CUDA(cudaMalloc(&ptr_, n * sizeof(T))); size_ = n; }
  void reset() { if (ptr_) cudaFree(ptr_); ptr_ = nullptr; size_ = 0; }
  void copy_from_host(const T* src, size_t n) { resize(n); NTFM_CUDA(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice)); }
  void copy_to_host(T* dst, size_t n) const { if (n > size_) throw std::runtime_error("DeviceBuffer copy exceeds allocation"); NTFM_CUDA(cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost)); }
  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t size() const { return size_; }
  size_t bytes() const { return size_ * sizeof(T); }
 private:
  void swap(DeviceBuffer& other) noexcept { std::swap(ptr_, other.ptr_); std::swap(size_, other.size_); }
  T* ptr_ = nullptr;
  size_t size_ = 0;
};
}
