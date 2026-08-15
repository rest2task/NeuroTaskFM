#pragma once
#include "neurotask/device_buffer.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ntfm {
struct Shape3D { int x = 0, y = 0, z = 0; size_t voxels() const { return static_cast<size_t>(x) * y * z; } };
struct Shape4D { int x = 0, y = 0, z = 0, t = 0; size_t voxels() const { return static_cast<size_t>(x) * y * z; } size_t elements() const { return voxels() * t; } };

struct HostImage {
  Shape4D shape;
  std::array<float, 4> spacing{1, 1, 1, 1};
  std::array<float, 16> affine{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<float> data;
};

struct DeviceImage {
  Shape4D shape;
  std::array<float, 4> spacing{1, 1, 1, 1};
  std::array<float, 16> affine{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  DeviceBuffer<float> data;
};

struct RigidTransform { std::array<float, 16> matrix{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; };

struct CompilerOutput {
  std::vector<uint32_t> dicom_tokens;
  std::vector<float> t1_slices;
  std::vector<float> epi_slices;
  std::vector<float> compiled_spatial;
  std::vector<float> compiled_temporal;
  std::vector<float> parcel_series;
  std::vector<float> anatomy_graph;
  std::vector<float> motion;
  std::vector<float> qc;
  int slice_count = 0;
  int slice_size = 0;
  int parcel_count = 0;
  int temporal_count = 0;
  int feature_channels = 0;
  std::string metadata_json;
};
}
