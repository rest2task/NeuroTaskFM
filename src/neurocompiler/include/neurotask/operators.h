#pragma once
#include "neurotask/types.h"
#include <array>
#include <vector>

namespace ntfm {
void robust_normalize(DeviceBuffer<float>& data, size_t n, float lower, float upper, int bins, cudaStream_t stream);
void temporal_reference(const DeviceImage& fmri, DeviceBuffer<float>& reference, cudaStream_t stream);
void temporal_reference_selected(const DeviceImage& fmri, const std::vector<int>& frames, DeviceBuffer<float>& reference, cudaStream_t stream);
void resample_volume_to_grid(const DeviceImage& input, Shape3D output_shape, const std::array<float, 16>& output_to_input, bool nearest, DeviceBuffer<float>& output, cudaStream_t stream);
void sobel3d(const float* input, float* output, Shape3D shape, cudaStream_t stream);
RigidTransform rigid_ncc(const float* moving, Shape3D moving_shape, const float* fixed, Shape3D fixed_shape, const std::vector<int>& iterations, float rotation_step, float translation_step, const std::array<float, 6>& initial, cudaStream_t stream);
std::vector<float> estimate_series_motion(const DeviceImage& fmri, const float* reference, cudaStream_t stream);
void resample_series(const DeviceImage& fmri, DeviceImage& output, const RigidTransform& epi_to_t1, const std::vector<float>& motion, cudaStream_t stream);
void nuisance_regression(DeviceImage& fmri, const std::vector<float>& motion, cudaStream_t stream);
void parcel_project(const DeviceImage& fmri, const DeviceImage& atlas, int parcels, DeviceBuffer<float>& series, cudaStream_t stream);
void compile_spatial_features(const float* parcel_series, int timepoints, int parcels, int channels, DeviceBuffer<float>& output, cudaStream_t stream);
void compile_temporal_features(const float* parcel_series, int timepoints, int parcels, int channels, const float* projection, DeviceBuffer<float>& output, cudaStream_t stream);
void extract_slices(const float* volume, Shape3D shape, int count, int size, DeviceBuffer<float>& output, cudaStream_t stream);
std::vector<float> quality_metrics(const DeviceImage& raw, const DeviceImage& clean, const std::vector<float>& motion, cudaStream_t stream);
}
