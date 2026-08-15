#include "neurotask/operator_graph.h"
#include "neurotask/cuda_check.h"
#include "neurotask/compiler_artifacts.h"
#include "neurotask/dicom_tokenizer.h"
#include "neurotask/geometry.h"
#include "neurotask/nifti_reader.h"
#include "neurotask/operators.h"
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace ntfm {
namespace {
std::vector<float> atlas_adjacency(const HostImage& atlas, int parcels) {
  std::vector<float> graph(static_cast<size_t>(parcels) * parcels, 0.0f);
  const int nx = atlas.shape.x, ny = atlas.shape.y, nz = atlas.shape.z;
  auto parcel_at = [&](int x, int y, int z) {
    const size_t index = (static_cast<size_t>(z) * ny + y) * nx + x;
    const float value = atlas.data[index];
    if (!std::isfinite(value)) return -1;
    const int parcel = static_cast<int>(std::lround(value)) - 1;
    return parcel >= 0 && parcel < parcels ? parcel : -1;
  };
  auto connect = [&](int first, int second) {
    if (first < 0 || second < 0 || first == second) return;
    graph[static_cast<size_t>(first) * parcels + second] = 1.0f;
    graph[static_cast<size_t>(second) * parcels + first] = 1.0f;
  };
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const int parcel = parcel_at(x, y, z);
        if (x + 1 < nx) connect(parcel, parcel_at(x + 1, y, z));
        if (y + 1 < ny) connect(parcel, parcel_at(x, y + 1, z));
        if (z + 1 < nz) connect(parcel, parcel_at(x, y, z + 1));
      }
    }
  }
  return graph;
}
}

OperatorGraph::OperatorGraph(const std::string& config_path): config_path_(config_path) {
  if (!std::filesystem::exists(config_path_)) throw std::runtime_error("Missing NeuroCompiler config: " + config_path_);
}

CompilerOutput OperatorGraph::run(const CompileRequest& request) const {
  require_b200();
  const YAML::Node config = YAML::LoadFile(config_path_);
  const int slice_size = config["output"]["slice_size"].as<int>();
  const int slice_count = config["output"]["slice_count_per_orientation"].as<int>();
  const int parcels = config["output"]["parcel_count"].as<int>();
  const int channels = config["output"]["temporal_channels"].as<int>();
  const std::filesystem::path config_dir = std::filesystem::absolute(config_path_).parent_path();
  std::filesystem::path artifact_path = config["artifacts"]["path"].as<std::string>();
  if (artifact_path.is_relative()) artifact_path = config_dir / artifact_path;
  const CompilerArtifacts artifacts = load_compiler_artifacts(artifact_path.string(), parcels, channels);
  float lower = 0.5f, upper = 99.5f, rotation_step = 0.0025f, translation_step = 0.25f, low_motion_fraction = 0.35f;
  int bins = 4096, minimum_reference_frames = 24;
  std::vector<int> iterations{40, 30, 20, 12};
  for (const auto& op : config["operators"]) {
    const std::string type = op["type"].as<std::string>();
    if (type == "robust_normalize") { lower = op["params"]["lower_percentile"].as<float>(); upper = op["params"]["upper_percentile"].as<float>(); bins = op["params"]["bins"].as<int>(); }
    if (type == "temporal_reference") { low_motion_fraction = op["params"]["low_motion_fraction"].as<float>(); minimum_reference_frames = op["params"]["minimum_frames"].as<int>(); }
    if (type == "rigid_ncc") {
      rotation_step = op["params"]["rotation_step"].as<float>(); translation_step = op["params"]["translation_step"].as<float>(); iterations.clear();
      for (const auto& item : op["params"]["iterations"]) iterations.push_back(item.as<int>());
    }
  }

  HostImage t1_host = read_nifti(request.t1_nifti), fmri_host = read_nifti(request.fmri_nifti), atlas_host = read_nifti(request.atlas_nifti);
  if (t1_host.shape.t != 1) throw std::runtime_error("T1 input must contain one 3D volume");
  if (fmri_host.shape.t < 2) throw std::runtime_error("fMRI input must contain at least two 3D volumes");
  if (atlas_host.shape.t != 1) throw std::runtime_error("Atlas input must contain one 3D label map");
  DeviceImage t1 = upload_image(t1_host), fmri = upload_image(fmri_host), atlas = upload_image(atlas_host);
  cudaStream_t stream;
  NTFM_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  robust_normalize(t1.data, t1.shape.elements(), lower, upper, bins, stream);
  robust_normalize(fmri.data, fmri.shape.elements(), lower, upper, bins, stream);

  const Shape3D epi_grid{fmri.shape.x, fmri.shape.y, fmri.shape.z};
  DeviceBuffer<float> t1_on_epi, atlas_on_epi;
  const Matrix4 epi_to_t1_header = target_to_source_voxel(t1.affine, fmri.affine);
  const Matrix4 epi_to_atlas_header = target_to_source_voxel(atlas.affine, fmri.affine);
  resample_volume_to_grid(t1, epi_grid, epi_to_t1_header, false, t1_on_epi, stream);
  resample_volume_to_grid(atlas, epi_grid, epi_to_atlas_header, true, atlas_on_epi, stream);
  DeviceImage atlas_epi; atlas_epi.shape = {fmri.shape.x, fmri.shape.y, fmri.shape.z, 1}; atlas_epi.spacing = fmri.spacing; atlas_epi.affine = fmri.affine; atlas_epi.data = std::move(atlas_on_epi);

  DeviceBuffer<float> epi_reference;
  temporal_reference(fmri, epi_reference, stream);
  const std::vector<float> preliminary_motion = estimate_series_motion(fmri, epi_reference.data(), stream);
  std::vector<std::pair<float, int>> ranked; ranked.reserve(fmri.shape.t);
  for (int t = 0; t < fmri.shape.t; ++t) { const size_t offset = static_cast<size_t>(t) * 6; const float magnitude = std::abs(preliminary_motion[offset + 3]) + std::abs(preliminary_motion[offset + 4]) + std::abs(preliminary_motion[offset + 5]); ranked.emplace_back(magnitude, t); }
  std::sort(ranked.begin(), ranked.end()); const int selected_count = std::min(fmri.shape.t, std::max(minimum_reference_frames, static_cast<int>(std::ceil(fmri.shape.t * low_motion_fraction))));
  std::vector<int> selected; selected.reserve(selected_count); for (int i = 0; i < selected_count; ++i) selected.push_back(ranked[i].second);
  temporal_reference_selected(fmri, selected, epi_reference, stream);
  DeviceBuffer<float> t1_edges(fmri.shape.voxels()), epi_edges(fmri.shape.voxels());
  sobel3d(t1_on_epi.data(), t1_edges.data(), epi_grid, stream);
  sobel3d(epi_reference.data(), epi_edges.data(), epi_grid, stream);
  const RigidTransform epi_to_t1 = rigid_ncc(epi_edges.data(), epi_grid, t1_edges.data(), epi_grid, iterations, rotation_step, translation_step, artifacts.alignment_prior, stream);
  const std::vector<float> motion = estimate_series_motion(fmri, epi_reference.data(), stream);
  DeviceImage clean;
  resample_series(fmri, clean, epi_to_t1, motion, stream);
  nuisance_regression(clean, motion, stream);

  DeviceBuffer<float> parcel_series, spatial, temporal, clean_reference;
  parcel_project(clean, atlas_epi, parcels, parcel_series, stream);
  compile_spatial_features(parcel_series.data(), clean.shape.t, parcels, channels, spatial, stream);
  DeviceBuffer<float> projection(artifacts.temporal_projection.size());
  projection.copy_from_host(artifacts.temporal_projection.data(), artifacts.temporal_projection.size());
  compile_temporal_features(parcel_series.data(), clean.shape.t, parcels, channels, projection.data(), temporal, stream);
  temporal_reference(clean, clean_reference, stream);
  DeviceBuffer<float> t1_slices, epi_slices;
  extract_slices(t1.data.data(), {t1.shape.x, t1.shape.y, t1.shape.z}, slice_count, slice_size, t1_slices, stream);
  extract_slices(clean_reference.data(), {clean.shape.x, clean.shape.y, clean.shape.z}, slice_count, slice_size, epi_slices, stream);
  NTFM_CUDA(cudaStreamSynchronize(stream));

  CompilerOutput out;
  out.slice_count = slice_count; out.slice_size = slice_size; out.parcel_count = parcels; out.temporal_count = clean.shape.t; out.feature_channels = channels;
  out.motion = motion;
  out.qc = quality_metrics(fmri, clean, motion, stream);
  out.t1_slices.resize(t1_slices.size()); t1_slices.copy_to_host(out.t1_slices.data(), out.t1_slices.size());
  out.epi_slices.resize(epi_slices.size()); epi_slices.copy_to_host(out.epi_slices.data(), out.epi_slices.size());
  out.compiled_spatial.resize(spatial.size()); spatial.copy_to_host(out.compiled_spatial.data(), out.compiled_spatial.size());
  out.compiled_temporal.resize(temporal.size()); temporal.copy_to_host(out.compiled_temporal.data(), out.compiled_temporal.size());
  out.parcel_series.resize(parcel_series.size()); parcel_series.copy_to_host(out.parcel_series.data(), out.parcel_series.size());
  out.anatomy_graph = atlas_adjacency(atlas_host, parcels);
  DicomTokenizer tokenizer({4096, 262144, true});
  if (!request.t1_dicom.empty() && std::filesystem::exists(request.t1_dicom)) out.dicom_tokens = tokenizer.tokenize_tree(request.t1_dicom);
  if (!request.fmri_dicom.empty() && std::filesystem::exists(request.fmri_dicom)) {
    auto f = tokenizer.tokenize_tree(request.fmri_dicom);
    if (!out.dicom_tokens.empty()) out.dicom_tokens.push_back(65541);
    out.dicom_tokens.insert(out.dicom_tokens.end(), f.begin(), f.end());
    if (out.dicom_tokens.size() > 8192) out.dicom_tokens.resize(8192);
  }
  nlohmann::json meta = {
    {"subject_key", request.subject_key}, {"task", request.task}, {"tr_seconds", request.tr_seconds},
    {"t1_shape", {t1.shape.x, t1.shape.y, t1.shape.z}}, {"fmri_shape", {clean.shape.x, clean.shape.y, clean.shape.z, clean.shape.t}},
    {"header_resampling", "t1_and_atlas_to_epi_lattice"},
    {"compiler", config["name"].as<std::string>()}, {"compiler_artifact", artifact_path.string()}, {"artifact_provenance", artifacts.provenance},
    {"representation", "model_aligned_not_conventional_activation"},
    {"physics_observation", "nuisance_residual_parcel_bold"},
    {"physics_anatomy", "atlas_face_adjacency"}
  };
  out.metadata_json = meta.dump();
  cudaStreamDestroy(stream);
  return out;
}
}
