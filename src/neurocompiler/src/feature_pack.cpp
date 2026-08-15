#include "neurotask/feature_pack.h"
#include <H5Cpp.h>
#include <filesystem>
#include <stdexcept>

namespace ntfm {
namespace {
void write_u32(H5::H5File& file, const std::string& path, const std::vector<uint32_t>& data) {
  hsize_t dims[1] = {data.size()};
  H5::DataSpace space(1, dims);
  auto dataset = file.createDataSet(path, H5::PredType::NATIVE_UINT32, space);
  if (!data.empty()) dataset.write(data.data(), H5::PredType::NATIVE_UINT32);
}

void write_f32(H5::H5File& file, const std::string& path, const std::vector<float>& data, const std::vector<hsize_t>& dims) {
  size_t expected = 1;
  for (hsize_t d : dims) expected *= d;
  if (expected != data.size()) throw std::runtime_error("HDF5 shape mismatch for " + path);
  H5::DataSpace space(static_cast<int>(dims.size()), dims.data());
  auto dataset = file.createDataSet(path, H5::PredType::NATIVE_FLOAT, space);
  if (!data.empty()) dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
}

void write_f32_compressed(H5::H5File& file, const std::string& path, const std::vector<float>& data, const std::vector<hsize_t>& dims) {
  size_t expected = 1;
  for (hsize_t d : dims) expected *= d;
  if (expected != data.size()) throw std::runtime_error("HDF5 shape mismatch for " + path);
  H5::DataSpace space(static_cast<int>(dims.size()), dims.data());
  H5::DSetCreatPropList properties;
  properties.setChunk(static_cast<int>(dims.size()), dims.data());
  properties.setDeflate(1);
  auto dataset = file.createDataSet(path, H5::PredType::NATIVE_FLOAT, space, properties);
  if (!data.empty()) dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
}

void write_string(H5::H5File& file, const std::string& path, const std::string& value) {
  H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSpace space(H5S_SCALAR);
  auto dataset = file.createDataSet(path, type, space);
  const char* ptr = value.c_str();
  dataset.write(&ptr, type);
}
}

void write_feature_pack(const std::string& path, const CompilerOutput& output) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  H5::H5File file(path, H5F_ACC_TRUNC);
  file.createGroup("/dicom");
  file.createGroup("/images");
  file.createGroup("/compiled");
  file.createGroup("/physics");
  file.createGroup("/motion");
  file.createGroup("/quality");
  file.createGroup("/meta");
  write_u32(file, "/dicom/tokens", output.dicom_tokens);
  write_f32(file, "/images/t1_slices", output.t1_slices, {3, static_cast<hsize_t>(output.slice_count), static_cast<hsize_t>(output.slice_size), static_cast<hsize_t>(output.slice_size)});
  write_f32(file, "/images/epi_slices", output.epi_slices, {3, static_cast<hsize_t>(output.slice_count), static_cast<hsize_t>(output.slice_size), static_cast<hsize_t>(output.slice_size)});
  write_f32(file, "/compiled/spatial", output.compiled_spatial, {static_cast<hsize_t>(output.parcel_count), static_cast<hsize_t>(output.feature_channels)});
  write_f32(file, "/compiled/temporal", output.compiled_temporal, {static_cast<hsize_t>(output.temporal_count), static_cast<hsize_t>(output.feature_channels)});
  write_f32(file, "/compiled/parcel_series", output.parcel_series, {static_cast<hsize_t>(output.temporal_count), static_cast<hsize_t>(output.parcel_count)});
  write_f32_compressed(file, "/physics/anatomy_graph", output.anatomy_graph, {static_cast<hsize_t>(output.parcel_count), static_cast<hsize_t>(output.parcel_count)});
  write_f32(file, "/motion/rigid", output.motion, {static_cast<hsize_t>(output.temporal_count), 6});
  write_f32(file, "/quality/metrics", output.qc, {static_cast<hsize_t>(output.qc.size())});
  write_string(file, "/meta/json", output.metadata_json);
  const int version = 3;
  H5::DataSpace attr_space(H5S_SCALAR);
  auto attr = file.createAttribute("ntfm_pack_version", H5::PredType::NATIVE_INT, attr_space);
  attr.write(H5::PredType::NATIVE_INT, &version);
}
}
