#include "neurotask/compiler_artifacts.h"
#include <H5Cpp.h>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace ntfm {
CompilerArtifacts load_compiler_artifacts(const std::string& path, int parcels, int channels) {
  if (!std::filesystem::exists(path)) throw std::runtime_error("Missing compiled NeuroCompiler artifact: " + path);
  H5::H5File file(path, H5F_ACC_RDONLY);
  CompilerArtifacts out; out.parcels = parcels; out.channels = channels;
  auto projection = file.openDataSet("/temporal_projection");
  H5::DataSpace space = projection.getSpace();
  hsize_t dims[2]{}; if (space.getSimpleExtentNdims() != 2) throw std::runtime_error("temporal_projection must be rank 2");
  space.getSimpleExtentDims(dims);
  if (static_cast<int>(dims[0]) != parcels || static_cast<int>(dims[1]) != channels) throw std::runtime_error("temporal_projection shape does not match compiler config");
  out.temporal_projection.resize(static_cast<size_t>(parcels) * channels); projection.read(out.temporal_projection.data(), H5::PredType::NATIVE_FLOAT);
  if (H5Lexists(file.getId(), "/alignment_prior", H5P_DEFAULT) > 0) file.openDataSet("/alignment_prior").read(out.alignment_prior.data(), H5::PredType::NATIVE_FLOAT);
  if (H5Lexists(file.getId(), "/provenance", H5P_DEFAULT) > 0) {
    auto dataset = file.openDataSet("/provenance"); H5::StrType type = dataset.getStrType(); char* text = nullptr; dataset.read(&text, type); if (text) { out.provenance = text; free(text); }
  }
  return out;
}
}
