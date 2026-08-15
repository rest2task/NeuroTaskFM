#pragma once
#include <array>
#include <string>
#include <vector>

namespace ntfm {
struct CompilerArtifacts {
  int parcels = 0;
  int channels = 0;
  std::vector<float> temporal_projection;
  std::array<float, 6> alignment_prior{0, 0, 0, 0, 0, 0};
  std::string provenance;
};
CompilerArtifacts load_compiler_artifacts(const std::string& path, int parcels, int channels);
}
