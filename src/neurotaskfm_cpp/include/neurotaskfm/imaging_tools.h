#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "neurotaskfm/cli.h"

namespace neurotaskfm {

struct Complex32 {
  float real = 0.0F;
  float imag = 0.0F;
};

struct CartesianKSpace {
  std::vector<Complex32> samples;
  std::int64_t frames = 1;
  std::int64_t coils = 1;
  std::int64_t depth = 1;
  std::int64_t height = 1;
  std::int64_t width = 1;
  int fft_dimensions = 2;
};

struct ReconstructedImage {
  std::vector<float> magnitude;
  std::int64_t frames = 1;
  std::int64_t depth = 1;
  std::int64_t height = 1;
  std::int64_t width = 1;
};

ReconstructedImage reconstruct_cartesian_cuda(const CartesianKSpace& input);
bool is_imaging_tool(const std::string& command);
int run_imaging_tool(const std::string& command, const Arguments& arguments);

}  // namespace neurotaskfm
