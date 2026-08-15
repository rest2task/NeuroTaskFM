#include "neurotask/geometry.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ntfm {
Matrix4 multiply_matrix4(const Matrix4& a, const Matrix4& b) {
  Matrix4 out{};
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) for (int k = 0; k < 4; ++k) out[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
  return out;
}

Matrix4 invert_matrix4(const Matrix4& value) {
  float aug[4][8]{};
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) { aug[r][c] = value[r * 4 + c]; aug[r][c + 4] = r == c ? 1.0f : 0.0f; }
  for (int col = 0; col < 4; ++col) {
    int pivot = col;
    for (int row = col + 1; row < 4; ++row) if (std::abs(aug[row][col]) > std::abs(aug[pivot][col])) pivot = row;
    if (std::abs(aug[pivot][col]) < 1e-8f) throw std::runtime_error("Image affine is singular");
    if (pivot != col) for (int c = 0; c < 8; ++c) std::swap(aug[pivot][c], aug[col][c]);
    const float scale = aug[col][col];
    for (int c = 0; c < 8; ++c) aug[col][c] /= scale;
    for (int row = 0; row < 4; ++row) if (row != col) {
      const float factor = aug[row][col];
      for (int c = 0; c < 8; ++c) aug[row][c] -= factor * aug[col][c];
    }
  }
  Matrix4 out{};
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out[r * 4 + c] = aug[r][c + 4];
  return out;
}

Matrix4 target_to_source_voxel(const Matrix4& source_affine, const Matrix4& target_affine) { return multiply_matrix4(invert_matrix4(source_affine), target_affine); }
}
