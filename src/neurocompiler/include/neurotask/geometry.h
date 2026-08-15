#pragma once
#include <array>

namespace ntfm {
using Matrix4 = std::array<float, 16>;
Matrix4 multiply_matrix4(const Matrix4& a, const Matrix4& b);
Matrix4 invert_matrix4(const Matrix4& value);
Matrix4 target_to_source_voxel(const Matrix4& source_affine, const Matrix4& target_affine);
}
