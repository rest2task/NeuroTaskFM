#include "neurotask/nifti_reader.h"
#include <nifti1_io.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ntfm {
namespace {
template <class T> void copy_scaled(const T* src, float* dst, size_t n, float slope, float intercept) {
  for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]) * slope + intercept;
}
}

HostImage read_nifti(const std::string& path) {
  nifti_image* nim = nifti_image_read(path.c_str(), 1);
  if (!nim) throw std::runtime_error("Unable to read NIfTI: " + path);
  HostImage out;
  out.shape = {nim->nx, nim->ny, nim->nz, std::max(1, nim->nt)};
  out.spacing = {nim->dx, nim->dy, nim->dz, nim->dt > 0 ? nim->dt : 1.0f};
  const mat44 affine = nim->sform_code > 0 ? nim->sto_xyz : nim->qto_xyz;
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out.affine[r * 4 + c] = affine.m[r][c];
  const size_t n = out.shape.elements();
  out.data.resize(n);
  const float slope = nim->scl_slope == 0.0f ? 1.0f : nim->scl_slope;
  const float intercept = nim->scl_inter;
  switch (nim->datatype) {
    case NIFTI_TYPE_UINT8: copy_scaled(static_cast<const uint8_t*>(nim->data), out.data.data(), n, slope, intercept); break;
    case NIFTI_TYPE_INT8: copy_scaled(static_cast<const int8_t*>(nim->data), out.data.data(), n, slope, intercept); break;
    case NIFTI_TYPE_UINT16: copy_scaled(static_cast<const uint16_t*>(nim->data), out.data.data(), n, slope, intercept); break;
    case NIFTI_TYPE_INT16: copy_scaled(static_cast<const int16_t*>(nim->data), out.data.data(), n, slope, intercept); break;
    case NIFTI_TYPE_UINT32: copy_scaled(static_cast<const uint32_t*>(nim->data), out.data.data(), n, slope, intercept); break;
    case NIFTI_TYPE_INT32: copy_scaled(static_cast<const int32_t*>(nim->data), out.data.data(), n, slope, intercept); break;
    case NIFTI_TYPE_FLOAT32: copy_scaled(static_cast<const float*>(nim->data), out.data.data(), n, slope, intercept); break;
    case NIFTI_TYPE_FLOAT64: copy_scaled(static_cast<const double*>(nim->data), out.data.data(), n, slope, intercept); break;
    default: nifti_image_free(nim); throw std::runtime_error("Unsupported NIfTI datatype in " + path);
  }
  nifti_image_free(nim);
  return out;
}

DeviceImage upload_image(const HostImage& image) {
  DeviceImage out;
  out.shape = image.shape;
  out.spacing = image.spacing;
  out.affine = image.affine;
  out.data.copy_from_host(image.data.data(), image.data.size());
  return out;
}
}
