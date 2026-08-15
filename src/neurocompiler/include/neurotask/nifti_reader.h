#pragma once
#include "neurotask/types.h"
#include <string>

namespace ntfm {
HostImage read_nifti(const std::string& path);
DeviceImage upload_image(const HostImage& image);
}
