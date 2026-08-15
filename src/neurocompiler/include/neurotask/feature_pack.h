#pragma once
#include "neurotask/types.h"
#include <string>

namespace ntfm {
void write_feature_pack(const std::string& path, const CompilerOutput& output);
}
