#pragma once
#include "neurotask/types.h"
#include <string>

namespace ntfm {
struct CompileRequest {
  std::string t1_nifti;
  std::string fmri_nifti;
  std::string atlas_nifti;
  std::string t1_dicom;
  std::string fmri_dicom;
  std::string config;
  std::string output;
  std::string subject_key;
  std::string task;
  float tr_seconds = 0.0f;
};

class OperatorGraph {
 public:
  explicit OperatorGraph(const std::string& config_path);
  CompilerOutput run(const CompileRequest& request) const;
 private:
  std::string config_path_;
};
}
