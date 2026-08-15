#include "neurotask/feature_pack.h"
#include "neurotask/operator_graph.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
  std::unordered_map<std::string, std::string> out;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (!key.starts_with("--") || i + 1 >= argc) throw std::runtime_error("Arguments use --key value form");
    out[key.substr(2)] = argv[++i];
  }
  return out;
}

std::string required(const std::unordered_map<std::string, std::string>& args, const std::string& key) {
  const auto it = args.find(key);
  if (it == args.end() || it->second.empty()) throw std::runtime_error("Missing --" + key);
  return it->second;
}
}

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);
    ntfm::CompileRequest request;
    request.t1_nifti = required(args, "t1"); request.fmri_nifti = required(args, "fmri"); request.atlas_nifti = required(args, "atlas");
    request.config = required(args, "config"); request.output = required(args, "out"); request.subject_key = required(args, "subject-key");
    if (auto it = args.find("t1-dicom"); it != args.end()) request.t1_dicom = it->second;
    if (auto it = args.find("fmri-dicom"); it != args.end()) request.fmri_dicom = it->second;
    if (auto it = args.find("task"); it != args.end()) request.task = it->second;
    if (auto it = args.find("tr"); it != args.end()) request.tr_seconds = std::stof(it->second);
    const auto start = std::chrono::steady_clock::now();
    ntfm::OperatorGraph graph(request.config);
    auto output = graph.run(request);
    ntfm::write_feature_pack(request.output, output);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"output\":\"" << request.output << "\",\"runtime_seconds\":" << seconds << ",\"dicom_tokens\":" << output.dicom_tokens.size() << "}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "neurocompile: " << e.what() << "\n";
    return 1;
  }
}
