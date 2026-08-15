#include "neurotaskfm/tools.h"

#include <H5Cpp.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include "neurotask/nifti_reader.h"
#include "neurotaskfm/config.h"
#include "neurotaskfm/data.h"
#include "neurotaskfm/imaging_tools.h"
#include "neurotaskfm/runtime.h"

using torch::indexing::Slice;

namespace neurotaskfm {
namespace {

nlohmann::json read_json(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open JSON: " + path.string());
  nlohmann::json value;
  stream >> value;
  return value;
}

void write_json(const std::filesystem::path& path, const nlohmann::json& value) {
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("cannot write JSON: " + path.string());
  stream << std::setw(2) << value << '\n';
}

bool h5_exists(H5::H5File& file, const std::string& path) {
  return H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

torch::Tensor h5_tensor(H5::H5File& file, const std::string& path) {
  auto dataset = file.openDataSet(path);
  auto space = dataset.getSpace();
  std::vector<hsize_t> dimensions(static_cast<std::size_t>(space.getSimpleExtentNdims()));
  space.getSimpleExtentDims(dimensions.data());
  std::vector<std::int64_t> shape;
  std::size_t count = 1;
  for (const auto dimension : dimensions) {
    shape.push_back(static_cast<std::int64_t>(dimension));
    count *= static_cast<std::size_t>(dimension);
  }
  std::vector<float> values(count);
  if (count) dataset.read(values.data(), H5::PredType::NATIVE_FLOAT);
  return torch::from_blob(values.data(), shape, torch::kFloat32).clone();
}

void h5_write(H5::H5File& file, const std::string& path, const torch::Tensor& source) {
  const auto parent = std::filesystem::path(path).parent_path().string();
  if (!parent.empty() && parent != "/" && !h5_exists(file, parent)) file.createGroup(parent);
  if (h5_exists(file, path)) H5Ldelete(file.getId(), path.c_str(), H5P_DEFAULT);
  const auto tensor = source.detach().to(torch::kCPU, torch::kFloat32).contiguous();
  std::vector<hsize_t> dimensions;
  for (const auto value : tensor.sizes()) dimensions.push_back(static_cast<hsize_t>(value));
  H5::DataSpace space(static_cast<int>(dimensions.size()), dimensions.data());
  auto dataset = file.createDataSet(path, H5::PredType::NATIVE_FLOAT, space);
  if (tensor.numel()) dataset.write(tensor.data_ptr<float>(), H5::PredType::NATIVE_FLOAT);
}

torch::Tensor nifti_tensor(const std::filesystem::path& path) {
  const auto image = ntfm::read_nifti(path.string());
  auto tensor = torch::from_blob(const_cast<float*>(image.data.data()),
      {image.shape.t, image.shape.z, image.shape.y, image.shape.x}, torch::kFloat32).clone();
  return image.shape.t == 1 ? tensor.select(0, 0) : tensor.permute({1, 2, 3, 0});
}

torch::Tensor load_numeric(const std::filesystem::path& path) {
  const auto extension = path.extension().string();
  if (extension == ".nii" || extension == ".gz") return nifti_tensor(path);
  if (extension == ".pt") {
    torch::serialize::InputArchive archive;
    archive.load_from(path.string(), torch::kCPU);
    torch::Tensor value;
    archive.read("value", value);
    return value;
  }
  if (extension == ".json") {
    const auto value = read_json(path);
    std::vector<float> values;
    for (const auto& item : value) values.push_back(item.get<float>());
    return torch::tensor(values);
  }
  std::ifstream stream(path);
  std::vector<float> values;
  float value;
  while (stream >> value) values.push_back(value);
  if (values.empty()) throw std::runtime_error("numeric input is empty: " + path.string());
  return torch::tensor(values);
}

std::string shell_status(const int status) {
  if (WIFEXITED(status)) return std::to_string(WEXITSTATUS(status));
  if (WIFSIGNALED(status)) return "signal-" + std::to_string(WTERMSIG(status));
  return "unknown";
}

int execute(const std::vector<std::string>& command, const std::string& gpu = {}) {
  const auto child = fork();
  if (child < 0) throw std::runtime_error("fork failed");
  if (child == 0) {
    if (!gpu.empty()) setenv("CUDA_VISIBLE_DEVICES", gpu.c_str(), 1);
    std::vector<char*> arguments;
    arguments.reserve(command.size() + 1);
    for (const auto& item : command) arguments.push_back(const_cast<char*>(item.c_str()));
    arguments.push_back(nullptr);
    execvp(arguments.front(), arguments.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(child, &status, 0) < 0) throw std::runtime_error("waitpid failed");
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

std::vector<std::int64_t> comma_shape(const std::string& text) {
  std::vector<std::int64_t> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto stop = text.find(',', start);
    result.push_back(std::stoll(text.substr(start, stop - start)));
    if (stop == std::string::npos) break;
    start = stop + 1;
  }
  return result;
}

int validate_manifest_tool(const Arguments& arguments) {
  nlohmann::json missing = nlohmann::json::array();
  nlohmann::json leakage = nlohmann::json::array();
  std::unordered_map<std::string, std::string> owners;
  std::set<std::string> subjects, visits, t1s;
  std::unordered_map<std::string, std::size_t> datasets, tasks, splits, resource_kinds;
  std::size_t rows = 0;
  for (const auto& manifest : arguments.all("manifest")) {
    const auto root = std::filesystem::absolute(manifest).parent_path();
    const auto resource_root = arguments.has("resource-root")
        ? std::filesystem::absolute(arguments.require("resource-root")) : root;
    for (const auto& row : load_manifest(manifest)) {
      ++rows;
      subjects.insert(row.subject_id);
      visits.insert(row.subject_id + "\x1f" + row.visit_id);
      if (!row.t1_nifti.empty()) t1s.insert(row.t1_nifti);
      ++datasets[row.dataset]; ++tasks[row.task.empty() ? "unknown" : row.task]; ++splits[row.split];
      const auto owner = row.family_id.empty() ? row.subject_id : row.family_id;
      const auto item = owners.find(owner);
      if (item != owners.end() && item->second != row.split) leakage.push_back({{"owner", owner}, {"splits", {item->second, row.split}}});
      owners[owner] = row.split;
      for (const auto& [field, text] : std::array<std::pair<const char*, std::string>, 3>{{
               {"t1_nifti", row.t1_nifti}, {"fmri_nifti", row.fmri_nifti}, {"atlas_nifti", row.atlas_nifti}}}) {
        if (text.empty()) continue;
        auto path = std::filesystem::path(text);
        if (path.is_relative()) path = root / path;
        if (!std::filesystem::exists(path)) missing.push_back({{"sample", row.sample_id}, {"field", field}, {"path", path.string()}});
      }
      if (arguments.has("check-packs") && !row.output_pack.empty()) {
        auto path = std::filesystem::path(row.output_pack);
        if (path.is_relative()) path = root / path;
        if (!std::filesystem::exists(path)) missing.push_back({{"sample", row.sample_id}, {"field", "output_pack"}, {"path", path.string()}});
        else {
          H5::H5File file(path.string(), H5F_ACC_RDONLY);
          for (const auto* key : {"/dicom/tokens", "/images/t1_slices", "/images/epi_slices",
                                  "/compiled/spatial", "/compiled/temporal", "/quality/metrics"}) {
            if (!h5_exists(file, key)) missing.push_back({{"sample", row.sample_id}, {"field", "pack_dataset"}, {"path", key}});
          }
        }
      }
      for (const auto& resource : row.mr_resources) {
        const auto kind = resource.value("kind", "unknown");
        ++resource_kinds[kind];
        auto path = std::filesystem::path(resource.at("path").get<std::string>());
        if (path.is_relative()) path = resource_root / path;
        if (!std::filesystem::exists(path)) {
          missing.push_back({{"sample", row.sample_id}, {"field", "mr_resources"},
                             {"kind", kind}, {"path", path.string()}});
          continue;
        }
        if (kind == "kspace" && (path.extension() == ".h5" || path.extension() == ".hdf5")) {
          H5::H5File file(path.string(), H5F_ACC_RDONLY);
          const auto dataset = resource.value("dataset", "/kspace");
          if (!h5_exists(file, dataset)) {
            missing.push_back({{"sample", row.sample_id}, {"field", "kspace_dataset"},
                               {"path", path.string()}, {"dataset", dataset}});
          }
        }
      }
    }
  }
  nlohmann::json result{{"rows", rows}, {"unique_subjects", subjects.size()},
                        {"unique_visits", visits.size()}, {"t1_examinations", t1s.size()},
                        {"datasets", datasets}, {"tasks", tasks}, {"splits", splits},
                        {"resource_kinds", resource_kinds},
                        {"missing", missing}, {"leakage", leakage}};
  std::cout << std::setw(2) << result << '\n';
  if (arguments.has("output")) write_json(arguments.require("output"), result);
  return missing.empty() && leakage.empty() ? 0 : 2;
}

int candidate_to_yaml_tool(const Arguments& arguments) {
  static const std::unordered_set<std::string> allowed{
      "robust_normalize", "temporal_reference", "sobel3d", "rigid_ncc", "rigid_series",
      "compose_resample", "nuisance_regression", "fixed_temporal_projection", "parcel_project",
      "slice_extract", "quality_metrics"};
  const auto candidate = read_json(arguments.require("candidate"));
  const auto graph = candidate.at("graph");
  auto config = load_yaml(arguments.get("template", "configs/compiler/neurocompiler.yaml"));
  const auto operations = graph.at("operators");
  if (operations.empty()) throw std::invalid_argument("candidate graph has no operators");
  for (const auto& operation : operations) {
    if (!allowed.count(operation.at("type").get<std::string>())) throw std::invalid_argument("candidate uses a disallowed operator");
  }
  config["name"] = graph.value("name", candidate.value("candidate_id", "agent-candidate"));
  config["operators"] = YAML::Load(operations.dump());
  std::ofstream output(arguments.require("output"));
  output << config;
  return 0;
}

bool dominates(const nlohmann::json& first, const nlohmann::json& second) {
  bool better = false;
  for (const auto* key : {"spatial_cka", "temporal_cka", "epi_slice_ncc"}) {
    if (first.at(key).get<double>() < second.at(key).get<double>()) return false;
    better = better || first.at(key).get<double>() > second.at(key).get<double>();
  }
  for (const auto* key : {"motion_mae", "quality_mae", "runtime_seconds"}) {
    if (first.at(key).get<double>() > second.at(key).get<double>()) return false;
    better = better || first.at(key).get<double>() < second.at(key).get<double>();
  }
  return better;
}

int select_candidate_tool(const Arguments& arguments) {
  nlohmann::json candidates = nlohmann::json::array();
  const auto maximum_runtime = arguments.number("max-runtime", 130.0);
  for (const auto& path : arguments.positional()) {
    auto value = read_json(path);
    value["path"] = path;
    if (value.at("runtime_seconds").get<double>() <= maximum_runtime) candidates.push_back(std::move(value));
  }
  nlohmann::json frontier = nlohmann::json::array();
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    bool dominated = false;
    for (std::size_t other = 0; other < candidates.size(); ++other) {
      if (other != index && dominates(candidates[other], candidates[index])) { dominated = true; break; }
    }
    if (!dominated) frontier.push_back(candidates[index]);
  }
  if (frontier.empty()) throw std::runtime_error("no candidate met the runtime constraint");
  const auto selected = *std::max_element(frontier.begin(), frontier.end(), [](const auto& a, const auto& b) {
    return a.value("pareto_score", 0.0) < b.value("pareto_score", 0.0);
  });
  write_json(arguments.require("output"), {{"selected", selected}, {"frontier", frontier}});
  std::cout << selected.at("path").get<std::string>() << '\n';
  return 0;
}

int estimate_model_tool(const Arguments& arguments) {
  const auto config = ModelConfig::load(arguments.require("config"));
  const auto estimate = config.estimate_parameters();
  std::cout << std::fixed << std::setprecision(2) << config.name << ": "
            << estimate.at("total_b") << "B total, " << estimate.at("active_b") << "B active\n";
  return 0;
}

int estimate_training_tool(const Arguments& arguments) {
  const auto config = load_yaml(arguments.require("config"));
  const auto cluster = load_yaml(config["cluster"].as<std::string>());
  const auto gpus = cluster["nodes"].as<std::int64_t>() * cluster["gpus_per_node"].as<std::int64_t>();
  const auto steps = config["max_steps"].as<std::int64_t>();
  const auto seconds = arguments.number("step-seconds", 1.0) * steps;
  nlohmann::json value{{"steps", steps}, {"gpus", gpus}, {"wall_days", seconds / 86400.0},
                       {"gpu_hours", seconds * gpus / 3600.0}};
  std::cout << std::setw(2) << value << '\n';
  return 0;
}

int build_manifest_tool(const Arguments& arguments) {
  const auto root = std::filesystem::absolute(arguments.require("root"));
  const auto dataset = arguments.require("dataset");
  const auto atlas = std::filesystem::absolute(arguments.require("atlas"));
  const auto train_fraction = arguments.number("train-fraction", 0.80);
  const auto validation_fraction = arguments.number("validation-fraction", 0.10);
  const auto resource_index = arguments.has("resource-index")
      ? read_json(arguments.require("resource-index")) : nlohmann::json::object();
  std::unordered_map<std::string, std::filesystem::path> t1_by_visit;
  std::regex subject_pattern("sub-[A-Za-z0-9]+"), visit_pattern("ses-[A-Za-z0-9]+"), task_pattern("task-([A-Za-z0-9]+)");
  const auto identifiers = [&](const std::filesystem::path& path) {
    const auto text = path.string(); std::smatch match;
    const auto subject = std::regex_search(text, match, subject_pattern) ? match.str() : path.parent_path().filename().string();
    const auto visit = std::regex_search(text, match, visit_pattern) ? match.str() : std::string("ses-01");
    const auto task = std::regex_search(path.filename().string(), match, task_pattern) ? match[1].str() : std::string("rest");
    return std::array<std::string, 3>{subject, visit, task};
  };
  for (const auto& item : std::filesystem::recursive_directory_iterator(root)) {
    if (!item.is_regular_file() || item.path().filename().string().find("T1w.nii") == std::string::npos) continue;
    const auto ids = identifiers(item.path()); t1_by_visit[ids[0] + "/" + ids[1]] = item.path();
  }
  nlohmann::json rows = nlohmann::json::array();
  for (const auto& item : std::filesystem::recursive_directory_iterator(root)) {
    if (!item.is_regular_file() || item.path().filename().string().find("bold.nii") == std::string::npos) continue;
    const auto ids = identifiers(item.path()); const auto key = ids[0] + "/" + ids[1];
    if (!t1_by_visit.count(key)) continue;
    const auto hash = stable_id(item.path().string(), 0x7fffffff);
    const auto sample = ids[0] + "_" + ids[1] + "_task-" + ids[2] + "_" + std::to_string(hash);
    const auto fraction = static_cast<double>(stable_id(dataset + ":" + ids[0], 1000000)) / 1000000.0;
    const auto split = fraction < train_fraction ? "train" :
                       fraction < train_fraction + validation_fraction ? "validation" : "test";
    const auto fmri = ntfm::read_nifti(item.path().string());
    nlohmann::json row{{"sample_id", sample}, {"subject_id", ids[0]}, {"family_id", ids[0]},
        {"visit_id", ids[1]}, {"dataset", dataset}, {"site_id", nullptr}, {"split", split},
        {"task", ids[2]}, {"contrast", nullptr}, {"t1_nifti", t1_by_visit.at(key).string()},
        {"fmri_nifti", item.path().string()}, {"atlas_nifti", atlas.string()},
        {"tr_seconds", fmri.spacing[3] > 0 ? fmri.spacing[3] : 1.0}, {"metadata", nlohmann::json::object()},
        {"mr_resources", nlohmann::json::array()}};
    for (const auto& resource_key : std::array<std::string, 4>{
             sample, ids[0] + "/" + ids[1] + "/" + ids[2], ids[0] + "/" + ids[1], ids[0]}) {
      if (resource_index.contains(resource_key)) {
        row["mr_resources"] = resource_index.at(resource_key);
        break;
      }
    }
    if (arguments.has("compiled-root")) row["output_pack"] =
        (std::filesystem::absolute(arguments.require("compiled-root")) / (sample + ".h5")).string();
    else row["output_pack"] = nullptr;
    rows.push_back(std::move(row));
  }
  const auto output_path = std::filesystem::path(arguments.require("output"));
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  std::ofstream output(output_path); std::set<std::string> subjects;
  for (const auto& row : rows) { output << row.dump() << '\n'; subjects.insert(row["subject_id"].get<std::string>()); }
  std::cout << nlohmann::json{{"rows", rows.size()}, {"subjects", subjects.size()}, {"output", output_path.string()}}.dump() << '\n';
  return 0;
}

int export_compiler_artifacts_tool(const Arguments& arguments) {
  const auto channels = arguments.integer("channels", 32);
  const auto ridge = arguments.number("ridge", 1e-3);
  torch::Tensor xtx, xty;
  std::int64_t observations = 0;
  for (const auto& row : load_manifest(arguments.require("manifest"))) {
    H5::H5File file(row.output_pack, H5F_ACC_RDONLY);
    if (!h5_exists(file, "/compiled/parcel_series") || !h5_exists(file, "/teacher/temporal_basis")) continue;
    auto x = h5_tensor(file, "/compiled/parcel_series").to(torch::kCUDA, torch::kFloat64);
    auto y = h5_tensor(file, "/teacher/temporal_basis").to(torch::kCUDA, torch::kFloat64);
    const auto count = std::min(x.size(0), y.size(0));
    x = x.narrow(0, 0, count); y = y.narrow(0, 0, count).index({Slice(), Slice(0, channels)});
    if (!xtx.defined()) { xtx = torch::zeros({x.size(1), x.size(1)}, x.options()); xty = torch::zeros({x.size(1), channels}, x.options()); }
    xtx += x.transpose(0, 1).matmul(x); xty += x.transpose(0, 1).matmul(y); observations += count;
  }
  if (!xtx.defined()) throw std::runtime_error("no pack contained teacher temporal basis");
  const auto scale = xtx.trace() / xtx.size(0);
  const auto projection = torch::linalg_solve(xtx + ridge * scale * torch::eye(xtx.size(0), xtx.options()), xty).to(torch::kFloat32).cpu();
  const auto output_path = std::filesystem::path(arguments.require("output"));
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  H5::H5File output(output_path.string(), H5F_ACC_TRUNC);
  h5_write(output, "/temporal_projection", projection);
  h5_write(output, "/alignment_prior", torch::zeros({6}));
  std::cout << nlohmann::json{{"output", output_path.string()}, {"observations", observations},
                              {"shape", projection.sizes().vec()}}.dump() << '\n';
  return 0;
}

int prepare_packs_tool(const Arguments& arguments) {
  const auto rows = load_manifest(arguments.require("manifest"));
  const auto compiler = arguments.get("compiler", "build/bin/neurocompile");
  const auto config = arguments.get("config", "configs/compiler/neurocompiler.yaml");
  const auto gpu_text = arguments.get("gpus", "0");
  std::vector<std::string> gpus;
  std::size_t start = 0;
  while (start <= gpu_text.size()) {
    const auto stop = gpu_text.find(',', start);
    gpus.push_back(gpu_text.substr(start, stop - start));
    if (stop == std::string::npos) break;
    start = stop + 1;
  }
  std::ofstream log(arguments.get("log", "compile-results.jsonl"));
  bool failed = false;
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const auto& row = rows[index];
    nlohmann::json result{{"sample_id", row.sample_id}};
    if (row.output_pack.empty() || row.t1_nifti.empty() || row.fmri_nifti.empty()) result["status"] = "direct_resource_only";
    else if (std::filesystem::exists(row.output_pack) && !arguments.has("overwrite")) result["status"] = "exists";
    else {
      std::vector<std::string> command{compiler, "--t1", row.t1_nifti, "--fmri", row.fmri_nifti,
          "--atlas", row.atlas_nifti, "--config", config, "--out", row.output_pack,
          "--subject-key", row.subject_id, "--task", row.task.empty() ? "unknown" : row.task,
          "--tr", std::to_string(row.tr_seconds)};
      if (!row.t1_dicom.empty()) command.insert(command.end(), {"--t1-dicom", row.t1_dicom});
      if (!row.fmri_dicom.empty()) command.insert(command.end(), {"--fmri-dicom", row.fmri_dicom});
      const auto status = execute(command, gpus[index % gpus.size()]);
      result["gpu"] = gpus[index % gpus.size()];
      result["status"] = status == 0 ? "complete" : "failed";
      if (status != 0) { result["exit_code"] = status; failed = true; }
    }
    std::cout << result.dump() << '\n';
    log << result.dump() << '\n';
  }
  return failed ? 2 : 0;
}

int attach_targets_tool(const Arguments& arguments) {
  const auto index = read_json(arguments.require("index"));
  for (const auto& row : load_manifest(arguments.require("manifest"))) {
    if (!index.contains(row.sample_id)) continue;
    H5::H5File file(row.output_pack, H5F_ACC_RDWR);
    for (const auto& [name, path_value] : index.at(row.sample_id).items()) {
      std::string group = "targets";
      std::string dataset = name;
      if (name.rfind("teacher_", 0) == 0) { group = "teacher"; dataset = name.substr(8); }
      else if (name.rfind("input_", 0) == 0) { group = "inputs"; dataset = name.substr(6); }
      else if (name.rfind("physics_", 0) == 0) {
        group = "physics"; dataset = name.substr(8);
        if (dataset == "anatomy") dataset = "anatomy_graph";
        if (dataset == "drive") dataset = "neural_drive";
        if (dataset == "dt") dataset = "dt_seconds";
      }
      h5_write(file, "/" + group + "/" + dataset, load_numeric(path_value.get<std::string>()));
    }
    std::cout << row.sample_id << '\n';
  }
  return 0;
}

int pack_raw_inputs_tool(const Arguments& arguments) {
  if (!torch::cuda::is_available()) throw std::runtime_error("pack-raw-inputs requires CUDA");
  const auto t1_shape = comma_shape(arguments.get("t1-shape", "192,224,192"));
  const auto fmri_shape = comma_shape(arguments.get("fmri-shape", "96,112,96"));
  const auto frame_count = arguments.integer("fmri-frames", 8);
  torch::InferenceMode guard;
  for (const auto& row : load_manifest(arguments.require("manifest"))) {
    auto t1 = nifti_tensor(row.t1_nifti).to(torch::kCUDA);
    auto fmri = nifti_tensor(row.fmri_nifti).to(torch::kCUDA);
    const auto normalize = [](torch::Tensor value) {
      const auto finite = value.masked_select(torch::isfinite(value));
      const auto bounds = torch::quantile(finite, torch::tensor({0.005, 0.995}, finite.options()));
      value = value.clamp(bounds[0].item<double>(), bounds[1].item<double>());
      return (value - bounds.mean()) / ((bounds[1] - bounds[0]) / 4.0).clamp_min(1e-6);
    };
    t1 = torch::nn::functional::interpolate(normalize(t1).unsqueeze(0).unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions().size(t1_shape).mode(torch::kTrilinear).align_corners(false)).squeeze();
    std::vector<torch::Tensor> frames;
    const auto available = fmri.dim() == 4 ? fmri.size(3) : 1;
    for (std::int64_t index = 0; index < std::min(frame_count, available); ++index) {
      const auto position = available == 1 ? 0 : index * (available - 1) / std::max<std::int64_t>(frame_count - 1, 1);
      const auto frame = fmri.dim() == 4 ? fmri.select(3, position) : fmri;
      frames.push_back(torch::nn::functional::interpolate(normalize(frame).unsqueeze(0).unsqueeze(0),
          torch::nn::functional::InterpolateFuncOptions().size(fmri_shape).mode(torch::kTrilinear).align_corners(false)).squeeze());
    }
    H5::H5File file(row.output_pack, H5F_ACC_RDWR);
    h5_write(file, "/raw/t1_volume", t1);
    h5_write(file, "/raw/fmri_volume", torch::stack(frames));
    std::cout << row.sample_id << '\n';
  }
  return 0;
}

int build_map_residuals_tool(const Arguments& arguments) {
  const auto train_split = arguments.get("train-split", "train");
  auto train_rows = load_manifest(arguments.require("train-manifest"), train_split);
  std::unordered_map<std::string, std::vector<ManifestRow>> groups;
  for (const auto& row : train_rows) groups[(row.task.empty() ? "unknown" : row.task) + "\x1f" +
                                             (row.contrast.empty() ? "unspecified" : row.contrast)].push_back(row);
  std::unordered_map<std::string, torch::Tensor> templates;
  for (const auto& [key, rows] : groups) {
    std::vector<torch::Tensor> maps;
    for (const auto& row : rows) { H5::H5File file(row.output_pack, H5F_ACC_RDONLY); maps.push_back(h5_tensor(file, "/targets/map").to(torch::kCUDA)); }
    templates[key] = torch::nanmean(torch::stack(maps).to(torch::kFloat64), {0}).to(torch::kFloat32).cpu();
  }
  const auto artifact_path = std::filesystem::path(arguments.require("artifact"));
  if (!artifact_path.parent_path().empty()) std::filesystem::create_directories(artifact_path.parent_path());
  H5::H5File artifact(artifact_path.string(), H5F_ACC_TRUNC);
  artifact.createGroup("/templates");
  std::size_t index = 0;
  for (const auto& [key, value] : templates) h5_write(artifact, "/templates/template_" + std::to_string(index++), value);
  std::vector<ManifestRow> apply = train_rows;
  for (const auto& manifest : arguments.all("apply-manifest")) {
    auto rows = load_manifest(manifest); apply.insert(apply.end(), rows.begin(), rows.end());
  }
  for (const auto& row : apply) {
    const auto key = (row.task.empty() ? "unknown" : row.task) + "\x1f" +
                     (row.contrast.empty() ? "unspecified" : row.contrast);
    if (!templates.count(key)) continue;
    H5::H5File file(row.output_pack, H5F_ACC_RDWR);
    const auto target = h5_tensor(file, "/targets/map");
    h5_write(file, "/targets/map_template", templates.at(key));
    h5_write(file, "/targets/map_residual", target - templates.at(key));
  }
  std::cout << nlohmann::json{{"artifact", artifact_path.string()}, {"templates", templates.size()},
                              {"applied_rows", apply.size()}}.dump(2) << '\n';
  return 0;
}

int fit_calibration_tool(const Arguments& arguments) {
  torch::serialize::InputArchive archive;
  archive.load_from(arguments.require("input"), torch::kCPU);
  const auto read = [&](const char* key) { torch::Tensor value; archive.read(key, value); return value.to(torch::kFloat64); };
  const auto temperature = [](const torch::Tensor& error, const torch::Tensor& variance) {
    const auto valid = torch::isfinite(error) & torch::isfinite(variance) & variance.gt(0);
    return torch::sqrt(error.masked_select(valid).square().mean() / variance.masked_select(valid).mean()).item<double>();
  };
  const auto map_variance = read("map_variance");
  const auto behavior_variance = read("behavior_variance");
  const auto map_temperature = temperature(read("map_error"), map_variance);
  const auto behavior_temperature = temperature(read("behavior_error"), behavior_variance);
  const auto map_confidence = torch::exp(-torch::sqrt(map_variance) * map_temperature);
  const auto behavior_confidence = torch::exp(-torch::sqrt(behavior_variance) * behavior_temperature);
  const auto ood = read("ood_score");
  const auto quantile = [](const torch::Tensor& value, const double probability) { return torch::quantile(value, probability).item<double>(); };
  nlohmann::json result{{"version", 1}, {"state", "locked_validation"},
      {"map_temperature", map_temperature}, {"behavior_temperature", behavior_temperature},
      {"ood_index", 15}, {"ood_temperature", 1.0},
      {"thresholds", {{"minimum_map_confidence", quantile(map_confidence, 1.0 - arguments.number("map-coverage", 0.90))},
                      {"minimum_behavior_confidence", quantile(behavior_confidence, 1.0 - arguments.number("behavior-coverage", 0.90))},
                      {"maximum_ood_score", quantile(ood, arguments.number("ood-specificity", 0.95))}}},
      {"provenance", "Fitted on a development-validation native Torch archive."}};
  write_json(arguments.require("output"), result);
  return 0;
}

int checkpoint_copy_tool(const Arguments& arguments) {
  const auto input = std::filesystem::path(arguments.has("input") ? arguments.require("input")
                                                                   : arguments.require("checkpoint"));
  const auto output = std::filesystem::path(arguments.require("out"));
  auto count = arguments.integer("target-ep", 0);
  if (count == 0 && arguments.has("cluster")) {
    count = load_yaml(arguments.require("cluster"))["expert_parallel_size"].as<std::int64_t>();
  }
  if (count == 0) count = 1;
  std::filesystem::create_directories(output);
  std::vector<std::filesystem::path> sources;
  for (const auto& item : std::filesystem::directory_iterator(input)) {
    if (item.path().extension() == ".pt") sources.push_back(item.path());
  }
  if (sources.empty()) throw std::runtime_error("checkpoint directory contains no .pt shards");
  std::sort(sources.begin(), sources.end());
  for (std::int64_t rank = 0; rank < count; ++rank) {
    std::ostringstream name; name << "rank-" << std::setw(2) << std::setfill('0') << rank << ".pt";
    std::filesystem::copy_file(sources[static_cast<std::size_t>(rank) % sources.size()], output / name.str(),
                               std::filesystem::copy_options::overwrite_existing);
  }
  return 0;
}

int check_repo_tool(const Arguments& arguments) {
  const auto root = std::filesystem::path(arguments.get("root", "."));
  std::vector<std::string> errors;
  for (const auto& item : std::filesystem::recursive_directory_iterator(root)) {
    const auto filename = item.path().filename().string();
    if (item.is_directory() && (filename == "__pycache__" || filename == ".pytest_cache")) {
      errors.push_back("Python cache remains: " + item.path().string());
      continue;
    }
    if (!item.is_regular_file()) continue;
    const auto extension = item.path().extension().string();
    try {
      if (extension == ".json") (void)read_json(item.path());
      if (extension == ".yaml" || extension == ".yml") (void)load_yaml(item.path());
      const bool python_extension = extension == ".py" || extension == ".pyi" || extension == ".pyc" ||
                                    extension == ".pyo" || extension == ".pyd" || extension == ".whl";
      const bool python_metadata = filename == "pyproject.toml" || filename == "setup.py" ||
                                   filename == "tox.ini" || filename.starts_with("requirements") ||
                                   filename.starts_with("Pipfile") || filename.starts_with("poetry.lock");
      if (python_extension || python_metadata) {
        errors.push_back("Python artifact remains: " + item.path().string());
      }
    } catch (const std::exception& error) { errors.push_back(item.path().string() + ": " + error.what()); }
  }
  for (const auto& item : std::filesystem::directory_iterator(root / "configs/models")) {
    if (item.path().extension() != ".yaml") continue;
    try { auto config = ModelConfig::load(item.path()); config.validate(); }
    catch (const std::exception& error) { errors.push_back(item.path().string() + ": " + error.what()); }
  }
  for (const auto& error : errors) std::cerr << "ERROR: " << error << '\n';
  std::cout << (errors.empty() ? "repository checks passed\n" : "repository checks failed\n");
  return errors.empty() ? 0 : 2;
}

}  // namespace

int run_tool(const std::string& command, const Arguments& arguments) {
  if (is_imaging_tool(command)) return run_imaging_tool(command, arguments);
  if (command == "validate-manifest") return validate_manifest_tool(arguments);
  if (command == "candidate-to-yaml") return candidate_to_yaml_tool(arguments);
  if (command == "select-compiler-candidate") return select_candidate_tool(arguments);
  if (command == "estimate-model") return estimate_model_tool(arguments);
  if (command == "estimate-training-time") return estimate_training_tool(arguments);
  if (command == "build-manifest") return build_manifest_tool(arguments);
  if (command == "export-compiler-artifacts") return export_compiler_artifacts_tool(arguments);
  if (command == "prepare-packs") return prepare_packs_tool(arguments);
  if (command == "attach-targets") return attach_targets_tool(arguments);
  if (command == "pack-raw-inputs") return pack_raw_inputs_tool(arguments);
  if (command == "build-map-residuals") return build_map_residuals_tool(arguments);
  if (command == "fit-calibration") return fit_calibration_tool(arguments);
  if (command == "repartition-checkpoint" || command == "export-expert-checkpoint") return checkpoint_copy_tool(arguments);
  if (command == "check-repo") return check_repo_tool(arguments);
  if (command == "compiler-metrics") return compiler_metrics(arguments);
  throw std::invalid_argument("unknown native tool command: " + command);
}

}  // namespace neurotaskfm
