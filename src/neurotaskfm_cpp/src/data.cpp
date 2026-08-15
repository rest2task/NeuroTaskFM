#include "neurotaskfm/data.h"

#include <H5Cpp.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

#include "neurotask/nifti_reader.h"

using torch::indexing::Slice;

namespace neurotaskfm {
namespace {

std::string string_value(const nlohmann::json& value, const char* key) {
  return value.contains(key) && !value[key].is_null() ? value[key].get<std::string>() : std::string{};
}

torch::Tensor read_dataset(H5::H5File& file, const std::string& path, const torch::Dtype dtype) {
  auto dataset = file.openDataSet(path);
  auto space = dataset.getSpace();
  const auto dimensions = space.getSimpleExtentNdims();
  std::vector<hsize_t> hdf_shape(static_cast<std::size_t>(dimensions));
  space.getSimpleExtentDims(hdf_shape.data());
  std::vector<std::int64_t> shape;
  shape.reserve(hdf_shape.size());
  std::size_t count = 1;
  for (const auto size : hdf_shape) {
    shape.push_back(static_cast<std::int64_t>(size));
    count *= static_cast<std::size_t>(size);
  }
  if (dtype == torch::kLong) {
    std::vector<std::uint32_t> values(count);
    if (count != 0) dataset.read(values.data(), H5::PredType::NATIVE_UINT32);
    auto tensor = torch::empty({static_cast<std::int64_t>(count)}, torch::kLong);
    auto* output = tensor.data_ptr<std::int64_t>();
    std::transform(values.begin(), values.end(), output,
                   [](const std::uint32_t value) { return static_cast<std::int64_t>(value); });
    return tensor.reshape(shape);
  }
  if (dtype == torch::kBool) {
    std::vector<std::uint8_t> values(count);
    if (count != 0) dataset.read(values.data(), H5::PredType::NATIVE_UINT8);
    return torch::from_blob(values.data(), shape, torch::kUInt8).to(torch::kBool).clone();
  }
  std::vector<float> values(count);
  if (count != 0) dataset.read(values.data(), H5::PredType::NATIVE_FLOAT);
  return torch::from_blob(values.data(), shape, torch::kFloat32).clone();
}

bool exists(H5::H5File& file, const std::string& path) {
  return H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

void add_native_resources(FeaturePack& pack, const ManifestRow& row,
                          const std::filesystem::path& root) {
  std::vector<torch::Tensor> volumes;
  std::vector<torch::Tensor> images;
  std::vector<std::int64_t> image_types, image_views, image_positions;
  std::vector<torch::Tensor> video;
  std::vector<std::int64_t> video_types, video_views, video_positions;
  const auto normalized_frame = [](const cv::Mat& source) {
    if (source.empty()) throw std::runtime_error("unable to decode image frame");
    cv::Mat gray;
    if (source.channels() == 1) gray = source;
    else if (source.channels() == 3) cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    else if (source.channels() == 4) cv::cvtColor(source, gray, cv::COLOR_BGRA2GRAY);
    else throw std::runtime_error("unsupported image channel count");
    cv::Mat resized, floating;
    cv::resize(gray, resized, cv::Size(224, 224), 0.0, 0.0, cv::INTER_AREA);
    resized.convertTo(floating, CV_32F);
    auto value = torch::from_blob(floating.data, {floating.rows, floating.cols}, torch::kFloat32).clone();
    const auto finite = value.masked_select(torch::isfinite(value));
    if (finite.numel() == 0) throw std::runtime_error("image frame contains no finite pixels");
    const auto bounds = torch::quantile(finite, torch::tensor({0.005, 0.995}, finite.options()));
    value = value.clamp(bounds[0].item<double>(), bounds[1].item<double>());
    value = (value - bounds.mean()) / ((bounds[1] - bounds[0]) / 4.0).clamp_min(1e-6);
    return torch::nan_to_num(value).unsqueeze(0);
  };
  for (const auto& resource : row.mr_resources) {
    const auto kind = resource.value("kind", std::string{});
    if (kind == "kspace") {
      throw std::runtime_error("k-space resource must be preprocessed with convert-kspace-manifest: " +
                               resource.at("path").get<std::string>());
    }
    auto path = std::filesystem::path(resource.at("path").get<std::string>());
    if (path.is_relative() && !root.empty()) path = root / path;
    const auto type = modality_id(resource.value("modality", "unknown"));
    const auto view_name = resource.contains("view") && resource["view"].is_string()
        ? resource["view"].get<std::string>() : (kind == "video" ? "cine" : "unknown");
    const auto view = view_id(view_name);
    if (kind == "image") {
      images.push_back(normalized_frame(cv::imread(path.string(), cv::IMREAD_UNCHANGED)));
      image_types.push_back(type); image_views.push_back(view); image_positions.push_back(0);
      continue;
    }
    if (kind == "image_series") {
      if (!std::filesystem::is_directory(path)) throw std::runtime_error("image_series is not a directory: " + path.string());
      std::vector<std::filesystem::path> files;
      for (const auto& item : std::filesystem::directory_iterator(path)) if (item.is_regular_file()) files.push_back(item.path());
      std::sort(files.begin(), files.end());
      const auto stride = static_cast<std::size_t>(std::max(resource.value("frame_stride", 1), 1));
      const auto maximum = static_cast<std::size_t>(resource.value("max_frames", 32));
      std::size_t accepted = 0;
      for (std::size_t index = 0; index < files.size() && accepted < maximum; index += stride) {
        const auto frame = cv::imread(files[index].string(), cv::IMREAD_UNCHANGED);
        if (frame.empty()) continue;
        video.push_back(normalized_frame(frame)); video_types.push_back(type); video_views.push_back(view);
        video_positions.push_back(static_cast<std::int64_t>(accepted++));
      }
      continue;
    }
    if (kind == "video") {
      cv::VideoCapture capture(path.string());
      if (!capture.isOpened()) throw std::runtime_error("unable to open video: " + path.string());
      const auto stride = std::max(resource.value("frame_stride", 1), 1);
      const auto maximum = resource.value("max_frames", 32);
      cv::Mat frame;
      std::int64_t index = 0;
      std::int64_t accepted = 0;
      while (accepted < maximum && capture.read(frame)) {
        if (index++ % stride != 0) continue;
        video.push_back(normalized_frame(frame)); video_types.push_back(type); video_views.push_back(view);
        video_positions.push_back(accepted++);
      }
      continue;
    }
    if (kind == "dicom" || kind == "dicom_series") {
      throw std::runtime_error("DICOM pixel resources must be preprocessed with dicom-to-nifti: " + path.string());
    }
    if (kind == "array") {
      throw std::runtime_error("array resources must be preprocessed with hdf5-to-nifti: " + path.string());
    }
    if (kind != "volume" && kind != "volume_series") continue;
    const auto name = path.string();
    if (!(name.ends_with(".nii") || name.ends_with(".nii.gz"))) {
      throw std::runtime_error("native volume resources must be NIfTI; convert first: " + name);
    }
    const auto image = ntfm::read_nifti(name);
    auto source = torch::from_blob(const_cast<float*>(image.data.data()),
        {image.shape.t, image.shape.z, image.shape.y, image.shape.x}, torch::kFloat32).clone();
    const auto maximum = resource.value("max_frames", kind == "volume" ? 1 : 8);
    const auto count = std::min<std::int64_t>(source.size(0), maximum);
    auto positions = count == source.size(0)
        ? torch::arange(count, torch::kLong)
        : torch::linspace(0, source.size(0) - 1, count).round().to(torch::kLong);
    source = source.index_select(0, positions);
    for (std::int64_t frame = 0; frame < source.size(0); ++frame) {
      auto value = source[frame];
      const auto finite = value.masked_select(torch::isfinite(value));
      if (finite.numel() == 0) continue;
      const auto bounds = torch::quantile(finite, torch::tensor({0.005, 0.995}, finite.options()));
      value = value.clamp(bounds[0].item<double>(), bounds[1].item<double>());
      value = (value - bounds.mean()) / ((bounds[1] - bounds[0]) / 4.0).clamp_min(1e-6);
      value = torch::nn::functional::interpolate(value.unsqueeze(0).unsqueeze(0),
          torch::nn::functional::InterpolateFuncOptions().size(std::vector<std::int64_t>{96, 112, 96})
              .mode(torch::kTrilinear).align_corners(false)).squeeze(0);
      volumes.push_back(torch::nan_to_num(value));
    }
  }
  if (!volumes.empty()) pack.values["mr_volumes"] = torch::stack(volumes);
  if (!images.empty()) {
    pack.values["mr_images"] = torch::stack(images);
    pack.values["mr_image_types"] = torch::tensor(image_types, torch::kLong);
    pack.values["mr_image_views"] = torch::tensor(image_views, torch::kLong);
    pack.values["mr_image_positions"] = torch::tensor(image_positions, torch::kLong);
  }
  if (!video.empty()) {
    pack.values["mr_video"] = torch::stack(video);
    pack.values["mr_video_types"] = torch::tensor(video_types, torch::kLong);
    pack.values["mr_video_views"] = torch::tensor(video_views, torch::kLong);
    pack.values["mr_video_positions"] = torch::tensor(video_positions, torch::kLong);
  }
}

std::pair<torch::Tensor, torch::Tensor> pad_first_dimension(const std::vector<torch::Tensor>& values) {
  if (values.empty()) return {};
  std::int64_t maximum = 0;
  for (const auto& value : values) maximum = std::max(maximum, value.size(0));
  auto shape = values.front().sizes().vec();
  shape.insert(shape.begin(), static_cast<std::int64_t>(values.size()));
  shape[1] = maximum;
  auto output = torch::zeros(shape, values.front().options());
  auto mask = torch::zeros({static_cast<std::int64_t>(values.size()), maximum},
                           values.front().options().dtype(torch::kBool));
  for (std::size_t index = 0; index < values.size(); ++index) {
    const auto length = values[index].size(0);
    output[static_cast<std::int64_t>(index)].index_put_({Slice(0, length)}, values[index]);
    mask[static_cast<std::int64_t>(index)].index_put_({Slice(0, length)}, true);
  }
  return {output, mask};
}

void add_padded(TensorMap& output, const std::vector<FeaturePack>& packs,
                const std::string& key, const std::string& mask_key) {
  std::vector<torch::Tensor> values;
  values.reserve(packs.size());
  bool any = false;
  for (const auto& pack : packs) {
    const auto item = pack.values.find(key);
    if (item == pack.values.end()) {
      values.push_back(torch::empty({0}, torch::kFloat32));
    } else {
      values.push_back(item->second);
      any = any || item->second.size(0) > 0;
    }
  }
  if (!any) return;
  const auto prototype = std::find_if(values.begin(), values.end(), [](const auto& value) {
    return value.dim() > 1 || value.numel() > 0;
  });
  for (auto& value : values) {
    if (value.dim() != prototype->dim()) {
      auto empty_shape = prototype->sizes().vec();
      empty_shape[0] = 0;
      value = torch::empty(empty_shape, prototype->options());
    }
  }
  auto padded = pad_first_dimension(values);
  output[key] = padded.first;
  output[mask_key] = padded.second;
}

}  // namespace

ManifestRow ManifestRow::from_json(nlohmann::json value) {
  ManifestRow row;
  row.sample_id = value.at("sample_id").get<std::string>();
  row.subject_id = value.at("subject_id").get<std::string>();
  row.visit_id = value.at("visit_id").get<std::string>();
  row.dataset = value.at("dataset").get<std::string>();
  row.split = value.at("split").get<std::string>();
  row.t1_nifti = string_value(value, "t1_nifti");
  row.fmri_nifti = string_value(value, "fmri_nifti");
  row.output_pack = string_value(value, "output_pack");
  row.family_id = string_value(value, "family_id");
  row.site_id = string_value(value, "site_id");
  row.task = string_value(value, "task");
  row.contrast = string_value(value, "contrast");
  row.t1_dicom = string_value(value, "t1_dicom");
  row.fmri_dicom = string_value(value, "fmri_dicom");
  row.atlas_nifti = string_value(value, "atlas_nifti");
  row.tr_seconds = value.value("tr_seconds", 1.0);
  row.metadata = value.value("metadata", nlohmann::json::object());
  row.mr_resources = value.value("mr_resources", nlohmann::json::array());
  row.source = std::move(value);
  return row;
}

std::vector<ManifestRow> load_manifest(const std::filesystem::path& path,
                                       const std::optional<std::string>& split) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open manifest: " + path.string());
  std::vector<ManifestRow> rows;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    try {
      auto row = ManifestRow::from_json(nlohmann::json::parse(line));
      if (!split || row.split == *split) rows.push_back(std::move(row));
    } catch (const std::exception& error) {
      throw std::runtime_error("invalid manifest row " + std::to_string(line_number) + ": " + error.what());
    }
  }
  if (rows.empty()) throw std::runtime_error("manifest contains no selected rows");
  return rows;
}

void assert_no_subject_leakage(const std::vector<std::vector<ManifestRow>>& manifests) {
  std::unordered_map<std::string, std::size_t> seen;
  for (std::size_t group = 0; group < manifests.size(); ++group) {
    for (const auto& row : manifests[group]) {
      const auto& key = row.family_id.empty() ? row.subject_id : row.family_id;
      const auto [item, inserted] = seen.emplace(key, group);
      if (!inserted && item->second != group) throw std::runtime_error("participant/family leakage: " + key);
    }
  }
}

torch::Tensor pair_tokens(const std::string& text, const std::size_t limit) {
  std::vector<std::int64_t> tokens{65536};
  const auto maximum = std::min(text.size(), limit * 2);
  for (std::size_t index = 0; index < maximum; index += 2) {
    const auto first = static_cast<unsigned char>(text[index]);
    const auto second = index + 1 < maximum ? static_cast<unsigned char>(text[index + 1]) : 0;
    tokens.push_back(first | (static_cast<std::int64_t>(second) << 8));
  }
  tokens.push_back(65537);
  return torch::tensor(tokens, torch::kLong);
}

std::int64_t stable_id(const std::string& text, const std::int64_t modulo) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto value : text) {
    hash ^= static_cast<unsigned char>(value);
    hash *= 1099511628211ULL;
  }
  return static_cast<std::int64_t>(hash % static_cast<std::uint64_t>(modulo));
}

std::int64_t modality_id(const std::string& input) {
  static const std::unordered_map<std::string, std::int64_t> values{
      {"t1", 0}, {"bold", 1}, {"fmri", 1}, {"t2", 2}, {"flair", 3}, {"dwi", 4},
      {"swi", 5}, {"asl", 6}, {"qsm", 7}, {"magnitude", 8}, {"angiography", 9},
      {"parametric", 10}, {"perfusion", 11}, {"spectroscopy", 12}, {"phase", 13},
      {"fieldmap", 14}, {"segmentation", 14}};
  auto name = input;
  std::transform(name.begin(), name.end(), name.begin(), [](const unsigned char c) { return std::tolower(c); });
  const auto item = values.find(name);
  return item == values.end() ? 15 : item->second;
}

std::int64_t view_id(const std::string& input) {
  static const std::unordered_map<std::string, std::int64_t> values{
      {"axial", 0}, {"transverse", 0}, {"coronal", 1}, {"sagittal", 2},
      {"oblique", 3}, {"projection", 4}, {"mip", 4}, {"cine", 5}};
  auto name = input;
  std::transform(name.begin(), name.end(), name.begin(), [](const unsigned char c) { return std::tolower(c); });
  const auto item = values.find(name);
  return item == values.end() ? 7 : item->second;
}

FeaturePack load_feature_pack(const ManifestRow& row, const bool load_physics,
                              const std::filesystem::path& root, const bool load_resources) {
  auto path = std::filesystem::path(row.output_pack);
  if (path.is_relative() && !root.empty()) path = root / path;
  if (row.output_pack.empty() || !std::filesystem::exists(path)) {
    FeaturePack direct;
    direct.sample_id = row.sample_id; direct.subject_id = row.subject_id;
    direct.visit_id = row.visit_id; direct.dataset = row.dataset; direct.metadata = row.metadata;
    const auto load_volume = [&](const std::string& value) {
      auto source = std::filesystem::path(value);
      if (source.is_relative() && !root.empty()) source = root / source;
      const auto image = ntfm::read_nifti(source.string());
      return torch::from_blob(const_cast<float*>(image.data.data()),
          {image.shape.t, 1, image.shape.z, image.shape.y, image.shape.x}, torch::kFloat32).clone();
    };
    if (!row.t1_nifti.empty()) direct.values["t1_volume"] = load_volume(row.t1_nifti).index({Slice(0, 1)});
    if (!row.fmri_nifti.empty()) {
      auto volumes = load_volume(row.fmri_nifti);
      const auto maximum = std::min<std::int64_t>(volumes.size(0), 8);
      if (volumes.size(0) > maximum) {
        const auto positions = torch::linspace(0, volumes.size(0) - 1, maximum).round().to(torch::kLong);
        volumes = volumes.index_select(0, positions);
      }
      direct.values["fmri_volume"] = volumes;
    }
    direct.values["quality"] = torch::zeros({16});
    direct.values["quality_mask"] = torch::zeros({16}, torch::kBool);
    direct.values["query_tokens"] = pair_tokens("task=" + (row.task.empty() ? "unknown" : row.task) +
        ";contrast=" + (row.contrast.empty() ? "unspecified" : row.contrast), 192);
    if (load_resources) add_native_resources(direct, row, root);
    return direct;
  }
  H5::H5File file(path.string(), H5F_ACC_RDONLY);
  FeaturePack pack;
  pack.sample_id = row.sample_id;
  pack.subject_id = row.subject_id;
  pack.visit_id = row.visit_id;
  pack.dataset = row.dataset;
  pack.values["dicom_tokens"] = read_dataset(file, "/dicom/tokens", torch::kLong);
  const auto t1_raw = read_dataset(file, "/images/t1_slices", torch::kFloat32);
  const auto t1 = t1_raw.reshape({-1, 1, t1_raw.size(-2), t1_raw.size(-1)});
  const auto epi_raw = read_dataset(file, "/images/epi_slices", torch::kFloat32);
  const auto epi = epi_raw.reshape({-1, 1, epi_raw.size(-2), epi_raw.size(-1)});
  pack.values["slices"] = torch::cat({t1, epi}, 0);
  pack.values["slice_types"] = torch::cat({torch::zeros({t1.size(0)}, torch::kLong),
                                            torch::ones({epi.size(0)}, torch::kLong)});
  pack.values["compiled_spatial"] = read_dataset(file, "/compiled/spatial", torch::kFloat32);
  pack.values["compiled_temporal"] = read_dataset(file, "/compiled/temporal", torch::kFloat32);
  pack.values["motion"] = read_dataset(file, "/motion/rigid", torch::kFloat32);
  const auto quality = read_dataset(file, "/quality/metrics", torch::kFloat32);
  pack.values["quality_mask"] = torch::isfinite(quality);
  pack.values["quality"] = torch::nan_to_num(quality);
  pack.values["query_tokens"] = pair_tokens("task=" + (row.task.empty() ? "unknown" : row.task) +
                                             ";contrast=" + (row.contrast.empty() ? "unspecified" : row.contrast), 192);
  pack.values["subject_hash"] = torch::tensor(stable_id(row.subject_id));
  pack.values["visit_hash"] = torch::tensor(stable_id(row.visit_id));
  pack.values["task_hash"] = torch::tensor(stable_id(row.task));
  const std::array<std::pair<const char*, const char*>, 12> target_paths{{
      {"map", "/targets/map"}, {"map_template", "/targets/map_template"},
      {"map_residual", "/targets/map_residual"}, {"behavior", "/targets/behavior"},
      {"state", "/targets/state"}, {"clinical", "/targets/clinical"},
      {"future", "/targets/future"}, {"concepts", "/targets/concepts"},
      {"teacher_hidden", "/teacher/hidden"}, {"teacher_map", "/teacher/map"},
      {"teacher_behavior", "/teacher/behavior"}, {"teacher_temporal_basis", "/teacher/temporal_basis"}}};
  for (const auto& [key, target_path] : target_paths) {
    if (exists(file, target_path)) pack.targets[key] = read_dataset(file, target_path, torch::kFloat32);
  }
  if (load_physics) {
    const auto bold_path = exists(file, "/physics/bold") ? "/physics/bold" : "/compiled/parcel_series";
    if (exists(file, bold_path)) pack.targets["physics_bold"] = read_dataset(file, bold_path, torch::kFloat32);
    if (exists(file, "/physics/anatomy_graph")) pack.targets["physics_anatomy"] = read_dataset(file, "/physics/anatomy_graph", torch::kFloat32);
    if (exists(file, "/physics/neural_drive")) pack.targets["physics_drive"] = read_dataset(file, "/physics/neural_drive", torch::kFloat32);
    if (exists(file, "/physics/mask")) pack.targets["physics_mask"] = read_dataset(file, "/physics/mask", torch::kBool);
    if (exists(file, "/physics/dt_seconds")) pack.targets["physics_dt"] = read_dataset(file, "/physics/dt_seconds", torch::kFloat32);
  }
  if (exists(file, "/inputs/biomarkers")) {
    const auto values = read_dataset(file, "/inputs/biomarkers", torch::kFloat32);
    pack.values["biomarkers"] = torch::nan_to_num(values);
    pack.values["biomarker_mask"] = exists(file, "/inputs/biomarker_mask")
        ? read_dataset(file, "/inputs/biomarker_mask", torch::kBool) : torch::isfinite(values);
  }
  if (exists(file, "/inputs/clinical")) {
    const auto values = read_dataset(file, "/inputs/clinical", torch::kFloat32);
    pack.values["clinical_inputs"] = torch::nan_to_num(values);
    pack.values["clinical_input_mask"] = exists(file, "/inputs/clinical_mask")
        ? read_dataset(file, "/inputs/clinical_mask", torch::kBool) : torch::isfinite(values);
  }
  if (exists(file, "/raw/t1_volume")) {
    auto value = read_dataset(file, "/raw/t1_volume", torch::kFloat32);
    pack.values["t1_volume"] = value.dim() == 3 ? value.unsqueeze(0).unsqueeze(0) : value;
  }
  if (exists(file, "/raw/fmri_volume")) {
    auto value = read_dataset(file, "/raw/fmri_volume", torch::kFloat32);
    pack.values["fmri_volume"] = value.dim() == 4 ? value.unsqueeze(1) : value;
  }
  pack.metadata = row.metadata;
  pack.metadata["tr_seconds"] = row.tr_seconds;
  if (load_resources) add_native_resources(pack, row, root);
  return pack;
}

TensorMap collate_feature_packs(const std::vector<FeaturePack>& packs, TensorMap* targets) {
  if (packs.empty()) throw std::invalid_argument("cannot collate an empty batch");
  TensorMap result;
  add_padded(result, packs, "query_tokens", "query_mask");
  add_padded(result, packs, "dicom_tokens", "dicom_mask");
  add_padded(result, packs, "compiled_spatial", "spatial_mask");
  add_padded(result, packs, "compiled_temporal", "temporal_mask");
  add_padded(result, packs, "slices", "slice_mask");
  add_padded(result, packs, "t1_volume", "t1_volume_mask");
  add_padded(result, packs, "fmri_volume", "fmri_volume_mask");
  add_padded(result, packs, "mr_volumes", "mr_volume_mask");
  add_padded(result, packs, "mr_images", "mr_image_mask");
  add_padded(result, packs, "mr_video", "mr_video_mask");
  for (const auto* key : {"mr_image_types", "mr_image_views", "mr_image_positions",
                          "mr_video_types", "mr_video_views", "mr_video_positions"}) {
    std::vector<torch::Tensor> values;
    for (const auto& pack : packs) {
      values.push_back(pack.values.count(key) ? pack.values.at(key) : torch::empty({0}, torch::kLong));
    }
    const auto padded = pad_first_dimension(values);
    if (padded.first.defined() && padded.first.numel() != 0) result[key] = padded.first;
  }
  std::vector<torch::Tensor> slice_types;
  std::vector<torch::Tensor> quality;
  for (const auto& pack : packs) {
    slice_types.push_back(pack.values.count("slice_types") ? pack.values.at("slice_types")
                                                            : torch::empty({0}, torch::kLong));
    quality.push_back(pack.values.at("quality"));
  }
  if (result.count("slices")) result["slice_types"] = pad_first_dimension(slice_types).first;
  result["quality"] = pad_first_dimension(quality).first;
  if (targets != nullptr) {
    std::set<std::string> shared;
    for (const auto& [key, value] : packs.front().targets) shared.insert(key);
    for (std::size_t index = 1; index < packs.size(); ++index) {
      for (auto item = shared.begin(); item != shared.end();) {
        if (packs[index].targets.find(*item) == packs[index].targets.end()) item = shared.erase(item);
        else ++item;
      }
    }
    for (const auto& key : shared) {
      std::vector<torch::Tensor> values;
      for (const auto& pack : packs) values.push_back(pack.targets.at(key));
      const auto same = std::all_of(values.begin(), values.end(), [&](const auto& value) {
        return value.sizes() == values.front().sizes();
      });
      (*targets)[key] = same ? torch::stack(values) : pad_first_dimension(values).first;
    }
  }
  return result;
}

TensorMap apply_context_limits(TensorMap batch, const YAML::Node& limits) {
  if (!limits) return batch;
  const auto cut = [&](const std::string& key, const char* limit_name) {
    if (!limits[limit_name]) return;
    const auto item = batch.find(key);
    if (item == batch.end()) return;
    const auto length = limits[limit_name].as<std::int64_t>();
    if (length > 0 && item->second.dim() > 1) item->second = item->second.index({Slice(), Slice(0, length)});
  };
  cut("dicom_tokens", "dicom_tokens"); cut("dicom_mask", "dicom_tokens");
  cut("slices", "slices"); cut("slice_types", "slices"); cut("slice_mask", "slices");
  cut("compiled_spatial", "spatial_tokens"); cut("spatial_mask", "spatial_tokens");
  cut("compiled_temporal", "temporal_tokens"); cut("temporal_mask", "temporal_tokens");
  return batch;
}

TensorMap apply_training_augmentation(TensorMap batch, const YAML::Node& config) {
  if (!config) return batch;
  if (batch.count("dicom_tokens") && config["dicom_token_mask"]) {
    const auto probability = config["dicom_token_mask"].as<double>();
    const auto selected = torch::rand_like(batch.at("dicom_tokens"), torch::kFloat32) < probability;
    batch["dicom_tokens"] = torch::where(selected & batch.at("dicom_mask"),
                                          torch::full_like(batch.at("dicom_tokens"), 65544),
                                          batch.at("dicom_tokens"));
  }
  if (batch.count("slice_mask") && config["slice_mask"]) {
    batch["slice_mask"] = batch.at("slice_mask") &
        (torch::rand_like(batch.at("slice_mask"), torch::kFloat32) >= config["slice_mask"].as<double>());
  }
  if (batch.count("compiled_temporal") && config["motion_corruption_probability"]) {
    const auto probability = config["motion_corruption_probability"].as<double>();
    if (torch::rand({1}).item<double>() < probability) {
      const auto scale = batch.at("compiled_temporal").to(torch::kFloat32).std(1, true, true).clamp_min(1e-4);
      batch["compiled_temporal"] = batch.at("compiled_temporal") +
          torch::randn_like(batch.at("compiled_temporal")) * scale * 0.08;
    }
  }
  return batch;
}

FeaturePackDataset::FeaturePackDataset(std::filesystem::path manifest,
                                       std::optional<std::string> split,
                                       const bool load_physics, const bool load_resources)
    : root_(std::filesystem::absolute(manifest).parent_path()),
      rows_(load_manifest(manifest, split)), load_physics_(load_physics),
      load_resources_(load_resources) {}

FeaturePack FeaturePackDataset::get(const std::size_t index) {
  return load_feature_pack(rows_.at(index), load_physics_, root_, load_resources_);
}

std::optional<std::size_t> FeaturePackDataset::size() const { return rows_.size(); }

HierarchicalDistributedSampler::HierarchicalDistributedSampler(
    const std::vector<ManifestRow>& rows, const std::int64_t replicas, const std::int64_t rank,
    const std::uint64_t seed, const double dataset_temperature)
    : rows_(&rows), replicas_(replicas), rank_(rank), seed_(seed), temperature_(dataset_temperature) {
  if (replicas <= 0 || rank < 0 || rank >= replicas) throw std::invalid_argument("invalid distributed sampler rank");
}

void HierarchicalDistributedSampler::set_epoch(const std::int64_t epoch) { epoch_ = epoch; }

std::vector<std::size_t> HierarchicalDistributedSampler::indices() const {
  std::unordered_map<std::string, std::vector<std::size_t>> groups;
  for (std::size_t index = 0; index < rows_->size(); ++index) groups[rows_->at(index).dataset].push_back(index);
  std::mt19937_64 generator(seed_ + static_cast<std::uint64_t>(epoch_));
  std::vector<std::size_t> ordered;
  for (auto& [dataset, values] : groups) {
    std::shuffle(values.begin(), values.end(), generator);
    const auto repeats = std::max<std::size_t>(1, static_cast<std::size_t>(
        std::pow(static_cast<double>(rows_->size()) / values.size(), 1.0 - temperature_)));
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) ordered.insert(ordered.end(), values.begin(), values.end());
  }
  std::shuffle(ordered.begin(), ordered.end(), generator);
  std::vector<std::size_t> local;
  for (std::size_t index = static_cast<std::size_t>(rank_); index < ordered.size(); index += replicas_) {
    local.push_back(ordered[index]);
  }
  return local;
}

}  // namespace neurotaskfm
