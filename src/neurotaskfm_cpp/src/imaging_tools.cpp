#include "neurotaskfm/imaging_tools.h"

#include <H5Cpp.h>
#include <nifti1_io.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include "neurotask/nifti_reader.h"

namespace neurotaskfm {
namespace {

std::vector<hsize_t> hdf_shape(const H5::DataSet& dataset) {
  auto space = dataset.getSpace();
  const auto rank = space.getSimpleExtentNdims();
  if (rank <= 0) throw std::runtime_error("scalar HDF5 datasets are not image arrays");
  std::vector<hsize_t> shape(static_cast<std::size_t>(rank));
  space.getSimpleExtentDims(shape.data());
  return shape;
}

std::size_t element_count(const std::vector<hsize_t>& shape) {
  std::size_t count = 1;
  for (const auto dimension : shape) {
    const auto size = static_cast<std::size_t>(dimension);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) {
      throw std::overflow_error("array is too large");
    }
    count *= size;
  }
  return count;
}

std::vector<float> read_real_dataset(H5::H5File& file, const std::string& path,
                                     std::vector<hsize_t>* output_shape = nullptr) {
  if (H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) <= 0) {
    throw std::runtime_error("HDF5 dataset does not exist: " + path);
  }
  auto dataset = file.openDataSet(path);
  if (dataset.getTypeClass() != H5T_INTEGER && dataset.getTypeClass() != H5T_FLOAT) {
    throw std::runtime_error("HDF5 dataset is not a real numeric array: " + path);
  }
  auto shape = hdf_shape(dataset);
  std::vector<float> values(element_count(shape));
  if (!values.empty()) dataset.read(values.data(), H5::PredType::NATIVE_FLOAT);
  if (output_shape != nullptr) *output_shape = std::move(shape);
  return values;
}

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::vector<Complex32> read_complex_dataset(H5::H5File& file, const std::string& path,
                                            std::vector<hsize_t>& logical_shape,
                                            const std::optional<std::string>& real_path,
                                            const std::optional<std::string>& imaginary_path) {
  if (real_path || imaginary_path) {
    if (!real_path || !imaginary_path) throw std::invalid_argument("both real and imaginary datasets are required");
    std::vector<hsize_t> imaginary_shape;
    auto real = read_real_dataset(file, *real_path, &logical_shape);
    auto imaginary = read_real_dataset(file, *imaginary_path, &imaginary_shape);
    if (logical_shape != imaginary_shape) throw std::runtime_error("real and imaginary dataset shapes differ");
    std::vector<Complex32> values(real.size());
    for (std::size_t index = 0; index < values.size(); ++index) values[index] = {real[index], imaginary[index]};
    return values;
  }

  if (H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) <= 0) {
    throw std::runtime_error("HDF5 dataset does not exist: " + path);
  }
  auto dataset = file.openDataSet(path);
  logical_shape = hdf_shape(dataset);
  if (dataset.getTypeClass() == H5T_COMPOUND) {
    auto type = dataset.getCompType();
    const auto members = type.getNmembers();
    if (members < 2) throw std::runtime_error("complex compound dataset has fewer than two members");
    int real_member = -1;
    int imaginary_member = -1;
    for (int index = 0; index < members; ++index) {
      const auto name = lowercase(type.getMemberName(static_cast<unsigned int>(index)));
      if (name == "r" || name == "real" || name == "re") real_member = index;
      if (name == "i" || name == "imag" || name == "imaginary" || name == "im") imaginary_member = index;
    }
    if (real_member < 0 || imaginary_member < 0) { real_member = 0; imaginary_member = 1; }
    H5::CompType memory_type(sizeof(Complex32));
    memory_type.insertMember(type.getMemberName(static_cast<unsigned int>(real_member)),
                             HOFFSET(Complex32, real), H5::PredType::NATIVE_FLOAT);
    memory_type.insertMember(type.getMemberName(static_cast<unsigned int>(imaginary_member)),
                             HOFFSET(Complex32, imag), H5::PredType::NATIVE_FLOAT);
    std::vector<Complex32> values(element_count(logical_shape));
    if (!values.empty()) dataset.read(values.data(), memory_type);
    return values;
  }
  if (dataset.getTypeClass() != H5T_INTEGER && dataset.getTypeClass() != H5T_FLOAT) {
    throw std::runtime_error("unsupported k-space HDF5 datatype");
  }
  if (logical_shape.empty() || logical_shape.back() != 2) {
    throw std::runtime_error("real k-space array must have a final [real, imaginary] dimension of length two");
  }
  const auto scalar_count = element_count(logical_shape);
  std::vector<float> interleaved(scalar_count);
  if (!interleaved.empty()) dataset.read(interleaved.data(), H5::PredType::NATIVE_FLOAT);
  logical_shape.pop_back();
  std::vector<Complex32> values(interleaved.size() / 2);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = {interleaved[index * 2], interleaved[index * 2 + 1]};
  }
  return values;
}

std::int64_t normalized_axis(std::int64_t axis, const std::int64_t rank) {
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) throw std::invalid_argument("axis lies outside the k-space rank");
  return axis;
}

CartesianKSpace canonicalize(std::vector<Complex32> source, const std::vector<hsize_t>& shape,
                             const int fft_dimensions,
                             const std::optional<std::int64_t> requested_frame_axis,
                             const std::optional<std::int64_t> requested_coil_axis) {
  const auto rank = static_cast<std::int64_t>(shape.size());
  if (fft_dimensions != 2 && fft_dimensions != 3) throw std::invalid_argument("--fft-dims must be 2 or 3");
  if (rank < fft_dimensions || rank > fft_dimensions + 2) {
    throw std::invalid_argument("k-space rank must contain spatial axes and at most frame and coil axes");
  }
  const auto prefix = rank - fft_dimensions;
  std::int64_t default_frame = prefix == 2 ? 0 : -1;
  std::int64_t default_coil = prefix >= 1 ? prefix - 1 : -1;
  auto frame_axis = requested_frame_axis.value_or(default_frame);
  auto coil_axis = requested_coil_axis.value_or(default_coil);
  if (frame_axis >= 0) frame_axis = normalized_axis(frame_axis, rank);
  if (coil_axis >= 0) coil_axis = normalized_axis(coil_axis, rank);
  if (frame_axis >= prefix || coil_axis >= prefix) {
    throw std::invalid_argument("frame and coil axes must precede the trailing spatial axes");
  }
  if (frame_axis >= 0 && coil_axis >= 0 && frame_axis == coil_axis) {
    throw std::invalid_argument("frame and coil axes must be distinct");
  }
  for (std::int64_t axis = 0; axis < prefix; ++axis) {
    if (axis != frame_axis && axis != coil_axis) throw std::invalid_argument("unassigned non-spatial k-space axis");
  }

  CartesianKSpace output;
  output.fft_dimensions = fft_dimensions;
  output.frames = frame_axis >= 0 ? static_cast<std::int64_t>(shape[static_cast<std::size_t>(frame_axis)]) : 1;
  output.coils = coil_axis >= 0 ? static_cast<std::int64_t>(shape[static_cast<std::size_t>(coil_axis)]) : 1;
  output.width = static_cast<std::int64_t>(shape.back());
  output.height = static_cast<std::int64_t>(shape[shape.size() - 2]);
  output.depth = fft_dimensions == 3 ? static_cast<std::int64_t>(shape[shape.size() - 3]) : 1;
  const auto canonical_order = prefix <= 1 || (frame_axis == 0 && coil_axis == 1);
  if (canonical_order) {
    output.samples = std::move(source);
    return output;
  }
  output.samples.resize(source.size());

  std::vector<std::size_t> strides(shape.size(), 1);
  for (std::size_t axis = shape.size() - 1; axis > 0; --axis) {
    strides[axis - 1] = strides[axis] * static_cast<std::size_t>(shape[axis]);
  }
  std::vector<std::size_t> coordinate(shape.size(), 0);
  for (std::int64_t frame = 0; frame < output.frames; ++frame) {
    for (std::int64_t coil = 0; coil < output.coils; ++coil) {
      for (std::int64_t z = 0; z < output.depth; ++z) {
        for (std::int64_t y = 0; y < output.height; ++y) {
          for (std::int64_t x = 0; x < output.width; ++x) {
            std::fill(coordinate.begin(), coordinate.end(), 0);
            if (frame_axis >= 0) coordinate[static_cast<std::size_t>(frame_axis)] = static_cast<std::size_t>(frame);
            if (coil_axis >= 0) coordinate[static_cast<std::size_t>(coil_axis)] = static_cast<std::size_t>(coil);
            coordinate[shape.size() - 1] = static_cast<std::size_t>(x);
            coordinate[shape.size() - 2] = static_cast<std::size_t>(y);
            if (fft_dimensions == 3) coordinate[shape.size() - 3] = static_cast<std::size_t>(z);
            std::size_t source_index = 0;
            for (std::size_t axis = 0; axis < shape.size(); ++axis) source_index += coordinate[axis] * strides[axis];
            const auto destination = static_cast<std::size_t>(
                ((((frame * output.coils + coil) * output.depth + z) * output.height + y) * output.width + x));
            output.samples[destination] = source[source_index];
          }
        }
      }
    }
  }
  return output;
}

void ensure_parent(const std::filesystem::path& path) {
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
}

void write_nifti(const ReconstructedImage& source, const std::filesystem::path& path,
                 const std::optional<std::filesystem::path>& reference = std::nullopt) {
  ensure_parent(path);
  int dimensions[8]{source.frames > 1 ? 4 : 3, static_cast<int>(source.width),
                    static_cast<int>(source.height), static_cast<int>(source.depth),
                    static_cast<int>(source.frames), 1, 1, 1};
  auto* image = nifti_make_new_nim(dimensions, NIFTI_TYPE_FLOAT32, 1);
  if (image == nullptr) throw std::runtime_error("failed to allocate NIfTI output");
  if (reference) {
    auto* geometry = nifti_image_read(reference->string().c_str(), 0);
    if (geometry == nullptr) { nifti_image_free(image); throw std::runtime_error("failed to read reference NIfTI geometry"); }
    image->dx = geometry->dx; image->dy = geometry->dy; image->dz = geometry->dz; image->dt = geometry->dt;
    image->pixdim[1] = geometry->pixdim[1]; image->pixdim[2] = geometry->pixdim[2];
    image->pixdim[3] = geometry->pixdim[3]; image->pixdim[4] = geometry->pixdim[4];
    image->qform_code = geometry->qform_code; image->sform_code = geometry->sform_code;
    image->qto_xyz = geometry->qto_xyz; image->qto_ijk = geometry->qto_ijk;
    image->sto_xyz = geometry->sto_xyz; image->sto_ijk = geometry->sto_ijk;
    nifti_image_free(geometry);
  }
  std::memcpy(image->data, source.magnitude.data(), source.magnitude.size() * sizeof(float));
  nifti_set_filenames(image, path.string().c_str(), 0, 1);
  nifti_image_write(image);
  nifti_image_free(image);
}

void write_hdf_volume(const ReconstructedImage& source, const std::filesystem::path& path,
                      const std::string& dataset_path) {
  ensure_parent(path);
  H5::H5File file(path.string(), H5F_ACC_TRUNC);
  auto parent = std::filesystem::path(dataset_path).parent_path().string();
  if (!parent.empty() && parent != "/") file.createGroup(parent);
  const std::array<hsize_t, 4> shape{static_cast<hsize_t>(source.frames),
                                    static_cast<hsize_t>(source.depth),
                                    static_cast<hsize_t>(source.height),
                                    static_cast<hsize_t>(source.width)};
  H5::DataSpace space(4, shape.data());
  auto dataset = file.createDataSet(dataset_path, H5::PredType::NATIVE_FLOAT, space);
  if (!source.magnitude.empty()) dataset.write(source.magnitude.data(), H5::PredType::NATIVE_FLOAT);
  const std::string layout = "frame,depth,height,width";
  H5::StrType text(H5::PredType::C_S1, layout.size());
  H5::DataSpace scalar(H5S_SCALAR);
  auto attribute = dataset.createAttribute("axis_order", text, scalar);
  attribute.write(text, layout);
}

bool is_hdf5_path(const std::filesystem::path& path) {
  const auto extension = lowercase(path.extension().string());
  return extension == ".h5" || extension == ".hdf5" || extension == ".mrd";
}

nlohmann::json statistics(const std::vector<float>& values) {
  double sum = 0.0;
  double square_sum = 0.0;
  std::size_t finite_count = 0;
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  for (const auto value : values) {
    if (!std::isfinite(value)) continue;
    ++finite_count; sum += value; square_sum += static_cast<double>(value) * value;
    minimum = std::min(minimum, value); maximum = std::max(maximum, value);
  }
  nlohmann::json result{{"elements", values.size()}, {"finite_elements", finite_count},
                        {"finite_fraction", values.empty() ? 0.0 : static_cast<double>(finite_count) / values.size()}};
  if (finite_count != 0) {
    const auto mean = sum / static_cast<double>(finite_count);
    result.update({{"minimum", minimum}, {"maximum", maximum}, {"mean", mean},
                   {"standard_deviation", std::sqrt(std::max(0.0, square_sum / finite_count - mean * mean))}});
  }
  return result;
}

int kspace_tool(const Arguments& arguments) {
  const auto input_path = std::filesystem::path(arguments.require("input"));
  H5::H5File file(input_path.string(), H5F_ACC_RDONLY);
  std::vector<hsize_t> shape;
  const auto real = arguments.has("real-dataset")
      ? std::optional<std::string>(arguments.require("real-dataset")) : std::nullopt;
  const auto imaginary = arguments.has("imag-dataset")
      ? std::optional<std::string>(arguments.require("imag-dataset")) : std::nullopt;
  auto values = read_complex_dataset(file, arguments.get("dataset", "/kspace"), shape, real, imaginary);
  const auto frame_axis = arguments.has("frame-axis")
      ? std::optional<std::int64_t>(arguments.integer("frame-axis", -1)) : std::nullopt;
  const auto coil_axis = arguments.has("coil-axis")
      ? std::optional<std::int64_t>(arguments.integer("coil-axis", -1)) : std::nullopt;
  auto canonical = canonicalize(std::move(values), shape, static_cast<int>(arguments.integer("fft-dims", 2)),
                                frame_axis, coil_axis);
  auto reconstructed = reconstruct_cartesian_cuda(canonical);
  const auto output = std::filesystem::path(arguments.require("output"));
  if (is_hdf5_path(output)) {
    write_hdf_volume(reconstructed, output, arguments.get("output-dataset", "/reconstruction/magnitude"));
  } else {
    const auto reference = arguments.has("reference")
        ? std::optional<std::filesystem::path>(arguments.require("reference")) : std::nullopt;
    write_nifti(reconstructed, output, reference);
  }
  std::cout << nlohmann::json{{"input", input_path.string()}, {"output", output.string()},
      {"input_shape", shape}, {"frames", reconstructed.frames}, {"coils", canonical.coils},
      {"depth", reconstructed.depth}, {"height", reconstructed.height},
      {"width", reconstructed.width}, {"fft_dimensions", canonical.fft_dimensions},
      {"coil_combination", "root_sum_of_squares"}}.dump() << '\n';
  return 0;
}

int convert_manifest_kspace_tool(const Arguments& arguments) {
  const auto manifest = std::filesystem::absolute(arguments.require("manifest"));
  const auto manifest_root = manifest.parent_path();
  const auto resource_root = arguments.has("resource-root")
      ? std::filesystem::absolute(arguments.require("resource-root")) : manifest_root;
  const auto output_root = std::filesystem::absolute(arguments.require("output-root"));
  const auto output_manifest = std::filesystem::path(arguments.require("output-manifest"));
  std::filesystem::create_directories(output_root);
  ensure_parent(output_manifest);
  std::ifstream input(manifest);
  std::ofstream output(output_manifest);
  if (!input || !output) throw std::runtime_error("unable to open k-space manifest conversion stream");
  std::string line;
  std::size_t line_number = 0;
  std::size_t converted = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    auto row = nlohmann::json::parse(line);
    if (!row.contains("mr_resources") || !row["mr_resources"].is_array()) {
      output << row.dump() << '\n';
      continue;
    }
    const auto sample_id = row.at("sample_id").get<std::string>();
    for (std::size_t index = 0; index < row["mr_resources"].size(); ++index) {
      auto& resource = row["mr_resources"][index];
      if (resource.value("kind", "") != "kspace") continue;
      auto source_path = std::filesystem::path(resource.at("path").get<std::string>());
      if (source_path.is_relative()) source_path = resource_root / source_path;
      if (!is_hdf5_path(source_path)) {
        throw std::runtime_error("manifest row " + std::to_string(line_number) +
                                 " uses non-HDF5 k-space; convert it to HDF5 first");
      }
      H5::H5File file(source_path.string(), H5F_ACC_RDONLY);
      std::vector<hsize_t> shape;
      auto complex = read_complex_dataset(file, resource.value("dataset", "/kspace"), shape,
                                          std::nullopt, std::nullopt);
      const auto fft_dimensions = resource.value("reconstruction_dims", 2);
      const auto frame_axis = resource.contains("frame_axis")
          ? std::optional<std::int64_t>(resource.at("frame_axis").get<std::int64_t>()) : std::nullopt;
      const auto coil_axis = resource.contains("coil_axis")
          ? std::optional<std::int64_t>(resource.at("coil_axis").get<std::int64_t>()) : std::nullopt;
      auto canonical = canonicalize(std::move(complex), shape, fft_dimensions, frame_axis, coil_axis);
      auto image = reconstruct_cartesian_cuda(canonical);
      const auto destination = output_root /
          (sample_id + "_resource-" + std::to_string(index) + "_magnitude.nii.gz");
      write_nifti(image, destination);
      auto metadata = resource.value("metadata", nlohmann::json::object());
      metadata["source_kspace"] = source_path.string();
      metadata["reconstruction"] = {
          {"algorithm", "centered_inverse_fft"}, {"fft_dimensions", fft_dimensions},
          {"coil_combination", "root_sum_of_squares"}, {"source_shape", shape}};
      resource["path"] = destination.string();
      resource["kind"] = image.frames > 1 ? "volume_series" : "volume";
      resource["layout"] = image.frames > 1 ? "volume_series" : "volume";
      resource["metadata"] = std::move(metadata);
      resource.erase("dataset"); resource.erase("coil_axis");
      resource.erase("frame_axis"); resource.erase("reconstruction_dims");
      ++converted;
    }
    output << row.dump() << '\n';
  }
  std::cout << nlohmann::json{{"input_manifest", manifest.string()},
      {"output_manifest", output_manifest.string()}, {"converted_resources", converted}}.dump() << '\n';
  return 0;
}

ReconstructedImage host_image(const ntfm::HostImage& image, std::vector<float> values = {}) {
  ReconstructedImage output;
  output.frames = image.shape.t; output.depth = image.shape.z;
  output.height = image.shape.y; output.width = image.shape.x;
  output.magnitude = values.empty() ? image.data : std::move(values);
  return output;
}

int normalize_volume_tool(const Arguments& arguments) {
  const auto input_path = std::filesystem::path(arguments.require("input"));
  const auto image = ntfm::read_nifti(input_path.string());
  auto tensor = torch::from_blob(const_cast<float*>(image.data.data()),
                                 {static_cast<std::int64_t>(image.data.size())}, torch::kFloat32).clone().to(torch::kCUDA);
  const auto finite_mask = torch::isfinite(tensor);
  const auto finite = tensor.masked_select(finite_mask);
  if (finite.numel() == 0) throw std::runtime_error("volume contains no finite values");
  const auto mode = lowercase(arguments.get("mode", "robust"));
  if (mode == "robust") {
    const auto lower = arguments.number("lower-quantile", 0.005);
    const auto upper = arguments.number("upper-quantile", 0.995);
    if (!(lower >= 0.0 && lower < upper && upper <= 1.0)) throw std::invalid_argument("invalid quantile range");
    const auto bounds = torch::quantile(finite, torch::tensor({lower, upper}, finite.options()));
    tensor = tensor.clamp(bounds[0].item<double>(), bounds[1].item<double>());
    tensor = (tensor - bounds.mean()) / ((bounds[1] - bounds[0]) / 4.0).clamp_min(1e-6);
  } else if (mode == "zscore") {
    tensor = (tensor - finite.mean()) / finite.std(false).clamp_min(1e-6);
  } else if (mode == "minmax") {
    tensor = (tensor - finite.min()) / (finite.max() - finite.min()).clamp_min(1e-6);
  } else {
    throw std::invalid_argument("--mode must be robust, zscore, or minmax");
  }
  tensor = torch::where(finite_mask, tensor, torch::zeros_like(tensor)).to(torch::kCPU).contiguous();
  std::vector<float> values(image.data.size());
  std::memcpy(values.data(), tensor.data_ptr<float>(), values.size() * sizeof(float));
  write_nifti(host_image(image, std::move(values)), arguments.require("output"), input_path);
  std::cout << nlohmann::json{{"input", input_path.string()}, {"output", arguments.require("output")},
                              {"mode", mode}}.dump() << '\n';
  return 0;
}

int inspect_data_tool(const Arguments& arguments) {
  const auto input = std::filesystem::path(arguments.require("input"));
  nlohmann::json result{{"path", input.string()}};
  if (is_hdf5_path(input)) {
    H5::H5File file(input.string(), H5F_ACC_RDONLY);
    const auto path = arguments.require("dataset");
    auto dataset = file.openDataSet(path);
    auto shape = hdf_shape(dataset);
    result.update({{"format", "hdf5"}, {"dataset", path}, {"shape", shape}});
    if (dataset.getTypeClass() == H5T_COMPOUND || arguments.has("complex")) {
      std::vector<hsize_t> logical;
      const auto values = read_complex_dataset(file, path, logical, std::nullopt, std::nullopt);
      std::vector<float> magnitude(values.size());
      std::transform(values.begin(), values.end(), magnitude.begin(),
                     [](const Complex32 value) { return std::hypot(value.real, value.imag); });
      result["logical_shape"] = logical;
      result["value"] = "complex_magnitude";
      result["statistics"] = statistics(magnitude);
    } else {
      result["statistics"] = statistics(read_real_dataset(file, path));
    }
  } else {
    const auto image = ntfm::read_nifti(input.string());
    result.update({{"format", "nifti"},
                   {"shape", {image.shape.x, image.shape.y, image.shape.z, image.shape.t}},
                   {"spacing", image.spacing}, {"affine", image.affine},
                   {"statistics", statistics(image.data)}});
  }
  if (arguments.has("output")) {
    ensure_parent(arguments.require("output"));
    std::ofstream stream(arguments.require("output"));
    stream << std::setw(2) << result << '\n';
  }
  std::cout << std::setw(2) << result << '\n';
  return 0;
}

int nifti_to_hdf5_tool(const Arguments& arguments) {
  const auto input = std::filesystem::path(arguments.require("input"));
  const auto image = ntfm::read_nifti(input.string());
  write_hdf_volume(host_image(image), arguments.require("output"), arguments.get("dataset", "/volume"));
  return 0;
}

int hdf5_to_nifti_tool(const Arguments& arguments) {
  H5::H5File file(arguments.require("input"), H5F_ACC_RDONLY);
  std::vector<hsize_t> shape;
  auto values = read_real_dataset(file, arguments.get("dataset", "/volume"), &shape);
  if (shape.size() < 2 || shape.size() > 4) throw std::invalid_argument("volume dataset rank must be 2, 3, or 4");
  ReconstructedImage image;
  image.width = static_cast<std::int64_t>(shape.back());
  image.height = static_cast<std::int64_t>(shape[shape.size() - 2]);
  image.depth = shape.size() >= 3 ? static_cast<std::int64_t>(shape[shape.size() - 3]) : 1;
  image.frames = shape.size() == 4 ? static_cast<std::int64_t>(shape[0]) : 1;
  image.magnitude = std::move(values);
  const auto reference = arguments.has("reference")
      ? std::optional<std::filesystem::path>(arguments.require("reference")) : std::nullopt;
  write_nifti(image, arguments.require("output"), reference);
  return 0;
}

int dicom_convert_tool(const Arguments& arguments) {
  const auto output = std::filesystem::absolute(arguments.require("output-dir"));
  std::filesystem::create_directories(output);
  std::vector<std::string> command{arguments.get("executable", "dcm2niix"),
      "-b", arguments.get("bids", "y"), "-z", arguments.get("compress", "y"),
      "-f", arguments.get("filename", "%p_%s"), "-o", output.string(), arguments.require("input")};
  std::vector<char*> raw;
  raw.reserve(command.size() + 1);
  for (auto& value : command) raw.push_back(value.data());
  raw.push_back(nullptr);
  const auto child = fork();
  if (child < 0) throw std::runtime_error("unable to fork dcm2niix process");
  if (child == 0) { execvp(raw[0], raw.data()); _exit(127); }
  int status = 0;
  if (waitpid(child, &status, 0) < 0) throw std::runtime_error("unable to wait for dcm2niix process");
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("dcm2niix failed with status " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
  }
  return 0;
}

}  // namespace

bool is_imaging_tool(const std::string& command) {
  static const std::array<std::string_view, 10> commands{
      "kspace-reconstruct", "kspace-convert", "normalize-volume", "inspect-data", "data-summary",
      "nifti-to-hdf5", "hdf5-to-nifti", "dicom-to-nifti", "dicom-convert",
      "convert-kspace-manifest"};
  return std::find(commands.begin(), commands.end(), command) != commands.end();
}

int run_imaging_tool(const std::string& command, const Arguments& arguments) {
  if (command == "kspace-reconstruct" || command == "kspace-convert") return kspace_tool(arguments);
  if (command == "convert-kspace-manifest") return convert_manifest_kspace_tool(arguments);
  if (command == "normalize-volume") return normalize_volume_tool(arguments);
  if (command == "inspect-data" || command == "data-summary") return inspect_data_tool(arguments);
  if (command == "nifti-to-hdf5") return nifti_to_hdf5_tool(arguments);
  if (command == "hdf5-to-nifti") return hdf5_to_nifti_tool(arguments);
  if (command == "dicom-to-nifti" || command == "dicom-convert") return dicom_convert_tool(arguments);
  throw std::invalid_argument("unknown native imaging command: " + command);
}

}  // namespace neurotaskfm
