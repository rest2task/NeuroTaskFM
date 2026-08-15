#include "neurotask/dicom_tokenizer.h"
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>

namespace ntfm {
namespace {
constexpr uint32_t kBos = 65536;
constexpr uint32_t kEos = 65537;
constexpr uint32_t kTag = 65538;
constexpr uint32_t kValue = 65539;
constexpr uint32_t kPixel = 65540;
constexpr uint32_t kSeparator = 65541;
constexpr uint32_t kTruncated = 65542;
constexpr uint32_t kHashed = 65543;

uint64_t fnv1a(const std::string& value) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char c : value) { hash ^= c; hash *= 1099511628211ULL; }
  return hash;
}

void append_pairs(std::vector<uint32_t>& tokens, const uint8_t* bytes, size_t n, size_t limit) {
  for (size_t i = 0; i < n && tokens.size() < limit; i += 2) {
    const uint32_t lo = bytes[i];
    const uint32_t hi = i + 1 < n ? static_cast<uint32_t>(bytes[i + 1]) << 8U : 0U;
    tokens.push_back(lo | hi);
  }
}

void append_u64(std::vector<uint32_t>& tokens, uint64_t value, size_t limit) { append_pairs(tokens, reinterpret_cast<const uint8_t*>(&value), sizeof(value), limit); }

bool numeric_tag(const DcmTagKey& key) {
  static const std::array<DcmTagKey, 24> tags = {
    DCM_MagneticFieldStrength, DCM_RepetitionTime, DCM_EchoTime, DCM_InversionTime, DCM_FlipAngle,
    DCM_ImagingFrequency, DCM_PixelBandwidth, DCM_Rows, DCM_Columns, DCM_SliceThickness,
    DCM_SpacingBetweenSlices, DCM_PixelSpacing, DCM_ImageOrientationPatient, DCM_ImagePositionPatient,
    DCM_AcquisitionMatrix, DCM_NumberOfTemporalPositions, DCM_TemporalPositionIdentifier, DCM_EchoTrainLength,
    DCM_PercentSampling, DCM_PercentPhaseFieldOfView, DCM_SAR, DCM_BitsAllocated, DCM_BitsStored, DCM_PixelRepresentation
  };
  return std::find(tags.begin(), tags.end(), key) != tags.end();
}

const std::array<DcmTagKey, 38> kTags = {
  DCM_Modality, DCM_MagneticFieldStrength, DCM_Manufacturer, DCM_ManufacturerModelName, DCM_SoftwareVersions,
  DCM_ProtocolName, DCM_SequenceName, DCM_ScanningSequence, DCM_SequenceVariant, DCM_ScanOptions,
  DCM_MRAcquisitionType, DCM_RepetitionTime, DCM_EchoTime, DCM_InversionTime, DCM_FlipAngle,
  DCM_ImagingFrequency, DCM_PixelBandwidth, DCM_Rows, DCM_Columns, DCM_SliceThickness,
  DCM_SpacingBetweenSlices, DCM_PixelSpacing, DCM_ImageOrientationPatient, DCM_ImagePositionPatient,
  DCM_AcquisitionMatrix, DCM_InPlanePhaseEncodingDirection, DCM_NumberOfTemporalPositions,
  DCM_TemporalPositionIdentifier, DCM_EchoTrainLength, DCM_PercentSampling, DCM_PercentPhaseFieldOfView,
  DCM_SAR, DCM_BitsAllocated, DCM_BitsStored, DCM_HighBit, DCM_PixelRepresentation,
  DCM_SamplesPerPixel, DCM_PhotometricInterpretation
};
}

std::vector<uint32_t> DicomTokenizer::tokenize_file(const std::string& path) const {
  DcmFileFormat file;
  const OFCondition status = file.loadFile(path.c_str());
  if (!status.good()) throw std::runtime_error("Unable to parse DICOM: " + path + ": " + status.text());
  DcmDataset* ds = file.getDataset();
  std::vector<uint32_t> tokens;
  tokens.reserve(std::min<size_t>(config_.max_tokens, 8192));
  tokens.push_back(kBos);
  for (const auto& key : kTags) {
    OFString value;
    if (!ds->findAndGetOFStringArray(key, value).good()) continue;
    tokens.push_back(kTag);
    const uint16_t group = key.getGroup(), element = key.getElement();
    append_pairs(tokens, reinterpret_cast<const uint8_t*>(&group), sizeof(group), config_.max_tokens);
    append_pairs(tokens, reinterpret_cast<const uint8_t*>(&element), sizeof(element), config_.max_tokens);
    tokens.push_back(kValue);
    const std::string text(value.c_str());
    if (numeric_tag(key)) append_pairs(tokens, reinterpret_cast<const uint8_t*>(text.data()), text.size(), config_.max_tokens);
    else { tokens.push_back(kHashed); append_u64(tokens, fnv1a(text), config_.max_tokens); }
    tokens.push_back(kSeparator);
    if (tokens.size() >= config_.max_tokens) break;
  }
  if (config_.include_pixel_bytes && tokens.size() + 16 < config_.max_tokens) {
    const Uint8* bytes8 = nullptr;
    unsigned long count8 = 0;
    if (ds->findAndGetUint8Array(DCM_PixelData, bytes8, &count8).good() && bytes8 && count8) {
      tokens.push_back(kPixel);
      append_pairs(tokens, bytes8, std::min<size_t>(count8, config_.max_pixel_bytes), config_.max_tokens);
    } else {
      const Uint16* bytes16 = nullptr;
      unsigned long count16 = 0;
      if (ds->findAndGetUint16Array(DCM_PixelData, bytes16, &count16).good() && bytes16 && count16) {
        tokens.push_back(kPixel);
        append_pairs(tokens, reinterpret_cast<const uint8_t*>(bytes16), std::min<size_t>(count16 * sizeof(Uint16), config_.max_pixel_bytes), config_.max_tokens);
      }
    }
  }
  if (tokens.size() >= config_.max_tokens) { tokens.resize(config_.max_tokens - 1); tokens.push_back(kTruncated); }
  tokens.push_back(kEos);
  return tokens;
}

std::vector<uint32_t> DicomTokenizer::tokenize_tree(const std::string& root) const {
  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) if (entry.is_regular_file()) files.push_back(entry.path().string());
  std::sort(files.begin(), files.end());
  std::vector<uint32_t> all;
  all.reserve(config_.max_tokens);
  for (const auto& path : files) {
    try {
      auto part = tokenize_file(path);
      if (!all.empty()) all.push_back(kSeparator);
      const size_t room = config_.max_tokens > all.size() ? config_.max_tokens - all.size() : 0;
      all.insert(all.end(), part.begin(), part.begin() + std::min(room, part.size()));
      if (all.size() >= config_.max_tokens) break;
    } catch (const std::exception&) {
      continue;
    }
  }
  if (all.empty()) throw std::runtime_error("No readable de-identified DICOM files under " + root);
  return all;
}
}
