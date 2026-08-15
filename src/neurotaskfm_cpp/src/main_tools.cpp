#include <exception>
#include <iostream>

#include "neurotaskfm/cli.h"
#include "neurotaskfm/tools.h"

int main(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
      std::cout
          << "usage: ntfm-tool COMMAND [OPTIONS]\n\n"
          << "Native imaging and data commands:\n"
          << "  dicom-to-nifti             Convert a de-identified DICOM tree with dcm2niix\n"
          << "  kspace-reconstruct         CUDA/cuFFT Cartesian reconstruction from HDF5\n"
          << "  convert-kspace-manifest    Reconstruct and rewrite all k-space resources\n"
          << "  normalize-volume           CUDA robust/z-score/min-max NIfTI normalization\n"
          << "  nifti-to-hdf5              Convert NIfTI to a numeric HDF5 dataset\n"
          << "  hdf5-to-nifti              Convert a numeric HDF5 dataset to NIfTI\n"
          << "  inspect-data               Report NIfTI or HDF5 shape and statistics\n"
          << "  validate-manifest          Validate paths, splits, resources, and packs\n"
          << "  build-manifest             Build a leakage-stable BIDS-style manifest\n"
          << "  prepare-packs              Run native NeuroCompiler over a manifest\n"
          << "  pack-raw-inputs            Add normalized teacher inputs to feature packs\n"
          << "  attach-targets             Attach supervised arrays to feature packs\n";
      return argc < 2 ? 2 : 0;
    }
    const std::string command = argv[1];
    neurotaskfm::Arguments arguments(argc, argv, 2);
    return neurotaskfm::run_tool(command, arguments);
  } catch (const std::exception& error) {
    std::cerr << "ntfm-tool: " << error.what() << '\n';
    return 1;
  }
}
