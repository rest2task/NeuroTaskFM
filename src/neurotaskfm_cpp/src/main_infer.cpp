#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>

#include "neurotaskfm/cli.h"
#include "neurotaskfm/runtime.h"

int main(int argc, char** argv) {
  try {
    neurotaskfm::Arguments arguments(argc, argv);
    std::optional<std::filesystem::path> output;
    if (arguments.has("out")) output = arguments.require("out");
    neurotaskfm::infer_request(arguments.require("config"), arguments.require("request"), output);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ntfm-infer: " << error.what() << '\n';
    return 1;
  }
}
