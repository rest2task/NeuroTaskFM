#include <exception>
#include <filesystem>
#include <iostream>
#include <vector>

#include "neurotaskfm/cli.h"
#include "neurotaskfm/runtime.h"

int main(int argc, char** argv) {
  try {
    neurotaskfm::Arguments arguments(argc, argv);
    std::vector<std::filesystem::path> packs;
    for (const auto& value : arguments.all("pack")) packs.emplace_back(value);
    if (packs.empty()) throw std::invalid_argument("at least one --pack is required");
    neurotaskfm::personalize(arguments.require("config"), packs, arguments.all("query"),
                             arguments.require("output"), arguments.integer("steps", 160),
                             arguments.number("learning-rate", 0.015));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ntfm-personalize: " << error.what() << '\n';
    return 1;
  }
}
