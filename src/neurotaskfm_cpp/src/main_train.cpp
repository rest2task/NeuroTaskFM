#include <exception>
#include <iostream>

#include "neurotaskfm/cli.h"
#include "neurotaskfm/config.h"
#include "neurotaskfm/runtime.h"

int main(int argc, char** argv) {
  try {
    neurotaskfm::Arguments arguments(argc, argv);
    const auto config = neurotaskfm::load_yaml(arguments.require("config"));
    const auto expert_parallel = config["parallel"] && config["parallel"]["expert_parallel_size"]
        ? config["parallel"]["expert_parallel_size"].as<std::int64_t>()
        : neurotaskfm::load_yaml(config["cluster"].as<std::string>())["expert_parallel_size"].as<std::int64_t>();
    neurotaskfm::Trainer trainer(config, neurotaskfm::init_distributed(expert_parallel));
    trainer.run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ntfm-train: " << error.what() << '\n';
    return 1;
  }
}
