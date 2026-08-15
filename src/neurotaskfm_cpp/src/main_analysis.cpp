#include <exception>
#include <iostream>
#include <string>

#include "neurotaskfm/cli.h"
#include "neurotaskfm/tools.h"

#ifndef NTFM_ANALYSIS_COMMAND
#define NTFM_ANALYSIS_COMMAND "evaluate"
#endif

int main(int argc, char** argv) {
  try {
    neurotaskfm::Arguments arguments(argc, argv);
    const std::string command = NTFM_ANALYSIS_COMMAND;
    if (command == "probe") return neurotaskfm::run_probe(arguments);
    if (command == "cross-state") return neurotaskfm::run_cross_state(arguments);
    return neurotaskfm::run_evaluate(arguments);
  } catch (const std::exception& error) {
    std::cerr << "ntfm-" << NTFM_ANALYSIS_COMMAND << ": " << error.what() << '\n';
    return 1;
  }
}
