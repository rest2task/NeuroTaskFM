#include <exception>
#include <iostream>

#include "neurotaskfm/cli.h"
#include "neurotaskfm/tools.h"

int main(int argc, char** argv) {
  try {
    return neurotaskfm::run_web_server(neurotaskfm::Arguments(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "neurotask-web: " << error.what() << '\n';
    return 1;
  }
}
