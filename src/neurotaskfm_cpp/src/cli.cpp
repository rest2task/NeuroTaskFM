#include "neurotaskfm/cli.h"

#include <stdexcept>

namespace neurotaskfm {

Arguments::Arguments(const int argc, char** argv, const int start) {
  for (int index = start; index < argc; ++index) {
    std::string value = argv[index];
    if (value.rfind("--", 0) != 0) {
      positional_.push_back(std::move(value));
      continue;
    }
    value.erase(0, 2);
    const auto separator = value.find('=');
    if (separator != std::string::npos) {
      options_[value.substr(0, separator)].push_back(value.substr(separator + 1));
    } else if (index + 1 < argc && std::string(argv[index + 1]).rfind("--", 0) != 0) {
      options_[value].push_back(argv[++index]);
    } else {
      options_[value].push_back("true");
    }
  }
}

bool Arguments::has(const std::string& name) const { return options_.find(name) != options_.end(); }

std::string Arguments::require(const std::string& name) const {
  const auto item = options_.find(name);
  if (item == options_.end() || item->second.empty()) throw std::invalid_argument("missing --" + name);
  return item->second.back();
}

std::string Arguments::get(const std::string& name, std::string fallback) const {
  const auto item = options_.find(name);
  return item == options_.end() || item->second.empty() ? std::move(fallback) : item->second.back();
}

std::int64_t Arguments::integer(const std::string& name, const std::int64_t fallback) const {
  return has(name) ? std::stoll(get(name)) : fallback;
}

double Arguments::number(const std::string& name, const double fallback) const {
  return has(name) ? std::stod(get(name)) : fallback;
}

std::vector<std::string> Arguments::all(const std::string& name) const {
  const auto item = options_.find(name);
  return item == options_.end() ? std::vector<std::string>{} : item->second;
}

}  // namespace neurotaskfm
