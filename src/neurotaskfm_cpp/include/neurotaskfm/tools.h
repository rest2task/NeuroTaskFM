#pragma once

#include <string>

#include "neurotaskfm/cli.h"

namespace neurotaskfm {

int run_tool(const std::string& command, const Arguments& arguments);
int run_probe(const Arguments& arguments);
int run_cross_state(const Arguments& arguments);
int run_evaluate(const Arguments& arguments);
int compiler_metrics(const Arguments& arguments);
int run_web_server(const Arguments& arguments);

}  // namespace neurotaskfm
