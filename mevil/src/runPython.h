#pragma once

#include "buildEnv.h"
#include <optional>
#include <string>
#include <variant>

// module object
namespace mevil {

std::variant<BuildEnv, std::string>
runPython(const std::string &workspace_path);
} // namespace mevil