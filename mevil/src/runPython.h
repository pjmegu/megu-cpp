#pragma once

#include <optional>
#include <string>
#include <variant>

// module object
namespace mevil {

std::variant<std::nullopt_t, std::string>
runPython(const std::string &workspace_path);
} // namespace mevil