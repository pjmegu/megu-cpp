#pragma once

#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

namespace mevil {

namespace fs = std::filesystem;
const std::string DIALECT_MAP_FILE = "dialect_map.json";
class IdentMap {
  public:
    struct PassMap {
        std::string pass_path;
        std::string pass_name;
    };

  private:
    // identifier: dialect path (relative)
    std::unordered_map<std::string, std::string> dialect_map;
    // identifier: {pass path (relative), pass name}
    std::unordered_map<std::string, PassMap> pass_map;

  public:
    inline IdentMap() : dialect_map(), pass_map() {}

    inline IdentMap(std::unordered_map<std::string, std::string> dialect_map,
                    std::unordered_map<std::string, PassMap> pass_map)
        : dialect_map(dialect_map), pass_map(pass_map) {}

    inline IdentMap(std::string workspace_path) {
        // load dialect map
        std::ifstream map_file(
            fs::absolute(fs::path(workspace_path) / DIALECT_MAP_FILE));
        std::string s((std::istreambuf_iterator<char>(map_file)),
                      std::istreambuf_iterator<char>());
        nlohmann::json js = nlohmann::json::parse(s);
        if (js.find("dialect") != js.end()) {
            for (auto &[key, value] : js["dialect"].items()) {
                dialect_map[key] =
                    fs::absolute(fs::path(workspace_path) / value).string();
            }
        }

        if (js.find("pass") != js.end()) {
            for (auto &[key, value] : js["pass"].items()) {
                pass_map[key] = {
                    fs::absolute(fs::path(workspace_path) / value["path"])
                        .string(),
                    value["name"]};
            }
        }
    }

    inline std::string get_dialect_path(std::string identifier) {
        return dialect_map[identifier];
    }

    inline bool has_dialect(std::string identifier) {
        return dialect_map.find(identifier) != dialect_map.end();
    }

    inline std::string get_pass_path(std::string identifier) {
        return pass_map[identifier].pass_path;
    }

    inline std::string get_pass_name(std::string identifier) {
        return pass_map[identifier].pass_name;
    }
};

} // namespace mevil