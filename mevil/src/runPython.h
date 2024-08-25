#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// module object
namespace mevil {

struct BuildEnv;

struct Module {
    std::vector<std::string> sources;
    
    struct Event {
        std::string name;
        std::vector<std::string> passes;
    };

    std::vector<Event> events;

    Module();
    Module(std::vector<std::string> sources);

    std::vector<std::string>& getSources() const;
    std::vector<Event>& getEvents() const;
};

struct ModuleRef {
    std::uint64_t id;
    ModuleRef(std::uint64_t id);
};

struct BuildEnv {
    std::vector<std::string> dialects;
    std::unordered_map<std::uint64_t, Module> modules;
    BuildEnv();
};

std::variant<std::nullopt_t, std::string>
runPython(const std::string &workspace_path);
} // namespace mevil