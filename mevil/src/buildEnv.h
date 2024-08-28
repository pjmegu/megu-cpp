#pragma once

#include "identMap.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mevil {
struct BuildEnv;

struct Module {
    std::vector<std::string> sources;

    struct Event {
        std::string name;
        std::vector<std::string> passes;
    };

    std::vector<Event> events;

    inline Module() {
        sources = std::vector<std::string>();
        events = std::vector<Event>();
    }

    inline Module(std::vector<std::string> sources) {
        this->sources = sources;
        events = std::vector<Event>();
    }

    const std::vector<std::string> &getSources() const;
    const std::vector<Event> &getEvents() const;
};

struct ModuleRef {
    std::uint64_t id;
    inline ModuleRef(std::uint64_t id) { this->id = id; }
};

struct BuildEnv {
    std::vector<std::string> dialects;
    std::unordered_map<std::uint64_t, Module> modules;

    IdentMap ident_map;

    inline BuildEnv() {
        dialects = std::vector<std::string>();
        modules = std::unordered_map<std::uint64_t, Module>();
        ident_map = IdentMap();
    }
};

} // namespace mevil