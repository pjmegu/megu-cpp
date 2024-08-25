#include "./runPython.h"
#include "./util/util.h"
#include "glob/glob.hpp"
#include "pybind11/embed.h"
#include <filesystem>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <vector>
namespace py = pybind11;
namespace fs = std::filesystem;

namespace {
static mevil::BuildEnv build_env;

template <typename T>
std::vector<T> append(std::vector<T> &vec, std::vector<T> &app) {
    vec.insert(vec.end(), app.begin(), app.end());
    return vec;
}
} // namespace

PYBIND11_EMBEDDED_MODULE(mevil, mod) {
    py::class_<mevil::ModuleRef>(mod, "MevilModule")
        .def("addEvent",
             [](mevil::ModuleRef &module, py::str event_name, py::list passes) {
                 std::string cpp_event_name = event_name.cast<std::string>();
                 std::vector<std::string> cpp_passes;
                 for (auto pass : passes) {
                     cpp_passes.push_back(pass.cast<std::string>());
                 }

                 build_env.modules.at(module.id).events.push_back(
                     mevil::Module::Event{cpp_event_name, cpp_passes});
             });

    mod.def("addDialects", [](py::list dialects) {
        std::vector<std::string> cpp_dialects;

        for (auto dialect : dialects) {
            cpp_dialects.push_back(dialect.cast<std::string>());
        }

        append(build_env.dialects, cpp_dialects);
    });

    mod.def("createModule", [](py::list sources) {
        std::vector<std::string> cpp_sources;

        auto dirname = py::module::import("os").attr("path").attr("dirname");

        auto dir_path = fs::path(dirname(py::globals()["__file__"]).cast<std::string>());

        std::cout << "Dir Path: " << dir_path << std::endl;

        auto glob = py::module::import("glob").attr("glob");
        using namespace py::literals;

        for (auto source : sources) {
            std::vector<fs::path> paths;
            std::string source_str = source.cast<std::string>();
            std::cout << "Source: " << (dir_path.string() + "/" + source_str) << std::endl;
            paths = glob::rglob(dir_path.string() + "/" + source_str);
            for (auto path : paths) {
                std::cout << "Found: " << path << std::endl;
                cpp_sources.push_back(path.string());
            }
        }

        auto id = mevil::random();
        build_env.modules[id] = mevil::Module(cpp_sources);
        return mevil::ModuleRef(id);
    });
}

std::variant<std::nullopt_t, std::string>
mevil::runPython(const std::string &workspace_path) {
    py::scoped_interpreter guard{};

    auto top_path = fs::path(workspace_path) / "mevil.py";

    // eval toplevel file(.py)
    try {
        py::eval_file(top_path.string());
        std::cout << "Python Evaluated" << std::endl;
        std::cout << "Dialects: ";
        for (auto dialect : build_env.dialects) {
            std::cout << dialect << " ";
        }
        std::cout << "\n";
        std::cout << "Modules: ";
        for (auto [id, module] : build_env.modules) {
            std::cout << "Module " << id << "\n    " << " Sources: ";
            for (auto source : module.sources) {
                std::cout << source << " ";
            }
            std::cout << "Events: ";
            for (auto event : module.events) {
                std::cout << "Event " << event.name << " Passes: ";
                for (auto pass : event.passes) {
                    std::cout << pass << " ";
                }
            }
        }
        std::cout << std::endl;
    } catch (const py::error_already_set &e) {
        std::cout << "Python Error: " << e.what() << std::endl;
        return e.what();
    }

    return std::nullopt;
}

mevil::BuildEnv::BuildEnv() {
    dialects = std::vector<std::string>();
    modules = std::unordered_map<std::uint64_t, Module>();
}

mevil::Module::Module() {
    sources = std::vector<std::string>();
    events = std::vector<Event>();
}

mevil::Module::Module(std::vector<std::string> sources) {
    this->sources = sources;
    events = std::vector<Event>();
}

mevil::ModuleRef::ModuleRef(std::uint64_t id) { this->id = id; }