#include "./loadlib/load_test.h"
#include "./runPython.h"
#include "CLI/CLI.hpp"
#include "buildEnv.h"
#include "constMLIR/constMLIR.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include "mokey/MokeyToken/MokeyTokenDialect.h"

int loadAndConstructMLIR(mevil::BuildEnv &build_env) {
    mlir::DialectRegistry registory;
    for (auto dialect : build_env.dialects) {
        if (build_env.ident_map.has_dialect(dialect)) {
            std::string dialect_path =
                build_env.ident_map.get_dialect_path(dialect);
            auto plugin = mlir::DialectPlugin::load(dialect_path);
            if (auto E = plugin.takeError()) {
                std::cout << "Failed to load dialect: "
                          << llvm::toString(std::move(E)) << std::endl;
                return 1;
            }

            plugin->registerDialectRegistryCallbacks(registory);
        } else {
            std::cout << "Error: dialect not found: " << dialect << std::endl;
            return 1;
        }
    }

    registory.insert<mokey::token::MokeyTokenDialect>();

    mlir::MLIRContext context(registory);

    mevil::constMLIR(build_env.modules, context);

    return 0;
}

int main(int argc, char **argv) {
    CLI::App app("mevil - Mokey Project Manager", "mevil");

    // mevil lint
    CLI::App *sub_lint = app.add_subcommand("lint", "Lint the project");
    std::string workspace_path = ".";
    sub_lint->add_option("-w,--workspace", workspace_path, "Workspace path")
        ->capture_default_str();

    // mevil tool
    CLI::App *sub_tool = app.add_subcommand("tool", "Tool for the project");

    // mevil tool load
    CLI::App *sub_tool_load =
        sub_tool->add_subcommand("load", "Load the project");
    std::string load_file_path;
    sub_tool_load->add_option("-f,--file", load_file_path, "Load file path")
        ->required();
    bool load_dialect;
    sub_tool_load->add_flag("-d,!-p", load_dialect, "dialect or pass")
        ->required();

    // parse state
    CLI11_PARSE(app, argc, argv);

    // run lint
    if (sub_lint->parsed()) {
        std::cout << "workspace path: " << workspace_path << std::endl;
        auto e = mevil::runPython(workspace_path);
        if (std::holds_alternative<std::string>(e)) {
            std::cout << "Error: " << std::get<std::string>(e) << std::endl;
            return 1;
        } else {
            mevil::BuildEnv build_env = std::get<mevil::BuildEnv>(e);
            if (loadAndConstructMLIR(build_env)) {
                return 1;
            }
        }
    }

    // run tool
    if (sub_tool->parsed()) {
        if (sub_tool_load->parsed()) {
            std::cout << "load file path: " << load_file_path << std::endl;
            std::cout << "load dialect: " << load_dialect << std::endl;

            if (load_dialect) {
                mevil::load_dialect_test(load_file_path);
            } else {
                mevil::load_pass_test(load_file_path);
            }
        }
    }

    return 0;
}