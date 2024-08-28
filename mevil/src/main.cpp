#include "./loadlib/load_test.h"
#include "./runPython.h"
#include "CLI/CLI.hpp"
#include <iostream>
#include <string>
#include <variant>

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
            std::cout << "Success" << std::endl;
            return 0;
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