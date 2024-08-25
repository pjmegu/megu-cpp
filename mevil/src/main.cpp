#include "./runPython.h"
#include "CLI/CLI.hpp"
#include <iostream>
#include <string>


int main(int argc, char **argv) {
    CLI::App app("mevil - Mokey Project Manager", "mevil");

    // mevil lint
    CLI::App *sub_lint = app.add_subcommand("lint", "Lint the project");
    std::string workspace_path = ".";
    sub_lint->add_option("-w,--workspace", workspace_path, "Workspace path")
        ->capture_default_str();

    // parse state
    CLI11_PARSE(app, argc, argv);

    std::cout << "workspace path: " << workspace_path << std::endl;

    // run lint
    if (sub_lint->parsed()) {
        mevil::runPython(workspace_path);
    }

    return 0;
}