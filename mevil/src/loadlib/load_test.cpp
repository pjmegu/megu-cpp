#include "dylib.hpp"
#include <filesystem>
#include <iostream>
#include <string>
#include "./load_test.h"

namespace fs = std::filesystem;

namespace mevil {
// TODO: Dialect読み込みテスト
void load_dialect_test(std::string path) {
    auto lib_path = fs::absolute(fs::path(path));
    try {
        dylib lib(lib_path);
    } catch (dylib::load_error &e) {
        std::cout << "Failed to load library: " << e.what() << std::endl;
        return;
    }

    std::cout << "Library loaded successfully" << std::endl;
}

// TODO: Passを読み込むテストを追加
void load_pass_test(std::string path) {
    auto lib_path = fs::absolute(fs::path(path));
    try {
        dylib lib(lib_path);
    } catch (dylib::load_error &e) {
        std::cout << "Failed to load library: " << e.what() << std::endl;
        return;
    }

    std::cout << "Library loaded successfully" << std::endl;
}
} // namespace mevil