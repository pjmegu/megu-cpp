#include "dylib.hpp"
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace mevil {
void load_dialect_test(std::string path) {
    auto lib_path = fs::absolute(fs::path(path));
    try {
        dylib lib(lib_path);
    } catch (dylib::load_error &e) {
        std::cout << "Failed to load library: " << e.what() << std::endl;
    }
}
} // namespace mevil