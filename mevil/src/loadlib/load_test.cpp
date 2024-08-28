#include "dylib.hpp"
#include <filesystem>
#include <iostream>
#include <string>
#include "./load_test.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

namespace fs = std::filesystem;

namespace mevil {
// TODO: Dialect読み込みテスト
void load_dialect_test(std::string path) {
    auto lib_path = fs::absolute(fs::path(path));

    auto plugin = mlir::DialectPlugin::load(lib_path.string());
    if (auto E = plugin.takeError()) {
        std::cout << "Failed to load dialect: " << llvm::toString(std::move(E)) << std::endl;
        return;
    }
    std::cout << "Dialect loaded successfully" << "\n";
    std::cout << "Dialect name: " << plugin->getPluginName().str() << "\n";
    std::cout << "Dialect version: " << plugin->getPluginVersion().str() << "\n";
    std::cout << "Dialect api version: " << plugin->getAPIVersion() << "\n";

    mlir::DialectRegistry registory;
    plugin->registerDialectRegistryCallbacks(registory);
    plugin->registerDialectRegistryCallbacks(registory);
    mlir::MLIRContext ctx(registory);
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