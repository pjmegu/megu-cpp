#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mokey/MokeyToken/MokeyTokenDialect.h"
#include "llvm/Support/Compiler.h"
#include <llvm/Config/llvm-config.h>

using namespace mlir;

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "mokey.token",
        LLVM_VERSION_STRING,
        [](DialectRegistry *registry) {
            registry->insert<mokey::token::MokeyTokenDialect>();
        }
    };
}