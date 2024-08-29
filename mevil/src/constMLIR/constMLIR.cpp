#include "constMLIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mokey/MokeyToken/MokeyTokenDialect.h"
#include "mokey/MokeyToken/MokeyTokenOps.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <llvm/ADT/StringRef.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace mevil {

void constMLIR(std::unordered_map<std::uint64_t, Module> &modules,
               mlir::MLIRContext &context) {
    context.getOrLoadDialect<mokey::token::MokeyTokenDialect>();

    mlir::OpBuilder builder(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module_op =
        mlir::ModuleOp::create(builder.getUnknownLoc());

    builder.setInsertionPointToEnd(module_op->getBody());

    for (auto &[id, mod] : modules) {
        auto token_module =
            builder.create<mokey::token::Module>(builder.getUnknownLoc(), id);
        token_module->getRegion(0);
        builder.setInsertionPointToEnd(&token_module.getBody().emplaceBlock());
        for (auto &file : mod.sources) {
            std::ifstream fstr(file);
            std::string source((std::istreambuf_iterator<char>(fstr)),
                               std::istreambuf_iterator<char>());
            auto token_file = builder.create<mokey::token::File>(
                builder.getUnknownLoc(), llvm::StringRef(file),
                llvm::StringRef(source));
        }
    }

    if (mlir::failed(module_op->verify())) {
        std::cout << "Failed to verify module" << std::endl;
        return;
    }

    module_op->dump();
}

} // namespace mevil