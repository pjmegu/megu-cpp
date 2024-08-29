#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "../buildEnv.h"
#include "mlir/IR/MLIRContext.h"

namespace mevil {

void constMLIR(std::unordered_map<std::uint64_t, Module> &modules, mlir::MLIRContext &context);
    
}