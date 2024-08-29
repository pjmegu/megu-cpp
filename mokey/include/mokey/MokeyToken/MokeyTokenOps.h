#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/RegionKindInterface.h"

#define GET_OP_CLASSES
#include "mokey/MokeyToken/MokeyTokenOps.h.inc"