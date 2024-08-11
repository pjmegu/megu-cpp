#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "MokeyToken/MokeyTokenOps.h.inc"