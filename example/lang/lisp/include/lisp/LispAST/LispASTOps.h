#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "lisp/LispAST/LispASTType.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "lisp/LispAST/LispASTOps.h.inc"