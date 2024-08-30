#pragma once

#include "mlir/Pass/Pass.h"

namespace mokey::example::lisp {
#define GEN_PASS_DECL
#include "lisp/Pass/ConvToLispAST/.h.inc"

#define GEN_PASS_REGISTRATION
#include "lisp/Pass/ConvToLispAST/.h.inc"
}