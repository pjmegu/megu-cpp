#include "lisp/LispAST/LispASTDialect.h"
#include "lisp/LispAST/LispASTOps.h"
#include "lisp/LispAST/LispASTType.h"

using namespace mlir;
using namespace mokey::example::lisp::ast;

#include "lisp/LispAST/LispASTOpsDialect.cpp.inc"

void LispASTDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "lisp/LispAST/LispASTOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "lisp/LispAST/LispASTOpsTypes.cpp.inc"
        >();
}