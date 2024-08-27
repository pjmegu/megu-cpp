#include "lisp/LispAST/LispASTType.h"

#include "lisp/LispAST/LispASTDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mokey::example::lisp::ast;

#define GET_TYPEDEF_CLASSES
#include "lisp/LispAST/LispASTOpsTypes.cpp.inc"

// void LispASTDialect::registerTypes() {
//     addTypes<
//     #define GET_TYPEDEF_LIST
//     #include "lisp/LispAST/LispASTOpsTypes.cpp.inc"
//     >();
// }