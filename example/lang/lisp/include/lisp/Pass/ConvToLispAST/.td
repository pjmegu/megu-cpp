#ifndef EXAMPLE_LISP_PASS_CONVTOLISPAST
#define EXAMPLE_LISP_PASS_CONVTOLISPAST

include "mlir/Pass/PassBase.td"

def ConvToLispAST: Pass<"conv-to-lisp-ast", "::mokey::token::Module"> {}

#endif // EXAMPLE_LISP_PASS_CONVTOLISPAST
