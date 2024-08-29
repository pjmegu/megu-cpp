#ifndef EXAMPLE_LISP_AST_DIALECT
#define EXAMPLE_LISP_AST_DIALECT

include "mlir/IR/OpBase.td"

def LispAST_Dialect : Dialect {
    let name = "lisp_ast";
    let cppNamespace = "::mokey::example::lisp::ast";
}

class LispAST_Op<string mnemonic, list<Trait> traits = []> :
    Op<LispAST_Dialect, mnemonic, traits>;

#endif // EXAMPLE_LISP_AST_DIALECT
