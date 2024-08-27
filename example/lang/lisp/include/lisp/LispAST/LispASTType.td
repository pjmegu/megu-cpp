#ifndef EXAMPLE_LISP_AST_TYPE
#define EXAMPLE_LISP_AST_TYPE

include "mlir/IR/AttrTypeBase.td"
include "lisp/LispAST/LispASTDialect.td"

class LispAST_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LispAST_Dialect, name, traits> {
        let mnemonic = typeMnemonic;
    }

def LispAST_NumType : LispAST_Type<"Num", "num"> {}

#endif // EXAMPLE_LISP_AST_TYPE
