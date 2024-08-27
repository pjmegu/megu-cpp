#ifndef EXAMPLE_LISP_AST_OPS
#define EXAMPLE_LISP_AST_OPS

include "lisp/LispAST/LispASTDialect.td"
include "lisp/LispAST/LispASTType.td"
include "mlir/IR/RegionKindInterface.td"

def LispAST_Package: LispAST_Op<"package", [] # GraphRegionNoTerminator.traits> {
    let arguments = (ins StrAttr:$name);
    let regions = (region SizedRegion<1>:$body);
}

def LispAST_Def: LispAST_Op<"def.func"> {
    let arguments = (ins StrAttr:$name);
    let regions = (region SizedRegion<1>:$body);
}

def LispAST_Let: LispAST_Op<"let"> {
    let arguments = (ins StrAttr:$name, LispAST_NumType:$value);
}

def LispAST_Math_Plus: LispAST_Op<"math.plus"> {
    let arguments = (ins LispAST_NumType:$lhs, LispAST_NumType:$rhs);
    let results = (outs LispAST_NumType:$result);
}

def LispAST_Num_Init: LispAST_Op<"num.init"> {
    let arguments = (ins I32Attr:$value);
    let results = (outs LispAST_NumType:$result);
}

def LispAST_Get: LispAST_Op<"get"> {
    let arguments = (ins StrAttr:$name);
    let results = (outs LispAST_NumType:$result);
}

def LispAST_Return: LispAST_Op<"return", [Terminator]> {
    let arguments = (ins Optional<LispAST_NumType>:$value);
}

#endif // EXAMPLE_LISP_AST_OPS
