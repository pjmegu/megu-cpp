#ifndef MOKEY_TOKEN_OPS
#define MOKEY_TOKEN_OPS

include "mokey/MokeyToken/MokeyTokenDialect.td"
include "mlir/IR/RegionKindInterface.td"

def MokeyToken_Module: MokeyToken_Op<"module", [] # GraphRegionNoTerminator.traits> {
    let arguments = (ins I64Attr:$id);
    let regions = (region SizedRegion<1>:$body);
}

def MokeyToken_File: MokeyToken_Op<"file"> {
    let arguments = (ins StrAttr:$file_path, StrAttr:$file_source);
}

#endif // MOKEY_TOKEN_OPS
