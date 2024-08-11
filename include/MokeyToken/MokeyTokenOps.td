#ifndef MOKEY_TOKEN_OPS
#define MOKEY_TOKEN_OPS

include "MokeyToken/MokeyTokenDialect.td"

def MokeyToken_TokenStr: MokeyToken_Op<"tokenstr"> {
    let arguments = (ins StrAttr:$token);
}

def MokeyToken_NewLine: MokeyToken_Op<"newline"> {
    let arguments = (ins I32Attr:$num);
}

def MokeyToken_Space: MokeyToken_Op<"space"> {
    let arguments = (ins I32Attr:$num);
}

#endif // MOKEY_TOKEN_OPS
