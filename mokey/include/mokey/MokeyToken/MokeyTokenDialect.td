#ifndef MOKEY_TOKEN_DIALECT
#define MOKEY_TOKEN_DIALECT

include "mlir/IR/OpBase.td"

def MokeyToken_Dialect : Dialect {
    let name = "mokeytoken";
    let cppNamespace = "::mokey::token";
}

class MokeyToken_Op<string mnemonic, list<Trait> traits = []> :
    Op<MokeyToken_Dialect, mnemonic, traits>;

#endif // MOKEY_TOKEN_DIALECT
