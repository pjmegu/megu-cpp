#include "mokey/MokeyToken/MokeyTokenOps.h"
#include "mokey/MokeyToken/MokeyTokenDialect.h"

#include "mokey/MokeyToken/MokeyTokenOpsDialect.cpp.inc"

void mokey::token::MokeyTokenDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "mokey/MokeyToken/MokeyTokenOps.cpp.inc"
    >();
}