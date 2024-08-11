#include "MokeyToken/MokeyTokenOps.h"
#include "MokeyToken/MokeyTokenDialect.h"

#include "MokeyToken/MokeyTokenOpsDialect.cpp.inc"

void mokey::token::MokeyTokenDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "MokeyToken/MokeyTokenOps.cpp.inc"
    >();
}