add_mlir_dialect_library(MLIRLispAST
    LispASTDialect.cpp
    LispASTOps.cpp
    LispASTType.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/example/lang/lisp/include

    DEPENDS
    MLIRLispASTOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)