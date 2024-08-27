add_llvm_library(MLIRLispASTPlugin
    MODULE
    LispASTPlugin.cpp

    DEPENDS
    MLIRLispAST
    PLUGIN_TOOL
    mlir-opt

    LINK_LIBS PUBLIC
    MLIRLispAST
    MLIRIR
)