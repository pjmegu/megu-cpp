add_llvm_library(MLIRMokeyTokenPlugin
    MODULE
    MokeyTokenPlugin.cpp

    DEPENDS
    MLIRMokeyToken
    PLUGIN_TOOL
    mlir-opt

    LINK_LIBS PUBLIC
    MLIRMokeyToken
    MLIRIR
)

target_include_directories(MLIRMokeyTokenPlugin PRIVATE ${PROJECT_BINARY_DIR}/include)