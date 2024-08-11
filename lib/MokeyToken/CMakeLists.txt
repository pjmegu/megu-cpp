add_mlir_dialect_library(
    MLIRMokeyToken
        MokeyTokenOps.cpp
    
    ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/MokeyToken
    
    DEPENDS
        MLIRMokeyTokenOpsIncGen
    
    LINK_LIBS PUBLIC
        MLIRIR
)