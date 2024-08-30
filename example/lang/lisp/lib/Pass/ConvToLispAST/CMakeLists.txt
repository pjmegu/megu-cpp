add_mlir_dialect_library(MLIRConvToLispAST
  ConvToLispAST.cpp
  Parser.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/example/lang/lisp/include

  DEPENDS
  MLIRConvToLispASTIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLispAST
  MLIRMokeyToken
)