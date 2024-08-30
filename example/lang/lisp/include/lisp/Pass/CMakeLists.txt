function(add_lisp_pass name)
    # cmake_parse_arguments(ARG "" "SOURCE" "" ${ARGN})
    set(LLVM_TARGET_DEFINITIONS ${name}/.td)
    mlir_tablegen(${name}/.h.inc -gen-pass-decls)
    add_public_tablegen_target(MLIR${name}IncGen)
endfunction()

add_lisp_pass(ConvToLispAST)
