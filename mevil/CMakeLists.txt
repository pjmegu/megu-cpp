include(FetchContent)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed)

set(PYBIND11_FINDPYTHON ON)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.4
)

FetchContent_MakeAvailable(pybind11)

# for llvm/mlir
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
        Support
)

# target
add_llvm_executable(mevil
        main.cpp
)

target_link_libraries(mevil PRIVATE pybind11::embed)
llvm_update_compile_flags(mevil)
mlir_check_all_link_libraries(mevil)