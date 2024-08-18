include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.4
)
FetchContent_MakeAvailable(pybind11)


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
        Support
        )
set(LIBS
        # ${dialect_libs}
        # ${conversion_libs}
        pybind11::embed
        )

add_llvm_executable(mevil
        main.cpp
        )
llvm_update_compile_flags(mevil)
target_link_libraries(mevil PRIVATE ${LIBS})
mlir_check_all_link_libraries(mevil)