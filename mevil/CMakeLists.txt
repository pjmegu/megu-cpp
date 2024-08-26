include(FetchContent)

# for cli
FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.4.2
)

FetchContent_MakeAvailable(CLI11)

# python
find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed)
# FetchContent_Populate(
#         cpython
#         GIT_REPOSITORY https://github.com/python/cpython.git
#         GIT_TAG v3.13.0rc1
# )
# どうにかしてpythonのインストール結果を参照する()

set(PYBIND11_FINDPYTHON ON)

# python binding
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.4
)

FetchContent_MakeAvailable(pybind11)

# for glob
FetchContent_Populate(
        glob
        GIT_REPOSITORY https://github.com/p-ranav/glob.git
)

# for dylib
FetchContent_Declare(
    dylib
    GIT_REPOSITORY "https://github.com/martin-olivier/dylib"
    GIT_TAG        "v2.2.1"
)

FetchContent_MakeAvailable(dylib)

# for llvm/mlir
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# target
add_llvm_executable(mevil
        src/main.cpp
        src/runPython.cpp
        src/util/util.cpp
        src/loadlib/dialect.cpp
)

target_link_libraries(mevil PRIVATE pybind11::embed CLI11::CLI11 dylib)
target_include_directories(mevil PRIVATE ${glob_SOURCE_DIR}/single_include)
llvm_update_compile_flags(mevil)
target_compile_options(mevil PUBLIC -fexceptions)
mlir_check_all_link_libraries(mevil)