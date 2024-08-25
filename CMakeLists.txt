cmake_minimum_required(VERSION 3.20)
project(Megu)

# mlir setup
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# 各種Includeフォルダの追加
include_directories(${PROJECT_SOURCE_DIR}/mokey/include)
include_directories(${PROJECT_BINARY_DIR}/mokey/include)

# llvm::clにてlibc++を使うとSEGVをキメます なんだそれ
# libstdc++はだいじょぶらしい
add_compile_options(-stdlib=libc++)
add_link_options(-stdlib=libc++)

add_subdirectory(mokey)
add_subdirectory(mevil)