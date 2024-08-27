FROM debian:trixie

ARG LLVM_VERSION=18

RUN apt-get update && \
    apt-get install -y curl

RUN curl https://apt.llvm.org/llvm-snapshot.gpg.key >> /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb [signed-by=/etc/apt/trusted.gpg.d/apt.llvm.org.asc] http://apt.llvm.org/unstable/ llvm-toolchain-${LLVM_VERSION} main" \
    >> /etc/apt/sources.list.d/llvm.list

RUN apt-get update
# mlir
RUN apt-get install -y libmlir-${LLVM_VERSION}-dev mlir-${LLVM_VERSION}-tools
# tools
RUN apt-get install -y clangd-${LLVM_VERSION} clang-tidy-${LLVM_VERSION} clang-format-${LLVM_VERSION} clang-tools-${LLVM_VERSION}
# compiler
# RUN apt-get install -y clang-${LLVM_VERSION} lldb-${LLVM_VERSION} lld-${LLVM_VERSION}
# libc++
# RUN apt-get install -y libc++-${LLVM_VERSION}-dev libc++abi-${LLVM_VERSION}-dev libllvmlibc-${LLVM_VERSION}-dev
# compiler (gcc)
RUN apt-get install -y build-essential mold
# cmake 
RUN apt-get install -y cmake ninja-build git
# python
RUN apt-get install -y python3 python3-dev