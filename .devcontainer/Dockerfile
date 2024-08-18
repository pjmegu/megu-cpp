FROM debian:trixie

ARG LLVM_VERSION=19

RUN apt-get update && \
    apt-get install -y curl

RUN curl https://apt.llvm.org/llvm-snapshot.gpg.key >> /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb [signed-by=/etc/apt/trusted.gpg.d/apt.llvm.org.asc] http://apt.llvm.org/unstable/ llvm-toolchain-${LLVM_VERSION} main" \
        >> /etc/apt/sources.list.d/llvm.list

RUN apt-get update && apt-get install -y \
    # mlir
    libmlir-${LLVM_VERSION}-dev mlir-${LLVM_VERSION}-tools \
    # tools
    clangd-${LLVM_VERSION} clang-tidy-${LLVM_VERSION} clang-format-${LLVM_VERSION} clang-tools-${LLVM_VERSION} \
    # compiler
    clang-${LLVM_VERSION} lldb-${LLVM_VERSION} lld-${LLVM_VERSION} \
    # libc++
    libc++-${LLVM_VERSION}-dev libc++abi-${LLVM_VERSION}-dev libllvmlibc-${LLVM_VERSION}-dev \
    # cmake 
    cmake ninja-build git \
    # python
    python3