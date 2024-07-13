FROM debian:trixie

RUN apt-get update && \
    apt-get install -y curl

RUN curl https://apt.llvm.org/llvm-snapshot.gpg.key >> /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb [signed-by=/etc/apt/trusted.gpg.d/apt.llvm.org.asc] http://apt.llvm.org/unstable/ llvm-toolchain-18 main" \
        >> /etc/apt/sources.list.d/llvm.list

RUN apt-get update && apt-get install -y \
    # mlir
    libmlir-18-dev mlir-18-tools \
    # tools
    clangd-18 clang-tidy-18 clang-format-18 clang-tools-18 \
    # compiler
    clang-18 lldb-18 lld-18 \
    # libc++
    libc++-18-dev libc++abi-18-dev libllvmlibc-18-dev \
    # cmake 
    cmake ninja-build git