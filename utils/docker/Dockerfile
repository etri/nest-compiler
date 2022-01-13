FROM ubuntu:20.04

ARG WORKDIR=/root/dev

# Create working folder
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# Update and install tools
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y clang clang-8 cmake graphviz libpng-dev \
        libprotobuf-dev llvm-8 llvm-8-dev ninja-build protobuf-compiler wget \
        opencl-headers libgoogle-glog-dev libboost-all-dev \
        libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
        libjemalloc-dev libpthread-stubs0-dev \
        ocl-icd-opencl-dev gcc-aarch64-linux-gnu g++-aarch64-linux-gnu curl unzip\
        # Additional dependencies
        git python python-numpy python3-pip && \
    # Delete outdated llvm to avoid conflicts
    apt-get autoremove -y llvm-6.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install

# Point clang to llvm-8 version
RUN update-alternatives --install /usr/bin/clang clang \
        /usr/lib/llvm-8/bin/clang 50 && \
    update-alternatives --install /usr/bin/clang++ clang++ \
        /usr/lib/llvm-8/bin/clang++ 50

RUN pip3 install numpy decorator attrs pytest onnx scipy

# Point default C/C++ compiler to clang
RUN update-alternatives --set cc /usr/bin/clang && \
    update-alternatives --set c++ /usr/bin/clang++

# Install fmt
RUN git clone https://github.com/fmtlib/fmt && \
    cd fmt && \
    git reset --hard efe3694f150a1f307d014e68cd88350067769b19 && \
    mkdir build && \
    cd build && \
    cmake .. && make -j32 && \
    make install

# Clean up
RUN rm -rf fmt
