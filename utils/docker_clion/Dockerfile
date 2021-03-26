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
        # Additional dependencies
        git python python-numpy && \
    # Delete outdated llvm to avoid conflicts
    apt-get autoremove -y llvm-6.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Point clang to llvm-8 version
RUN update-alternatives --install /usr/bin/clang clang \
        /usr/lib/llvm-8/bin/clang 50 && \
    update-alternatives --install /usr/bin/clang++ clang++ \
        /usr/lib/llvm-8/bin/clang++ 50

# Point default C/C++ compiler to clang
RUN update-alternatives --set cc /usr/bin/clang && \
    update-alternatives --set c++ /usr/bin/clang++

# Install fmt
RUN git clone https://github.com/fmtlib/fmt && \
    mkdir fmt/build && \
    cd fmt/build && \
    cmake .. && make -j32 && \
    make install

# Clean up
RUN rm -rf fmt

# install packages for CLion
RUN apt-get update \
  && apt-get install -y ssh \
      build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      cmake \
      rsync \
      tar \
      python \
      ssh \
  && apt-get clean

# sshd
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]

# change password
RUN echo 'root:root' | chpasswd