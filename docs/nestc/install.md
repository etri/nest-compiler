## Installation

### System Requirements

NEST-C builds and runs on Linux. The software depends on a modern C++
compiler that supports C++11, on CMake, LLVM (>=7.0), glog, protocol buffers, and
libpng.

#### Get NEST-C!

  ```bash
  git clone https://github.com/etri/nest-compiler.git
  cd nest-compiler
  ```

#### Submodules

NEST-C depends on a few submodules: googletest, onnx, and a library
for FP16 conversions.

To get them, from the NEST-C directory, run:

  ```bash
  git submodule update --init --recursive glow
  git submodule update --init --recursive tvm
  ```

#### Source dependencies

NEST-C depends on `fmt`, which must be built from source:
```bash
git clone https://github.com/fmtlib/fmt
mkdir fmt/build
cd fmt
git reset --hard efe3694f150a1f307d014e68cd88350067769b19
cd build
cmake ..
make
sudo make install
```
[The following instructions have been tested on 20.04]

In order to build NEST-C on Ubuntu it is necessary to install a few packages. The
following command should install the required dependencies:

  ```bash
  sudo apt-get install clang clang-8 cmake graphviz libpng-dev \
      libprotobuf-dev llvm-8 llvm-8-dev ninja-build protobuf-compiler wget \
      opencl-headers libgoogle-glog-dev libboost-all-dev \
      libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
      libjemalloc-dev libpthread-stubs0-dev python python3-pip
  pip3 install numpy decorator attrs pytest onnx scipy
  pip install --upgrade protobuf
  ```


It may be desirable to use `update-alternatives` to manage the version of
clang/clang++:

  ```bash
  sudo update-alternatives --install /usr/bin/clang clang \
      /usr/lib/llvm-8/bin/clang 50
  sudo update-alternatives --install /usr/bin/clang++ clang++ \
      /usr/lib/llvm-8/bin/clang++ 50
  ```

NEST-C uses the system default C/C++ compiler (/usr/bin/c++), and so you may also
want to switch your default C/C++ compiler to clang:

  ```bash
  sudo update-alternatives --config cc
      # Select the option corresponding to /usr/bin/clang ...
  sudo update-alternatives --config c++
      # Select the option corresponding to /usr/bin/clang++ ...
  ```

NEST-C *should* build just fine with gcc (e.g. gcc 5.4), but we mostly use clang
and are more attentive to compatibility with clang.

Finally, in order to support the ONNX net serialization format, NEST-C requires
`protobuf >= 2.6.1`, but the above command may install older
version on older Ubuntu (e.g. 14.04). If this is the case, we suggest to look
at `utils/install_protobuf.sh` to install a newer version from source.


### Configure and Build

To build the compiler, create a build directory and run cmake on the source
directory. It's a good idea to build two configurations (Release and Debug)
because some programs take a really long time to run in Debug mode. It's also a
good idea to build the project outside of the source directory.

  ```bash
  mkdir build
  cd build
  export TVM_HOME=/path/to/nest_compiler/tvm
  export PYTHONPATH=/path/to/nest_compiler/tvm/python
  export TVM_LIBRARY_PATH=/path/to/build/tvm
  cmake -G Ninja /path/to/nest_compiler -DCMAKE_BUILD_TYPE=Release -DNESTC_WITH_EVTA=ON -DNESTC_EVTA_BUNDLE_TEST=ON -DGLOW_WITH_BUNDLES=ON -DNESTC_USE_PRECOMPILED_EVTA_LIBRARY=ON 
  ninja check_nestc
  ```

It's possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.

#### Building with dependencies (LLVM)

By default, NEST-C will use a system provided LLVM.  Note that NEST-C requires LLVM
7.0 or later. If you have LLVM installed in a non-default location, you need to tell CMake
where to find llvm using `-DLLVM_DIR`. For example, if LLVM were
installed in `/usr/local/opt`:

  ```bash
  -DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm
  ```

If LLVM is not available on your system you'll need to build it manually.  Run
the script '`/utils/build_llvm.sh` to clone, build and install LLVM in a local
directory. You will need to configure NEST-C with the flag `-DLLVM_DIR` to tell
the build system where to find LLVM given the local directory you installed it
in (e.g. `-DLLVM_DIR=/path/to/llvm_install/lib/cmake/llvm` if using
`build_llvm.sh`).

