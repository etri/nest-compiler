## Installation

### System Requirements

NEST-C builds and runs on macOS and Linux. The software depends on a modern C++
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
  git submodule update --init --recursive
  ```

#### Source dependencies

NEST-C depends on `fmt`, which must be built from source:
```bash
git clone https://github.com/fmtlib/fmt
mkdir fmt/build
cd fmt/build
cmake ..
make
sudo make install
```

#### macOS

Install the required dependencies using either [Homebrew](https://brew.sh/) or
[MacPorts](https://www.macports.org/). If using Homebrew, run:

  ```bash
  brew install cmake graphviz libpng zlib ninja protobuf wget glog autopep8 llvm   \
      boost double-conversion gflags jemalloc libevent lz4 openssl pkg-config \
      snappy xz
  ```

If using MacPorts, run:

  ```bash
  port install cmake graphviz libpng zlib ninja protobuf-cpp wget google-glog \
      boost double-conversion gflags jemalloc libevent lz4 openssl snappy xz
  # Choose version >= 7
  export LLVM_VERSION=7
  port install llvm-$LLVM_VERSION.0 
  ```


Note that LLVM is installed in a non-default location to avoid conflicts with
the system's LLVM --Homebrew usually installs LLVM in `/usr/local/opt/llvm/`,
whereas MacPorts installs it in `/opt/local/libexec/llvm-$LLVM_VERSION.0/`. This means that
CMake will need to be told where to find LLVM when building; instructions on
that can be found [here](#building-with-dependencies-llvm).

Finally, create a symbolic link to the Homebrew- or MacPorts-installed
`clang-*` tools so that the `utils/format.sh` script is able to find them later
on. For a Homebrew-managed installation, run:
  ```
  ln -s "/usr/local/opt/llvm/bin/clang-format" "/usr/local/bin/clang-format"
  ln -s "/usr/local/opt/llvm/bin/clang-tidy" "/usr/local/bin/clang-tidy"
  ```
For MacPorts, run:
  ```
  ln -s "/opt/local/libexec/llvm-$LLVM_VERSION.0/bin/clang-format" "/usr/local/bin/clang-format"
  ln -s "/opt/local/libexec/llvm-$LLVM_VERSION.0/bin/clang-tidy" "/usr/local/bin/clang-tidy"
```

> **Note:** Starting with macOS Mojave, Xcode's command line tools changed header layout. 
> In order for NEST-C to build on Mojave, you might need to install
> `macOS_SDK_headers_for_macOS_10.14.pkg`, located in 
> `/Library/Developer/CommandLineTools/Packages/`.
> For macOS Catalina you might need to explicitly specify SDKROOT: 
> `export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"`


#### Ubuntu

[The following instructions have been tested on Ubuntu 16.04 and 18.04]

In order to build NEST-C on Ubuntu it is necessary to install a few packages. The
following command should install the required dependencies:

  ```bash
  sudo apt-get install clang clang-8 cmake graphviz libpng-dev \
      libprotobuf-dev llvm-8 llvm-8-dev ninja-build protobuf-compiler wget \
      opencl-headers libgoogle-glog-dev libboost-all-dev \
      libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
      libjemalloc-dev libpthread-stubs0-dev
  ```

[Note: Ubuntu 16.04 and 18.04 ship with llvm-6 and need to be upgraded before building NEST-C. Building NEST-C on Ubuntu 16.04 with llvm-7 fails because llvm-7 xenial distribution uses an older c++ ABI, however building NEST-C on Ubuntu 18.04 with llvm-7 has been tested and is successful]

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
  mkdir build_Debug
  cd build_Debug
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ../nest-compiler
  ninja all
  ```

It's possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.


If you're running macOS v10.14 (Mojave) and `ninja all` fails because it can't
find headers (e.g. `string.h`), run this command to fix it, and try again.
More information is available [here](https://developer.apple.com/documentation/xcode_release_notes/xcode_10_release_notes)
under "Command Line Tools".

  ```bash
  open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
  ```

For macOS v10.15 (Catalina) you might need to explicitly specify SDKROOT:

   ```bash
   export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
   ```


#### Building with dependencies (LLVM)

By default, NEST-C will use a system provided LLVM.  Note that NEST-C requires LLVM
7.0 or later. If you have LLVM installed in a non-default location (for
example, if you installed it using Homebrew on macOS), you need to tell CMake
where to find llvm using `-DLLVM_DIR`. For example, if LLVM were
installed in `/usr/local/opt`:

  ```bash
  cmake -G Ninja ../nest-compiler \
      -DCMAKE_BUILD_TYPE=Debug \
      -DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm
  ```

If LLVM is not available on your system you'll need to build it manually.  Run
the script '`/utils/build_llvm.sh` to clone, build and install LLVM in a local
directory. You will need to configure NEST-C with the flag `-DLLVM_DIR` to tell
the build system where to find LLVM given the local directory you installed it
in (e.g. `-DLLVM_DIR=/path/to/llvm_install/lib/cmake/llvm` if using
`build_llvm.sh`).

