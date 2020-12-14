![NEST-C Logo](docs/nestc/logo_nestc.png)


NEST Compiler (NEST-C) is an open source project led by ETRI, which is based on GLOW project.
NEST-C is a machine learning compiler and execution engine for hardware
accelerators. It is designed to be used as a backend for high-level machine
learning frameworks.  The compiler is designed to allow state of the art
compiler optimizations and code generation of neural network graphs. This
library is in active development. 

## Demo

![Nestc-demo](docs/nestc/nestc_demo.gif)

## Partners

Contributions to NEST-C are welcomed and encouraged! 


## How does it work?

NEST-C lowers a traditional neural network dataflow graph into a two-phase
strongly-typed [intermediate representation (IR)](./docs/IR.md). The high-level
IR allows the optimizer to perform domain-specific optimizations. The
lower-level instruction-based address-only IR allows the compiler to perform
memory-related optimizations, such as instruction scheduling, static memory
allocation and copy elimination. At the lowest level, the optimizer performs
machine-specific code generation to take advantage of specialized hardware
features. NEST-C features a lowering phase which enables the compiler to support a
high number of input operators as well as a large number of hardware targets by
eliminating the need to implement all operators on all targets. The lowering
phase is designed to reduce the input space and allow new hardware backends to
focus on a small number of linear algebra primitives.

![](./docs/nestc/nestc_overview.png)

## Getting Started

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


## EVTA
![EVTA](docs/nestc/EVTA_overview.png)
EVTA는 Ultra96-V2와 ZCU102 보드에서 다양한 특성을 가진 NPU를 타겟으로 코드 생성 실험을 할수 있도록 개발한 레퍼런스 하드웨어다. [nest-data](https://github.com/etri/nest-data/tree/master/bitstreams) 저장소에서 비트스트림을 다운로드하여 각 보드에서 실행이 가능하며, EVTA 시뮬레이터를 통한 검증도 가능하다.

### EVTA 시뮬레이터를 통한 Resnet18 빌드 및 실행 방법
```bash
cmake -DNESTC_WITH_EVTA=ON -DNESTC_WITH_EVTA_BUNDLE_TEST=ON -DNESTC_WITH_BUNDLES=ON [소스코드 위치]
make vtaMxnetResnet18Bundle
```

### ZCU102 보드 상에서의 Resnet18 빌드 및 실행 방법
호환 비트스트림 [다운로드](https://github.com/etri/nest-data/blob/master/bitstreams/zcu102_1x16_i8w8a32_16_16_19_18.bit) 및 설치
```bash
cmake -DNESTC_WITH_EVTA=ON -DNESTC_WITH_EVTASIM=OFF -DNESTC_USE_PREBUILT_LIB=ON -DNESTC_WITH_EVTA_BUNDLE_TEST=ON -DNESTC_WITH_BUNDLES=ON -DNESTC_EVTA_RUN_ON_AARCH64=ON [소스코드 위치]
make vtaMxnetResnet18Bundle
```

## Quantization

### image-selection

```bash
python image_selection.py --file-name=test.txt --num-of-images=1 --train-images=/home/jemin/hdd/imagenet/train_processed
```

### profling
```bash
# resnet 18
cd /home/jemin/development/nest_compiler/utils/profile
./profile.sh
```

### Parameter Search

```bash
cd /home/jemin/development/nest_compiler/utils/imagenet_val_script

# 1,000, Max, non-fusion
./resnet18.sh 3

# 10,000, KL, fusion
./resnet18.sh 4

# 10,000, KL, Fusion
./resnet18.sh 7
```

#### Show explored results

```bash
# All search results of resnet18 
ls | grep resnet18

# 3
ls | grep -E "nonfusion.*resnet18.*_1000_.*none" | xargs cat
# 4
ls | grep -E "nonfusion.*resnet18.*_10000_.*KL" | xargs cat
# 7
ls | grep -E "vtainterpreter-fusion.*resnet18.*_10000_.*KL" | xargs cat

# 전체 결과
cd /home/jemin/development/nest_compiler/utils/imagenet_val_script/quant_result
ls | grep resnet18 | xargs more
```

## Profile-based Graph Partitioning
See [Profile-based graph partitioning](docs/nestc/NestPartitioner-kor.md)


## Testing and Running

### Unit tests

The project has a few unit tests in the tests/unittests subdirectory. To run all
of them, simply run `ninja test`.

### C++ API examples

A few test programs that use NEST-C's C++ API are found under the `examples/`
subdirectory. The `mnist`, `cifar10`, `fr2en` and `ptb` programs train and run digit
recognition, image classification and language modeling benchmarks,
respectively.

To run these programs, build NEST-C in Release mode, then run the following commands
to download the cifar10, mnist and ptb databases.

  ```bash
  python ../nest-compiler/utils/download_datasets_and_models.py --all-datasets
  ```

Now run the examples. Note that the databases should be in the current working
directory.

  ```bash
  ./bin/mnist
  ./bin/cifar10
  ./bin/fr2en
  ./bin/ptb
  ./bin/char-rnn
  ```

If everything goes well you should see:
  * `mnist`: pictures from the mnist digits database
  * `cifar10`: image classifications that steadily improve
  * `fr2en`: an interactive French-to-English translator
  * `ptb`: decreasing perplexity on the dataset as the network trains
  * `char-rnn`: generates random text based on some document

Note that the default build mode is `Debug`, which means that the compiler
itself is easy to debug because the binary contains debug info, lots of
assertions, and the optimizations are disabled. It also means that the compiler
and runtime are very slow, and the execution time can be hundreds of times
slower than that of release builds. If you wish to benchmark the compiler, run
long benchmarks, or release the product then you should compile the compiler in
Release mode. Check the main CMake file for more details.


### Ahead-of-time Compilation

NEST-C can be used to compile neural networks into object files containing native
code.  We provide resnet18 and resnet50 (both quantized and non-quantized versions) as
examples of this capability in `examples/bundles/resnet18`, `examples/bundles/resnet50` and `examples/bundles/vta_cpu_partition_net`. 

See [Creating Standalone Executable Bundles](docs/AOT.md) for more detail on creating a bundle for a single backend..

See [Profile-based graph partitioning](docs/nestc/NestPartitioner-kor.md) on creating a bundle for a multiple backends.



## License

NEST-C is licensed under the [Apache 2.0 License](LICENSE).


