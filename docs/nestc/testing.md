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

