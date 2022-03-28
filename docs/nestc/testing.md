## Testing and Running

### Unit tests

The project has a few unit tests in the tests/unittests subdirectory. To run all
of them, simply run `ninja test`.

### Ahead-of-time Compilation

NEST-C can be used to compile neural networks into object files containing native
code.  There are bundle examples for Alexnet, DenseNet, EfficientNet, GoogleNet, Inception V1, MNasNet, MobileNet, Resnet18, Resnext50, Resnext101, ShuffleNet, Vgg19 and ZfNet with various options. 

To run these programs, build NEST-C in Release mode, then run the following commands
  ```bash
  mkdir build
  cd build
  cmake -G Ninja ../ -DCMAKE_BUILD_TYPE=Release -DNESTC_WITH_EVTA=ON -DNESTC_EVTA_BUNDLE_TEST=ON -DGLOW_WITH_BUNDLES=ON -DNESTC_USE_PRECOMPILED_EVTA_LIBRARY=ON 
  ninja check_nestc
  ```

Now run the examples. You can find built examples under the 'build/examples/' subdirectory. For example, you can run Resnet18 as following

  ```bash
  cd resnet18
  ./ResNet18Bundle ../../../../glow/tests/images/imagenet/cat_285.png 
  ```

See [Profile-based graph partitioning](docs/nestc/NestPartitioner-kor.md) on creating a bundle for a multiple backends.

