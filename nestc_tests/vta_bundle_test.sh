#!/bin/sh
echo 'PARAM:' $0
DIR="$( cd "$( dirname "$0" )" && pwd -P )"
echo $DIR
cmake $DIR/../../ -DLLVM_DIR=/usr/lib/llvm-8.0/lib/cmake/llvm -DGLOW_WITH_VTA=ON -DGLOW_WITH_VTASIM=OFF  -DCMAKE_BUILD_TYPE=Release -DGLOW_USE_PREBUILT_LIB=ON -DGLOW_WITH_VTA_BUNDLE_TEST=ON
cat $DIR/nestc_vta_bundle_tests.txt | xargs -I {} make {} -j4
cat $DIR/nestc_vta_bundle_tests.txt | xargs -I {} make {} -j4

