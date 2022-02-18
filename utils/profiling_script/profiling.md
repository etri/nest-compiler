## Profiling 결과 저장

### EVTA 시뮬레이터를 통한 Profiling 결과 저장
* ResConv1TestBundle을 예를 들어 설명한다.

0. vta/vtalib/lib/Bundle/VTABundle.cpp 의 아래 부분의 주석을 해제한다.
~~~
//#define PROFILE_AUTOTUNE
~~~

1. profiling을 하기 위해서는 cmake 옵션에 -DNESTC_EVTA_PROFILE_AUTOTUNE=ON 을 추가하여 빌드한다.

~~~
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNESTC_WITH_EVTA=ON -DNESTC_EVTA_BUNDLE_TEST=ON -DGLOW_WITH_BUNDLES=ON -DLLVM_DIR=/usr/lib/llvm-8/cmake -DNESTC_EVTA_PROFILE_AUTOTUNE=ON [소스코드 위치]
make vtaResConv1TestBundleProfiling
~~~

2. 해당 폴더에 있는 modify_csv.py와 run.sh를 build/vta/bundle/ResConv1Test 위치에 옮겨준다.

3. run.sh 를 실행한다.

4. ResConv1Test_data.csv에 profiling 결과가 저장된다.

~~~
chmod +x run.sh
bash run.sh ResConv1Test
~~~