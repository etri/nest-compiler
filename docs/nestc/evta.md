
## EVTA
![EVTA](docs/nestc/EVTA_overview.png)
EVTA는 Ultra96-V2와 ZCU102 보드에서 다양한 특성을 가진 NPU를 타겟으로 코드 생성 실험을 할수 있도록 개발한 레퍼런스 하드웨어다. [nest-data](https://github.com/etri/nest-data/tree/master/bitstreams) 저장소에서 비트스트림을 다운로드하여 각 보드에서 실행이 가능하며, EVTA 시뮬레이터를 통한 검증도 가능하다.

### EVTA 시뮬레이터를 통한 Resnet18 빌드 및 실행 방법
```bash
cmake -DGLOW_WITH_VTA=ON -DGLOW_WITH_VTA_BUNDLE_TEST=ON -DGLOW_WITH_BUNDLES=ON [소스코드 위치]
make vtaMxnetResnet18Bundle
```

### ZCU102 보드 상에서의 Resnet18 빌드 및 실행 방법
호환 비트스트림 [다운로드](https://github.com/etri/nest-data/blob/master/bitstreams/zcu102_1x16_i8w8a32_16_16_19_18.bit) 및 설치
```bash
cmake -DGLOW_WITH_VTA=ON -DGLOW_WITH_VTASIM=OFF -DGLOW_USE_PREBUILT_LIB=ON -DGLOW_WITH_VTA_BUNDLE_TEST=ON -DGLOW_WITH_BUNDLES=ON -DNESTC_VTA_RUN_ON_AARCH64=ON [소스코드 위치]
make vtaMxnetResnet18Bundle
```
