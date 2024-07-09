<!---![NEST-C Logo](docs/nestc/logo_nestc.png)-->
<img src="./docs/nestc/logo_nestc.png" width="300">

## Partners

Contributions to NEST-C are welcomed and encouraged! We are proud to collaborate with the following companies on our open-source project:
| [<img src="https://mis-prod-koce-skhynixhomepage-cdn-01-ep.azureedge.net/img/content/img_ci01.png" height="80">](https://www.skhynix.com/) | [<img src="https://static.wixstatic.com/media/fcd3ed_8d00a2b4d2a544cbbbe4c260c105a0a6~mv2.png/v1/fill/w_322,h_80,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/openEdges_logo_basic2.png" height="60">](https://www.openedges.com/) | [<img src="https://neowine.com/theme/a03/img/logo.gif" height="60">](https://neowine.com/) |
|---|---|---|
| [<img src="https://www.nextchip.com/images/common/footer_logo.png" height="60">](https://www.nextchip.com/) | [<img src="https://aimfuture.ai/wp-content/uploads/aimlogo_dark.svg" height="30">](https://aimfuture.ai/) | [<img src="https://surromind.ai/images/common/logoGrayFooter.png" height="30">](https://surromind.ai/) |

We appreciate their support and contributions to our project. Click on the logos to learn more about each partner and their involvement with our project.

## Contacts

### PI

- yongin.kwon@etri.re.kr

### Developers

- msyu@etri.re.kr
- yongin.kwon@etri.re.kr
- leejaymin@etri.re.kr
- jeman@etri.re.kr

## Overview

[![Video Label](https://img.youtube.com/vi/lxHYEtAfhrs/0.jpg)](https://youtu.be/lxHYEtAfhrs?t=0s)
[![ETRI Conference 2024, AI 컴파일러로 실용화에 한발짝 더 다가선 AI 반도체](https://img.youtube.com/vi/hQmy7vJVNEE/0.jpg)](https://www.youtube.com/watch?v=hQmy7vJVNEE&t=22724)

The NEST Compiler (NEST-C) is an open source project led by ETRI, which is based on [GLOW project] (https://github.com/pytorch/glow).

Glow is a machine learning compiler and execution engine for hardware
accelerators. It is designed to be used as a backend for high-level machine
learning frameworks. The compiler is designed to allow state of the art
compiler optimizations and code generation of neural network graphs.

The objective of NEST-C is to generate optimized code for various kinds of Neural-network Processing Uints (NPUs). Therefore, NEST-C provides automatic tuning functionalities and tools for each optimization step.

<center><b>Optimization modules and tools of NEST-Compiler</b>

<img src="docs/nestc/nestc_overview-1.png" width="800"></center>

<center><b> Input/output examples of NEST-C modules </b>

<img src="docs/nestc/nestc_overview-2.png" width="800"></center>

## Features

- NPU HW backend supporting EVTA series
- Automatic backend parameter tuning (under development)
- Profile-based dyanamic quantization for supporting various NPU backends
- C/C++ code generation for embedded HWs supporting gcc/llvm compilers
- Profile-based graph partitioning

## Documentation

- [Installation](docs/nestc/install.md)
- [Docker Setting](docs/nestc/DockerSetting.md)
- [Testing and Running](docs/nestc/testing.md)
- [EVTA](docs/nestc/evta.md)
- [Quantization](docs/nestc/quantization.md)
- [Profile-based Graph Partitioning](docs/nestc/NestPartitioner-kor.md)
- [C/C++ code generator](docs/nestc/KCC2020-ccodegen-publish.pdf)

## Research Papers

- [PartitionTuner: An operator scheduler for deep-learning compilers supporting multiple heterogeneous processing units](https://onlinelibrary.wiley.com/doi/full/10.4218/etrij.2021-0446)
  - ETRI Journal Vol. 45 No. 2 (JCR21 IF: 1.622) ISSN: 1225-6463, doi: https://doi.org/10.4218/etrij.2021-0446, Apr 2023.
- [Quantune: Post-training quantization of convolutional neural networks using extreme gradient boosting for fast deployment](https://www.sciencedirect.com/science/article/pii/S0167739X22000498?via%3Dihub)
  - Future Generation Computer Systems Vol. 132, 2022, pp. 124-135, July 01, 2022 (JCR21 IF: 7.307, Top 9.09%), Jul 2022
- [범용 AI 컴파일러의 비공개 NPU 코드생성을 위한 공통 인터페이스 설계 및 검증](https://paper.cricit.kr/user/listview/ieie2018/doc_rdoc.asp?catvalue=3&returnVal=RD_R&organCode=ieie&organCode2=ieie01&yearmonth=202310&page=1&dn=424671&step=&usernum=0&seid=)
  - 전자공학회논문지, v.60 no.10, pp.29-31, 2023.10.25, 우수등재지

## Demo

![Nestc-demo](docs/nestc/nestc_demo.gif)

## License

NEST-C is licensed under the [Apache 2.0 License](LICENSE).
