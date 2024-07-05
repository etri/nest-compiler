<!---![NEST-C Logo](docs/nestc/logo_nestc.png)-->
<img src="./docs/nestc/logo_nestc.png" width="300"> 

## Contacts
### PI
* yongin.kwon@etri.re.kr
### Developers
* msyu@etri.re.kr
* yongin.kwon@etri.re.kr
* leejaymin@etri.re.kr
* jeman@etri.re.kr
    
## Overview  

[![Video Label](https://img.youtube.com/vi/lxHYEtAfhrs/0.jpg)](https://youtu.be/lxHYEtAfhrs?t=0s)

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
* NPU HW backend supporting EVTA series
* Automatic backend parameter tuning (under development)
* Profile-based dyanamic quantization for supporting various NPU backends
* C/C++ code generation for embedded HWs supporting gcc/llvm compilers
* Profile-based graph partitioning


## Documentation

* [Installation](docs/nestc/install.md)
* [Docker Setting](docs/nestc/DockerSetting.md)
* [Testing and Running](docs/nestc/testing.md)
* [EVTA](docs/nestc/evta.md)
* [Quantization](docs/nestc/quantization.md)
* [Profile-based Graph Partitioning](docs/nestc/NestPartitioner-kor.md)
* [C/C++ code generator](docs/nestc/KCC2020-ccodegen-publish.pdf)


## Demo

![Nestc-demo](docs/nestc/nestc_demo.gif)



## Partners

We are proud to collaborate with the following companies on our open-source project:

<a href="https://www.skhynix.com/"><img src="https://mis-prod-koce-skhynixhomepage-cdn-01-ep.azureedge.net/img/content/img_ci01.png" height="80"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://www.openedges.com/"><img src="https://static.wixstatic.com/media/fcd3ed_8d00a2b4d2a544cbbbe4c260c105a0a6~mv2.png/v1/fill/w_322,h_80,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/openEdges_logo_basic2.png" height="50"></a>

<a href="https://neowine.com/"><img src="https://neowine.com/theme/a03/img/logo.gif" height="60"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://www.nextchip.com/"><img src="https://www.nextchip.com/images/common/footer_logo.png" height="60"></a>

<a href="https://aimfuture.ai/"><img src="https://aimfuture.ai/wp-content/uploads/aimlogo_dark.svg" height="30"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://surromind.ai/"><img src="https://surromind.ai/images/common/logoGrayFooter.png" height="30"></a>

We appreciate their support and contributions to our project. Click on the logos to learn more about each partner and their involvement with our project.

Contributions to NEST-C are welcomed and encouraged!

## License

NEST-C is licensed under the [Apache 2.0 License](LICENSE).
