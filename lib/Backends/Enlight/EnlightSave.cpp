/*****************************************************************************
*
* Copyright Next-Generation System Software Research Group, All rights reserved.
* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
* Electronics and Telecommunications Research Institute (ETRI)
*
* THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
* WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
* TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE
* REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
* IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
* COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
*
* LICENSE file : LICENSE_ETRI located in the top directory
*
*****************************************************************************/

#include "Enlight.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include <cstring>
#include <fstream>
#include <vector>
#include <time.h>

using namespace glow;

void Enlight::save(Function *F, llvm::StringRef outputDir,
               llvm::StringRef bundleName,
               llvm::StringRef mainEntryName) const {
  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

  /////////////////저장할 yaml file path & 내용
    std::string yamlFileName = outputDir;
    yamlFileName.append("/");
    yamlFileName.append(bundleName);
    yamlFileName.append(".yaml");

    std::ofstream profileFile(yamlFileName);
    if (profileFile.is_open()) {
        profileFile << "---" << std::endl;
        profileFile << "backendName: " << getBackendName() << std::endl; //BackendName variable
        profileFile << "..." << std::endl;
        profileFile << "---" << std::endl;
  /////////////////저장할 yaml file path

  //node --> yaml file
  for (const auto &I : IR->getInstrs()) {

    switch (I.getKind()) {
    case Kinded::Kind::FullyConnectedInstKind: {
      auto I2 = llvm::cast<FullyConnectedInst>(&I);
      //llvm::outs()<<"FullyConnectedInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
      profileFile << "- NodeOutputName:  \'"<< I.getName().str()<<'\''<<std::endl;
      break;
    }
      case Kinded::Kind::AllocActivationInstKind: {
        auto I2 = llvm::cast<AllocActivationInst>(&I);
          profileFile << "- NodeOutputName:  \'" << I.getName().str()<<'\''<<std::endl;
        break;
      }
      case Kinded::Kind::DeallocActivationInstKind: {
          //llvm::outs()<<"DeallocActivationInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
          profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
        break;
      }
      case Kinded::Kind::ElementAddInstKind:{//elementadd
          //llvm::outs()<<"ElementAddInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
          profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
            break;
      }
      case Kinded::Kind::ElementSubInstKind:{//elementsub
          //llvm::outs()<<"ElementSubInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
          profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
            break;
      }
      case Kinded::Kind::ElementMulInstKind:{//elementmul
          //llvm::outs()<<"ElementMulInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
          profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
          break;
      }
      case Kinded::Kind::TransposeInstKind: {// transpose
          //llvm::outs()<<"TransposeInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
          profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
          break;
      }
      case Kinded::Kind::ConvolutionInstKind: {//convolution
          //llvm::outs()<<"ConvolutionInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
          profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
          break;
      }
      case Kinded::Kind::ReluInstKind: { //relu
          //llvm::outs()<<"ReluInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
          profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
          break;
      }
        case Kinded::Kind::MaxPoolInstKind:{//maxpool
            //llvm::outs()<<"MaxPoolInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
            profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
            break;
        }
        case Kinded::Kind::AvgPoolInstKind: {//avgpool
            //llvm::outs()<<"AvgPoolInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
            profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
            break;
        }
        case Kinded::Kind::TensorViewInstKind:{//tensorview
            //llvm::outs()<<"TensorViewInstKind >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
            profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;
            break;
        }

        default:
            //llvm::outs()<<"default >> "<<I.getKindName()<<" : "<<I.getName() <<"\n";
            profileFile << "- NodeOutputName:  \'" <<I.getName().str()<<'\''<<std::endl;;
      std::string msg = I.getKindName();
      msg.append(" is an unhandled instruction");
      break;
      llvm_unreachable(msg.c_str());
    }

  }
        profileFile << "...";
        profileFile.close();
  }



  return;
}


