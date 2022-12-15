/*****************************************************************************
        *
        * Copyright Next-Generation System Software Research Group, All rights
reserved.
        * Future Computing Research Division, Artificial Intelligence Reserch
Laboratory
        * Electronics and Telecommunications Research Institute (ETRI)
        *
        * THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
        * WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
        * TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE"
        * REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
        "* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
        * IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
        * COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
        *
        * LICENSE file : LICENSE_ETRI located in the top directory
        *
*****************************************************************************/

#include "glow/Partitioner/NestPartitionerSchedule.h"
#include "llvm/Support/FileSystem.h"
#include <fstream>

using namespace glow;
using namespace std;

void NestPartitionerSchedule::setKeyName(std::string key, int i) {
  keyList_[i] = key;
}

void NestPartitionerSchedule::setProfileMode(int mode) { profileMode_ = mode; }

void NestPartitionerSchedule::setPartitionExeMode(int mode) {
  partitionExeMode_ = mode;
}

void NestPartitionerSchedule::setInputPartitionName(std::string name) {
  inputPartitionName_ = name;
}

std::string NestPartitionerSchedule::getInputPartitionName() {
  return inputPartitionName_;
}

int NestPartitionerSchedule::getProfileMode() { return profileMode_; }

// void NestPartitionerSchedule::generateDeclaration(string& wfilec, int
// partitionNum, const PartitionConfig &partitionConfig, int profileMode) {
void NestPartitionerSchedule::generateDeclaration(
    string &wfilec, const PartitionConfig &partitionConfig) {
  //  cout << "generateDeclaration: " <<  wfilec << endl;

  if (profileMode_ == 1) {
    wfilec.append("#define REPEAT 1\n\n");
  }

  bool vtaFlag = false, relayFlag = false;
  for (int i = 0; i < partitionConfig.partitionNames.size() - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;

    if (!partitionConfig.backendNames[i].compare("Relay"))
      relayFlag = true;
  }

  wfilec.append("#include \"main_template.h\"\n");

  if (vtaFlag) {
    wfilec.append("#include \"VTARuntime.h\"\n");
  }

  for (int i = 0; i < partitionConfig.partitionNames.size() - 1; i++) {
    wfilec.append("#include \"" + partitionConfig.partitionNames[i] + ".h\"\n");
    //    cout << "#include \"p" + std::to_string(i) + ".h\"" << endl;
  }
  wfilec.append("\n");

  wfilec.append("#include <sys/time.h>\n\n");

  if (relayFlag) {
    wfilec.append("#include <dlpack/dlpack.h>\n"
                  "#include <tvm/runtime/module.h>\n"
                  "#include <tvm/runtime/packed_func.h>\n"
                  "#include <tvm/runtime/registry.h>\n\n");
  }

  if (profileMode_ == 1) {
    wfilec.append("#define ARRAY_LENGTH " + std::to_string(partitionConfig.partitionNames.size()) +
                  "\n");
    wfilec.append("unsigned long p_time_ary[ARRAY_LENGTH], "
                  "comm_time_ary[ARRAY_LENGTH];\n\n");
  }
}

std::string NestPartitionerSchedule::addProfileKey(
    Function *function, std::set<std::string> *keySet, bool isFusion) {
  BFSLevel bfs = getBFSLevel(function);
  size_t level = bfs.size();
  std::string key = "";
  int cnt = 0;
  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node *node = bfs[i][j];
      if (node->getKind() == Kinded::Kind::SaveNodeKind ||
          node->getKind() == Kinded::Kind::PlaceholderKind) {
        continue;
      }
      key = key + getProfileKey(node) + "+";
      cnt++;
    }
  }

  if (isFusion && cnt == 1)
    return "";

  key.pop_back();

  if (keySet->find(key) == keySet->end()) {
    keySet->insert(key);
    return key;
  } else
    return "";
}

void NestPartitionerSchedule::generateYamlFile(string &wfilec, const PartitionConfig &partitionConfig, std::string inputPartitionName) {
  //  std::cout << "== [Start] generateYamlFile == " << std::endl;

   std::string backendName = partitionConfig.backendNames[0];
  std::string pTimeFilename, commTimeFilename;
  std::string keys;

  pTimeFilename = "./" + inputPartitionName + "Profile.yaml";
  commTimeFilename = "./" + inputPartitionName + "CommProfile.yaml";

  std::cout << "partition # : " << partitionConfig.partitionNames.size() << std::endl;
  std::cout << "inputPartitionName: " << inputPartitionName << std::endl;

  bool isFused = false;
  if (inputPartitionName.find("FusedNode") != std::string::npos) {
    isFused = true;
  }

  for (int idx = 0; idx < partitionConfig.partitionNames.size() - 1; idx++) {
    auto func = (*funcMap_)[partitionConfig.partitionNames[idx]];

    std::string key = addProfileKey(func, profileKeySet_, isFused);

    if (key.length() == 0) {
      keys = keys + "\"blank\", ";
    } else {
      keys = keys + "\"" + key + "\", ";
    }
  }

  keys.pop_back();
  keys.pop_back();

  wfilec.append("void createYamlFile(){\n");
  wfilec.append("\tprintf(\"== create Yaml files ==\\n\");\n");
  wfilec.append("\tstd::string keyList[" + std::to_string(partitionConfig.partitionNames.size()) +
                "] = {" + keys + "};\n");

  wfilec.append("\tstd::string pTimeFName = \"" + pTimeFilename + "\";\n");
  wfilec.append("\tstd::ofstream pTimeProfile(pTimeFName);\n\n");

  wfilec.append("\tstd::string commTimeFName = \"" + commTimeFilename +
                "\";\n");
  wfilec.append("\tstd::ofstream commTimeProfile(commTimeFName);\n\n");

  wfilec.append("\tchar* p_str = new char[10];\n");
  wfilec.append("\tchar* comm_str = new char[10];\n\n");

  wfilec.append(
      "\tif(pTimeProfile.is_open() && commTimeProfile.is_open()) {\n");
  wfilec.append("\t\tpTimeProfile << \"---\" << std::endl;\n");
  wfilec.append("\t\tcommTimeProfile << \"---\" << std::endl;\n");

  wfilec.append("\t\tpTimeProfile << \"Type: ExeTime\" << std::endl;\n");
  wfilec.append("\t\tcommTimeProfile << \"CommTime\" << std::endl;\n");

  wfilec.append("\t\tpTimeProfile << \"backendName: " + backendName +
                "\" << std::endl;\n");
  wfilec.append("\t\tcommTimeProfile << \"backendName: " + backendName +
                "\" << std::endl;\n");
  wfilec.append("\t\tfor(int i = 0; i < " + std::to_string(partitionConfig.partitionNames.size() - 1) +
                "; i++){\n");
  wfilec.append("\t\t\tif(keyList[i].compare(\"blank\")) {\n");
  wfilec.append("\t\t\t\tsprintf(p_str, \"%lu\", p_time_ary[i]);\n");
  wfilec.append("\t\t\t\tsprintf(comm_str, \"%lu\", comm_time_ary[i]);\n");
  wfilec.append("\t\t\t\tpTimeProfile << keyList[i] << \" : \" << p_str;\n");
  wfilec.append(
      "\t\t\t\tcommTimeProfile << keyList[i] << \" : \" << comm_str;\n");
  wfilec.append("\t\t\t\tpTimeProfile << std::endl;\n");
  wfilec.append("\t\t\t\tcommTimeProfile << std::endl;\n");
  wfilec.append("\t\t\t}\n");
  wfilec.append("\t\t}\n");
  wfilec.append("\t\tpTimeProfile << \"...\" << std::endl;\n");
  wfilec.append("\t\tcommTimeProfile << \"...\" << std::endl;\n");
  wfilec.append("\t\tpTimeProfile.close();\n");
  wfilec.append("\t\tcommTimeProfile.close();\n");
  wfilec.append("\t}\n");
  wfilec.append("}\n");
}

void NestPartitionerSchedule::generateFree(
  string &wfilec, const PartitionConfig &partitionConfig) {

  for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
    auto pname = partitionConfig.partitionNames.at(idx);
    if (!partitionConfig.backendNames[idx].compare("VTA") ||
        !partitionConfig.backendNames[idx].compare("Relay")) {
      wfilec.append("\t" + pname + "_destroy_module();\n");
    }
  }
  wfilec.append("\n");

  bool vtaFlag = false;
  for (int i = 0; i < partitionConfig.partitionNames.size() - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if (vtaFlag) {
    wfilec.append("\tdestroyVTARuntime();\n");
    wfilec.append("\tdestroyVTARuntime1();\n");
    wfilec.append("\tdestroyVTARuntime2();\n");
    wfilec.append("\tdestroyVTARuntime3();\n");
    //        wfilec.append("#endif\n");
  }
  wfilec.append("\n");

  wfilec.append("\t// Free all resources.\n");

  for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
    auto pname = partitionConfig.partitionNames.at(idx);
    std::string partitionID = pname.substr(1, pname.length());

    if (!partitionConfig.backendNames[idx].compare("VTA") ||
        !partitionConfig.backendNames[idx].compare("CPU")) {
      wfilec.append("\tfree(activationsAddr" + partitionID + ");\n");
    }
    wfilec.append("\tfree(constantWeightVarsAddr" + partitionID + ");\n");
    wfilec.append("\tfree(mutableWeightVarsAddr" + partitionID +
                  ");\n\n");
  }

  if (profileMode_ == 1) {
    wfilec.append("\tcreateYamlFile();\n");
  }

  wfilec.append("\treturn 0;\n");
  wfilec.append("}\n\n");
}

#include <boost/algorithm/string/replace.hpp>
void NestPartitionerSchedule::generateMainforEachPartition(const PartitionConfig &partitionConfig) {
  std::string partitionMainTemplate =
      std::string("#include <stdlib.h>\n") + "#include \"main_template.h\"\n" +
      "#include \"p%.h\"\n" + "$0" + //#include "VTARuntime.h"
      "\n" + "int main(int argc, char **argv) {\n" + "\n" +
      "$1" + // initVTARuntime();
      "\tuint8_t *constantWeightVarsAddr% = "
      "initConstantWeights(\"p%.weights.bin\", p%_config);\n" +
      "\tuint8_t *mutableWeightVarsAddr% =  "
      "allocateMutableWeightVars(p%_config);\n" +
      "\tuint8_t *activationsAddr% = initActivations(p%_config);\n" +
      "$4" + // load_module();
      "\tint repeat = 1;\n" + "\tif(argc > 1)\n" +
      "\t\trepeat = atoi(argv[1]);\n" + "\tprintf(\"\\npStart $3\\n\");\n" +
      "\tfflush(stdout);\n" + "\tfor(int i = 0; i < repeat; i++)\n" +
      "\t\tp%(constantWeightVarsAddr%, mutableWeightVarsAddr%, "
      "activationsAddr%);\n" +
      "\tprintf(\"pEnd $3\\n\");\n" + "\tfflush(stdout);\n" +
      "$5" + // load_module();
      "$2" + // destroyVTARuntime();
      "\n}";

  for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
    auto pname = partitionConfig.partitionNames.at(idx);
    auto func = (*funcMap_)[pname];
    std::string partitionID = pname.substr(1, pname.length());

    std::string key = getPartitionProfileKey(func);
    std::string partitionMainStr(partitionMainTemplate.c_str());
    std::string mainSubFileName = outputDir_ + "/" + pname + "_main.cpp";
    if (!llvm::sys::fs::is_directory(outputDir_)) {
      llvm::sys::fs::create_directory(outputDir_);
    }

    ofstream writeFileC;
    writeFileC.open(mainSubFileName.data());
    if (writeFileC.is_open()) {
      if (!partitionConfig.backendNames[idx].compare("VTA")) {
        boost::replace_all(partitionMainStr, "$0",
                           "#include \"VTARuntime.h\"\n");
        boost::replace_all(partitionMainStr, "$1", "\tinitVTARuntime();\n");
        boost::replace_all(partitionMainStr, "$2", "\tdestroyVTARuntime();\n");
        boost::replace_all(partitionMainStr, "$4",
                           "\t" + pname + "_load_module(constantWeightVarsAddr" +
                                   partitionID + ");\n");
        boost::replace_all(partitionMainStr, "$5",
                           "\t" + pname + "_destroy_module();\n");
      } else {
        boost::replace_all(partitionMainStr, "$0", "");
        boost::replace_all(partitionMainStr, "$1", "");
        boost::replace_all(partitionMainStr, "$2", "");
        boost::replace_all(partitionMainStr, "$4", "");
        boost::replace_all(partitionMainStr, "$5", "");
      }

      boost::replace_all(partitionMainStr, "$3", key);
      boost::replace_all(partitionMainStr, "%", partitionID);
      writeFileC << partitionMainStr;
      writeFileC.close();
    }
  }
}

bool isMyNode(BFSLevel bfs, string name) {
  size_t level = bfs.size();
  for (int bi = 0; bi < level; bi++) {
    for (int bii = 0; bii < bfs[bi].size(); bii++) {
      Node *node = bfs[bi][bii];
      if (!name.compare(node->getName()))
        return true;
    }
  }
  return false;
}

void NestPartitionerSchedule::setPartitionInputOutputList(
    Function *function, std::vector<std::string> *inputList,
    std::vector<std::string> *outputList, int pi) {
  BFSLevel bfs = getBFSLevel(function);
  size_t level = bfs.size();

  for (int bi = 0; bi < level; bi++) {
    for (int bii = 0; bii < bfs[bi].size(); bii++) {
      Node *node = bfs[bi][bii];
      if (!string(node->getKindName()).compare("Save")) {
        std::string name =
            node->getName().substr(0, node->getName().size() - 5);
        outputList->push_back(name);
        partitionOutList_[name] = pi;
//                        cout << "pi = " << pi << ", " << "name = " << name << ", output = " << node->getName().str() << endl;
//                        cout << "outputList[inputCount] = " <<
//                        outputList[*outputCount] << std::endl; cout <<
//                        "outputCount = " << *outputCount << std::endl;
//                        (*outputCount)++;
      } else {
        for (int count = 0; count < node->getNumInputs(); count++) {
          Node *inNode = node->getNthInput(count).getNode();
          string kindName = inNode->getKindName();

          if (!kindName.compare("Placeholder")) {

            if (isMyNode(bfs, inNode->getName())) {
              continue;
            }

            inputList->push_back(inNode->getName());

//                                    cout << "input = " << inNode->getKindName() << endl;
//                                    cout "input = " << inNode->getName().str() << endl;
//                                    cout << "inputList[inputCount] = "
            //                        << inputList[*inputCount] << std::endl;
            //                        cout << "inputCount = " << *inputCount <<
            //                        std::endl;

            //                        (*inputCount)++;
          }
        }
      }
    }
  }

  std::sort(inputList->begin(), inputList->end());
  inputList->erase(std::unique(inputList->begin(), inputList->end()),
                   inputList->end());

  std::sort(outputList->begin(), outputList->end());
  outputList->erase(std::unique(outputList->begin(), outputList->end()),
                    outputList->end());
}
void NestPartitionerSchedule::generateNonThreadCall(
    std::string &wfilec, int pi, bool profileMode,
    std::vector<std::string> *inputList, std::string backendName) {
//    std::cout << "[generateNonThreadCall]" << std::endl;


    for (int cnt = 0; cnt < inputList->size(); cnt++) {
    std::string varStr = "";
    std::string varPi = "";
    std::string nodeName = inputList->at(cnt);

    auto iter = partitionOutList_.find(nodeName);
    if (iter != partitionOutList_.end()) {
      varPi = std::to_string(iter->second);
      auto varIter = partitionOutVarList_.find(nodeName);
      if (varIter != partitionOutVarList_.end()) {
        varStr = partitionOutVarList_[nodeName];
      } else {
        if (inputList->size() == 1) {
          varStr = "inputVar" + std::to_string(pi);
        } else {
          varStr = "inputVar" + std::to_string(pi) + "_" + std::to_string(cnt);
        }
        partitionOutVarList_[nodeName] = varStr;

        if (pi > 0) {

          if (profileMode == 1) {
            wfilec.append("\n\tgettimeofday(&comm_time_start, NULL);\n");
          }
          std::string rVarName = wfilec.append(
              "\tfloat* " + varStr + " = getInferenceResultVar(p" + varPi +
              "_config, mutableWeightVarsAddr" + varPi + ", \"" +
              inputList->at(cnt) + "\");\n");
        }
      }
    } else {
//      std::cout << "No output: " << nodeName << std::endl;
    }

    if (pi > 0) {
      if(!backendName.compare("Newton")){
        wfilec.append("\tcopyMutableWeightVarsWithoutAllocNewton(p" +
                      std::to_string(pi) + "_config, mutableWeightVarsAddr" +
                      std::to_string(pi) + ", \"" + inputList->at(cnt) + "\", " +
                      varStr + ");\n");
      } else {
        wfilec.append("\tcopyMutableWeightVarsWithoutAlloc(p" +
                      std::to_string(pi) + "_config, mutableWeightVarsAddr" +
                      std::to_string(pi) + ", \"" + inputList->at(cnt) + "\", " +
                      varStr + ");\n");
      }

      if (profileMode == 1) {
        wfilec.append("\tgettimeofday(&comm_time_end, NULL);\n");
        wfilec.append("\tcomm_time = (comm_time_end.tv_sec - "
                      "comm_time_start.tv_sec)*1000000.0 + "
                      "(comm_time_end.tv_usec - comm_time_start.tv_usec);\n");
        wfilec.append("\tcomm_time_ary[" + std::to_string(pi) +
                      "] += comm_time;\n");
        wfilec.append("\tprintf(\"\\n====> [Inference_" +
                      std::to_string(pi - 1) +
                      " p time] %lu microsec.\\n\", p_time_ary[" +
                      std::to_string(pi - 1) + "]);\n");
        wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(pi) +
                      " comm time] %lu microsec.\\n\", comm_time_ary[" +
                      std::to_string(pi) + "]);\n\n");
      }
    }
  }

  if (profileMode == 1) {
    wfilec.append("\tgettimeofday(&p_time_start, NULL);\n");
  }

  if(!backendName.compare("Newton")){
    wfilec.append("\tp" + std::to_string(pi) + "((float*)constantWeightVarsAddr" +
                  std::to_string(pi) + ", (float*)mutableWeightVarsAddr" +
                  std::to_string(pi) + ", (float*)activationsAddr" + std::to_string(pi) +
                  ");\n");
  } else {
    wfilec.append("\tp" + std::to_string(pi) + "(constantWeightVarsAddr" +
                  std::to_string(pi) + ", mutableWeightVarsAddr" +
                  std::to_string(pi) + ", activationsAddr" + std::to_string(pi) +
                  ");\n");
  }

  if (profileMode == 1) {
    wfilec.append("\tgettimeofday(&p_time_end, NULL);\n");
    wfilec.append(
        "\tp_time = (p_time_end.tv_sec - p_time_start.tv_sec)*1000000.0 + "
        "(p_time_end.tv_usec - p_time_start.tv_usec);\n");
    wfilec.append("\tp_time_ary[" + std::to_string(pi) + "] = p_time;\n");
  }
}

bool NestPartitionerSchedule::generateThreadCall(
        std::string &wfilec, int pi, std::set<std::string> *pGroup,
    bool profileMode, std::vector<std::string> *inputList, std::string backendName) {

//    std::cout << "[generateThreadCall]" << std::endl;

  for (int cnt = 0; cnt < inputList->size(); cnt++) {
    std::string varStr = "";
    std::string varPi = "";
    std::string nodeName = inputList->at(cnt);

    auto iter = partitionOutList_.find(nodeName);
    if (iter != partitionOutList_.end()) {
      varPi = std::to_string(iter->second);
      auto varIter = partitionOutVarList_.find(nodeName);
      if (varIter != partitionOutVarList_.end()) {
        varStr = partitionOutVarList_[nodeName];
      } else {
        if (inputList->size() == 1) {
          varStr = "inputVar" + std::to_string(pi);
        } else {
          varStr = "inputVar" + std::to_string(pi) + "_" + std::to_string(cnt);
        }
        partitionOutVarList_[nodeName] = varStr;
        std::string rVarName =
            wfilec.append("\tfloat* " + varStr + " = getInferenceResultVar(p" +
                          varPi + "_config, mutableWeightVarsAddr" + varPi +
                          ", \"" + inputList->at(cnt) + "\");\n");
      }
    } else {
//      std::cout << "(No output) " << "pi = " << pi << ", node name = " << nodeName << std::endl;
      return false;
    }

    for (auto gpname : *pGroup) {
      std::string gpid = gpname.substr(1, gpname.length());
      wfilec.append("\tcopyMutableWeightVarsWithoutAlloc(p" + gpid +
                    "_config, mutableWeightVarsAddr" + gpid + ", \"" +
                    inputList->at(cnt) + "\", " + varStr + ");\n");
    }
  }

  wfilec.append("#ifdef THREAD\n");
  for (auto gpname : *pGroup) {
    std::string gpid = gpname.substr(1, gpname.length());
    wfilec.append("\tstd::thread thread_p" + gpid + "(p" + gpid +
                  ", constantWeightVarsAddr" + gpid +
                  ", mutableWeightVarsAddr" + gpid + ", activationsAddr" +
                  gpid + ");\n");
  }

  for (auto gpname : *pGroup) {
    std::string gpid = gpname.substr(1, gpname.length());
    wfilec.append("\tthread_p" + gpid + ".join();\n");
  }

  wfilec.append("#else\n");
  for (auto gpname : *pGroup) {
    std::string gpid = gpname.substr(1, gpname.length());
    wfilec.append("\t" + gpname + "(constantWeightVarsAddr" + gpid +
                  ", mutableWeightVarsAddr" + gpid + ", activationsAddr" +
                  gpid + ");\n");
  }
  wfilec.append("#endif\n");

  if (profileMode == 1) {
    wfilec.append("\ttimeval now_" + std::to_string(pi) + ";\n");
    wfilec.append("\tgettimeofday(&now_" + std::to_string(pi) + ", NULL);\n");
    wfilec.append("\tdouble inftime_" + std::to_string(pi) + " = (now_" +
                  std::to_string(pi) + ".tv_sec - start_" + std::to_string(pi) +
                  ".tv_sec)*1000000.0 + (now_" + std::to_string(pi) +
                  ".tv_usec - start_" + std::to_string(pi) + ".tv_usec);\n");

    wfilec.append("\tdelay_time[" + std::to_string(pi) + "] = inftime_" +
                  std::to_string(pi) + ";\n");
    wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(pi) +
                  " time] %lu microsec.\\n\", delay_time[" +
                  std::to_string(pi) + "]);\n\n");
  }
  return true;
}

void NestPartitionerSchedule::generateVarInit(std::string &wfilec, const PartitionConfig &partitionConfig) {
  wfilec.append("\n\t//======== Declaration & initialization of weight "
                "variables =======//\n\n");
  //  cout << "//======== initialize variables start =======//" << endl;
  std::string inputName;
  BFSLevel bfs = getBFSLevel((*funcMap_)[partitionConfig.partitionNames[0]]);
  size_t level = bfs.size();
  std::string kindName_check;
  for (int i2 = level - 1; i2 >= 0; i2--) {
    Node *node = bfs[i2][0];
    kindName_check = node->getNthInput(0).getNode()->getKindName();
    if (i2 != 0 && !kindName_check.compare("Placeholder")) {
      inputName = node->getNthInput(0).getNode()->getName();
      break;
    }
  }

  for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
    auto pname = partitionConfig.partitionNames.at(idx);
    std::string partitionID = pname.substr(1, pname.length());
    if (!partitionConfig.backendNames[idx].compare("Relay")) {
      wfilec.append("\tuint8_t *constantWeightVarsAddr" + partitionID +
                    " = NULL;\n");
    } else {
      wfilec.append("\tuint8_t *constantWeightVarsAddr" + partitionID +
                    " = \n\t\tinitConstantWeights(\"" + pname +
                    ".weights.bin\", "+ pname + "_config);\n");
    }
    if (idx == 0) {
      wfilec.append("\tuint8_t *mutableWeightVarsAddr" + partitionID +
                    " = initMutableWeightVars(" + pname +
                    "_config, \"" + inputName + "\");\n");
    }

    wfilec.append("\tuint8_t *activationsAddr" + partitionID +
                  " = initActivations(" + pname + "_config);\n\n");
  }

  //  cout << "//======== initialize mutableWeightVarsAddr start =======//" <<
  //  endl;
  wfilec.append("\t//======== Declaration and initialization of mutable "
                "variables =======//\n\n");

  for(int idx = 1; idx < partitionConfig.partitionNames.size()-1; idx++) {
    auto pname = partitionConfig.partitionNames.at(idx);
    if(idx > 0) {
        std::string partitionID = pname.substr(1, pname.length());
        wfilec.append("\tuint8_t *mutableWeightVarsAddr" + partitionID +
                      " =  allocateMutableWeightVars(" + pname +
                      "_config);\n");
    }
  }
  wfilec.append("\n");

  //  cout << "//======== call xx_load_module() =======//" << endl;
  for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
    auto pname = partitionConfig.partitionNames.at(idx);
      std::string partitionID = pname.substr(1, pname.length());
    if (!partitionConfig.backendNames[idx].compare("Relay")) {
      wfilec.append("\t" + pname + "_load_module(0);\n");
    } else if (!partitionConfig.backendNames[idx].compare("VTA")) {
      wfilec.append("\t" + pname +
                    "_load_module(constantWeightVarsAddr" + partitionID +
                    ");\n");
    } else if (!partitionConfig.backendNames[idx].compare("Newton")) {
      wfilec.append("\t" + pname +
                    "_load_module((float *)constantWeightVarsAddr" + partitionID +
                    ");\n");
    }
  }
  wfilec.append("\n");
}

void getParallelGroup(std::vector<std::set<std::string>> parallelPartitions,
                      int pid, std::set<std::string> *pGroup) {

  std::string pidStr = "p" + std::to_string(pid);
  for (auto idSet : parallelPartitions) {
    if (idSet.find(pidStr) != idSet.end()) {
      pGroup->insert(idSet.begin(), idSet.end());
      return;
    }
  }
}



void NestPartitionerSchedule::generatePartitionCall(
  std::string &wfilec, const PartitionConfig &partitionConfig) {
    std::cout << "[generatePartitionCall]" << std::endl;
    // first partition
  wfilec.append("\ttimeval t1, t2;\n");
  if (profileMode_ == 1) {
    wfilec.append("\ttimeval p_time_start, p_time_end;\n");
    wfilec.append("\ttimeval comm_time_start, comm_time_end;\n");
    wfilec.append("\tdouble p_time = 0.0, comm_time = 0.0;\n");
  }

  wfilec.append("\tgettimeofday(&t1, NULL);\n\n");

  std::vector<std::string> processedPP;
  std::string finalOutStr;

  for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
    std::string pname = partitionConfig.partitionNames.at(idx);
    size_t pid = std::stoi(pname.substr(1, pname.length()));

    auto func = (*funcMap_)[pname];
    pid = std::stoi(pname.substr(1, pname.length()));

    std::vector<std::string> inputList;
    std::vector<std::string> outputList;

    setPartitionInputOutputList(func, &inputList, &outputList, pid);

    std::set<std::string> pGroup;
    getParallelGroup(partitionConfig.parallelPartitions, pid, &pGroup);
    if (std::find(processedPP.begin(), processedPP.end(),
                  pname) != processedPP.end()) {
        continue;
    }

    // partition call
    if (pGroup.size() == 0) {
      generateNonThreadCall(wfilec, pid, profileMode_, &inputList, partitionConfig.backendNames[idx]);
    } else {
      // same input
      bool processed = generateThreadCall(wfilec, pid, &pGroup, profileMode_, &inputList, partitionConfig.backendNames[idx]);
      if(processed) {
          processedPP.insert(processedPP.end(), pGroup.begin(), pGroup.end());
      }else {
//          std::cout << "Error generating thread code!" << pname << std::endl;
      }
    }

    // last partition
    if(idx == partitionConfig.partitionNames.size() - 2) {
      finalOutStr = outputList.at(0);
      break;
    }
  }

    auto plastPName = partitionConfig.partitionNames.at(partitionConfig.partitionNames.size() - 1);
    std::string plastPartitionID = plastPName.substr(1, plastPName.length());

    auto lastPName = partitionConfig.partitionNames.at(partitionConfig.partitionNames.size() - 2);
    std::string lastPartitionID = lastPName.substr(1, lastPName.length());

  if (profileMode_ == 1) {
    wfilec.append("\tgettimeofday(&comm_time_end, NULL);\n");
    wfilec.append("\tcomm_time = (comm_time_end.tv_sec - "
                  "comm_time_start.tv_sec)*1000000.0 + (comm_time_end.tv_usec "
                  "- comm_time_start.tv_usec);\n");
    wfilec.append("\tcomm_time_ary[" + std::to_string(partitionConfig.partitionNames.size() - 1) +
                  "] = comm_time;\n");
    wfilec.append("\tprintf(\"\\n====> [Inference_" +
                  lastPartitionID +
                  " p time] %lu microsec.\\n\", p_time_ary[" +
                  std::to_string(partitionConfig.partitionNames.size() - 2) + "]);\n");

    wfilec.append("\tprintf(\"\\n====> [Inference_" +
                          plastPartitionID +
                  " comm time] %lu microsec.\\n\", comm_time_ary[" +
                  std::to_string(partitionConfig.partitionNames.size() - 1) + "]);\n\n");
  }

  wfilec.append("\tgettimeofday(&t2, NULL);\n");
  wfilec.append("\tdouble inftime_total = (t2.tv_sec - t1.tv_sec)*1000.0 + "
                "(t2.tv_usec - t1.tv_usec)/1000.0;\n");
  wfilec.append("\tprintf(\"\\n====> [Total inference time] %.4f msec.\\n\", "
                "inftime_total);\n\n");

  wfilec.append("\t// Report the results.\n");
  wfilec.append(
      "\tprintf(\"====== final result of this neural net ========\\n\");\n");
  wfilec.append("\tdumpInferenceResults(" + lastPName +
                "_config, mutableWeightVarsAddr" +
                        lastPartitionID + ", \"" + finalOutStr +
                "\");\n\n");
}

void NestPartitionerSchedule::generateMain(
    std::string &wfilec, const PartitionConfig &partitionConfig) {
  //  cout << "generateMain: " << wfilec << endl;

  std::cout << "Partition plan fileName : " << getPartitionPlanFile()
            << std::endl;
  std::map<std::string, std::string> pmap =
      deserializeStrStrMapFromYaml(getPartitionPlanFile());

  if (partitionConfig.parallelPartitions.size() > 0) {
    wfilec.append("#define THREAD 1\n\n");
    wfilec.append("#include <thread>\n\n");
  }

  wfilec.append("int main(int argc, char **argv) {\n");
  if (profileMode_ == 1) {
    wfilec.append(
        "\tmemset(comm_time_ary, 0, ARRAY_LENGTH*sizeof(unsigned long));\n");
  }
  wfilec.append("\tparseCommandLineOptions(argc, argv);\n\n");

  bool vtaFlag = false;
  for (int i = 0; i < partitionConfig.partitionNames.size() - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if (vtaFlag) {
    wfilec.append("\tinitVTARuntime();\n");
    wfilec.append("\tinitVTARuntime1();\n");
    wfilec.append("\tinitVTARuntime2();\n");
    wfilec.append("\tinitVTARuntime3();\n");
  }

  generateVarInit(wfilec, partitionConfig);
  generatePartitionCall(wfilec, partitionConfig);
}

void NestPartitionerSchedule::generateCodeFromModels(const PartitionConfig &partitionConfig, std::string inputPartitionName) {
  std::cout << "generateCodeFromModels Start" << std::endl;

  generateMainFile(partitionConfig, inputPartitionName);
  generateCMakeListsFile(partitionConfig);
  generateRelayFile(partitionConfig);
}

#include <stdlib.h>
void NestPartitionerSchedule::generateRelayFile(const PartitionConfig &partitionConfig) {
  string genRelayFileName = outputDir_ + "/genRelay.sh";
  bool isRelay = false;

  for (int i = 0; i < partitionConfig.partitionNames.size() - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("Relay")) {
      isRelay = true;
      break;
    }
  }

  if (!isRelay)
    return;

  ofstream writeRelaySHFile;
  writeRelaySHFile.open(genRelayFileName.data());
  if (writeRelaySHFile.is_open()) {
    string relaySH;

    for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
      auto pname = partitionConfig.partitionNames.at(idx);
      if (!partitionConfig.backendNames[idx].compare("Relay")) {
        relaySH.append("python3 relay__" + pname +
                       "/genCode.py\n");
      }
    }
    writeRelaySHFile << relaySH;
    writeRelaySHFile.close();

    string cmd = "chmod 755 " + genRelayFileName;
    system(cmd.data());
  }

  string genBundleFileName = outputDir_ + "/genBundleObj.sh";
  ofstream writeBundleSHFile;
  writeBundleSHFile.open(genBundleFileName.data());
  if (writeBundleSHFile.is_open()) {
    string bundleSH;

    for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
      auto pname = partitionConfig.partitionNames.at(idx);
      if (!partitionConfig.backendNames[idx].compare("Relay")) {
        bundleSH.append("cd module__" + pname + ";make;cd ..\n");
      }
    }

    writeBundleSHFile << bundleSH;
    writeBundleSHFile.close();

    string cmd = "chmod 755 " + genBundleFileName;
    system(cmd.data());
  }

  string cmd = "chmod 755 " + genBundleFileName;
  system(cmd.data());
}

void NestPartitionerSchedule::generateMainFile(const PartitionConfig &partitionConfig, std::string inputPartitionName) {

  string mainFileName = outputDir_ + "/main.cpp";

  if (!llvm::sys::fs::is_directory(outputDir_)) {
    llvm::sys::fs::create_directory(outputDir_);
  }
  ofstream writeMainFile;
  writeMainFile.open(mainFileName.data());

  if (writeMainFile.is_open()) {
    string declare;
    string initialize;
    string inference_main;
    string memfree;
    string yamlFile;
    std::vector<std::string> resultVarList;

    generateDeclaration(declare, partitionConfig);
    generateMain(inference_main, partitionConfig);
    generateFree(memfree, partitionConfig);

    writeMainFile << declare;
    writeMainFile << initialize;
    // writeFileC << inference;
    writeMainFile << inference_main;
    writeMainFile << memfree;

    if (profileMode_ == 1) {
      generateYamlFile(yamlFile, partitionConfig,
                       inputPartitionName);
      writeMainFile << yamlFile;
    }
    writeMainFile.close();
  }

  if (partitionExeMode_ == 1) {
    generateMainforEachPartition(partitionConfig);
  }

  std::cout << "main outfile position : " << mainFileName << std::endl;
}

#include <random>
void NestPartitionerSchedule::generateCMakeListsFile(const PartitionConfig &partitionConfig) {
  //  std::cout << "== [Start] generateCMakeListsFile == " << std::endl;

  string makeFileName = outputDir_ + "/CMakeLists.txt";
  ofstream writeMakeFile;
  writeMakeFile.open(makeFileName.data());
  if (writeMakeFile.is_open()) {
    string cmakeFile;

    cmakeFile.append("add_custom_command(\n");
    cmakeFile.append("\tOUTPUT\n");

    cmakeFile.append("\t");
    for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
      auto pname = partitionConfig.partitionNames.at(idx);
      if (!partitionConfig.backendNames[idx].compare("CPU")) {
        cmakeFile.append("${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".o ");
      } else if (!partitionConfig.backendNames[idx].compare("VTA")) {
        cmakeFile.append("${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".cpp ");
      } else if (!partitionConfig.backendNames[idx].compare("Relay")) {
        cmakeFile.append("${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".o ");
        cmakeFile.append("${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         "_tvm.so ");
      }
    }
    cmakeFile.append("\n\n");

    for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
      auto pname = partitionConfig.partitionNames.at(idx);
      cmakeFile.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");

      if (!partitionConfig.backendNames[idx].compare("CPU")) {
        cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/" + pname +
                         ".o\n");
        cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".o\n");
      } else if (!partitionConfig.backendNames[idx].compare("VTA")) {
        cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/" + pname +
                         ".cpp\n");
        cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".cpp\n");
      } else if (!partitionConfig.backendNames[idx].compare("Relay")) {
        cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/module__" + pname
        + "/" + pname + ".o\n");
        cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".o\n");

        cmakeFile.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
        cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/module__" + pname + "/" + pname +
                         "_tvm.so\n");
        cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         "_tvm.so\n");
      }
    }
    cmakeFile.append("\n");

    for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
      auto pname = partitionConfig.partitionNames.at(idx);
      cmakeFile.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");

      if (!partitionConfig.backendNames[idx].compare("Relay")) {
        cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/module__" + pname
        + "/" + pname + ".h\n");
        cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".h\n");
      } else {
        cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/" + pname +
                         ".h\n");
        cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".h\n");
      }
    }
    cmakeFile.append("\n");

    cmakeFile.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
    cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/main_header_test.h\n");
    cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/main_header_test.h\n\n");

    cmakeFile.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
    cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/main_template.h\n");
    cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/main_template.h\n\n");

    bool vtaFlag = false, relayFlag = false;
    for (int i = 0; i < partitionConfig.partitionNames.size() - 1; i++) {
      if (!partitionConfig.backendNames[i].compare("VTA"))
        vtaFlag = true;

      if (!partitionConfig.backendNames[i].compare("Relay"))
        relayFlag = true;
    }

    if (vtaFlag) {
      cmakeFile.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
      cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/VTARuntime.h\n");
      cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h\n\n");
    }

    for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
      auto pname = partitionConfig.partitionNames.at(idx);
      if (partitionConfig.backendNames[idx].compare("Relay")) {
        cmakeFile.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
        cmakeFile.append("\t${CMAKE_CURRENT_SOURCE_DIR}/" + pname +
                         ".weights.bin\n");
        cmakeFile.append("\t${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".weights.bin\n");
      }
    }
    cmakeFile.append("\t)\n");

    srand((int)time(0));
    std::string demoExeFileName =
        "demoExeFileName" + std::to_string(rand() % 100000);

    cmakeFile.append("add_custom_target(" + demoExeFileName + "Net DEPENDS\n");
    cmakeFile.append("\t");

    for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
      auto pname = partitionConfig.partitionNames.at(idx);
      if (!partitionConfig.backendNames[idx].compare("CPU") ||
          !partitionConfig.backendNames[idx].compare("Relay")) {
        cmakeFile.append("${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".o ");
      } else if (!partitionConfig.backendNames[idx].compare("VTA")) {
        cmakeFile.append("${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".cpp ");
      }
    }
    cmakeFile.append("\n)\n\n");

    cmakeFile.append("INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})\n");
    cmakeFile.append("add_executable(" + demoExeFileName + " main.cpp\n");
    cmakeFile.append("\t");

    for(int idx = 0; idx < partitionConfig.partitionNames.size()-1; idx++) {
      auto pname = partitionConfig.partitionNames.at(idx);
      if (!partitionConfig.backendNames[idx].compare("VTA")) {
        cmakeFile.append("${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".cpp ");
      } else {
        cmakeFile.append("${CMAKE_CURRENT_BINARY_DIR}/" + pname +
                         ".o ");
      }
    }

    cmakeFile.append("\t\n)\n");

    if (relayFlag) {
      cmakeFile.append("link_directories(${CMAKE_BINARY_DIR}/tvm)\n");
      cmakeFile.append("include_directories(${CMAKE_CURRENT_BINARY_DIR})\n");
      cmakeFile.append("include_directories(BEFORE\n"
                       "\t${NESTC_ROOT_DIR}/tvm/include\n"
                       "\t${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include\n"
                       "\t${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include\n"
                       "\t)\n");
    }

    // make an exe file for a partition
    if (partitionExeMode_ == 1) {
      std::string pExeStrCPUTemplate =
          std::string("add_executable(" + demoExeFileName +
                      "-p# p#_main.cpp ${CMAKE_CURRENT_BINARY_DIR}/p#.o)\n") +
          "add_dependencies(" + demoExeFileName + "-p# " + demoExeFileName +
          "Net)\n" + "target_link_libraries(" + demoExeFileName + "-p# png)\n" +
          "set_target_properties(" + demoExeFileName + "-p#\n" +
          "\tPROPERTIES\n" +
          "\tRUNTIME_OUTPUT_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")\n";

      std::string pExeStrVTATemplate =
          std::string("add_executable(" + demoExeFileName +
                      "-p# p#_main.cpp ${CMAKE_CURRENT_BINARY_DIR}/p#.cpp)\n") +
          "add_dependencies(" + demoExeFileName + "-p# " + demoExeFileName +
          "Net)\n" + "target_link_libraries(" + demoExeFileName +
          "-p# VTABundle png)\n" + "set_target_properties(" + demoExeFileName +
          "-p#\n" + "\tPROPERTIES\n" +
          "\tRUNTIME_OUTPUT_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")\n";

      std::string pExeStrRelayTemplate =
          std::string("add_executable(" + demoExeFileName +
                      "-p# p#_main.cpp ${CMAKE_CURRENT_BINARY_DIR}/p#.o)\n") +
          "add_dependencies(" + demoExeFileName + "-p# " + demoExeFileName +
          "Net)\n" + "target_link_libraries(" + demoExeFileName +
          "-p# -fPIC -I. -fopenmp -pthread tvm_runtime png)\n" +
          "set_target_properties(" + demoExeFileName + "-p#\n" +
          "\tPROPERTIES\n" +
          "\tRUNTIME_OUTPUT_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")\n";

      std::string pid;
      for (int i = 0; i < partitionConfig.partitionNames.size() - 1; i++) {
        std::string pExeStr = "";
        if (!partitionConfig.backendNames[i].compare("CPU")) {
          pExeStr = pExeStrCPUTemplate;
        } else if (!partitionConfig.backendNames[i].compare("VTA")) {
          pExeStr = pExeStrVTATemplate;
        } else if (!partitionConfig.backendNames[i].compare("Relay")) {
          pExeStr = pExeStrRelayTemplate;
        }

        pid = std::to_string(i);
        boost::replace_all(pExeStr, "#", pid);
        cmakeFile.append(pExeStr);
      }
    }

    cmakeFile.append("INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})\n");
    cmakeFile.append("add_dependencies(" + demoExeFileName + " " +
                     demoExeFileName + "Net)\n");


    if (relayFlag) {
      cmakeFile.append(
          "target_link_libraries(" + demoExeFileName +
          " VTABundle -fPIC -I. -fopenmp -pthread tvm_runtime png)\n");
    } else {
      cmakeFile.append("target_link_libraries(" + demoExeFileName +
                       " VTABundle -fPIC -I. -fopenmp -pthread png)\n");
    }
    cmakeFile.append("set_target_properties(" + demoExeFileName + "\n");
    cmakeFile.append("\tPROPERTIES\n");
    cmakeFile.append(
        "\tIMPORTED_LOCATION "
        "${CMAKE_SOURCE_DIR}/build_release/tvm/libtvm_runtime.so\n");
    cmakeFile.append(
        "\tRUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}\n");
    cmakeFile.append("\t)\n\n");

    cmakeFile.append(
        "add_custom_command(\n"
        "        TARGET " +
        demoExeFileName +
        " POST_BUILD\n"
        "        COMMAND ${CMAKE_COMMAND} -E copy\n"
        "        ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png\n"
        "        ${CMAKE_CURRENT_BINARY_DIR}/cat_285.png\n"
        "\t)\n");

    writeMakeFile << cmakeFile;
    writeMakeFile.close();
  }
  std::cout << "CMakeLists outfile position : " << makeFileName << std::endl;
}

std::string
NestPartitionerSchedule::getPartitionProfileKey(Function *function) {
  BFSLevel bfs = getBFSLevel(function);
  size_t level = bfs.size();
  std::string key = "";
  int cnt = 0;
  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node *node = bfs[i][j];
      if (node->getKind() == Kinded::Kind::SaveNodeKind ||
          node->getKind() == Kinded::Kind::PlaceholderKind) {
        continue;
      }
      key = key + getProfileKey(node) + "+";
      cnt++;
    }
  }

  if (key.length() > 1) {
    key.pop_back();
  }
  return key;
}

std::string NestPartitionerSchedule::getProfileKey(Node *node) {
  std::string key;
  std::string kindName = node->getKindName();

  if (node->getKind() == Kinded::Kind::ConvolutionNodeKind) {
    key =
        std::string(node->getKindName()) + "-" +
        getDimArrayStr(node->getNthInput(0).getType()) + "-" +
        getDataTypeStr(node->getNthInput(0).getElementType()) + "-" +
        std::to_string(((ConvolutionNode *)node)->getFilter().dims().front()) +
        "-" + std::to_string(((ConvolutionNode *)node)->getKernels().front()) +
        "-" + std::to_string(((ConvolutionNode *)node)->getStrides().front());
  } else {
    if (node->getNumInputs() > 0) {
      key = std::string(node->getKindName()) + "-" +
            getDimArrayStr(node->getNthInput(0).getType()) + "-" +
            getDataTypeStr(node->getNthInput(0).getElementType());
    } else {
      key = node->getKindName();
    }
  }

  return key;
}

string NestPartitionerSchedule::replaceAll(std::string &str,
                                           std::string pattern,
                                           std::string replace) {
  string result = str;
  string::size_type pos = 0;
  string::size_type offset = 0;
  while ((pos = result.find(pattern, offset)) != std::string::npos) {
    result.replace(result.begin() + pos, result.begin() + pos + pattern.size(),
                   replace);
    offset = pos + replace.size();
  }
  return result;
}
