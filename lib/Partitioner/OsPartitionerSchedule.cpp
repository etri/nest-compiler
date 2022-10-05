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

//#include "glow/Partitioner/NestPartitionerSchedule.h"
#include "llvm/Support/FileSystem.h"
#include <fstream>

#include "glow/Partitioner/OsPartitionerSchedule.h"
#include "llvm/Support/FileSystem.h"
#include <fstream>

#include <sys/stat.h>
#include <sys/types.h>

extern int os_partition_main_type;

using namespace glow;
using namespace std;

void NestPartitionerSchedule::setKeyName(std::string key, int i) {
  keyList_[i] = key;
}

double NestPartitionerSchedule::getDelayTime(int i) { return delayTime_[i]; }

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
    string &wfilec, int partitionNum, const PartitionConfig &partitionConfig) {
  //  cout << "generateDeclaration: " <<  wfilec << endl;

  if (profileMode_ == 1) {
    wfilec.append("#define REPEAT 1\n\n");
  }

  bool vtaFlag = false;
  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if (vtaFlag) {
    wfilec.append("#define VTAMAIN\n\n");
  }

  wfilec.append("#include \"main_template.h\"\n\n");

  if (vtaFlag) {
    wfilec.append("#ifdef VTAMAIN\n");
    wfilec.append("#include \"VTARuntime.h\"\n");
    wfilec.append("#endif\n\n");
  }

  for (int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("#include \"p" + std::to_string(i) + ".h\"\n");
    //    cout << "#include \"p" + std::to_string(i) + ".h\"" << endl;
  }
  wfilec.append("\n");

  wfilec.append("#include <time.h>\n\n");

  wfilec.append("//hkjun\n\n");
  if (profileMode_ == 1) {
    // wfilec.append("#include <sys/timeb.h>\n\n");
    // wfilec.append("#include <time.h>\n\n");

    wfilec.append("#define ARRAY_LENGTH " + std::to_string(partitionNum) +
                  "\n");
    wfilec.append("unsigned long delay_time[ARRAY_LENGTH];\n\n");
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

void NestPartitionerSchedule::generateYamlFile(
    string &wfilec, std::size_t partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig, std::string inputPartitionName) {
  //  std::cout << "== [Start] generateYamlFile == " << std::endl;

  std::string backendName = partitionConfig.backendNames[0];
  std::string filename;
  std::string keyList[partitionNum];
  std::string keys;

  filename = "./" + inputPartitionName + "Profile.yaml";

  std::cout << "partition # : " << partitionNum << std::endl;
  std::cout << "inputPartitionName: " << inputPartitionName << std::endl;

  bool isFused = false;
  if (inputPartitionName.find("FusedNode") != std::string::npos) {
    isFused = true;
  }

  for (int nodeN = 0; nodeN < partitionNum - 1; nodeN++) {
    auto func = funcList[nodeN];

    std::string key = addProfileKey(func, profileKeySet_, isFused);

    // std::cout << "key: " << key << std::endl;

    if (key.length() == 0) {
      keys = keys + "\"blank\", ";
    } else {
      keys = keys + "\"" + key + "\", ";
    }
  }

  keys.pop_back();
  keys.pop_back();

  wfilec.append("void createYamlFile(){\n");
  wfilec.append("\tprintf(\"== createYamlFile start ==\\n\");\n");
  wfilec.append("\tstd::string keyList[" + std::to_string(partitionNum) +
                "] = {" + keys + "};\n");
  wfilec.append("\tstd::string filename = \"" + filename + "\";\n");
  wfilec.append("\tstd::ofstream profileFile(filename);\n\n");
  wfilec.append("\tchar* str = new char[10];\n\n");
  wfilec.append("\tif (profileFile.is_open()) {\n");
  wfilec.append("\t\tprofileFile << \"---\" << std::endl;\n");
  wfilec.append("\t\tprofileFile << \"backendName: " + backendName +
                "\" << std::endl;\n");
  wfilec.append("\t\tfor(int i = 0; i < " + std::to_string(partitionNum - 1) +
                "; i++){\n");
  wfilec.append("\t\t\tif(keyList[i].compare(\"blank\")) {\n");
  wfilec.append("\t\t\t\tsprintf(str, \"%lu\", delay_time[i]);\n");
  wfilec.append("\t\t\t\tprofileFile << keyList[i] << \" : \" << str;\n");
  wfilec.append("\t\t\t\tprofileFile << std::endl;\n");
  wfilec.append("\t\t\t}\n");
  wfilec.append("\t\t}\n");
  wfilec.append("\t\tprofileFile << \"...\" << std::endl;\n");
  wfilec.append("\t\tprofileFile.close();\n");
  wfilec.append("\t}\n");
  wfilec.append("}\n");
}

void NestPartitionerSchedule::generateFree(
    string &wfilec, int partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig) {

  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA")) {
      wfilec.append("\tp" + std::to_string(i) + "_destroy_module();\n");
    }
  }
  wfilec.append("\n");

  wfilec.append("\t// Free all resources.\n");

  //  const char *fileName;
  std::string filePath;

  for (int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("\tfree(activationsAddr" + std::to_string(i) + ");\n");
    wfilec.append("\tfree(constantWeightVarsAddr" + std::to_string(i) + ");\n");
    wfilec.append("\tfree(mutableWeightVarsAddr" + std::to_string(i) +
                  ");\n\n");
  }

  bool vtaFlag = false;
  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if (vtaFlag) {
    wfilec.append("#ifdef VTAMAIN\n");
    wfilec.append("\tdestroyVTARuntime();\n");
    wfilec.append("#endif\n");
  }

  if (profileMode_ == 1) {
    wfilec.append("\tcreateYamlFile();\n");
  }

  wfilec.append("\treturn 0;\n");
  wfilec.append("}\n\n");
}

#include <boost/algorithm/string/replace.hpp>
void NestPartitionerSchedule::generateMainforEachPartition(
    int partitionNum, const PartitionConfig &partitionConfig) {
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
      "\tint repeat = 1;\n" + "\tif(argc > 1)\n" +
      "\t\trepeat = atoi(argv[1]);\n" + "\tprintf(\"pStart %\\n\");\n" +
      "\tfor(int i = 0; i < repeat; i++)\n" +
      "\t\tp%(constantWeightVarsAddr%, mutableWeightVarsAddr%, "
      "activationsAddr%);\n" +
      "\tprintf(\"pEnd %\\n\");\n" + "$2" + // destroyVTARuntime();
      "\n}";

  for (int i = 0; i < partitionNum - 1; i++) {
    std::string partitionMainStr(partitionMainTemplate.c_str());
    std::string pid = std::to_string(i);
    std::string outfile_c = outputDir_ + "/p" + pid + "_main.cpp";
    if (!llvm::sys::fs::is_directory(outputDir_)) {
      llvm::sys::fs::create_directory(outputDir_);
    }

    ofstream writeFileC;
    writeFileC.open(outfile_c.data());
    if (writeFileC.is_open()) {
      if (!partitionConfig.backendNames[i].compare("VTA")) {
        boost::replace_all(partitionMainStr, "$0",
                           "#include \"VTARuntime.h\"\n");
        boost::replace_all(partitionMainStr, "$1", "\tinitVTARuntime();\n");
        boost::replace_all(partitionMainStr, "$2", "\tdestroyVTARuntime();\n");
      } else {
        boost::replace_all(partitionMainStr, "$0", "");
        boost::replace_all(partitionMainStr, "$1", "");
        boost::replace_all(partitionMainStr, "$2", "");
      }

      boost::replace_all(partitionMainStr, "%", pid.c_str());
      //   std::cout << partitionMainStr << std::endl;
      writeFileC << partitionMainStr;
      writeFileC.close();
    }
  }
}

// by jun
int clientPartitionFlag = 0;
int clientPartitionNumber;

void NestPartitionerSchedule::generateMain(
    std::string &wfilec, std::string &wfiled, std::string &wfilee,
    int partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig) {
  //  cout << "generateMain: " << wfilec << endl;

  std::cout << "generateMain os_partition_main_tyep = "
            << os_partition_main_type << std::endl;

  wfilee.append(
      "START LAYER: 0  END LAYER: " + std::to_string(partitionNum - 2) + "\n");

  int os_partition_flag = 0;
  std::cout << "Partition plan fileName : " << getPartitionPlanFile()
            << std::endl;
  std::vector<std::string> parallerPartitions;
  std::map<std::string, std::string> pmap =
      deserializeStrStrMapFromYaml(getPartitionPlanFile());
  if (pmap.find("parallelPartitions") != pmap.end()) {
    boost::split(parallerPartitions, pmap.find("parallelPartitions")->second,
                 [](char c) { return c == ':'; });
    for (int i = 0; i < parallerPartitions.size(); i++) {
      parallerPartitions[i] = replaceAll(parallerPartitions[i], "(", "");
      parallerPartitions[i] = replaceAll(parallerPartitions[i], ")", "");
      parallerPartitions[i] =
          parallerPartitions[i].substr(0, parallerPartitions[i].find("||"));
    }
  }

  if (parallerPartitions.size() > 0) {
    wfilec.append("#define THREAD 1\n\n");

    wfilec.append("#include <thread>\n\n");
  }
  /*
    bool vtaFlag = false;
    for(int i = 0; i < partitionNum - 1; i++)
    {
      if(!partitionConfig.backendNames[i].compare("VTA"))
        vtaFlag = true;
    }

    if(vtaFlag)
    {
      wfilec.append("#ifdef VTAMAIN\n");
      wfilec.append("\tinitVTARuntime();\n");
      wfilec.append("#endif\n");
    }
  */

  //  cout << "//======== initialize variables start =======//" << endl;

  //  const char *fileName;
  std::string filePath;
  std::string inputName;

  BFSLevel bfs = getBFSLevel(funcList[0]);
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

  /*
  for(int i = 0; i < partitionNum - 1; i++) {

    wfilec.append("\tconstantWeightVarsAddr" + std::to_string(i) +
                  " = \n\t\tinitConstantWeights(\"p" + std::to_string(i) +
                  ".weights.bin\", p" + std::to_string(i) + "_config);\n");

    if (i == 0) {
      wfilec.append("\tmutableWeightVarsAddr" + std::to_string(i) +
                    " = initMutableWeightVars(p" + std::to_string(i) +
                    "_config, \"" + inputName + "\");\n");
    }

    wfilec.append("\tactivationsAddr" + std::to_string(i) +
                  " = initActivations(p" + std::to_string(i) + "_config);\n\n");
  }
*/
  //  cout << "//======== initialize mutableWeightVarsAddr start =======//" <<
  //  endl;

  /*
    for(int i = 0; i < partitionNum - 1; i++)
    {
      if(!partitionConfig.backendNames[i].compare("VTA"))
      {
        wfilec.append("\tp" + std::to_string(i) +
    "_load_module(constantWeightVarsAddr" + std::to_string(i) + ");\n");
      }
    }
    wfilec.append("\n");
  */
  // cout << "//======== multiple partitions start =======//" << endl;
  wfilec.append("\t//======== multiple partitions =======//\n\n");

  if (profileMode_ == 1) {
    wfilec.append("\tunsigned long totalInftime;\n\n");
  }

  int resultNum = 0;
  int resultVarNum = 0;
  string resultList[partitionNum + partitionNum];
  string resultCheckList[partitionNum + partitionNum];
  std::vector<std::string> paraller;

  for (int i = 0; i < partitionNum - 1; i++) {
    auto func = funcList[i];

    int inputCount = 0;
    int outputCount = 0;
    int checkResultNum = 0;
    bool flag = false;

    //    const char *fileName;
    std::string filePath;

    string inputList[partitionNum];
    string outputList[partitionNum];
    BFSLevel bfs = getBFSLevel(func);
    size_t level = bfs.size();

    // std::string kindName_check;
    //    cout << "//======== inputName List Node " << i << " =======//" <<
    //    endl;
    for (int i2 = level - 1; i2 >= 0; i2--) {
      Node *node = bfs[i2][0];

      /*    if (node->getNumInputs() == 2) {
              wfilec.append("\t//node get number inputs " +
         std::to_string(node->getNumInputs()) + "\n");
              wfilec.append("\t//partition layer " + std::to_string(i) + "\n");
          }
      */
      for (int count = 0; count < node->getNumInputs(); count++) {
        kindName_check = node->getNthInput(count).getNode()->getKindName();
        if (i2 != 0 && !kindName_check.compare("Placeholder")) {
          //          cout << "node inputCount : " << count << ", inputName = "
          //          << node->getNthInput(count).getNode()->getName().str() <<
          //          endl; cout << "node input KindName = " <<
          //          node->getNthInput(count).getNode()->getKindName() << endl;
          inputList[inputCount] = node->getNthInput(count).getNode()->getName();
          //        std::cout << inputList[inputCount] << std::endl;
          inputCount++;
        }
      }
    }

    // outputName 초기화
    std::string outputName;
    outputName = func->getNodes().back().getName().trim("_save");
    // cout << "//======== outputName List Node " << i << " =======//" << endl;
    NodesList &nodeList = func->getNodes();
    for (auto &n : nodeList) {
      kindName_check = n.getKindName();
      if (!kindName_check.compare("Save")) {
        outputList[outputCount] = n.getName().substr(0, n.getName().size() - 5);
        outputCount++;
      }
    }

    if (i == 0) {
      wfilec.append("static void predicting_resnet18_layer" +
                    std::to_string(i) +
                    "(ManagerPredictingLayer *manager_layer){\n\n");
      wfilec.append("\tprintf(\"layer " + std::to_string(i) + "\\n\");\n\n");

      wfilec.append("\tchar *image = manager_layer->input_image;\n");
      wfilec.append("\tinputImageFilenames.push_back(image);\n\n");

      //    wfilec.append("\tclock_t start_total = 0;\n");
      //    wfilec.append("\tstart_total = clock();\n");

      if (profileMode_ == 1) {
        wfilec.append("\tif(REPEAT > 0){\n");
        wfilec.append("\t\ttotalInftime = 0;\n");
        wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; "
                      "repeatCount++){\n");

        wfilec.append("\t\t\tclock_t start_" + std::to_string(i) + ";\n");
        wfilec.append("\t\t\tstart_" + std::to_string(i) + " = clock();\n\n");
      }

      wfilec.append("\tmutableWeightVarsAddr" + std::to_string(0) +
                    " = initMutableWeightVars(p" + std::to_string(0) +
                    "_config, \"" + inputName + "\");\n\n");

      wfilec.append("\tp" + std::to_string(i) + "(constantWeightVarsAddr" +
                    std::to_string(i) + ", mutableWeightVarsAddr" +
                    std::to_string(i) + ", activationsAddr" +
                    std::to_string(i) + ");\n\n");

      if (profileMode_ == 1) {
        wfilec.append("\t\t\tclock_t now_" + std::to_string(i) + ";\n");
        wfilec.append("\t\t\tnow_" + std::to_string(i) + " = clock();\n");
        wfilec.append("\t\t\tunsigned long inftime_" + std::to_string(i) +
                      " = (unsigned long)(now_" + std::to_string(i) +
                      " - start_" + std::to_string(i) + ");\n");
        // wfilec.append("\t\t\t\t\t(float)(t_now_" + std::to_string(i) +
        // ".millitm - t_start_" + std::to_string(i) + ".millitm);\n\n");

        wfilec.append("\t\t\tif(repeatCount != 0){\n");
        wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" +
                      std::to_string(i) + ";\n");
        wfilec.append("\t\t\t}\n");
        wfilec.append("\t\t}\n");
        wfilec.append("\t}\n\n");

        wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
        wfilec.append("\tdelay_time[" + std::to_string(i) +
                      "] = totalInftime/(REPEAT);\n");
        wfilec.append("\t}else {\n");
        wfilec.append("\t\tdelay_time[" + std::to_string(i) +
                      "] = totalInftime;\n");
        wfilec.append("\t}\n");
        wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i) +
                      " time] %lu microsec.\\n\", delay_time[" +
                      std::to_string(i) + "]);\n\n");
      }

      if (outputCount == 1) {

        wfilec.append("\tresultVar" + std::to_string(resultVarNum) +
                      " = getInferenceResultVar(p" + std::to_string(i) +
                      "_config, mutableWeightVarsAddr" + std::to_string(i) +
                      ", \"" + outputList[0] + "\");\n");

        wfilec.append("\tinputImageFilenames.pop_back();\n");

        resultCheckList[resultNum] = std::to_string(resultVarNum);
        resultList[resultNum] = outputList[0];
        resultNum++;

      } else {
        for (int outputNum = 0; outputNum < outputCount; outputNum++) {
          //          wfilec.append("\tchar *image =
          //          manager_layer->input_image;\n");
          //         wfilec.append("\tinputImageFilenames.push_back(image);\n");
          wfiled.append("float *resultVar" + std::to_string(resultVarNum) +
                        "_" + std::to_string(outputNum + 1) + ";\n");
          wfilec.append("\tresultVar" + std::to_string(resultVarNum) + "_" +
                        std::to_string(outputNum + 1) +
                        " = getInferenceResultVar(p" + std::to_string(i) +
                        "_config, mutableWeightVarsAddr" + std::to_string(i) +
                        ", \"" + outputList[outputNum] + "\");\n");

          //         wfilec.append("\tinputImageFilenames.pop_back();\n");

          resultCheckList[resultNum] = std::to_string(resultVarNum) + "_" +
                                       std::to_string(outputNum + 1);
          resultList[resultNum] = outputList[outputNum];
          resultNum++;
        }
      }

      wfilec.append("}\n\n");

      resultVarNum++;

    } else {
      wfilec.append("static void predicting_resnet18_layer" +
                    std::to_string(i) +
                    "(ManagerPredictingLayer *manager_layer){\n\n");
      wfilec.append("\tprintf(\"layer " + std::to_string(i) + "\\n\");\n\n");
      for (int checkCount = 0; checkCount < resultNum; checkCount++) {
        // if(!resultList[checkCount].compare(func->getNodes().front().getNthInput(0).getNode()->getName()))
        if (!resultList[checkCount].compare(inputList[0])) {
          flag = true;
          checkResultNum = checkCount;
        }
      }

      if (!flag) { // 기존 선언된 resultVar 중 중복된 inputName이 없다면, 해당
                   // 로직 필요 여부 체크
        if (resultNum != 0)
          checkResultNum =
              resultNum - 1; // 앞에서 선언한 resultVar 호출, inputName 중복이면
                             // checkCount resultVar 호출
      }
      wfilec.append("\tcopyMutableWeightVarsWithoutAlloc(p" +
                    std::to_string(i) + "_config, mutableWeightVarsAddr" +
                    std::to_string(i) + ", \"" + inputList[0] +
                    "\", resultVar" + resultCheckList[checkResultNum] + ");\n");
    }

    if (i != 0 && inputList->size() > 1) {
      for (int count = 1; count < inputList->size(); count++) {
        if (!inputList[count].compare(""))
          break;

        for (int checkCount = 0; checkCount < resultNum; checkCount++) {
          if (!resultList[checkCount].compare(inputList[count])) {
            flag = true;
            checkResultNum = checkCount;
          }
        }

        if (!flag) { // 기존 선언된 resultVar 중 중복된 inputName이 없다면, 해당
                     // 로직 필요 여부 체크
          if (resultNum != 0)
            checkResultNum =
                resultNum - 1; // 앞에서 선언한 resultVar 호출, inputName
                               // 중복이면 checkCount resultVar 호출
        }

        flag = true;
        for (int checkCount = 0; checkCount < resultNum; checkCount++) {
          if ((checkCount != count) &&
              (!inputList[checkCount].compare(inputList[count]))) {
            flag = false;
          }
        }
        if (flag) {
          wfilec.append("\tcopyMutableWeightVarsWithoutAlloc(p" +
                        std::to_string(i) + "_config, mutableWeightVarsAddr" +
                        std::to_string(i) + ", \"" + inputList[count] +
                        "\", resultVar" + resultCheckList[checkResultNum] +
                        ");\n");
          if (i != (partitionNum - 2)) {
            wfilee.append("PARTITION LAYER NUMBER:" + std::to_string(i) +
                          "  RETURN VALUE:resultVar" + std::to_string(i) +
                          "  OUTPUT FUNCTION:" + outputList[0] + ";\n");
            os_partition_flag = 1;
            if (clientPartitionFlag == 0) {
              clientPartitionNumber = i;
              clientPartitionFlag = 1;
            }
          }
        }
        flag = false;
      }
      wfilec.append("\n");
    }

    if (parallerPartitions.size() > 0) {
      for (int partitionCount = 0; partitionCount < parallerPartitions.size();
           partitionCount++) {
        if (!parallerPartitions[partitionCount].compare(
                "p" + std::to_string(i - 1))) {
          wfilec.append("#ifdef THREAD\n");
          if (profileMode_ == 1) {
            wfilec.append("\tif(REPEAT > 0){\n");
            wfilec.append("\t\ttotalInftime = 0;\n");
            wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; "
                          "repeatCount++){\n");

            wfilec.append("\t\t\tclock_t start_" + std::to_string(i - 1) +
                          ";\n");
            wfilec.append("\t\t\tstart_" + std::to_string(i - 1) +
                          " = clock();\n\n");
          }
          wfilec.append("\tstd::thread t" + std::to_string(i - 1) + "(p" +
                        std::to_string(i - 1) + ", constantWeightVarsAddr" +
                        std::to_string(i - 1) + ", mutableWeightVarsAddr" +
                        std::to_string(i - 1) + ", activationsAddr" +
                        std::to_string(i - 1) + ");\n");
          wfilec.append("\tstd::thread t" + std::to_string(i) + "(p" +
                        std::to_string(i) + ", constantWeightVarsAddr" +
                        std::to_string(i) + ", mutableWeightVarsAddr" +
                        std::to_string(i) + ", activationsAddr" +
                        std::to_string(i) + ");\n");
          wfilec.append("\tt" + std::to_string(i - 1) + ".join();\n");
          wfilec.append("\tt" + std::to_string(i) + ".join();\n");
          if (profileMode_ == 1) {
            wfilec.append("\t\t\tclock_t now_" + std::to_string(i - 1) + ";\n");
            wfilec.append("\t\t\tnow_" + std::to_string(i - 1) +
                          " = clock();\n");
            wfilec.append("\t\t\tunsigned long inftime_" +
                          std::to_string(i - 1) + " = (unsigned long)(now_" +
                          std::to_string(i) + " - start_" +
                          std::to_string(i - 1) + ");\n");

            wfilec.append("\t\t\tif(repeatCount != 0){\\n");
            wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" +
                          std::to_string(i - 1) + ";\n");
            wfilec.append("\t\t\t}\n");
            wfilec.append("\t\t}\n");
            wfilec.append("\t}\n\n");

            wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i - 1) +
                          "] = totalInftime/(REPEAT);\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i) +
                          "] = totalInftime/(REPEAT);\n");
            wfilec.append("\t}else {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i - 1) +
                          "] = totalInftime;\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i) +
                          "] = totalInftime;\n");
            wfilec.append("\t}\n");
            wfilec.append("\tprintf(\"\\n====> [Inference_" +
                          std::to_string(i) +
                          " time] %lu microsec.\\n\", delay_time[" +
                          std::to_string(i - 1) + "]);\n\n");
            //            wfilec.append("\tprintf(\"\\n====> [Inference_" +
            //            std::to_string(i) + " time] %.4lf ms.\\n\",
            //            delay_time[" + std::to_string(i - 1) + "]);\n\n");
          }
          wfilec.append("#else\n");
          if (profileMode_ == 1) {
            wfilec.append("\tif(REPEAT > 0){\n");
            wfilec.append("\t\ttotalInftime = 0;\n");
            wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; "
                          "repeatCount++){\n");

            wfilec.append("\t\t\tclock_t start_" + std::to_string(i - 1) +
                          ";\n");
            wfilec.append("\t\t\tstart_" + std::to_string(i - 1) +
                          " = clock();\n\n");
          }
          wfilec.append("\tp" + std::to_string(i - 1) +
                        "(constantWeightVarsAddr" + std::to_string(i - 1) +
                        ", mutableWeightVarsAddr" + std::to_string(i - 1) +
                        ", activationsAddr" + std::to_string(i - 1) + ");\n");
          if (profileMode_ == 1) {
            wfilec.append("\t\t\tclock_t now_" + std::to_string(i - 1) + ";\n");
            wfilec.append("\t\t\tnow_" + std::to_string(i - 1) +
                          " = clock();\n");
            wfilec.append("\t\t\tunsigned long inftime_" +
                          std::to_string(i - 1) + " = (unsigned long)(now_" +
                          std::to_string(i) + " - start_" +
                          std::to_string(i - 1) + ");\n");

            wfilec.append("\t\t\tif(repeatCount != 0){\\n");
            wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" +
                          std::to_string(i - 1) + ";\n");
            wfilec.append("\t\t\t}\n");
            wfilec.append("\t\t}\n");
            wfilec.append("\t}\n\n");

            wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i - 1) +
                          "] = totalInftime/(REPEAT);\n");
            wfilec.append("\t}else {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i - 1) +
                          "] = totalInftime;\n");
            wfilec.append("\t}\n");
            wfilec.append("\tprintf(\"\\n====> [Inference_" +
                          std::to_string(i - 1) +
                          " time] %lu microsec.\\n\", delay_time[" +
                          std::to_string(i - 1) + "]);\n\n");
            //            wfilec.append("\tprintf(\"\\n====> [Inference_" +
            //            std::to_string(i - 1) + " time] %.4lf ms.\\n\",
            //            delay_time[" + std::to_string(i - 1) + "]);\n\n");
          }
          if (profileMode_ == 1) {
            wfilec.append("\tif(REPEAT > 0){\n");
            wfilec.append("\t\ttotalInftime = 0;\n");
            wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; "
                          "repeatCount++){\n");

            wfilec.append("\t\t\tclock_t start_" + std::to_string(i) + ";\n");
            wfilec.append("\t\t\tstart_" + std::to_string(i) +
                          " = clock();\n\n");
          }
          wfilec.append("\tp" + std::to_string(i) + "(constantWeightVarsAddr" +
                        std::to_string(i) + ", mutableWeightVarsAddr" +
                        std::to_string(i) + ", activationsAddr" +
                        std::to_string(i) + ");\n");
          if (profileMode_ == 1) {
            wfilec.append("\t\t\tclock_t now_" + std::to_string(i) + ";\n");
            wfilec.append("\t\t\tnow_" + std::to_string(i) + " = clock();\n");
            wfilec.append("\t\t\tunsigned long inftime_" + std::to_string(i) +
                          " = (unsigned long)(now_" + std::to_string(i) +
                          " - start_" + std::to_string(i - 1) + ");\n");

            wfilec.append("\t\t\tif(repeatCount != 0){\\n");
            wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" +
                          std::to_string(i) + ";\n");
            wfilec.append("\t\t\t}\n");
            wfilec.append("\t\t}\n");
            wfilec.append("\t}\n\n");

            wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i) +
                          "] = totalInftime/(REPEAT);\n");
            wfilec.append("\t}else {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i) +
                          "] = totalInftime;\n");
            wfilec.append("\t}\n");
            wfilec.append("\tprintf(\"\\n====> [Inference_" +
                          std::to_string(i) +
                          " time] %lu microsec.\\n\", delay_time[" +
                          std::to_string(i) + "]);\n\n");
          }
          wfilec.append("#endif\n\n");
        }
      }
    }

    if (i != 0) {
      // wfilec.append("\t// Perform the computation.\n");

      bool parallerFlag = true;
      if (parallerPartitions.size() > 0) {
        for (int partitionCount = 0; partitionCount < parallerPartitions.size();
             partitionCount++) {
          if (!parallerPartitions[partitionCount].compare(
                  "p" + std::to_string(i - 1))) {
            parallerFlag = false;
          }
          if (!parallerPartitions[partitionCount].compare("p" +
                                                          std::to_string(i))) {
            parallerFlag = false;
          }
        }
      }

      if (parallerFlag) {
        if (profileMode_ == 1) {
          wfilec.append("\tif(REPEAT > 0){\n");
          wfilec.append("\t\ttotalInftime = 0;\n");
          wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; "
                        "repeatCount++){\n");

          wfilec.append("\t\t\tclock_t start_" + std::to_string(i) + ";\n");
          wfilec.append("\t\t\tstart_" + std::to_string(i) + " = clock();\n\n");
        }
        wfilec.append("\tp" + std::to_string(i) + "(constantWeightVarsAddr" +
                      std::to_string(i) + ", mutableWeightVarsAddr" +
                      std::to_string(i) + ", activationsAddr" +
                      std::to_string(i) + ");\n\n");
        if (profileMode_ == 1) {
          wfilec.append("\t\t\tclock_t now_" + std::to_string(i) + ";\n");
          wfilec.append("\t\t\tnow_" + std::to_string(i) + " = clock();\n");
          wfilec.append("\t\t\tunsigned long inftime_" + std::to_string(i) +
                        " = (unsigned long)(now_" + std::to_string(i) +
                        " - start_" + std::to_string(i) + ");\n");

          //          wfilec.append("\t\t\ttimeb t_now_" + std::to_string(i) +
          //          ";\n"); wfilec.append("\t\t\tftime(&t_now_" +
          //          std::to_string(i) + ");\n"); wfilec.append("\t\t\tfloat
          //          inftime_" + std::to_string(i) + " = (float)(t_now_" +
          //          std::to_string(i) + ".time - t_start_" + std::to_string(i)
          //          + ".time) * 1000.0f +\n");
          //          wfilec.append("\t\t\t\t\t(float)(t_now_" +
          //          std::to_string(i) + ".millitm - t_start_" +
          //          std::to_string(i) + ".millitm);\n\n");

          // wfilec.append("\tdelay_time[" + std::to_string(i) + "] = inftime_"
          // + std::to_string(i) + ";\n"); wfilec.append("\tprintf(\"\\n====>
          // [Inference_" + std::to_string(i) + " time] %.4lf ms.\\n\",
          // inftime_"
          // + std::to_string(i) + ");\n\n");

          wfilec.append("\t\t\tif(repeatCount != 0){\n");
          wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" +
                        std::to_string(i) + ";\n");
          wfilec.append("\t\t\t}\n");
          wfilec.append("\t\t}\n");
          wfilec.append("\t}\n\n");

          wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
          wfilec.append("\tdelay_time[" + std::to_string(i) +
                        "] = totalInftime/(REPEAT);\n");
          wfilec.append("\t}else {\n");
          wfilec.append("\t\tdelay_time[" + std::to_string(i) +
                        "] = totalInftime;\n");
          wfilec.append("\t}\n");
          wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i) +
                        " time] %lu microsec.\\n\", delay_time[" +
                        std::to_string(i) + "]);\n\n");
        }
      }

      if (i == partitionNum - 2) {

        //        wfilec.append("\tclock_t now_total;\n");
        //        wfilec.append("\tnow_total = clock();\n");
        //        wfilec.append("\tunsigned long inftime_total = (unsigned
        //        long)(now_total - start_total);\n");
        //        wfilec.append("\tprintf(\"\\n====> [Total inference time] %lu
        //        microsec.\\n\", inftime_total);\n\n");

        wfilec.append("\t// Report the results.\n");
        wfilec.append("\tprintf(\"====== final result of this neural net "
                      "========\\n\");\n");

        if (os_partition_main_type == 0) {
          wfilec.append("\tdumpInferenceResults(p" + std::to_string(i) +
                        "_config, mutableWeightVarsAddr" + std::to_string(i) +
                        ", \"" + outputName + "\");\n\n");

          wfilec.append("}\n\n");
        } else if (os_partition_main_type == 1) {

          wfilec.append("\tint resultLabel = dumpInferenceResults(p" +
                        std::to_string(i) + "_config, mutableWeightVarsAddr" +
                        std::to_string(i) + ", \"" + outputName + "\");\n\n");

          wfilec.append("\tmanager_layer->returnLabel = resultLabel;\n\n");
          wfilec.append("}\n\n");
        }

      } else {
        bool flag = true;
        for (int checkCount = 0; checkCount < resultNum; checkCount++) {
          // outputName 중복 여부 체크
          if (!resultList[checkCount].compare(outputName))
            flag = false;
        }

        // output 개수에 따른 resultVar값 선언 부분 수정 필요
        if (flag) // 중복된 inputName이 없다면(전 파티션 outputName들 비교)
        {
          if (outputCount == 1) {
            wfilec.append("\tresultVar" + std::to_string(resultVarNum) +
                          " = getInferenceResultVar(p" + std::to_string(i) +
                          "_config, mutableWeightVarsAddr" + std::to_string(i) +
                          ", \"" + outputList[0] + "\");\n\n");

            if (os_partition_flag == 1) {
              wfilec.append("\tmanager_layer->returnVal = resultVar" +
                            std::to_string(i) + ";\n");
              wfilec.append("\tmanager_layer->returnValSize = p" +
                            std::to_string(i) +
                            "_config.symbolTable[2].size * 4;\n\n");
              //                wfilec.append("\tprintf(\" name = %s \\n\", p" +
              //                std::to_string(i)
              //                +"_config.symbolTable[2].name);\n");
              //                wfilec.append("\tprintf(\" size = %d \\n\",
              //                manager_layer->returnValSize);\n");
              os_partition_flag = 0;
            }

            //            wfilec.append("\tmanager_layer->freeVarAddrList[manager_layer->count_freeVarAddrList++]
            //            = mutableWeightVarsAddr" + std::to_string(i) + ";\n");

            resultCheckList[resultNum] = std::to_string(resultVarNum);
            resultList[resultNum] = outputList[0];
            resultNum++;
          } else {
            for (int outputNum = 0; outputNum < outputCount; outputNum++) {
              wfiled.append("float *resultVar" + std::to_string(resultVarNum) +
                            "_" + std::to_string(outputNum + 1) + ";\n");

              wfilec.append("\tresultVar" + std::to_string(resultVarNum) + "_" +
                            std::to_string(outputNum + 1) +
                            " = getInferenceResultVar(p" + std::to_string(i) +
                            "_config, mutableWeightVarsAddr" +
                            std::to_string(i) + ", \"" + outputList[outputNum] +
                            "\");\n");

              // std::cout << "check : " << resultVarNum << ", outputNum : " <<
              // outputNum << std::endl; std::cout << "check 1 : " <<
              // std::to_string(resultVarNum) + "_" + std::to_string(outputNum +
              // 1) << std::endl;
              resultCheckList[resultNum] = std::to_string(resultVarNum) + "_" +
                                           std::to_string(outputNum + 1);
              resultList[resultNum] = outputList[outputNum];
              resultNum++;
            }
          }

          wfilec.append("}\n\n");
          resultVarNum++;
        }
      }
    }
  }

  // set_input_layer_data
  wfilec.append(
      "void set_input_layer_data(ManagerPredictingLayer *manager_layer){\n\n");

  for (int i = 1; i < partitionNum - 1; i++) {
    wfilec.append("\tif (manager_layer->start_layer == " + std::to_string(i) +
                  ") {\n");
    wfilec.append("\tresultVar" + std::to_string(i - 1) +
                  " = manager_layer->input_layer_data;\n\n");
    wfilec.append("\t}\n\n");
  }

  wfilec.append("}\n\n");

  // prediction_layer_manager_create
  wfilec.append("ManagerPredictingLayer * prediction_layer_manager_create(char "
                "*input_image, float *input_layer_data, int start_layer, int "
                "end_layer) {\n\n");
  wfilec.append("\tManagerPredictingLayer *manager_layer;\n\n");
  wfilec.append("\tprintf(\"prediction_layer_create\\n\");\n");
  wfilec.append("\tmanager_layer = (ManagerPredictingLayer *) "
                "malloc(sizeof(ManagerPredictingLayer));\n\n");
  wfilec.append("\tmanager_layer->start_layer = start_layer;\n");
  wfilec.append("\tmanager_layer->end_layer = end_layer;\n\n");
  //    wfilec.append("\tmanager_layer->count_freeVarAddrList =  0;\n\n");
  wfilec.append("\tif (manager_layer->start_layer == 0) {\n\n");
  wfilec.append("\t\tmanager_layer->input_image = input_image;\n\n");
  wfilec.append("\t}\n\n");
  wfilec.append("\telse {\n\n");
  wfilec.append("\t\tmanager_layer->input_layer_data = input_layer_data;\n");
  wfilec.append("\t\tset_input_layer_data(manager_layer);\n\n");
  wfilec.append("\t}\n\n");

  for (int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("\tmanager_layer->predicting_layers[" + std::to_string(i) +
                  "] = predicting_resnet18_layer" + std::to_string(i) + ";\n");
  }

  wfilec.append("\treturn manager_layer;\n");
  wfilec.append("}\n");
  wfilec.append("\n");

  // prediction_layer_exit
  wfilec.append("void prediction_layer_exit(ManagerPredictingLayer "
                "*manager_layer) {\n\n");
  wfilec.append("\tprintf(\"prediction_layer_exit\\n\");\n\n");
  wfilec.append(
      "\t//for (int i=0; i<manager_layer->count_freeVarAddrList; i++)\n");
  wfilec.append("\t\t//free(manager_layer->freeVarAddrList[i]);\n\n");
  wfilec.append("\tfree(manager_layer);\n\n");
  wfilec.append("}\n\n");

  // extern variable
  if (os_partition_main_type == 0) {
    wfilec.append("float *resultVar;\n");
    wfilec.append("int resultLabel;\n");
    wfilec.append("int resultSize;\n\n");
  } else if (os_partition_main_type == 1) {
    wfilec.append("extern float *resultVar;\n");
    wfilec.append("extern int resultLabel;\n");
    wfilec.append("extern int resultSize;\n\n");
  }
  // running_resnet18
  wfilec.append("void running_resnet18(char *input_image, float "
                "*input_layer_data, int start_layer, int end_layer) {\n\n");
  wfilec.append("\tint i;\n");
  wfilec.append("\tManagerPredictingLayer *manager_layer;\n\n");
  wfilec.append(
      "\tmanager_layer = prediction_layer_manager_create(input_image, "
      "input_layer_data, start_layer, end_layer);\n\n");
  wfilec.append("\tfor ( i = start_layer; i < end_layer + 1; i++) {\n\n");
  wfilec.append("\t\tmanager_layer->predicting_layers[i](manager_layer);\n\n");
  wfilec.append("\t}\n\n");
  wfilec.append("\tif (end_layer == " + std::to_string(partitionNum - 2) +
                ") {\n\n");
  wfilec.append("\t\tprintf(\"end layer  %d \\n\", end_layer);\n");
  wfilec.append("\t\tresultLabel = manager_layer->returnLabel;\n\n");
  wfilec.append("\t}\n");
  wfilec.append("\telse {\n\n");
  wfilec.append(
      "\t\tprintf(\"end layer size %d\\n\", manager_layer->returnValSize);\n");
  wfilec.append("\t\tresultVar = (float *)malloc(sizeof(float) * "
                "manager_layer->returnValSize);\n");
  wfilec.append("\t\tmemcpy(resultVar, manager_layer->returnVal, "
                "manager_layer->returnValSize);\n\n");
  wfilec.append("\t}\n\n");
  wfilec.append("\tresultSize = manager_layer->returnValSize;\n\n");
  wfilec.append("\tif (start_layer == 0) {\n\n");
  wfilec.append("\t\tfree(mutableWeightVarsAddr0);\n\n");
  wfilec.append("\t}\n\n");
  wfilec.append("\tprediction_layer_exit(manager_layer);\n");
  wfilec.append("}\n\n");

  // main
  if (os_partition_main_type != 1) {

    wfilec.append("int main(int argc, char **argv) {\n");
    wfilec.append("\tparseCommandLineOptions(argc, argv);\n\n");

    wfilec.append("\t//======== loading resnet18 =======//\n\n");
    wfilec.append("\tloading_resnet18();\n");

    wfilec.append(
        "\t//======== initialize mutableWeightVarsAddr =======//\n\n");

    wfilec.append("\n");
    wfilec.append("\tchar *input_image = \"cat_285.png\";\n");
    wfilec.append("\tfloat *input_layer_data = NULL;\n\n");
    //    for(int i = 0; i < partitionNum - 1; i++)
    //    {
    //        wfilec.append("\tpredicting_resnet18_layer" + std::to_string(i) +
    //        "(NULL);\n\n");
    //    }

    wfilec.append("\trunning_resnet18(input_image, input_layer_data, 0," +
                  std::to_string(partitionNum - 2) + ");\n");
    wfilec.append("\tunloading_resnet18();\n");
    wfilec.append("\treturn 0;\n");
    wfilec.append("}\n\n");
  }
  // extern "C"
  if (os_partition_main_type == 1)
    wfilec.append("} //extern \"c\"\n\n");
}

void NestPartitionerSchedule::generateUnloadingResnet18(
    string &wfilec, int partitionNum, const PartitionConfig &partitionConfig) {

  wfilec.append("void unloading_resnet18(void) {\n\n");

  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA")) {
      wfilec.append("\tp" + std::to_string(i) + "_destroy_module();\n");
    }
  }
  wfilec.append("\n");

  wfilec.append("\t// Free all resources.\n");

  //  const char *fileName;
  std::string filePath;

  for (int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("\tfree(activationsAddr" + std::to_string(i) + ");\n");
    wfilec.append("\tfree(constantWeightVarsAddr" + std::to_string(i) + ");\n");
    if (i != 0) {
      wfilec.append("\tfree(mutableWeightVarsAddr" + std::to_string(i) +
                    ");\n");
    }
    wfilec.append("\n");
  }

  bool vtaFlag = false;
  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if (vtaFlag) {
    wfilec.append("#ifdef VTAMAIN\n");
    wfilec.append("\tdestroyVTARuntime();\n");
    wfilec.append("#endif\n");
  }

  if (profileMode_ == 1) {
    wfilec.append("\tcreateYamlFile();\n");
  }

  //    wfilec.append("\treturn 0;\n");
  wfilec.append("}\n\n");
}

void NestPartitionerSchedule::generateLoadingResnet18(
    string &wfilec, int partitionNum, const PartitionConfig &partitionConfig) {
  cout << "generateLoadingResnet18: " << endl;

  wfilec.append("void loading_resnet18(void) {\n\n");

  bool vtaFlag = false;
  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if (vtaFlag) {
    wfilec.append("#ifdef VTAMAIN\n");
    wfilec.append("\tinitVTARuntime();\n");
    wfilec.append("#endif\n");
  }

  for (int i = 0; i < partitionNum - 1; i++) {

    wfilec.append("\tconstantWeightVarsAddr" + std::to_string(i) +
                  " = \n\t\tinitConstantWeights(\"p" + std::to_string(i) +
                  ".weights.bin\", p" + std::to_string(i) + "_config);\n");

    wfilec.append("\tactivationsAddr" + std::to_string(i) +
                  " = initActivations(p" + std::to_string(i) + "_config);\n\n");
  }

  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA")) {
      wfilec.append("\tp" + std::to_string(i) +
                    "_load_module(constantWeightVarsAddr" + std::to_string(i) +
                    ");\n");
    }
  }
  wfilec.append("\n");

  for (int i = 1; i < partitionNum - 1; i++) {
    wfilec.append("\tmutableWeightVarsAddr" + std::to_string(i) +
                  " =  allocateMutableWeightVars(p" + std::to_string(i) +
                  "_config);\n");
  }

  wfilec.append("}\n\n");
}

void NestPartitionerSchedule::generateLayerManager(
    string &wfilec, int partitionNum, const PartitionConfig &partitionConfig) {
  cout << "generateLayerManager: " << endl;

  wfilec.append("typedef struct manager_predicting_layer {\n\n"
                "\tchar * input_image;\n"
                "\tfloat * input_layer_data;\n"
                "\tint returnLabel;\n"
                "\tint returnValSize;\n"
                "\tfloat *returnVal;\n\n"
                "\tint start_layer;\n"
                "\tint end_layer;\n\n");
  //                        "\tuint8_t *freeVarAddrList[100];\n"
  //                        "\tint count_freeVarAddrList = 0;\n\n");

  wfilec.append("\tvoid (*predicting_layers[" +
                std::to_string(partitionNum - 1) +
                "])(struct manager_predicting_layer *);\n\n"
                "} ManagerPredictingLayer;\n\n");
}

void NestPartitionerSchedule::generateParseLayerConfiguration(
    string &wfilec, int partitionNum, const PartitionConfig &partitionConfig) {
  cout << "generateParseLayerConfiguration: " << endl;

  wfilec.append("int return_layer_sizes[" + std::to_string(partitionNum - 1) +
                "];\n\n");

  wfilec.append("int parse_layer_configuration( int *layer_sizes) {\n\n"
                "\tchar strline[100];\n\n"
                "\tFILE *fp = fopen(\"layer_configuration.txt\", \"r\");\n\n"
                "\tif(fp == NULL){\n\n"
                "\tputs(\"FAIL\");\n"
                "\treturn -1;\n\n"
                "\t}\n\n"
                "\twhile(fgets(strline, sizeof(strline), fp)) {\n\n"
                "\t\tchar *ptr = strtok(strline, \":\");\n"
                "\t\tint layer_num = atoi(ptr);\n"
                "\t\tptr = strtok(NULL,\"\\n\");\n"
                "\t\tint layer_size = atoi(ptr);\n\n"
                "\t\tprintf(\"layer_num %d, layer_size %d \\n\", layer_num, "
                "layer_size);\n"
                "\t\tlayer_sizes[layer_num] = layer_size;\n\n"
                "\t}\n\n"
                "\tfclose(fp);\n"
                "\treturn 0;\n\n"
                "} \n\n");
}

void NestPartitionerSchedule::generateGlobalVariables(
    string &wfilec, int partitionNum, const PartitionConfig &partitionConfig) {
  cout << "generateGlobalVariables: " << endl;
  if (os_partition_main_type == 1) {
    wfilec.append("extern \"C\" {\n");
  }

  for (int i = 0; i < partitionNum - 1; i++) {

    wfilec.append("uint8_t *constantWeightVarsAddr" + std::to_string(i) +
                  ";\n");

    wfilec.append("uint8_t *mutableWeightVarsAddr" + std::to_string(i) + ";\n");

    wfilec.append("uint8_t *activationsAddr" + std::to_string(i) + ";\n");

    wfilec.append("float *resultVar" + std::to_string(i) + ";\n\n");
  }
}

void NestPartitionerSchedule::generateCodeFromModels(
    std::size_t partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig, std::string inputPartitionName) {
  std::cout << "getnerateCodeFromModels Start" << std::endl;

  //  std::cout << "out dir = " << outputDir_ << std::endl;

  string outfile_c = outputDir_ + "/main.cpp"; //
  string make_outfile_c = outputDir_ + "/CMakeLists.txt";
  string make_outfileos_c = outputDir_ + "/Makefile";
  string make_os_partition_layer_info_outfile_c =
      outputDir_ + "/os_partition_layer_info.txt"; //
  string make_os_partition_nnobject_outfile_c =
      outputDir_ + "/nn_object/nn_object_23.c"; //
  string make_os_partition_nnobject_outdir = outputDir_ + "/nn_object";

  if (!llvm::sys::fs::is_directory(outputDir_)) {
    llvm::sys::fs::create_directory(outputDir_);
  }
  ofstream writeFileC;
  writeFileC.open(outfile_c.data());

  ofstream writeOsPartitionLayerC;
  writeOsPartitionLayerC.open(make_os_partition_layer_info_outfile_c.data());

  if (writeFileC.is_open() && writeOsPartitionLayerC.is_open()) {
    string declare;
    string layerManager;
    string globalVariable;
    string loadingResnet;
    //    string parseLayerConfiguration;

    string initialize;
    string inference_main;
    string osPartitionLayer;
    string UnloadingResnet;
    //    string memfree;
    string yamlFile;

    generateDeclaration(declare, partitionNum, partitionConfig);
    generateGlobalVariables(globalVariable, partitionNum, partitionConfig);
    generateLayerManager(layerManager, partitionNum, partitionConfig);
    // generateParseLayerConfiguration(parseLayerConfiguration, partitionNum,
    // partitionConfig);
    generateLoadingResnet18(loadingResnet, partitionNum, partitionConfig);

    generateMain(inference_main, globalVariable, osPartitionLayer, partitionNum,
                 funcList, partitionConfig);
    generateUnloadingResnet18(UnloadingResnet, partitionNum, partitionConfig);

    // generateFree(memfree, partitionNum, funcList, partitionConfig);

    writeFileC << declare;
    writeFileC << globalVariable;
    writeFileC << layerManager;
    //    writeFileC << parseLayerConfiguration;
    writeFileC << loadingResnet;

    writeFileC << initialize;
    // writeFileC << inference;
    writeFileC << UnloadingResnet;

    writeFileC << inference_main;
    writeOsPartitionLayerC << osPartitionLayer;

    //    writeFileC << memfree;

    if (profileMode_ == 1) {
      generateYamlFile(yamlFile, partitionNum, funcList, partitionConfig,
                       inputPartitionName);
      writeFileC << yamlFile;
    }

    writeFileC.close();
    writeOsPartitionLayerC.close();
  }

  if (partitionExeMode_ == 1) {
    generateMainforEachPartition(partitionNum, partitionConfig);
  }

  // CMakeLists.txt
  ofstream writeMakeFileC;
  writeMakeFileC.open(make_outfile_c.data());
  if (writeMakeFileC.is_open()) {
    string cmakeFile;

    generateCMakeListsFile(cmakeFile, partitionNum, funcList, partitionConfig);
    writeMakeFileC << cmakeFile;
    writeMakeFileC.close();
  }

  // NEST-OS nn_object23.c
  if (os_partition_main_type == 1) {
    std::cout << "os_partition_main_type Start" << std::endl;

    mkdir(make_os_partition_nnobject_outdir.c_str(), 0766);

    ofstream writeNnobjectC;
    writeNnobjectC.open(make_os_partition_nnobject_outfile_c.data());

    if (writeNnobjectC.is_open()) {

      string nnobjectOs;

      generateNnobjectFile(nnobjectOs, partitionNum, funcList, partitionConfig);
      writeNnobjectC << nnobjectOs;
      writeNnobjectC.close();
    }
  }

  // NEST-OS local/nestos Makefile
  ofstream writeMakeFileOsC;
  writeMakeFileOsC.open(make_outfileos_c.data());
  if (writeMakeFileOsC.is_open()) {
    string makeFileOs;

    if (os_partition_main_type == 0) {
      generateMakeFileLocal(makeFileOs, partitionNum, funcList,
                            partitionConfig);
    } else if (os_partition_main_type == 1) {
      generateMakeFileNestOs(makeFileOs, partitionNum, funcList,
                             partitionConfig);
    }
    writeMakeFileOsC << makeFileOs;
    writeMakeFileOsC.close();
  }

  std::cout << "main outfile position : " << outfile_c << std::endl;
  std::cout << "CMakeLists outfile position : " << make_outfile_c << std::endl;

  // NEST-OS client code generation

  if (os_partition_main_type == 1) {
    std::cout << "os_partition_main_type Start" << std::endl;

    string make_os_client_outdir = outputDir_ + "/client";
    string make_os_client_nnload_outfile_c = outputDir_ + "/client/nnload.c"; //
    string make_os_client_resnet18_outfile_c =
        outputDir_ + "/client/resnet18.c"; //

    mkdir(make_os_client_outdir.c_str(), 0766);

    ofstream writeNnloadC;

    writeNnloadC.open(make_os_client_nnload_outfile_c.data());

    if (writeNnloadC.is_open()) {

      string nnLoadOs;

      generateNnloadFile(nnLoadOs, partitionNum, funcList, partitionConfig);
      writeNnloadC << nnLoadOs;
      writeNnloadC.close();
    }

    ofstream writeResnetC;

    writeResnetC.open(make_os_client_resnet18_outfile_c.data());

    if (writeResnetC.is_open()) {

      string nnResnetOs;

      generateResnetFile(nnResnetOs, partitionNum, funcList, partitionConfig);
      writeResnetC << nnResnetOs;
      writeResnetC.close();
    }
  }
}

void NestPartitionerSchedule::generateNnloadFile(
    string &wfilec, std::size_t partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig) {

  wfilec.append("#include <stdio.h>\n");
  wfilec.append("#include <stdlib.h>\n");
  wfilec.append("#include <string.h>\n");
  wfilec.append("#include \"error.h\"\n");
  wfilec.append("#include \"loader.h\"\n");
  wfilec.append("#include \"support_model.h\"\n");
  wfilec.append("#include \"config.h\"\n\n");
  wfilec.append("int main(int argc, char *argv[])\n");
  wfilec.append("{\n");
  wfilec.append("\tLoader *loader=loader_create();\n\n");
  wfilec.append("\tif (argc == 1)\n");
  wfilec.append("\t{\n");
  wfilec.append(
      "\t\tloader->load_partition_range(loader, MODEL_PART_DYNAMIC_NAME, " +
      std::to_string(0) + ", " + std::to_string(partitionNum - 2) +
      ", 0); // install into NPU OS #0\n");
  wfilec.append("\t\tprintf(\"MODEL NAME %d\\n\", MODEL_PART_DYNAMIC_NAME);\n");
  wfilec.append("\t}\n");
  wfilec.append("\telse if (argc == 2)\n");
  wfilec.append("\t{\n");
  wfilec.append("\t\tif (!strcmp(argv[1], \"-undo\"))\n");
  wfilec.append("\t\t{\n");
  wfilec.append(
      "\t\t\tloader->unload_partition_range(loader, MODEL_PART_DYNAMIC_NAME, " +
      std::to_string(0) + ", " + std::to_string(partitionNum - 2) +
      ", 0); // uninstall into NPU OS #0\n");
  wfilec.append("\t\t}\n");
  wfilec.append("\t}\n");
  wfilec.append("}\n");
}

void NestPartitionerSchedule::generateResnetFile(
    string &wfilec, std::size_t partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig) {

  wfilec.append("#include <stdio.h>\n");
  wfilec.append("#include <stdlib.h>\n");
  wfilec.append("#include <string.h>\n");
  wfilec.append("#include <dirent.h>\n");
  wfilec.append("#include \"ntask.h\"\n");
  wfilec.append("#include \"nprocess.h\"\n");
  wfilec.append("#include \"demo/image.h\"\n");
  wfilec.append("#include \"util_time.h\"\n");
  wfilec.append("#include \"config.h\"\n\n");
  wfilec.append("#define MAX 1000\n");
  wfilec.append("//#define TRANSFER_DATA_SIZE 401408\n\n");
  wfilec.append("#ifdef OPENCV\n\n");
  wfilec.append("static volatile int demo_done = 0;\n");
  wfilec.append("Image resized_im;\n");
  wfilec.append("void *wsmem;\n");
  wfilec.append("static char *image_path;\n\n");
  wfilec.append("void get_labels(char *names[])\n");
  wfilec.append("{\n");
  wfilec.append("\tchar *label = NULL;\n\n");
  wfilec.append("\tFILE *fp;\n");
  wfilec.append("\tchar *path = \"labels.txt\";\n");
  wfilec.append("\tint i = 0;\n\n");
  wfilec.append("\tfp = fopen(path, \"r\");\n\n");
  wfilec.append("\tif (fp == NULL)\n");
  wfilec.append("\t{\n");
  wfilec.append("\t\tfprintf(stderr, \"File Open Error!\\n\");\n");
  wfilec.append("\t\texit(1);\n");
  wfilec.append("\t}\n\n");
  wfilec.append("\twhile (!feof(fp))\n");
  wfilec.append("\t{\n");
  wfilec.append("\t\tlabel = (char*)malloc(sizeof(char) * MAX);\n");
  wfilec.append("\t\tfgets(label, MAX, fp);\n");
  wfilec.append("\t\tnames[i] = label;\n");
  wfilec.append("\t\ti++;\n");
  wfilec.append("\t}\n\n");
  wfilec.append("\tif (label)\n");
  wfilec.append("\t\tfree(label);\n");
  wfilec.append("\tfclose(fp);\n");
  wfilec.append("}\n\n");

  wfilec.append("void print_label(char *names[], int num)\n");
  wfilec.append("{\n");
  wfilec.append("\tprintf(\"label : %s\\n\", names[num]);\n");
  wfilec.append("}\n\n");

  wfilec.append("static void ntask_input_handler1(Ntask *nt)\n");
  wfilec.append("{\n\n");
  wfilec.append("\tnt->set_file_input(nt, image_path);\n\n");
  wfilec.append("}\n\n");

  wfilec.append("int output_offset, output_size;\n");
  wfilec.append("static void ntask_output_handler1(Ntask *nt)\n");
  wfilec.append("{\n");
  wfilec.append("\tnt->get_data_output(nt, &output_offset, &output_size);\n");
  wfilec.append("\tprintf(\"output data : offset = %d size = %d\\n\", "
                "output_offset, output_size);\n\n");
  wfilec.append("}\n\n");

  if (clientPartitionFlag == 1) {
    wfilec.append("static void ntask_input_handler2(Ntask *nt)\n");
    wfilec.append("{\n\n");
    wfilec.append("\tnt->set_data_input(nt, 0, output_size);\n\n");
    wfilec.append("}\n\n");

    wfilec.append("int output_offset2, output_size2;\n");
    wfilec.append("static void ntask_output_handler2(Ntask *nt)\n");
    wfilec.append("{\n\n");
    wfilec.append(
        "\tnt->get_data_output(nt, &output_offset2, &output_size2);\n");
    wfilec.append("\tprintf(\"output data : offset = %d size = %d\\n\", "
                  "output_offset2, output_size2);\n");
    wfilec.append("}\n\n");
  }
  wfilec.append("int main(int argc, char *argv[])\n");
  wfilec.append("{\n\n");
  wfilec.append("\tchar* name[1000];\n");
  wfilec.append("\tchar file_name[100][100];\n");
  wfilec.append("\tint i = 0;\n\n");
  wfilec.append("\tImage **alphabet = NULL;\n");
  wfilec.append("\tif (!alphabet)\n");
  wfilec.append("\t\talphabet = load_alphabet();\n\n");
  wfilec.append("\tDIR *dir;\n");
  wfilec.append("\tstruct dirent *ent;\n");
  wfilec.append("\tdir = opendir (\"100/\");\n\n");
  wfilec.append("\tif (dir != NULL) {\n");
  wfilec.append(
      "\t/* print all the files and directories within directory */\n");
  wfilec.append("\t\twhile ((ent = readdir (dir)) != NULL)\n");
  wfilec.append("\t{\n");
  wfilec.append("\t\tchar *s1 = \".\";\n");
  wfilec.append("\t\tchar *s2 = \"..\";\n");
  wfilec.append("\t\tchar str1[100] = \"100/\";\n");
  wfilec.append("\t\tchar* str2 = ent->d_name;\n\n");
  wfilec.append("\t\tif(!strcmp(str2, s1) || !strcmp(str2, s2))\n");
  wfilec.append("\t\t{\n");
  wfilec.append("\t\t\tcontinue;\n");
  wfilec.append("\t\t}\n\n");
  wfilec.append("\t\tstrcat(str1, str2);\n");
  wfilec.append("\t\tstrcpy(file_name[i], str1);\n");
  wfilec.append("\t\tprintf (\"%s\\n\",file_name[i]);\n");
  wfilec.append("\t\ti++;\n");
  wfilec.append("\t\t}\n");
  wfilec.append("\t\tclosedir (dir);\n");
  wfilec.append("\t}\n");
  wfilec.append("\telse {\n");
  wfilec.append("\t/* could not open directory */\n");
  wfilec.append("\t\tperror (\"\");\n");
  wfilec.append("\t\treturn EXIT_FAILURE;\n");
  wfilec.append("\t}\n\n");
  wfilec.append("\t// get labels\n");
  wfilec.append("\tget_labels(name);\n\n");

  if (clientPartitionFlag == 1) {
    wfilec.append("\tNprocess *np = nprocess_create(\"np\");\n\n");
    wfilec.append("\t// ntask create and do setting if necessary (affinity or "
                  "priority)\n");
    wfilec.append("\twsmem = malloc(64*1024*1024); // 64MB\n");
    wfilec.append("\tNtask *nt1 = ntask_partition_range_create(\"nt1\", "
                  "RESNET18_NPU_DYNAMIC, " +
                  std::to_string(0) + ", " +
                  std::to_string(clientPartitionNumber) +
                  ", ntask_input_handler1, ntask_output_handler1, wsmem);\n");
    wfilec.append("\tNtask *nt2 = ntask_partition_range_create(\"nt2\", "
                  "RESNET18_NPU_DYNAMIC, " +
                  std::to_string(clientPartitionNumber + 1) + ", " +
                  std::to_string(partitionNum - 2) +
                  ", ntask_input_handler2, ntask_output_handler2, wsmem);\n");
    wfilec.append("\tnp->add_ntask(np, nt1); // link: nt1->nt2\n");
    wfilec.append("\tnp->add_ntask(np, nt2); // link: nt1->nt2\n\n");
  } else {
    wfilec.append("\n");
    wfilec.append("\tNtask *nt1 = ntask_partition_range_create(\"nt1\", "
                  "RESNET18_NPU_DYNAMIC, " +
                  std::to_string(0) + ", " + std::to_string(partitionNum - 2) +
                  ", ntask_input_handler1, ntask_output_handler1, wsmem);\n");
  }
  wfilec.append("\t//nt->set_affinity(nt, MASK(1));\n");
  wfilec.append("\t//nt->set_priority(nt, 10);\n\n");
  wfilec.append("\tmake_window(\"predictions\", 448, 448, 0);\n\n");
  wfilec.append("\tImage label;\n");
  wfilec.append("loop:\n");
  wfilec.append("\t// infinite loop\n");
  wfilec.append("\tfor(int j = 0; j < 80; j++)\n");
  wfilec.append("\t{\n");
  wfilec.append("\t\timage_path = file_name[j];\n");
  wfilec.append("\t\tImage im = load_image_color(file_name[j], 224, 224);\n\n");
  wfilec.append("\t\tprintf(\"> prediction starts\\n\");\n");
  wfilec.append("\t\tdouble time=what_time_is_it_now();\n\n");
  wfilec.append("\t\t// run input_handler+prediction is made\n");

  if (clientPartitionFlag == 1) {
    wfilec.append("\t\tnp->run(np);\n\n");
    wfilec.append("\t\tint* num = "
                  "(int*)(nt2->get_memory_pointer(nt2)+output_offset2);\n");
  } else {

    wfilec.append("\t\tnt1->run(nt1);\n\n");
    wfilec.append("\t\tint* num = "
                  "(int*)(nt1->get_memory_pointer(nt1)+output_offset);\n\n");
  }

  wfilec.append("\t\tprintf(\"> prediction ends -> took %f seconds\\n\", "
                "what_time_is_it_now() - time);\n\n");
  wfilec.append("\t\tprint_label(name, *num);\n\n");
  wfilec.append("\t\tlabel = get_label(alphabet, name[*num], (im.h*.03));\n\n");
  wfilec.append("\t\tfloat rgb[3] = {1, 1, 1};\n");
  wfilec.append("\t\tdraw_label(im, 220, 0, label, rgb);\n");
  wfilec.append("\t\tint c = show_image(im, \"predictions\", 1);\n\n");
  wfilec.append("\t\tif (c != -1)\n");
  wfilec.append("\t\t{\n");
  wfilec.append("\t\t\tc = c%256;\n");
  wfilec.append("\t\t}\n");
  wfilec.append("\t\t//print_label(name, *num);\n");
  wfilec.append("\t\timage_free(im);\n");
  wfilec.append("\t\timage_free(label);\n\n");
  wfilec.append("\t\tif (c == 27)  // ESC key; terminal condition\n");
  wfilec.append("\t\t{\n");
  wfilec.append("\t\t\tprintf(\"ESC key pressed!\\n\");\n");
  wfilec.append("\t\t\tgoto out;\n");
  wfilec.append("\t\t}\n");
  wfilec.append("\t}\n");
  wfilec.append("\tgoto loop;\n");
  wfilec.append("out:\n");
  wfilec.append("\tfree(wsmem);\n\n");
  wfilec.append("#ifdef OPENCV\n");
  wfilec.append("#endif\n");

  if (clientPartitionFlag == 1) {
    wfilec.append("\tnp->exit(np);// ntask destroy\n");
  } else {
    wfilec.append("\tnt1->exit(nt1);\n");
  }

  wfilec.append("}\n");
  wfilec.append("#else\n");
  wfilec.append("int main(int argc, char *argv[])\n");
  wfilec.append("{\n");
  wfilec.append(
      "\tfprintf(stderr, \"Demo needs OpenCV for mp4 images.\\n\");\n");
  wfilec.append("}\n");
  wfilec.append("#endif\n");
}

void NestPartitionerSchedule::generateNnobjectFile(
    string &wfilec, std::size_t partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig) {

  wfilec.append("#include <stdio.h>\n");
  wfilec.append("#include <malloc.h>\n");
  wfilec.append("#include <stdlib.h>\n");
  wfilec.append("#include <pthread.h>\n");
  wfilec.append("#include <string.h>\n");
  wfilec.append("#include \"nn_object.h\"\n");
  wfilec.append("//#include \"neural_net.h\"\n");
  wfilec.append("#include \"error.h\"\n");
  wfilec.append("#include \"hwtype.h\"\n");
  wfilec.append("#include \"nn_object_header.h\"\n");
  wfilec.append("#include \"stdint.h\"\n");
  wfilec.append("#include \"demo/image.h\"\n\n");

  wfilec.append("float *resultVar;\n");
  wfilec.append("int resultLabel;\n");
  wfilec.append("int resultSize;\n\n");

  wfilec.append("static void load(NnObject *object, NetInfo *info)\n");
  wfilec.append("{\n");
  wfilec.append("\tloading_resnet18();\n");
  wfilec.append("\tobject->loaded = 1; // mark it as loaded\n");
  wfilec.append("\tprintf(\"NNID %d Kernel is loaded\\n\", object->nnid);\n");
  wfilec.append("}\n\n");

  wfilec.append("static void unload(NnObject *object, NetInfo *info)\n");
  wfilec.append("{\n");
  wfilec.append("\tif (object->loaded) {\n");
  wfilec.append("\t\tunloading_resnet18();\n");
  wfilec.append("\t\tobject->loaded = 0;\n");
  wfilec.append("\t}\n");
  wfilec.append("\telse {\n");
  wfilec.append("\t\tprintf(\"object is not loaded yet\\n\");\n");
  wfilec.append("\t}\n");
  wfilec.append("}\n\n");

  wfilec.append("extern void save_image_options(Image im, const char *name, "
                "IMTYPE f, int quality);\n");
  wfilec.append("static char *image_path;\n\n");

  wfilec.append("static void predict(NnObject *object, NetInfo *info)\n");
  wfilec.append("{\n");
  wfilec.append("\tchar * image = NULL;\n");
  wfilec.append("\tfloat * input_layer_data = NULL;\n\n");
  wfilec.append("\tint start_layer = info->part_num_start;\n");
  wfilec.append(
      "\tint end_layer = info->part_num_start + info->part_num_delta;\n\n");
  wfilec.append(
      "\tprintf(\" predict start %d end %d\\n\", start_layer, end_layer);\n\n");
  wfilec.append("\tif (info->part_num_start == 0) {\n");
  wfilec.append("\t\timage = image_path;\n");
  wfilec.append("\t\tinput_layer_data = NULL;\n");
  wfilec.append("\t}\n");
  wfilec.append("\telse {\n");
  wfilec.append("\t\tinput_layer_data = (float*) object->wsmem;\n");
  wfilec.append("\t\timage = NULL;\n");
  wfilec.append("\t}\n");
  wfilec.append("\t// else if \"middle layer\"\n\n");
  wfilec.append(
      "\trunning_resnet18(image, input_layer_data, start_layer, end_layer);\n");
  wfilec.append("}\n\n");

  wfilec.append(
      "static void set_input(NnObject *object, NetInfo *info, Input *input)\n");
  wfilec.append("{\n");
  wfilec.append("\tprintf(\"set_input start layer %d , end layer %d\\n\", "
                "info->part_num_start, info->part_num_start + "
                "info->part_num_delta);\n\n");
  wfilec.append("\tif (info->part_num_start  == 0) {\n");
  wfilec.append("\t\timage_path = input->filename;\n");
  wfilec.append("\t} else {\n");
  wfilec.append("\t\tprintf(\"start middle layer prediction\\n\");\n");
  wfilec.append("\t}\n\n");
  wfilec.append("}\n\n");

  wfilec.append("static void get_output(NnObject *object, NetInfo *info, "
                "Output *output)\n");
  wfilec.append("{\n\n");
  wfilec.append(
      "\tint end_layer = info->part_num_start + info->part_num_delta;\n\n");
  wfilec.append("\tif (end_layer == " + std::to_string(partitionNum - 2) +
                ") {\n\n");
  wfilec.append("\t\toutput->data_offset = 0;\n");
  wfilec.append("\t\toutput->data_size = 4;\n");
  wfilec.append("\t\tprintf(\"jun result : %d output size %d\\n\", "
                "resultLabel, output->data_size);\n");
  wfilec.append("\t\tmemcpy(object->wsmem + output->data_offset, &resultLabel, "
                "output->data_size);\n");
  wfilec.append("\t}\n");
  wfilec.append("\telse {\n");
  wfilec.append("\t\toutput->data_offset = 0;\n");
  wfilec.append("\t\toutput->data_size = resultSize;\n");
  wfilec.append("\t\tmemcpy(object->wsmem + output->data_offset, resultVar, "
                "output->data_size);\n\n");
  wfilec.append("\tfree(resultVar);\n");
  wfilec.append("\t}\n\n");
  wfilec.append("}\n\n");

  wfilec.append("NnObject nn_object_23 = {23, 0, 0, {0, 0, 0, 0}, 0, load, "
                "unload, predict, set_input, get_output};\n");
}

/*
void NestPartitionerSchedule::generateNnobjectFile(string& wfilec, std::size_t
partitionNum, std::vector<Function *> funcList, const PartitionConfig
&partitionConfig) {

    wfilec.append("#include <stdio.h>\n");
    wfilec.append("#include <malloc.h>\n");
    wfilec.append("#include <stdlib.h>\n");
    wfilec.append("#include <pthread.h>\n");
    wfilec.append("#include <string.h>\n");
    wfilec.append("#include \"nn_object.h\"\n");
    wfilec.append("//#include \"neural_net.h\"\n");
    wfilec.append("#include \"error.h\"\n");
    wfilec.append("#include \"hwtype.h\"\n");
    wfilec.append("#include \"nn_object_header.h\"\n");
    wfilec.append("#include \"stdint.h\"\n");
    wfilec.append("#include \"demo/image.h\"\n\n");

    wfilec.append("Image* imp2;\n");
    wfilec.append("Image im3;\n\n");

    wfilec.append("uint8_t *constantWeightVarsAddr;\n");
    wfilec.append("uint8_t *mutableWeightVarsAddr;\n");
    wfilec.append("uint8_t *activationsAddr;\n\n");

    wfilec.append("float *resultVar;\n");
    wfilec.append("int resultLabel;\n");
    wfilec.append("int resultSize;\n\n");

    wfilec.append("int start_layer;\n");
    wfilec.append("int end_layer;\n\n");

    wfilec.append("static void load(NnObject *object)\n");
    wfilec.append("{\n");
    wfilec.append("\tloading_resnet18();\n");
    wfilec.append("\tobject->loaded = 1; // mark it as loaded\n\n");
    wfilec.append("\tprintf(\"NNID %d Kernel is loaded\\n\", object->nnid);\n");
    wfilec.append("}\n\n");

    wfilec.append("static void unload(NnObject *object)\n");
    wfilec.append("{\n");
    wfilec.append("\tif (object->loaded)\n");
    wfilec.append("\t{\n");
    wfilec.append("\t\tunloading_resnet18();\n");
    wfilec.append("\t\tobject->loaded = 0;\n");
    wfilec.append("\t}\n");
    wfilec.append("\telse {\n");
    wfilec.append("\t\tprintf(\"object is not loaded yet\\n\");\n");
    wfilec.append("\t}\n");
    wfilec.append("}\n");

    wfilec.append("extern void save_image_options(Image im, const char *name,
IMTYPE f, int quality);\n\n");

    wfilec.append("static char *image_path;\n\n");

    wfilec.append("static void predict(NnObject *object)\n");
    wfilec.append("{\n");
    wfilec.append("#if 0\n\n");
    wfilec.append("#else\n");
    wfilec.append("\tchar * image = NULL;\n");
    wfilec.append("\tfloat * input_layer_data = NULL;\n\n");
    wfilec.append("\tif (start_layer == 0) {\n\n");
    wfilec.append("\t\timage = image_path;\n");
    wfilec.append("\t\tinput_layer_data = NULL;\n\n");
    wfilec.append("\t}\n");
    wfilec.append("else {\n\n");
    wfilec.append("\t\tinput_layer_data = (float*) object->wsmem;\n");
    wfilec.append("\t\timage = NULL;\n\n");
    wfilec.append("}\n");
    wfilec.append("\t// else if \"middle layer\"\n\n");
    wfilec.append("\trunning_resnet18(image, input_layer_data, start_layer,
end_layer);\n\n"); wfilec.append("#endif\n"); wfilec.append("}\n");

    wfilec.append("static void set_input(NnObject *object, Input *input)\n");
    wfilec.append("{\n");
    wfilec.append("#if 0\n");
    wfilec.append("#else\n");
    wfilec.append("\tint *dataptr = object->wsmem + (input->data_size -
8);\n\n"); wfilec.append("\tstart_layer = dataptr[0];\n");
    wfilec.append("\tend_layer = dataptr[1];\n\n");
    wfilec.append("\tprintf(\"set_input start layer %d , end layer %d\\n\",
dataptr[0], dataptr[1]);\n\n"); wfilec.append("\tif (start_layer == 0) {\n\n");
    wfilec.append("\t\timage_path = input->filename;\n\n");
    wfilec.append("\t} else {\n");
    wfilec.append("\tprintf(\"start middle layer prediction\\n\");\n");
    wfilec.append("\t}\n");
    wfilec.append("#endif\n\n");
    wfilec.append("}\n");

    wfilec.append("static void get_output(NnObject *object, Output *output)\n");
    wfilec.append("{\n\n");

    wfilec.append("\tif (end_layer == " + std::to_string(partitionNum-2) + ")
{\n\n"); wfilec.append("\t\toutput->data_offset = 0;\n");
    wfilec.append("\t\t//output->data_size = resultSize;\n");
    wfilec.append("\t\toutput->data_size = 4;\n");
    wfilec.append("\t\tprintf(\"jun result : %d output size %d\\n\",
resultLabel, output->data_size);\n"); wfilec.append("\t\tmemcpy(object->wsmem +
output->data_offset, &resultLabel, output->data_size);\n\n");
    wfilec.append("\t}\n");
    wfilec.append("\telse {\n");
    wfilec.append("\t\toutput->data_offset = 0;\n");
    wfilec.append("\t\toutput->data_size = resultSize;\n");
    wfilec.append("\t\tmemcpy(object->wsmem + output->data_offset, resultVar,
output->data_size);\n\n"); wfilec.append("\t\t//printf(\"befor alloc_size
%ld\\n\", malloc_usable_size(resultVar));\n");
    wfilec.append("\t\t//printf(\"befor alloc_size %ld\\n\",
malloc_usable_size(imp2));\n\n"); wfilec.append("\t\tfree(imp2);\n");
    wfilec.append("\t\tfree(resultVar);\n\n");
    wfilec.append("\t\t//printf(\" after alloc_size %ld\\n\",
malloc_usable_size(imp2));\n"); wfilec.append("\t}\n"); wfilec.append("}\n");

    wfilec.append("NnObject nn_object_23 = {23, 0, 0, 0, load, unload, predict,
set_input, get_output};\n");

}
*/

void NestPartitionerSchedule::generateMakeFileNestOs(
    string &wfilec, std::size_t partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig) {
  //  std::cout << "== [Start] generateCMakeListsFile == " << std::endl;

  wfilec.append("# if you need debugging, set it to 1, else 0\n");
  wfilec.append("#include config\n");
  wfilec.append(
      "NOS_HOME=$(NEST_HOME)/os_dynamic_resnet18_npu_os_partition\n\n");
  wfilec.append("CC=gcc\n");
  wfilec.append("CPP=g++\n");
  wfilec.append("DEBUG=1\n\n");
  wfilec.append("export DEBUG\n");
  wfilec.append("export NPU\n");
  wfilec.append("export CC\n");
  wfilec.append("export CPP\n\n");
  wfilec.append("LIBNAME=libresnet18.a\n");
  wfilec.append("ifeq ($(DEBUG), 1)\n");
  wfilec.append("OPT=-O0\n");
  wfilec.append("else\n");
  wfilec.append("OPT=-O3\n");
  wfilec.append("endif\n\n");
  wfilec.append("CFLAGS=-I. -I$(NOS_HOME)/include -I$(NEST_HOME)/include -Ios "
                "-Inn -Icpu -Idemo -g -std=c99 -Wall -Wno-unused-result "
                "-Wno-unknown-pragmas -Wfatal-errors -fPIC $(OPT)\n");
  wfilec.append("CFLAGS+=-D_GNU_SOURCE\n");
  wfilec.append("ifeq ($(NPU), 1)\n");
  wfilec.append("CFLAGS+=-Inpu\n");
  wfilec.append("CFLAGS+=-DUSENPU\n");
  wfilec.append("endif\n");
  wfilec.append(
      "CPPFLAGS=-I. -Iinclude -I../include -Ios -Inn -Icpu -Idemo -g -Wall "
      "-Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC $(OPT)\n");
  wfilec.append("CPPFLAGS+=-D_GNU_SOURCE\n");
  wfilec.append("ifeq ($(NPU), 1)\n");
  wfilec.append("CPPFLAGS+=-Inpu\n");
  wfilec.append("CPPFLAGS+=-DUSENPU\n");
  wfilec.append("endif\n");
  wfilec.append("CPPFLAGS+= -std=c++11\n");
  wfilec.append("CPPFLAGS+= -I/root/nestc_vtabundle/vtalib/include/zcu102/ "
                "-I/root/nestc_vtabundle/vtalib/include/Bundle\n\n");
  wfilec.append("LDFLAGS=-lm\n");
  wfilec.append("LDFLAGS+=-ldl\n");
  wfilec.append("LDFLAGS+=-lpthread #use pthread\n\n");
  wfilec.append("LDFLAGS+= -lrt\n");
  wfilec.append("//LDFLAGS+=-mavx\n");
  wfilec.append("LDFLAGS+=-mcpu=cortex-a53\n\n");
  wfilec.append("LDFLAGS+= -L/root/nestc_vtabundle/vtalib/zcu102 "
                "-Wl,-rpath,/root/nestc_vtabundle/vtalib/zcu102 -lVTABundle "
                "-lvta_runtime /usr/lib/llvm-8.0/lib/libLLVMSupport.a -lcma "
                "-lgomp -lpng -lz -lrt -ldl -ltinfo -lpthread -lm "
                "/usr/lib/llvm-8.0/lib/libLLVMDemangle.a\n\n");
  wfilec.append("all: objcreate1 objcreate libmake \n\n");
  wfilec.append("OBJ_FILES=main.o nn_object/nn_object_23.o\n");

  string libObjFilec;
  string objFilesc;

  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("CPU")) {
      libObjFilec.append("p" + std::to_string(i) + ".o ");
    } else if (!partitionConfig.backendNames[i].compare("VTA")) {
      objFilesc.append("p" + std::to_string(i) + ".o ");
    }
  }

  wfilec.append("OBJ_FILES1= " + objFilesc + "\n");
  wfilec.append("LIB_OBJ_FILES= " + libObjFilec + "\n\n");
  wfilec.append("$(TARGET) : $(OBJ_FILES1) $(OBJ_FILES) $(LIB_OBJ_FILES)\n");
  wfilec.append("\t$(CPP) $(OBJ_FILES1) $(OBJ_FILES) $(LIB_OBJ_FILES) -o $@ "
                "$(LDFLAGS)\n\n");
  wfilec.append("objcreate: $(OBJ_FILES)\n\n");
  wfilec.append("objcreate1: $(OBJ_FILES1)\n\n");
  wfilec.append(".c.o:\n");
  wfilec.append("\t$(CC) -c $(COMMON) $(CFLAGS) $< -o $@\n");
  wfilec.append(".cpp.o:\n");
  wfilec.append("\t$(CPP) -c $(COMMON) $(CPPFLAGS) $< -o $@\n");
  wfilec.append("clean:\n");
  wfilec.append("\techo $(OBJ_FILES)\n");
  wfilec.append("\trm -f $(OBJ_FILES) $(CPP_OBJ_FILES) $(CFG_OBJ_FILES) "
                "$(APP_OBJ_FILES) nn_object_2*.o\n");
  wfilec.append("\trm -f *.stackdump resenet_main libresenet18.a\n");
  wfilec.append("libmake:\n");
  wfilec.append("\tar rscv $(LIBNAME) $(OBJ_FILES) $(OBJ_FILES1) "
                "$(LIB_OBJ_FILES) $(AIWARE_OBJ_FILES)\n");
}

void NestPartitionerSchedule::generateMakeFileLocal(
    string &wfilec, std::size_t partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig) {
  //  std::cout << "== [Start] generateCMakeListsFile == " << std::endl;

  wfilec.append("# if you need debugging, set it to 1, else 0\n");
  wfilec.append("#include config\n");
  wfilec.append("#NOS_HOME=$(NEST_HOME)/os_resnet18_npu\n\n");
  wfilec.append("CC=gcc\n");
  wfilec.append("CPP=g++\n");
  wfilec.append("DEBUG=1\n\n");
  wfilec.append("export DEBUG\n");
  wfilec.append("export NPU\n");
  wfilec.append("export CC\n");
  wfilec.append("export CPP\n\n");
  wfilec.append("LIBNAME=libresnet18.a\n");
  wfilec.append("TARGET=dynamic_layer_test\n\n");
  wfilec.append("ifeq ($(DEBUG), 1)\n");
  wfilec.append("OPT=-O0\n");
  wfilec.append("else\n");
  wfilec.append("OPT=-O3\n");
  wfilec.append("endif\n\n");
  wfilec.append("CFLAGS=-I. -I$(NOS_HOME)/include -I$(NEST_HOME)/include -Ios "
                "-Inn -Icpu -Idemo -g -std=c99 -Wall -Wno-unused-result "
                "-Wno-unknown-pragmas -Wfatal-errors -fPIC $(OPT)\n");
  wfilec.append("CFLAGS+=-D_GNU_SOURCE\n");
  wfilec.append("ifeq ($(NPU), 1)\n");
  wfilec.append("CFLAGS+=-Inpu\n");
  wfilec.append("CFLAGS+=-DUSENPU\n");
  wfilec.append("endif\n");
  wfilec.append(
      "CPPFLAGS=-I. -Iinclude -I../include -Ios -Inn -Icpu -Idemo -g -Wall "
      "-Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC $(OPT)\n");
  wfilec.append("CPPFLAGS+=-D_GNU_SOURCE\n");
  wfilec.append("ifeq ($(NPU), 1)\n");
  wfilec.append("CPPFLAGS+=-Inpu\n");
  wfilec.append("CPPFLAGS+=-DUSENPU\n");
  wfilec.append("endif\n");
  wfilec.append("CPPFLAGS+= -std=c++11\n");
  wfilec.append("CPPFLAGS+= -I/root/nestc_vtabundle/vtalib/include/zcu102/ "
                "-I/root/nestc_vtabundle/vtalib/include/Bundle\n\n");
  wfilec.append("LDFLAGS=-lm\n");
  wfilec.append("LDFLAGS+=-ldl\n");
  wfilec.append("LDFLAGS+=-lpthread #use pthread\n\n");
  wfilec.append("LDFLAGS+= -lrt\n");
  wfilec.append("//LDFLAGS+=-mavx\n");
  wfilec.append("LDFLAGS+=-mcpu=cortex-a53\n\n");
  wfilec.append("LDFLAGS+= -L/root/nestc_vtabundle/vtalib/zcu102 "
                "-Wl,-rpath,/root/nestc_vtabundle/vtalib/zcu102 -lVTABundle "
                "-lvta_runtime /usr/lib/llvm-8.0/lib/libLLVMSupport.a -lcma "
                "-lgomp -lpng -lz -lrt -ldl -ltinfo -lpthread -lm "
                "/usr/lib/llvm-8.0/lib/libLLVMDemangle.a\n\n");
  wfilec.append("all: objcreate1 objcreate libmake $(TARGET)\n\n");
  wfilec.append("OBJ_FILES=main.o \n");

  string libObjFilec;
  string objFilesc;

  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("CPU")) {
      libObjFilec.append("p" + std::to_string(i) + ".o ");
    } else if (!partitionConfig.backendNames[i].compare("VTA")) {
      objFilesc.append("p" + std::to_string(i) + ".o ");
    }
  }

  wfilec.append("OBJ_FILES1= " + objFilesc + "\n");
  wfilec.append("LIB_OBJ_FILES= " + libObjFilec + "\n\n");
  wfilec.append("$(TARGET) : $(OBJ_FILES1) $(OBJ_FILES) $(LIB_OBJ_FILES)\n");
  wfilec.append("\t$(CPP) $(OBJ_FILES1) $(OBJ_FILES) $(LIB_OBJ_FILES) -o $@ "
                "$(LDFLAGS)\n\n");
  wfilec.append("objcreate: $(OBJ_FILES)\n\n");
  wfilec.append("objcreate1: $(OBJ_FILES1)\n\n");
  wfilec.append(".c.o:\n");
  wfilec.append("\t$(CC) -c $(COMMON) $(CFLAGS) $< -o $@\n");
  wfilec.append(".cpp.o:\n");
  wfilec.append("\t$(CPP) -c $(COMMON) $(CPPFLAGS) $< -o $@\n");
  wfilec.append("clean:\n");
  wfilec.append("\techo $(OBJ_FILES)\n");
  wfilec.append("\trm -f $(OBJ_FILES) $(CPP_OBJ_FILES) $(CFG_OBJ_FILES) "
                "$(APP_OBJ_FILES) nn_object_2*.o\n");
  wfilec.append("\trm -f *.stackdump resenet_main libresenet18.a\n");
  wfilec.append("libmake:\n");
  wfilec.append("\tar rscv $(LIBNAME) $(LIB_OBJ_FILES) $(AIWARE_OBJ_FILES)\n");
}

#include <random>
void NestPartitionerSchedule::generateCMakeListsFile(
    string &wfilec, std::size_t partitionNum, std::vector<Function *> funcList,
    const PartitionConfig &partitionConfig) {
  //  std::cout << "== [Start] generateCMakeListsFile == " << std::endl;

  wfilec.append("add_custom_command(\n");
  wfilec.append("\tOUTPUT\n");

  wfilec.append("\t");
  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("CPU")) {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                    ".o ");
    } else if (!partitionConfig.backendNames[i].compare("VTA")) {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                    ".cpp ");
    }
  }
  wfilec.append("\n\n");

  for (int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");

    if (!partitionConfig.backendNames[i].compare("CPU")) {
      wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/p" + std::to_string(i) +
                    ".o\n");
      wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                    ".o\n");
    } else if (!partitionConfig.backendNames[i].compare("VTA")) {
      wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/p" + std::to_string(i) +
                    ".cpp\n");
      wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                    ".cpp\n");
    }
  }
  wfilec.append("\n");

  for (int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
    wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/p" + std::to_string(i) +
                  ".h\n");
    wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                  ".h\n");
  }
  wfilec.append("\n");

  wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
  wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/main_header_test.h\n");
  wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/main_header_test.h\n\n");

  wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
  wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/main_template.h\n");
  wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/main_template.h\n\n");

  bool vtaFlag = false;
  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if (vtaFlag) {
    wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
    wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/VTARuntime.h\n");
    wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h\n\n");
  }

  for (int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
    wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/p" + std::to_string(i) +
                  ".weights.bin\n");
    wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                  ".weights.bin\n");
  }
  wfilec.append("\t)\n");

  srand((int)time(0));
  std::string demoExeFileName =
      "demoExeFileName" + std::to_string(rand() % 10000);

  // std::cout << demoExeFileName << std::endl;

  wfilec.append("add_custom_target(" + demoExeFileName + "Net DEPENDS\n");
  wfilec.append("\t");
  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("CPU")) {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                    ".o ");
    } else if (!partitionConfig.backendNames[i].compare("VTA")) {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                    ".cpp ");
    }
  }
  wfilec.append("\t)\n\n");

  wfilec.append("INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})\n");
  wfilec.append("add_executable(" + demoExeFileName + " main.cpp\n");
  wfilec.append("\t");
  for (int i = 0; i < partitionNum - 1; i++) {
    if (!partitionConfig.backendNames[i].compare("CPU")) {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                    ".o ");
    } else if (!partitionConfig.backendNames[i].compare("VTA")) {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) +
                    ".cpp ");
    }
  }

  wfilec.append(")\n");

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
        "-p# VTABundle vta_runtime arm_compute arm_compute_core "
        "arm_compute_graph LLVMSupport cma gomp png)\n" +
        "set_target_properties(" + demoExeFileName + "-p#\n" +
        "\tPROPERTIES\n" +
        "\tRUNTIME_OUTPUT_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")\n";
    std::string pid;
    for (int i = 0; i < partitionNum - 1; i++) {
      std::string pExeStr = "";
      if (!partitionConfig.backendNames[i].compare("CPU")) {
        pExeStr = pExeStrCPUTemplate;
      } else {
        pExeStr = pExeStrVTATemplate;
      }

      pid = std::to_string(i);
      boost::replace_all(pExeStr, "#", pid);
      wfilec.append(pExeStr);
    }
  }

  wfilec.append("INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})\n");
  wfilec.append("add_dependencies(" + demoExeFileName + " " + demoExeFileName +
                "Net)\n");
  wfilec.append("target_link_libraries(" + demoExeFileName +
                " VTABundle vta_runtime arm_compute arm_compute_core "
                "arm_compute_graph LLVMSupport cma gomp png)\n");
  wfilec.append("set_target_properties(" + demoExeFileName + "\n");
  wfilec.append("\tPROPERTIES\n");
  wfilec.append("\tRUNTIME_OUTPUT_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\"\n");
  wfilec.append("\t)\n\n");

  wfilec.append("#if(GLOW_WITH_VTA_BUNDLE_TEST)\n");
  wfilec.append("\tadd_custom_command(\n");
  wfilec.append("\t\tTARGET " + demoExeFileName + " POST_BUILD\n");
  wfilec.append("\t\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
  wfilec.append("\t\t${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png\n");
  wfilec.append("\t\t${CMAKE_CURRENT_BINARY_DIR}/cat_285.png)\n");

  wfilec.append("#\n");
  wfilec.append("#\tadd_custom_command(\n");
  wfilec.append("#\t\tTARGET demo8Mobilenet POST_BUILD\n");
  wfilec.append("#\t\tCOMMAND\n");
  wfilec.append("#\t\tmkdir -p debug)\n");
  wfilec.append("#\tadd_custom_command(\n");
  wfilec.append("#\t\tTARGET demo8Mobilenet POST_BUILD\n");
  wfilec.append("#\t\tCOMMAND\n");
  wfilec.append("#\t\tdemo8Mobilenet cat_285.png)\n");
  wfilec.append("#\tadd_custom_command(\n");
  wfilec.append("#\t\tTARGET demo8Mobilenet POST_BUILD\n");
  wfilec.append("#\t\tCOMMAND\n");
  wfilec.append("#\t\techo \"demo8Mobilenet Test Succeed\")\n");
  wfilec.append("#endif()\n");
}

std::string NestPartitionerSchedule::getProfileKey(Node *node) {
  std::string key;
  std::string kindName = node->getKindName();

  //  if(node->getNumInputs() == 0) {
  //    std::cout << "0 input: " << node->getName().str() << std::endl;
  //    std::cout << node->getKindName() << std::endl;
  //  }

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