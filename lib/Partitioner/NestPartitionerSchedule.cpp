//
// Created by 유미선 on 2020/12/09.
//

#include "glow/Partitioner/NestPartitionerSchedule.h"
#include <fstream>

using namespace glow;
using namespace std;

void NestPartitionerSchedule::setKeyName(std::string key, int i)
{
  keyList_[i] = key;
}

double NestPartitionerSchedule::getDelayTime(int i)
{
  return delayTime_[i];
}


void NestPartitionerSchedule::setProfileMode(int mode)
{
  profileMode_ = mode;
}

void NestPartitionerSchedule::setInputPartitionName(std::string name)
{
  inputPartitionName_ = name;
}

std::string NestPartitionerSchedule::getInputPartitionName()
{
  return inputPartitionName_;
}

int NestPartitionerSchedule::getProfileMode()
{
  return profileMode_;
}


void NestPartitionerSchedule::generateDeclaration(string& wfilec, int partitionNum, const PartitionConfig &partitionConfig, int profileMode) {
//  cout << "generateDeclaration: " <<  wfilec << endl;

  if(profileMode == 1) {
    wfilec.append("#define REPEAT 1\n\n");
  }

  bool vtaFlag = false;
  for(int i = 0; i < partitionNum - 1; i++)
  {
    if(!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if(vtaFlag)
  {
    wfilec.append("#define VTAMAIN\n\n");
  }

  wfilec.append("#include \"main_template.h\"\n\n");

  if(vtaFlag)
  {
    wfilec.append("#ifdef VTAMAIN\n");
    wfilec.append("#include \"VTARuntime.h\"\n");
    wfilec.append("#endif\n\n");
  }

  for(int i = 0; i < partitionNum - 1; i++)
  {
    wfilec.append("#include \"p" + std::to_string(i) + ".h\"\n");
//    cout << "#include \"p" + std::to_string(i) + ".h\"" << endl;
  }
  wfilec.append("\n");

  wfilec.append("#include <time.h>\n\n");
  if(profileMode == 1) {
    //wfilec.append("#include <sys/timeb.h>\n\n");
    //wfilec.append("#include <time.h>\n\n");

    wfilec.append("#define ARRAY_LENGTH " + std::to_string(partitionNum) + "\n");
    wfilec.append("unsigned long delay_time[ARRAY_LENGTH];\n\n");
  }
}

std::string NestPartitionerSchedule::addProfileKey(Function* function, std::set<std::string> *keySet, bool isFusion)
{
  BFSLevel bfs = getBFSLevel(function);
  size_t level = bfs.size();
  std::string key = "";
  int cnt = 0;
  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node *node = bfs[i][j];
      if(node->getKind() == Kinded::Kind::SaveNodeKind || node->getKind() == Kinded::Kind::PlaceholderKind){
        continue;
      }
      key = key + getProfileKey(node) + "+";
      cnt++;
    }
  }

  if(isFusion && cnt == 1)
    return "";

  key.pop_back();

  if(keySet->find(key) == keySet->end()) {
    keySet->insert(key);
    return key;
  } else
    return "";
}

void NestPartitionerSchedule::generateYamlFile(string& wfilec, std::size_t partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig, std::string inputPartitionName) {
//  std::cout << "== [Start] generateYamlFile == " << std::endl;

  std::string backendName = partitionConfig.backendNames[0];
  std::string filename;
  std::string keyList[partitionNum];
  std::string keys;

  filename = "./" + inputPartitionName + "Profile.yaml";

  std::cout << "partition # : " << partitionNum << std::endl;
  std::cout << "inputPartitionName: " << inputPartitionName << std::endl;

  bool isFused = false;
  if(inputPartitionName.find("FusedNode") != std::string::npos) {
    isFused = true;
  }

  for(int nodeN = 0; nodeN < partitionNum - 1; nodeN++) {
    auto func = funcList[nodeN];

    std::string key = addProfileKey(func, profileKeySet_, isFused);

    std::cout << "key: " << key << std::endl;

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
  wfilec.append("\tstd::string keyList[" + std::to_string(partitionNum) + "] = {" + keys + "};\n");
  wfilec.append("\tstd::string filename = \"" + filename + "\";\n");
  wfilec.append("\tstd::ofstream profileFile(filename);\n\n");
  wfilec.append("\tchar* str = new char[10];\n\n");
  wfilec.append("\tif (profileFile.is_open()) {\n");
  wfilec.append("\t\tprofileFile << \"---\" << std::endl;\n");
  wfilec.append("\t\tprofileFile << \"backendName: " + backendName + "\" << std::endl;\n");
  wfilec.append("\t\tprofileFile << \"commCost: \" << \"0.0\" << std::endl;\n\n");
  wfilec.append("\t\tfor(int i = 0; i < " + std::to_string(partitionNum-1) + "; i++){\n");
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

void NestPartitionerSchedule::generateFree(string& wfilec, int partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig, int profileMode) {

  for(int i = 0; i < partitionNum - 1; i++)
  {
    if(!partitionConfig.backendNames[i].compare("VTA"))
    {
      wfilec.append("\tp" + std::to_string(i) + "_destroy_module();\n");
    }
  }
  wfilec.append("\n");

  wfilec.append("\t// Free all resources.\n");

//  const char *fileName;
  std::string filePath;

  for(int i = 0; i < partitionNum - 1; i++)
  {
    wfilec.append("\tfree(activationsAddr" + std::to_string(i) + ");\n");
    wfilec.append("\tfree(constantWeightVarsAddr" + std::to_string(i) + ");\n");
    wfilec.append("\tfree(mutableWeightVarsAddr" + std::to_string(i) + ");\n\n");
  }

  bool vtaFlag = false;
  for(int i = 0; i < partitionNum - 1; i++)
  {
    if(!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if(vtaFlag)
  {
    wfilec.append("#ifdef VTAMAIN\n");
    wfilec.append("\tdestroyVTARuntime();\n");
    wfilec.append("#endif\n");
  }

  if(profileMode == 1)
  {
    wfilec.append("\tcreateYamlFile();\n");
  }

  wfilec.append("}\n\n");
}

void NestPartitionerSchedule::generateMain(std::string &wfilec, int partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig, int profileMode) {
//  cout << "generateMain: " << wfilec << endl;

  std::cout << "Partition plan fileName : " << getPartitionPlanFile() << std::endl;
  std::vector<std::string> parallerPartitions;
  std::map<std::string, std::string> pmap = deserializeStrStrMapFromYaml(getPartitionPlanFile());
  if(pmap.find("parallelPartitions") != pmap.end())
  {
    boost::split(parallerPartitions, pmap.find("parallelPartitions")->second, [](char c){return c == ':';});
    for(int i = 0; i < parallerPartitions.size(); i++)
    {
      parallerPartitions[i] = replaceAll(parallerPartitions[i], "(", "");
      parallerPartitions[i] = replaceAll(parallerPartitions[i], ")", "");
      parallerPartitions[i] = parallerPartitions[i].substr(0, parallerPartitions[i].find("||"));
    }
  }

  if(parallerPartitions.size() > 0)
  {
    wfilec.append("#define THREAD 1\n\n");

    wfilec.append("#include <thread>\n\n");
  }

  wfilec.append("int main(int argc, char **argv) {\n");
  wfilec.append("\tparseCommandLineOptions(argc, argv);\n\n");

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

 wfilec.append("\t//======== initialize variables =======//\n\n");

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
    if(i2 != 0 && !kindName_check.compare("Placeholder"))
    {
      inputName = node->getNthInput(0).getNode()->getName();
      break;
    }
  }

  for(int i = 0; i < partitionNum - 1; i++) {

    wfilec.append("\tuint8_t *constantWeightVarsAddr" + std::to_string(i) +
                  " = \n\t\tinitConstantWeights(\"p" + std::to_string(i) +
                  ".weights.bin\", p" + std::to_string(i) + "_config);\n");

    if (i == 0) {
      wfilec.append("\tuint8_t *mutableWeightVarsAddr" + std::to_string(i) +
                    " = initMutableWeightVars(p" + std::to_string(i) +
                    "_config, \"" + inputName + "\");\n");
    }

    wfilec.append("\tuint8_t *activationsAddr" + std::to_string(i) +
                  " = initActivations(p" + std::to_string(i) + "_config);\n\n");
  }

//  cout << "//======== initialize mutableWeightVarsAddr start =======//" << endl;
  wfilec.append("\t//======== initialize mutableWeightVarsAddr =======//\n\n");
  for(int i = 1; i < partitionNum - 1; i++) {
    wfilec.append("\tuint8_t *mutableWeightVarsAddr" + std::to_string(i) + " =  allocateMutableWeightVars(p" +
                  std::to_string(i) + "_config);\n");
  }
  wfilec.append("\n");

  for(int i = 0; i < partitionNum - 1; i++)
  {
    if(!partitionConfig.backendNames[i].compare("VTA"))
    {
      wfilec.append("\tp" + std::to_string(i) + "_load_module(constantWeightVarsAddr" + std::to_string(i) + ");\n");
    }
  }
  wfilec.append("\n");

  // cout << "//======== multiple partitions start =======//" << endl;
  wfilec.append("\t//======== multiple partitions =======//\n\n");

  if(profileMode == 1) {
    wfilec.append("\tunsigned long totalInftime;\n\n");
  }

  int resultNum = 0;
  int resultVarNum = 0;
  string resultList[partitionNum + partitionNum];
  string resultCheckList[partitionNum + partitionNum];
  std::vector<std::string> paraller;

  for(int i = 0; i < partitionNum - 1; i++) {
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

    //std::string kindName_check;
//    cout << "//======== inputName List Node " << i << " =======//" << endl;
    for (int i2 = level - 1; i2 >= 0; i2--) {
      Node *node = bfs[i2][0];

      for(int count = 0; count < node->getNumInputs(); count++)
      {
        kindName_check = node->getNthInput(count).getNode()->getKindName();
        if(i2 != 0 && !kindName_check.compare("Placeholder"))
        {
//          cout << "node inputCount : " << count << ", inputName = " << node->getNthInput(count).getNode()->getName().str() << endl;
//          cout << "node input KindName = " << node->getNthInput(count).getNode()->getKindName() << endl;
          inputList[inputCount] = node->getNthInput(count).getNode()->getName();
          inputCount++;
        }
      }
    }

    // outputName 초기화
    std::string outputName;
    outputName = func->getNodes().back().getName().trim("_save");
    //cout << "//======== outputName List Node " << i << " =======//" << endl;
    NodesList& nodeList = func->getNodes();
    for (auto &n : nodeList)
    {
      kindName_check = n.getKindName();
      if(!kindName_check.compare("Save"))
      {
        outputList[outputCount] = n.getName().substr(0, n.getName().size() - 5);
        outputCount++;
      }

    }

    if(i == 0)
    {
      //wfilec.append("\t// Perform the computation.\n");

      wfilec.append("\tclock_t start_total = 0;\n");
      wfilec.append("\tstart_total = clock();\n");

      if(profileMode == 1) {
        wfilec.append("\tif(REPEAT > 0){\n");
        wfilec.append("\t\ttotalInftime = 0;\n");
        wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; repeatCount++){\n");

        wfilec.append("\t\t\tclock_t start_" + std::to_string(i) + ";\n");
        wfilec.append("\t\t\tstart_" + std::to_string(i) + " = clock();\n\n");
      }

      wfilec.append("\tp" +
                    std::to_string(i) + "(constantWeightVarsAddr" +
                    std::to_string(i) + ", mutableWeightVarsAddr" +
                    std::to_string(i) + ", activationsAddr" +
                    std::to_string(i) + ");\n\n");

      if(profileMode == 1) {
        wfilec.append("\t\t\tclock_t now_" + std::to_string(i) + ";\n");
        wfilec.append("\t\t\tnow_" + std::to_string(i) + " = clock();\n");
        wfilec.append("\t\t\tunsigned long inftime_" + std::to_string(i) + " = (unsigned long)(now_" + std::to_string(i) + " - start_" + std::to_string(i) +");\n");
       // wfilec.append("\t\t\t\t\t(float)(t_now_" + std::to_string(i) + ".millitm - t_start_" + std::to_string(i) + ".millitm);\n\n");

        wfilec.append("\t\t\tif(repeatCount != 0){\n");
        wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" + std::to_string(i) + ";\n");
        wfilec.append("\t\t\t}\n");
        wfilec.append("\t\t}\n");
        wfilec.append("\t}\n\n");

        wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
        wfilec.append("\tdelay_time[" + std::to_string(i) + "] = totalInftime/(REPEAT);\n");
        wfilec.append("\t}else {\n");
        wfilec.append("\t\tdelay_time[" + std::to_string(i) + "] = totalInftime;\n");
        wfilec.append("\t}\n");
        wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i) + " time] %lu microsec.\\n\", delay_time[" + std::to_string(i) + "]);\n\n");
      }

      if(outputCount == 1) {
        wfilec.append("\tfloat* resultVar" + std::to_string(resultVarNum) +
                      " = getInferenceResultVar(p" + std::to_string(i) +
                      "_config, mutableWeightVarsAddr" + std::to_string(i) +
                      ", \"" + outputList[0] + "\");\n");

        resultCheckList[resultNum] = std::to_string(resultVarNum);
        resultList[resultNum] = outputList[0];
        resultNum++;
      }
      else {
        for(int outputNum = 0; outputNum < outputCount; outputNum++)
        {
          wfilec.append("\tfloat* resultVar" + std::to_string(resultVarNum) + "_" + std::to_string(outputNum + 1) +
                        " = getInferenceResultVar(p" + std::to_string(i) +
                        "_config, mutableWeightVarsAddr" + std::to_string(i) +
                        ", \"" + outputList[outputNum] + "\");\n");

          resultCheckList[resultNum] = std::to_string(resultVarNum) + "_" + std::to_string(outputNum + 1);
          resultList[resultNum] = outputList[outputNum];
          resultNum++;
        }
      }
      resultVarNum++;
    }
    else {
      for(int checkCount = 0; checkCount < resultNum; checkCount++)
      {
        //if(!resultList[checkCount].compare(func->getNodes().front().getNthInput(0).getNode()->getName()))
        if(!resultList[checkCount].compare(inputList[0]))
        {
          flag = true;
          checkResultNum = checkCount;
        }
      }

      if(!flag)  { // 기존 선언된 resultVar 중 중복된 inputName이 없다면, 해당 로직 필요 여부 체크
        if(resultNum != 0)
          checkResultNum = resultNum - 1; // 앞에서 선언한 resultVar 호출, inputName 중복이면 checkCount resultVar 호출
      }
      wfilec.append("\tcopyMutableWeightVarsWithoutAlloc(p" + std::to_string(i) +
                    "_config, mutableWeightVarsAddr" + std::to_string(i) + ", \"" + inputList[0] + "\", resultVar" + resultCheckList[checkResultNum] + ");\n");
    }

    if(i != 0 && inputList->size() > 1) {
      for(int count = 1; count < inputList->size(); count++) {
        if(!inputList[count].compare(""))
          break;

        for(int checkCount = 0; checkCount < resultNum; checkCount++) {
          if(!resultList[checkCount].compare(inputList[count])) {
            flag = true;
            checkResultNum = checkCount;
          }
        }

        if(!flag) { // 기존 선언된 resultVar 중 중복된 inputName이 없다면, 해당 로직 필요 여부 체크
          if(resultNum != 0)
            checkResultNum = resultNum - 1; // 앞에서 선언한 resultVar 호출, inputName 중복이면 checkCount resultVar 호출
        }

        flag = true;
        for(int checkCount = 0; checkCount < resultNum; checkCount++) {
          if((checkCount != count) && (!inputList[checkCount].compare(inputList[count]))) {
            flag = false;
          }
        }
        if(flag) {
          wfilec.append("\tcopyMutableWeightVarsWithoutAlloc(p" + std::to_string(i) + "_config, mutableWeightVarsAddr" + std::to_string(i) +
                        ", \"" + inputList[count] + "\", resultVar" + resultCheckList[checkResultNum] + ");\n");
        }
        flag = false;
      }
      wfilec.append("\n");
    }

    if(parallerPartitions.size() > 0) {
      for(int partitionCount = 0; partitionCount < parallerPartitions.size(); partitionCount++) {
        if(!parallerPartitions[partitionCount].compare("p" + std::to_string(i - 1))) {
          wfilec.append("#ifdef THREAD\n");
          if(profileMode == 1) {
            wfilec.append("\tif(REPEAT > 0){\n");
            wfilec.append("\t\ttotalInftime = 0;\n");
            wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; repeatCount++){\n");

            wfilec.append("\t\t\tclock_t start_" + std::to_string(i-1) + ";\n");
            wfilec.append("\t\t\tstart_" + std::to_string(i-1) + " = clock();\n\n");

          }
          wfilec.append("\tstd::thread t" + std::to_string(i - 1) + "(p" + std::to_string(i - 1) + ", constantWeightVarsAddr" + std::to_string(i - 1) + ", mutableWeightVarsAddr" + std::to_string(i - 1) + ", activationsAddr" + std::to_string(i - 1) + ");\n");
          wfilec.append("\tstd::thread t" + std::to_string(i) + "(p" + std::to_string(i) + ", constantWeightVarsAddr" + std::to_string(i) + ", mutableWeightVarsAddr" + std::to_string(i) + ", activationsAddr" + std::to_string(i) + ");\n");
          wfilec.append("\tt" + std::to_string(i - 1) + ".join();\n");
          wfilec.append("\tt" + std::to_string(i) + ".join();\n");
          if(profileMode == 1) {
            wfilec.append("\t\t\tclock_t now_" + std::to_string(i-1) + ";\n");
            wfilec.append("\t\t\tnow_" + std::to_string(i-1) + " = clock();\n");
            wfilec.append("\t\t\tunsigned long inftime_" + std::to_string(i-1) + " = (unsigned long)(now_" + std::to_string(i) + " - start_" + std::to_string(i-1) +");\n");

            wfilec.append("\t\t\tif(repeatCount != 0){\\n");
            wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" + std::to_string(i - 1) + ";\n");
            wfilec.append("\t\t\t}\n");
            wfilec.append("\t\t}\n");
            wfilec.append("\t}\n\n");

            wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i - 1) + "] = totalInftime/(REPEAT);\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i) + "] = totalInftime/(REPEAT);\n");
            wfilec.append("\t}else {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i - 1) + "] = totalInftime;\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i) + "] = totalInftime;\n");
            wfilec.append("\t}\n");
            wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i) + " time] %lu microsec.\\n\", delay_time[" + std::to_string(i-1) + "]);\n\n");
//            wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i) + " time] %.4lf ms.\\n\", delay_time[" + std::to_string(i - 1) + "]);\n\n");
          }
          wfilec.append("#else\n");
          if(profileMode == 1) {
            wfilec.append("\tif(REPEAT > 0){\n");
            wfilec.append("\t\ttotalInftime = 0;\n");
            wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; repeatCount++){\n");

            wfilec.append("\t\t\tclock_t start_" + std::to_string(i-1) + ";\n");
            wfilec.append("\t\t\tstart_" + std::to_string(i-1) + " = clock();\n\n");
          }
          wfilec.append("\tp" + std::to_string(i - 1) + "(constantWeightVarsAddr" + std::to_string(i - 1) + ", mutableWeightVarsAddr" + std::to_string(i - 1) + ", activationsAddr" + std::to_string(i - 1) + ");\n");
          if(profileMode == 1) {
            wfilec.append("\t\t\tclock_t now_" + std::to_string(i-1) + ";\n");
            wfilec.append("\t\t\tnow_" + std::to_string(i-1) + " = clock();\n");
            wfilec.append("\t\t\tunsigned long inftime_" + std::to_string(i-1) + " = (unsigned long)(now_" + std::to_string(i) + " - start_" + std::to_string(i-1) +");\n");

            wfilec.append("\t\t\tif(repeatCount != 0){\\n");
            wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" + std::to_string(i - 1) + ";\n");
            wfilec.append("\t\t\t}\n");
            wfilec.append("\t\t}\n");
            wfilec.append("\t}\n\n");

            wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i - 1) + "] = totalInftime/(REPEAT);\n");
            wfilec.append("\t}else {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i - 1) + "] = totalInftime;\n");
            wfilec.append("\t}\n");
            wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i-1) + " time] %lu microsec.\\n\", delay_time[" + std::to_string(i-1) + "]);\n\n");
//            wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i - 1) + " time] %.4lf ms.\\n\", delay_time[" + std::to_string(i - 1) + "]);\n\n");
          }
          if(profileMode == 1) {
            wfilec.append("\tif(REPEAT > 0){\n");
            wfilec.append("\t\ttotalInftime = 0;\n");
            wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; repeatCount++){\n");

            wfilec.append("\t\t\tclock_t start_" + std::to_string(i) + ";\n");
            wfilec.append("\t\t\tstart_" + std::to_string(i) + " = clock();\n\n");
          }
          wfilec.append("\tp" + std::to_string(i) + "(constantWeightVarsAddr" + std::to_string(i) + ", mutableWeightVarsAddr" + std::to_string(i) + ", activationsAddr" + std::to_string(i) + ");\n");
          if(profileMode == 1) {
            wfilec.append("\t\t\tclock_t now_" + std::to_string(i) + ";\n");
            wfilec.append("\t\t\tnow_" + std::to_string(i) + " = clock();\n");
            wfilec.append("\t\t\tunsigned long inftime_" + std::to_string(i) + " = (unsigned long)(now_" + std::to_string(i) + " - start_" + std::to_string(i-1) +");\n");

            wfilec.append("\t\t\tif(repeatCount != 0){\\n");
            wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" + std::to_string(i) + ";\n");
            wfilec.append("\t\t\t}\n");
            wfilec.append("\t\t}\n");
            wfilec.append("\t}\n\n");

            wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i) + "] = totalInftime/(REPEAT);\n");
            wfilec.append("\t}else {\n");
            wfilec.append("\t\tdelay_time[" + std::to_string(i) + "] = totalInftime;\n");
            wfilec.append("\t}\n");
            wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i) + " time] %lu microsec.\\n\", delay_time[" + std::to_string(i) + "]);\n\n");
          }
          wfilec.append("#endif\n\n");
        }
      }
    }

    if(i != 0)
    {
      //wfilec.append("\t// Perform the computation.\n");

      bool parallerFlag = true;
      if(parallerPartitions.size() > 0)
      {
        for(int partitionCount = 0; partitionCount < parallerPartitions.size(); partitionCount++) {
          if (!parallerPartitions[partitionCount].compare("p" + std::to_string(i - 1)))
          {
            parallerFlag = false;
          }
          if (!parallerPartitions[partitionCount].compare("p" + std::to_string(i)))
          {
            parallerFlag = false;
          }
        }
      }

      if(parallerFlag) {
        if(profileMode == 1)
        {
          wfilec.append("\tif(REPEAT > 0){\n");
          wfilec.append("\t\ttotalInftime = 0;\n");
          wfilec.append("\t\tfor(int repeatCount = 0; repeatCount <= REPEAT; repeatCount++){\n");

          wfilec.append("\t\t\tclock_t start_" + std::to_string(i) + ";\n");
          wfilec.append("\t\t\tstart_" + std::to_string(i) + " = clock();\n\n");
        }
        wfilec.append("\tp" +
                      std::to_string(i) + "(constantWeightVarsAddr" +
                      std::to_string(i) + ", mutableWeightVarsAddr" +
                      std::to_string(i) + ", activationsAddr" +
                      std::to_string(i) + ");\n\n");
        if(profileMode == 1)
        {
          wfilec.append("\t\t\tclock_t now_" + std::to_string(i) + ";\n");
          wfilec.append("\t\t\tnow_" + std::to_string(i) + " = clock();\n");
          wfilec.append("\t\t\tunsigned long inftime_" + std::to_string(i) + " = (unsigned long)(now_" + std::to_string(i) + " - start_" + std::to_string(i) +");\n");

//          wfilec.append("\t\t\ttimeb t_now_" + std::to_string(i) + ";\n");
//          wfilec.append("\t\t\tftime(&t_now_" + std::to_string(i) + ");\n");
//          wfilec.append("\t\t\tfloat inftime_" + std::to_string(i) + " = (float)(t_now_" + std::to_string(i) + ".time - t_start_" + std::to_string(i) + ".time) * 1000.0f +\n");
//          wfilec.append("\t\t\t\t\t(float)(t_now_" + std::to_string(i) + ".millitm - t_start_" + std::to_string(i) + ".millitm);\n\n");

          //wfilec.append("\tdelay_time[" + std::to_string(i) + "] = inftime_" + std::to_string(i) + ";\n");
          //wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i) + " time] %.4lf ms.\\n\", inftime_" + std::to_string(i) + ");\n\n");

          wfilec.append("\t\t\tif(repeatCount != 0){\n");
          wfilec.append("\t\t\t\ttotalInftime = totalInftime + inftime_" + std::to_string(i) + ";\n");
          wfilec.append("\t\t\t}\n");
          wfilec.append("\t\t}\n");
          wfilec.append("\t}\n\n");

          wfilec.append("\tif(totalInftime > 0 && REPEAT > 1) {\n");
          wfilec.append("\tdelay_time[" + std::to_string(i) + "] = totalInftime/(REPEAT);\n");
          wfilec.append("\t}else {\n");
          wfilec.append("\t\tdelay_time[" + std::to_string(i) + "] = totalInftime;\n");
          wfilec.append("\t}\n");
          wfilec.append("\tprintf(\"\\n====> [Inference_" + std::to_string(i) + " time] %lu microsec.\\n\", delay_time[" + std::to_string(i) + "]);\n\n");
        }
      }

      if (i == partitionNum - 2) {

        wfilec.append("\tclock_t now_total;\n");
        wfilec.append("\tnow_total = clock();\n");
        wfilec.append("\tunsigned long inftime_total = (unsigned long)(now_total - start_total);\n");
        wfilec.append("\tprintf(\"\\n====> [Total inference time] %lu microsec.\\n\", inftime_total);\n\n");

        wfilec.append("\t// Report the results.\n");
        wfilec.append("\tprintf(\"====== final result of this neural net ========\\n\");\n");
        wfilec.append("\tdumpInferenceResults(p" + std::to_string(i) +
                      "_config, mutableWeightVarsAddr" + std::to_string(i) +
                      ", \"" + outputName + "\");\n\n");
      } else {
        bool flag = true;
        for(int checkCount = 0; checkCount < resultNum; checkCount++)
        {
          // outputName 중복 여부 체크
          if(!resultList[checkCount].compare(outputName))
            flag = false;
        }

        // output 개수에 따른 resultVar값 선언 부분 수정 필요
        if(flag)  // 중복된 inputName이 없다면(전 파티션 outputName들 비교)
        {
          if(outputCount == 1)
          {
            wfilec.append("\tfloat* resultVar" + std::to_string(resultVarNum) +
                          " = getInferenceResultVar(p" + std::to_string(i) +
                          "_config, mutableWeightVarsAddr" + std::to_string(i) +
                          ", \"" + outputList[0] + "\");\n");

            resultCheckList[resultNum] = std::to_string(resultVarNum);
            resultList[resultNum] = outputList[0];
            resultNum++;
          }
          else
          {
            for(int outputNum = 0; outputNum < outputCount; outputNum++)
            {
              wfilec.append("\tfloat* resultVar" + std::to_string(resultVarNum) + "_" + std::to_string(outputNum + 1) +
                            " = getInferenceResultVar(p" + std::to_string(i) +
                            "_config, mutableWeightVarsAddr" + std::to_string(i) +
                            ", \"" + outputList[outputNum] + "\");\n");

              //std::cout << "check : " << resultVarNum << ", outputNum : " << outputNum << std::endl;
              //std::cout << "check 1 : " << std::to_string(resultVarNum) + "_" + std::to_string(outputNum + 1) << std::endl;
              resultCheckList[resultNum] = std::to_string(resultVarNum) + "_" + std::to_string(outputNum + 1);
              resultList[resultNum] = outputList[outputNum];
              resultNum++;
            }
          }
          resultVarNum++;
        }
      }
    }
  }
}

void NestPartitionerSchedule::generateCodeFromModels(std::size_t partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig, int profileMode, std::string inputPartitionName) {
  std::cout << "getnerateCodeFromModels Start" << std::endl;

//  std::cout << "out dir = " << outputDir_ << std::endl;

  string outfile_c = outputDir_ + "/main.cpp";
  string make_outfile_c = outputDir_ + "/CMakeLists.txt";

  if (!llvm::sys::fs::is_directory(outputDir_)) {
    llvm::sys::fs::create_directory(outputDir_);
  }
  ofstream writeFileC;
  writeFileC.open(outfile_c.data());

  if(writeFileC.is_open()) {
    string declare;
    string initialize;
    string inference_main;
    string memfree;
    string yamlFile;

    generateDeclaration(declare, partitionNum, partitionConfig, profileMode); // 헤더 선언
    generateMain(inference_main, partitionNum, funcList, partitionConfig, profileMode);
    generateFree(memfree, partitionNum, funcList, partitionConfig, profileMode);

    writeFileC << declare;
    writeFileC << initialize;
    //writeFileC << inference;
    writeFileC << inference_main;
    writeFileC << memfree;

    if(profileMode == 1)
    {
      generateYamlFile(yamlFile, partitionNum, funcList, partitionConfig, inputPartitionName);
      writeFileC << yamlFile;
    }

    writeFileC.close();
  }

  ofstream writeMakeFileC;
  writeMakeFileC.open(make_outfile_c.data());

  if(writeMakeFileC.is_open()) {
    string cmakeFile;

    generateCMakeListsFile(cmakeFile, partitionNum, funcList, partitionConfig);
    writeMakeFileC << cmakeFile;

    writeMakeFileC.close();
  }

  std::cout << "main outfile position : " << outfile_c << std::endl;
  std::cout << "CMakeLists outfile position : " << make_outfile_c << std::endl;
}

void NestPartitionerSchedule::generateCMakeListsFile(string& wfilec, std::size_t partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig) {
//  std::cout << "== [Start] generateCMakeListsFile == " << std::endl;

  wfilec.append("add_custom_command(\n");
  wfilec.append("\tOUTPUT\n");

  wfilec.append("\t");
  for(int i = 0; i < partitionNum - 1; i++)
  {
    if(!partitionConfig.backendNames[i].compare("CPU"))
    {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".o ");
    }
    else if(!partitionConfig.backendNames[i].compare("VTA"))
    {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".cpp ");
    }
  }
  wfilec.append("\n\n");

  for(int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");

    if(!partitionConfig.backendNames[i].compare("CPU"))
    {
      wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/p" + std::to_string(i) + ".o\n");
      wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".o\n");
    }
    else if(!partitionConfig.backendNames[i].compare("VTA"))
    {
      wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/p" + std::to_string(i) + ".cpp\n");
      wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".cpp\n");
    }
  }
  wfilec.append("\n");

  for(int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
    wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/p" + std::to_string(i) + ".h\n");
    wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".h\n");
  }
  wfilec.append("\n");

  wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
  wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/main_header_test.h\n");
  wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/main_header_test.h\n\n");

  wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
  wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/main_template.h\n");
  wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/main_template.h\n\n");

  bool vtaFlag = false;
  for(int i = 0; i < partitionNum - 1; i++)
  {
    if(!partitionConfig.backendNames[i].compare("VTA"))
      vtaFlag = true;
  }

  if(vtaFlag)
  {
    wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
    wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/VTARuntime.h\n");
    wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h\n\n");
  }

  for(int i = 0; i < partitionNum - 1; i++) {
    wfilec.append("\tCOMMAND ${CMAKE_COMMAND} -E copy\n");
    wfilec.append("\t${CMAKE_CURRENT_SOURCE_DIR}/p" + std::to_string(i) + ".weights.bin\n");
    wfilec.append("\t${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".weights.bin\n");
  }
  wfilec.append("\t)\n");

  wfilec.append("add_custom_target(demo9MobilenetNet DEPENDS\n");
  wfilec.append("\t");
  for(int i = 0; i < partitionNum - 1; i++)
  {
    if(!partitionConfig.backendNames[i].compare("CPU"))
    {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".o ");
    }
    else if(!partitionConfig.backendNames[i].compare("VTA"))
    {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".cpp ");
    }
  }
  wfilec.append("\t)\n\n");

  wfilec.append("INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})\n");
  wfilec.append("add_executable(demo9Mobilenet main.cpp\n");
  wfilec.append("\t");
  for(int i = 0; i < partitionNum - 1; i++)
  {
    if(!partitionConfig.backendNames[i].compare("CPU"))
    {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".o ");
    }
    else if(!partitionConfig.backendNames[i].compare("VTA"))
    {
      wfilec.append("${CMAKE_CURRENT_BINARY_DIR}/p" + std::to_string(i) + ".cpp ");
    }
  }
  wfilec.append("\t)\n");

  wfilec.append("INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})\n");
  wfilec.append("add_dependencies(demo9Mobilenet demo9MobilenetNet)\n");
  wfilec.append("target_link_libraries(demo9Mobilenet VTABundle vta_runtime arm_compute arm_compute_core arm_compute_graph LLVMSupport cma gomp png)\n");
  wfilec.append("set_target_properties(demo9Mobilenet\n");
  wfilec.append("\tPROPERTIES\n");
  wfilec.append("\tRUNTIME_OUTPUT_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\"\n");
  wfilec.append("\t)\n\n");

  wfilec.append("#if(GLOW_WITH_VTA_BUNDLE_TEST)\n");
  wfilec.append("\tadd_custom_command(\n");
  wfilec.append("\t\tTARGET demo9Mobilenet POST_BUILD\n");
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

  if(node->getKind() == Kinded::Kind::ConvolutionNodeKind) {
    key = std::string(node->getKindName()) + "-" + getDimArrayStr(node->getNthInput(0).getType())
          + "-" + getDataTypeStr(node->getNthInput(0).getElementType())
          + "-" + std::to_string(((ConvolutionNode*)node)->getFilter().dims().front())
          + "-" + std::to_string(((ConvolutionNode*)node)->getKernels().front())
          + "-" + std::to_string(((ConvolutionNode*)node)->getStrides().front());
  } else {
    if(node->getNumInputs() > 0) {
      key = std::string(node->getKindName()) + "-" +
            getDimArrayStr(node->getNthInput(0).getType()) + "-" +
            getDataTypeStr(node->getNthInput(0).getElementType());
    } else {
      key = node->getKindName();
    }
  }

  return key;
}

string NestPartitionerSchedule::replaceAll(std::string &str, std::string pattern, std::string replace) {
  string result = str;
  string::size_type pos = 0;
  string::size_type offset = 0;
  while ((pos = result.find(pattern, offset)) != std::string::npos)
  {
    result.replace(result.begin() + pos, result.begin() + pos + pattern.size(), replace);
    offset = pos + replace.size();
  }
  return result;
}
