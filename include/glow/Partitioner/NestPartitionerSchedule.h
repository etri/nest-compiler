/*****************************************************************************
 *
 * Copyright Next-Generation System Software Research Group, All rights
 *reserved. Future Computing Research Division, Artificial Intelligence Reserch
 *Laboratory Electronics and Telecommunications Research Institute (ETRI)
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

#ifndef GLOW_NESTPARTITIONERSCHEDULE_H
#define GLOW_NESTPARTITIONERSCHEDULE_H

#include "glow/Partitioner/PartitionerUtils.h"
#include "glow/Runtime/HostManager/HostManager.h"
#include "llvm/Support/CommandLine.h"

#include <boost/algorithm/string.hpp>
#include <glog/logging.h>

#define maxInOut 500

using namespace glow::runtime;

namespace glow {

class NestPartitionerSchedule {

  std::map<int, std::string> keyList_;
  std::map<int, double> delayTime_;
  std::map<int, int> insertCount_;

  bool flag_ = false;
  int profileMode_;
  int partitionExeMode_;
  std::string inputPartitionName_;
  std::string outputDir_;

  std::string profilePath_;
  std::string planFileName_;
  std::set<std::string> *profileKeySet_;

  std::map<std::string, int> partitionOutList_;
  std::map<std::string, std::string> partitionOutVarList_;
  std::map<std::string, Function *>* funcMap_;

public:
  void setProfilePath(std::string path) { profilePath_ = path; };
  std::string getProfilePath() { return profilePath_; };
  void setPartitionPlanFile(std::string file) { planFileName_ = file; };
  std::string getPartitionPlanFile() { return planFileName_; };
  void setProfileSet(std::set<std::string> *profileKeySet) {profileKeySet_ = profileKeySet;};
  void setFuncMap(std::map<std::string, Function *>* funcMap) { funcMap_ = funcMap; };

  void setKeyName(std::string key, int i);
  void setProfileMode(int mode);
  void setPartitionExeMode(int mode);
  int getProfileMode();
  void setOutputDir(std::string dir) { outputDir_ = dir; };
  void setInputPartitionName(std::string name);
  std::string getInputPartitionName();
  std::string getProfileKey(Node *node);
  std::string replaceAll(std::string &str, std::string pattern,
                         std::string replace);
  std::string addProfileKey(Function *function, std::set<std::string> *keySet,
                            bool isFusion);

  void generateVarInit(std::string &wfilec, const PartitionConfig &partitionConfig);
  void generateDeclaration(std::string &writeFileC,
                           const PartitionConfig &partitionConfig);
  void generateMainforEachPartition(const PartitionConfig &partitionConfig);
  void generatePartitionCall(std::string &wfilec, const PartitionConfig &partitionConfig);
  void generateMain(std::string &wfilec, const PartitionConfig &partitionConfig);
  void generateFree(std::string &wfilec, const PartitionConfig &partitionConfig);

  void generateCodeFromModels(const PartitionConfig &partitionConfig,
                              std::string inputPartitionName);
  void generateMainFile(const PartitionConfig &partitionConfig,
                        std::string inputPartitionName);
  void generateRelayFile(const PartitionConfig &partitionConfig);
  void generateCMakeListsFile(const PartitionConfig &partitionConfig);

  void generateYamlFile(std::string &wfilec, const PartitionConfig &partitionConfig,
                        std::string inputPartitionName);
  std::string getPartitionProfileKey(Function *function);
  void setPartitionInputOutputList(Function *function,
                                   std::vector<std::string> *inputList,
                                   std::vector<std::string> *outputList,
                                   int pi);
  void generateNonThreadCall(std::string &wfilec, int pi, bool profileMode,
                             std::vector<std::string> *inputList, std::string backendName);
  bool generateThreadCall(std::string &wfilec, int pi,
                          std::set<std::string> *pGroup, bool profileMode,
                          std::vector<std::string> *inputList, std::string backendName);
};

} // namespace glow
#endif // GLOW_NESTPARTITIONERSCHEDULE_H
