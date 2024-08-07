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

#ifndef GLOW_NESTOSPARTITIONERSCHEDULE_H
#define GLOW_NESTOSPARTITIONERSCHEDULE_H

#include "glow/Partitioner/PartitionerUtils.h"
#include "glow/Runtime/HostManager/HostManager.h"
#include "llvm/Support/CommandLine.h"

#include <boost/algorithm/string.hpp>
#include <glog/logging.h>

//#define ARRAY_LENGTH 100

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
  std::set<std::string> *fuseOperatorSet_;
  std::set<std::string> *profileKeySet_;

public:
  void setProfilePath(std::string path) { profilePath_ = path; };
  std::string getProfilePath() { return profilePath_; };
  void setPartitionPlanFile(std::string file) { planFileName_ = file; };
  std::string getPartitionPlanFile() { return planFileName_; };
  void setProfileSet(std::set<std::string> *fuseOperatorSet,
                     std::set<std::string> *profileKeySet) {
    fuseOperatorSet_ = fuseOperatorSet;
    profileKeySet_ = profileKeySet;
  };

  void setFunction(Function *function);
  void setKeyName(std::string key, int i);
  double getDelayTime(int i);
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

  void generateDeclaration(std::string &writeFileC, int partitionNum,
                           const PartitionConfig &partitionConfig);
  void generateGlobalVariables(std::string &writeFileC, int partitionNum,
                               const PartitionConfig &partitionConfig);
  void generateLayerManager(std::string &writeFileC, int partitionNum,
                            const PartitionConfig &partitionConfig);
  void generateParseLayerConfiguration(std::string &writeFileC,
                                       int partitionNum,
                                       const PartitionConfig &partitionConfig);
  void generateLoadingResnet18(std::string &writeFileC, int partitionNum,
                               const PartitionConfig &partitionConfig);
  void generateUnloadingResnet18(std::string &writeFileC, int partitionNum,
                                 const PartitionConfig &partitionConfig);

  void generateMainforEachPartition(int partitionNum,
                                    const PartitionConfig &partitionConfig);
  void generateMain(std::string &wfilec, std::string &wfiled,
                    std::string &wfilee, int partitionNum,
                    std::vector<Function *> funcList,
                    const PartitionConfig &partitionConfig);
  void generateFree(std::string &wfilec, int partitionNum,
                    std::vector<Function *> funcList,
                    const PartitionConfig &partitionConfig);
  void generateCodeFromModels(std::size_t partitionNum,
                              std::vector<Function *> funcList,
                              const PartitionConfig &partitionConfig,
                              std::string inputPartitionName);
  void generateYamlFile(std::string &wfilec, std::size_t partitionNum,
                        std::vector<Function *> funcList,
                        const PartitionConfig &partitionConfig,
                        std::string inputPartitionName);
  void generateNnobjectFile(std::string &wfilec, std::size_t partitionNum,
                            std::vector<Function *> funcList,
                            const PartitionConfig &partitionConfig);

  void generateNnloadFile(std::string &wfilec, std::size_t partitionNum,
                          std::vector<Function *> funcList,
                          const PartitionConfig &partitionConfig);
  void generateResnetFile(std::string &wfilec, std::size_t partitionNum,
                          std::vector<Function *> funcList,
                          const PartitionConfig &partitionConfig);

  void generateMakeFileLocal(std::string &wfilec, std::size_t partitionNum,
                             std::vector<Function *> funcList,
                             const PartitionConfig &partitionConfig);
  void generateMakeFileNestOs(std::string &wfilec, std::size_t partitionNum,
                              std::vector<Function *> funcList,
                              const PartitionConfig &partitionConfig);
  void generateCMakeListsFile(std::string &wfilec, std::size_t partitionNum,
                              std::vector<Function *> funcList,
                              const PartitionConfig &partitionConfig);
};

} // namespace glow
#endif // GLOW_NESTOSPARTITIONERSCHEDULE_H
