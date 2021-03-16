//
// Created by Misun Yu on 2020/12/09.
//

#ifndef GLOW_NESTPARTITIONERSCHEDULE_H
#define GLOW_NESTPARTITIONERSCHEDULE_H

#include "glow/Runtime/HostManager/HostManager.h"
#include "llvm/Support/CommandLine.h"
#include "glow/Partitioner/PartitionerUtils.h"

#include <glog/logging.h>
#include <boost/algorithm/string.hpp>

//#define ARRAY_LENGTH 100

using namespace glow::runtime;

namespace glow {

class NestPartitionerSchedule {

  std::map<int, std::string> keyList_;
  std::map<int, double> delayTime_;
  std::map<int, int> insertCount_;

  bool flag_ = false;
  bool profileMode_;
  std::string inputPartitionName_;
  std::string outputDir_;

  std::string profilePath_;
  std::string planFileName_;
  std::set<std::string> *fuseOperatorSet_;
  std::set<std::string> *profileKeySet_;

public:
  void setProfilePath(std::string path){profilePath_ = path;};
  std::string getProfilePath(){return profilePath_;};
  void setPartitionPlanFile(std::string file){planFileName_ = file;};
  std::string getPartitionPlanFile(){return planFileName_;};
  void setProfileSet(std::set<std::string> *fuseOperatorSet, std::set<std::string> *profileKeySet) {
    fuseOperatorSet_ = fuseOperatorSet;
    profileKeySet_ = profileKeySet;
  };

  void setFunction(Function* function);
  void setKeyName(std::string key, int i);
  double getDelayTime(int i);
  void setProfileMode(int mode);
  int getProfileMode();
  void setOutputDir(std::string dir) {outputDir_ = dir;};
  void setInputPartitionName(std::string name);
  std::string getInputPartitionName();
  std::string getProfileKey(Node *node);
  std::string replaceAll(std::string &str, std::string pattern, std::string replace);

  void generateDeclaration(std::string& writeFileC, int partitionNum, const PartitionConfig &partitionConfig, int profileMode);
  void generateMain(std::string &wfilec, int partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig, int profileMode);
  void generateFree(std::string &wfilec, int partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig, int profileMode);
  void generateCodeFromModels(std::size_t partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig, int profileMode, std::string inputPartitionName);
  void generateYamlFile(std::string &wfilec, std::size_t partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig, std::string inputPartitionName);
  void generateCMakeListsFile(std::string &wfilec, std::size_t partitionNum, std::vector<Function *> funcList, const PartitionConfig &partitionConfig);

};

} // namespace glow
#endif // GLOW_NESTPARTITIONERSCHEDULE_H