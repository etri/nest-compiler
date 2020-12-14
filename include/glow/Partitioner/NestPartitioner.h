/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GLOW_NEST_PARTITIONER_PARTITIONER_H
#define GLOW_NEST_PARTITIONER_PARTITIONER_H

#include "glow/Partitioner/PartitionerBase.h"
#include "glow/Partitioner/NestPartitionerSchedule.h"
#include "glow/Support/Error.h"

namespace glow {
using namespace runtime;

/// This is struct for saving the profiled information.
//struct PartitionProfileInfo {
//  std::map<std::string, std::map<std::string, float>> backendProfileMap_;
//  std::set<std::string> fuseOperatorSet_;
//  std::set<std::string> profileKeySet_;
//};

class CostNode {
public:
  Node* node_;
  size_t partitionID_ = -1;
  std::string name_;
  std::string backendName_;
  float cost_ = INFINITY;
  bool isFused_ = false;
};

class NodeGroup {
public:
  int partitionID = -1;
  std::string backendName_;
  std::set<CostNode*> outNodeSet_;
  std::vector<CostNode*> nodeList_;
  float totalCost_ = INFINITY;
  bool isParalleBranch = false;

  std::vector<NodeGroup*> branchList_;
};

/// Given a module, partitions each of the its functions into multiple ones
/// based on memory constraints and minimizes the communication cost.
class NestPartitioner final : public PartitionerBase {

    /// The module that needs to be decomposed.
    Module *module_;

    /// The representative function used for partition. We choose the function who
    /// has the largest memory size.
    Function *F_;

    /// True if there are more than 1 type of backends.
    bool multiBackendNames_;

    /// Number of copies of inputs/outputs to assume when calculating mem size.
    unsigned contextCount_{1};

    /// The cost model related to device.
    std::vector<DeviceInfo> deviceInfo_;

    /// The backends created in Partitioner. Used for function optimization.
    std::vector<std::unique_ptr<Backend>> backendHolder_;

    /// The raw backend pointers.
    std::vector<Backend *> backends_;

    /// The map between backend name and BackendInfo.
    std::map<std::string, BackendInfo> backendMap_;

    /// The map between partitions and the logicalDeviceID. The partitions with
    /// the same logicalDeviceID will be assigned into the same physical device.
    std::map<Function *, std::vector<DeviceIDTy>> logicalIDMap_;

    /// The number of logicalDevice IDs, i.e. the number of physical devices
    /// needed after partitions.
    DeviceIDTy logicalDeviceID_;

    /// Total memory (bytes) requested by one module.
    uint64_t memSize_;

    /// Flag to set if the funcitons in the module are areadly optimized. By
    /// default, the optimization should be done in Partitioner due to
    /// heterogeneous partition.
    bool optimized_;

    /// The struct contain user-defined partition info.
    PartitionConfig partitionConfig_;

    std::vector<NodeGroup*> nodeGroups_;
//    PartitionProfileInfo pInfo_;
    std::map<std::string, float> commCostMap_;
    std::set<std::string> backendSet_;
    std::string quantFileName_;

    NestPartitionerSchedule* appGen_ = nullptr;

    /// Initialization. Called in class constructor.
    void init();

    /// Verify the generated functions in module, and \returns error if any
    /// function is invalid. Dump partition logs from \p partitions and \p
    /// mapping.
    Error finalize(const DAGListTy &partitions, const NodeToFunctionMap &mapping);

    void genBackendMap(std::map<std::string, BackendInfo> &backendMap,
                       std::vector<std::unique_ptr<Backend>> &backendsHolder,
                       std::vector<Backend *> &backends);

    /// Returns info for the default device of the backend. If multiple devices,
    /// returns the first one.
    const DeviceInfo &getDeviceInfoForBackend(llvm::StringRef backendName);

public:
  /// \p parent is the module which contains the functions need to be divided.
  /// Here we assume that all the functions in one module belong to a same
  /// "Function Family", that is, without considerting the "dynamic stuff" (i.e.
  /// batch size, input/output shape of each op), all the functions are
  /// identical. The required memory and computation cost for each op can be
  /// found in Module.
  /// The \p devices provides the cost model related to devices.
  /// \p optimized is false by default, which means the functions in this module
  /// are not optimized. \p partitionConfig contains the user defined partition
  /// info.
//  NestPartitioner(Module *parent, const std::vector<DeviceInfo> &devices,
//              bool optimized = false,
//              PartitionConfig partitionConfig = PartitionConfig());

  NestPartitioner(Module *parent, const std::vector<DeviceInfo> &devices,
                  bool optimized = false,
                  PartitionConfig partitionConfig = PartitionConfig(), std::string quantFileName = "");

  void setContextCount(unsigned count) { contextCount_ = count; }

  /// Based on \p partitionConfig passed into Partitioner, do user-defined
  /// partition.
  Expected<DAGListTy>
  partitionFromConfig(const PartitionConfig &partitionConfig,
                      CompilationContext &cctx);

  //Based on \p cctx, setup all data structures needed for a DAG.
  //cctx.prepartitionedConfig contains the Functions which are already
  //partitioned and connected via Placeholders.
  Expected<DAGListTy> setupPrepartitionedModule(CompilationContext &cctx);

  Expected<DAGListTy> partition(CompilationContext &cctx) override;
  Expected<DAGListTy> partition(CompilationContext &cctx, size_t exeType, std::string profilePath, std::string partitionPlanFile, int profileMode);

  void printGraphNodes(Function* function);
  void outPartitionPlanForSingleNode(Function* function, DeviceInfo deviceInfo, std::string filename);
  void outPartitionPlanForFusion(Function* function, DeviceInfo deviceInfo, std::string filename);
  bool isUserDefinedFusable(Node *node);

  void outPartitionPlan(Function* function, std::string filename);
  void loadPartitionPlan(runtime::PartitionConfig* partitionConfig, std::string filename);
  void loadPerformProfileInfo(std::string pdir);
  void getMinCostOfSingleNode(CostNode *cnode);
  void getMinCostOfFusedNode(CostNode* prevCNode, CostNode* curCNode);
  void getMinCostOfFusedNode(CostNode* secondPrevCNode, CostNode* firstPrevCNode, CostNode* curCNode);
  void partitionBfs(std::vector<std::vector<CostNode>>* cnodeBfs);
  int getFusedPartitionCount(Function* function);

  void outPerformProfileForEachNode(Function* function, std::string filename, DeviceInfo device);
  void outPerformProfileForEachNode_1(std::string *keylist, std::string filename, std::string backendName, int array_size);
  void outPerformProfileForFusedNode(Function* function, std::string filename, DeviceInfo device);
  void allocOptimalPUSingleNode(std::vector<std::vector<CostNode>>* function);
  void allocOptimalPUFusedNodes(std::vector<std::vector<CostNode>>* function);
  void findOptimalPartitionsFusedNodes(Function* function);
  void loadFuseOperatorsConfig(std::string fname);
//  std::string getProfileKey(Node *node);
  void makeCostNode(Function* function, std::vector<std::vector<CostNode>>* cnodeBfs);
  bool isFusable(Node* node);
  Node* getFirstInputOpNode(Node* node);
  CostNode* findNodeFromCNodeList(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node);
  void allocatePUToParallelBranch(Function* function, std::vector<std::vector<CostNode>>* cnodeBfs);
//  int getInNodeCnt(Function* function, Node* node);
  //int getOutNodeCnt(Function* function, Node* node);
//  std::vector<CostNode> getOutNodeList(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node, std::string partitionBackend);
  std::set<CostNode*> getOutNodeSet(std::vector<std::vector<CostNode>>* cnodeBfs, NodeGroup* partition);
  NodeGroup* getBranchNodes(std::vector<std::vector<CostNode>>* cnodeBfs, CostNode* cnode, NodeGroup* partition);

  //  std::vector<CostNode> getEndNodeList(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node, int partitionID);
//  bool canSplit(NodeGroup* ng, Node* node);
  CostNode* getCostNode(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node);
  float getPartitionCost(NodeGroup* partition);
  void findAndSetParalleCPUBranch(NodeGroup* partition);
  float getCostOfSingleNode(CostNode *cnode, std::string backendName);

  DAGListTy doPartitioning(llvm::StringRef funcName,
                           std::vector<Function *> funcs, Module *module,
                           NodeToFunctionMap &mapping, bool saveDAG,
                           BackendSpecificNodeInfo &nodeInfo,
                           bool skipCloning = false);

  std::string getDataArrayStr(const Type* value);
//  std::string replaceAll(std::string &str, std::string pattern, std::string replace);
//  PartitionProfileInfo getPartitionProfileInfo(){return pInfo_;};

  NestPartitionerSchedule* getAppGenerator() {
    if(appGen_ == nullptr) {
      appGen_ = new NestPartitionerSchedule;
//      appGen_->setPartitioner(this);
    }
    return appGen_;
  };

};
} // namespace glow
#endif // GLOW_NEST_PARTITIONER_PARTITIONER_H
