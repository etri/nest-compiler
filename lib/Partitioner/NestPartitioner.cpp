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

#include "glow/Partitioner/NestPartitioner.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Partitioner/PartitionerOptimizer.h"
#include "glow/Partitioner/PartitionerUtils.h"
#include "glow/Partitioner/PartitionerValidation.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "glow/Quantization/Serialization.h"

#include <fstream>
namespace glow {
//bool GlowEnableLoadBalancedPartitioning = true;
bool GlowLogNestPartition = false;
bool GlowDumpNestPartition = false;
//static llvm::cl::opt<bool, /* ExternalStorage */ true>
//    GlowEnableLoadBalancedPartitioningOpt(
//        "glow_partitioner_enable_load_balance",
//        llvm::cl::desc(
//            "Enable a partitioner pass to optimize for "
//            "load balance in addition to memory capacity constraints"),
//        llvm::cl::location(GlowEnableLoadBalancedPartitioning));
} // namespace glow
// -log-partition - Command line option to dump Partitioner logs.
static llvm::cl::OptionCategory PartitionerCat("Glow Partitioner Options");
static llvm::cl::opt<bool, /* ExternalStorage */ true> logPartition(
    "nest-log-partition", llvm::cl::desc("Enable logging partition info"),
    llvm::cl::location(glow::GlowLogNestPartition), llvm::cl::cat(PartitionerCat));

/// -dump-partition - Command line option to dump the graph of each partitions
/// by calling F->dumpDAG().
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    dumpPartition("nest-dump-partition",
                  llvm::cl::desc("Enable dumping the graph of each partitions"),
                  llvm::cl::location(glow::GlowDumpNestPartition),
                  llvm::cl::cat(PartitionerCat));

using namespace glow;
using llvm::isa;


void NestPartitioner::init() {
  memSize_ = module_->getConstantsSize();
  logicalDeviceID_ = 0;
  multiBackendNames_ = false;
  for (size_t i = 1, e = deviceInfo_.size(); i < e; i++) {
    if (deviceInfo_[i].backendName != deviceInfo_[0].backendName) {
      multiBackendNames_ = true;
      break;
    }
  }
}

Error NestPartitioner::finalize(const DAGListTy &partitions,
                                const NodeToFunctionMap &mapping) {

  // NOTE: Cannot validate the functions after partitioning here. The validation
  // needs the backend specific verifier. Tensor layouts, for example, might
  // have gone from canonical form to backend specific form.

  if (logPartition) {
    LOG(INFO) << "The number of partitions is : "
              << module_->getFunctions().size();
    LOG(INFO) << "Dumping partitioning DAG to DAG.dot file.";
    dumpDAG("DAG.dot", partitions);
    logPartitionInfo(mapping);
  }

  // Dump the graph of each function after partitioning.
  if (dumpPartition) {
    for (const auto &node : partitions[0].nodes) {
      Function *subF = module_->getFunction(node->name);
      if (!subF) {
        // If we fail dump partition info for debugging.
        logPartitionInfo(mapping);
        return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                        "Invalid function name " + node->name);
      }
      subF->dumpDAG("partitionLogicalID" +
                    std::to_string(node->logicalDevices[0]) + "__" +
                    subF->getName().str() + "__" + node->backendName + ".dot");
    }
  }
  return Error::success();
}


// Current only partition the representative function.
DAGListTy NestPartitioner::doPartitioning(
    llvm::StringRef funcName, std::vector<Function *> funcs, Module *module,
    NodeToFunctionMap &mapping, bool saveDAG, BackendSpecificNodeInfo &nodeInfo,
    bool skipCloning) {
  DAGListTy partitions;
  // Add a dummy node to make sure that a DAG has a single entrance.
  DAGNodePtr DAGRoot = glow::make_unique<DAGNode>();
  DAGNodePtrVec nodes;
  DAGRoot->logicalDevices = {0};
  DAGRoot->name = funcName;
  DAGRoot->module = module;
  DAGNode *root = DAGRoot.get();

  llvm::DenseMap<Node *, Node *> currToNew;

  if (!skipCloning) {
    // Clone nodes into target partition. Update nodeInfo as necessary.
    for (size_t i = 0, e = funcs.size(); i < e; i++) {
      for (auto &N : funcs[i]->getNodes()) {
        auto *clone = N.clone();
        currToNew[&N] = clone;
        mapping[&N]->addNode(clone);

        // If needed, update NodeInfo to point old Node's info to clone.
        auto itF = nodeInfo.find(funcs[i]);
        if (itF == nodeInfo.end()) {
          continue;
        }
        auto &currNodeInfo = itF->second;
        auto itN = currNodeInfo.find(&N);
        if (itN != currNodeInfo.end()) {
          currNodeInfo[clone] = std::move(itN->second);
          // Erase old NodeInfo; current Nodes will be eliminated later when
          // input funcs will be erased.
          currNodeInfo.erase(itN);
        }
      }
    }
  }

  // For any dependency that crosses a partition, add a placeholder and save
  // node. Record the dependence in the function graph.
  std::unordered_map<NodeValue, Placeholder *> placeholders;
  llvm::DenseMap<Function *, DAGNode *> funcDAG;
  for (auto *subF : mapping.getPartitions()) {

    if (funcDAG.find(subF) == funcDAG.end()) {
      std::unique_ptr<DAGNode> subDAG = createDAGNodeFromFun(subF, mapping);
      funcDAG[subF] = subDAG.get();
      nodes.push_back(std::move(subDAG));
    }

    // Link subF to its parents.
    std::set<Function *> parents;
    for (auto &N : subF->getNodes()) {
      for (int inp = 0, e = N.getNumInputs(); inp < e; inp++) {
        auto input = N.getNthInput(inp);
        // No need to check Constant since it won't be the result of another
        // function.
        if (isa<Constant>(input.getNode())) {
          continue;
        }

        Function *inputF = nullptr;
        // It is possible that one input is the output of anther function.
        if (Placeholder *ph = llvm::dyn_cast<Placeholder>(input.getNode())) {
          //std::cout << "ph name = " << ph->getName().str() << std::endl;
          for (auto &user : ph->getUsers()) {
            if (auto *save = llvm::dyn_cast<SaveNode>(user.getUser())) {
              placeholders[input] = save->getPlaceholder();
              inputF = mapping[user.getUser()];
              break;
            }
          }
          if (!inputF) {
            continue;
          }
        }

        if (!inputF) {
          inputF = mapping[input.getNode()];
        }
        if (subF == inputF) {
          continue;
        }
        // Check if a DAGNode for subF's parent is created or not. If not,
        // create one.
        if (funcDAG.find(inputF) == funcDAG.end()) {
          std::unique_ptr<DAGNode> subDAG =
              createDAGNodeFromFun(inputF, mapping);
          funcDAG[inputF] = subDAG.get();
          nodes.push_back(std::move(subDAG));
        }

        // subF is a child of inputF, inputF is a parent of subF.
        if (parents.find(inputF) == parents.end()) {
          funcDAG[inputF]->children.push_back(funcDAG[subF]);
          funcDAG[subF]->parents.push_back(funcDAG[inputF]);
          parents.insert(inputF);
        }
        // If we've already created a placeholder for this dependence, use it.
        auto it = placeholders.find(input);
        if (it != placeholders.end()) {
          N.setNthInput(inp, it->second);
          continue;
        }

        // Create a new placeholder to represent this dependence.
        auto placeholderName = input.getNode()->getName().str();
        auto *save = inputF->createPartitionSave(placeholderName, input);
        auto *tmp = save->getPlaceholder();
        placeholders[input] = tmp;
        N.setNthInput(inp, tmp);

      }
    }
  }

  if (saveDAG) {
    DAG dag;
    dag.root = std::move(DAGRoot);
    dag.nodes = std::move(nodes);
    partitions.push_back(std::move(dag));
  }

  if (!skipCloning) {
    // Update links between nodes in the cloned functions. Add placeholders (and
    // save nodes) where a link crosses a partition boundary.
    for (auto *subF : mapping.getPartitions()) {
      for (auto &N : subF->getNodes()) {
        for (int inp = 0, e = N.getNumInputs(); inp < e; inp++) {
          auto input = N.getNthInput(inp);
          if (isa<Storage>(input.getNode())) {
            continue;
          }
          // Link this node to the clone of its input.
          auto *clone = currToNew[input.getNode()];
          N.setNthInput(inp, NodeValue(clone, input.getResNo()));
        }
      }
    }
  }

  // For all DAGNode without parents, link them to the root DAG.
  for (auto *subF : mapping.getPartitions()) {
    if (funcDAG[subF]->parents.size() == 0) {
      funcDAG[subF]->parents.push_back(root);
      root->children.push_back(funcDAG[subF]);
    }
  }
  return partitions;
}

NestPartitioner::NestPartitioner(Module *parent, const std::vector<DeviceInfo> &devices,
                                 bool optimized, PartitionConfig partitionConfig, std::string quantFileName)
    : module_(parent), deviceInfo_(devices), optimized_(optimized),
      partitionConfig_(partitionConfig), quantFileName_(quantFileName) {
  init();
}

void NestPartitioner::genBackendMap(
    std::map<std::string, BackendInfo> &backendMap,
    std::vector<std::unique_ptr<Backend>> &backendsHolder,
    std::vector<Backend *> &backends) {
  // If the backends are created already, we use them directly.
  bool hasBackends = backends_.size() != 0;
  if (hasBackends) {
    DCHECK(backends_.size() == deviceInfo_.size())
        << "number of backends and devices is not match.";
  }

  int n = 0;
  for (size_t i = 0, e = deviceInfo_.size(); i < e; i++) {
    std::string backendName = deviceInfo_[i].backendName;
    if (hasBackends) {
      DCHECK(backends_[i]->getBackendName() == backendName)
          << "Backend Type mismatch.";
    }
    if (backendMap.find(backendName) == backendMap.end()) {
      BackendInfo backendInfo;
      backendInfo.num = 1;
      // We assume that for the same type of devices, the available memory size
      // is the same.
      // TODO : will improve the algorithm for different memory size.
      backendInfo.memSize = deviceInfo_[i].availableMemory;
      backendInfo.peakDramBw = deviceInfo_[i].peakDramBw;
      backendInfo.peakSramBw = deviceInfo_[i].peakSramBw;
      backendInfo.sramCapacity = deviceInfo_[i].sramCapacity;
      backendInfo.peakCompute = deviceInfo_[i].peakCompute;
      backendInfo.nonSupportedNodesKinds =
          generateNodeKindsSet(deviceInfo_[i].nonSupportedNodes);
      backendInfo.supportedNodesKinds =
          generateNodeKindsSet(deviceInfo_[i].supportedNodes);
      if (hasBackends) {
        backendInfo.backend = backends_[i];
      } else {
        backendsHolder.emplace_back(createBackend(backendName));
        backendInfo.backend = backendsHolder[n++].get();
      }
      backendMap[backendName] = backendInfo;
      backends.push_back(backendMap[backendName].backend);
    } else {
      backendMap[backendName].num += 1;
      // Since we are currently assuming one value it should be the max.
      backendMap[backendName].memSize = std::max(
          backendMap[backendName].memSize, deviceInfo_[i].availableMemory);
    }
  }
}

const DeviceInfo &
NestPartitioner::getDeviceInfoForBackend(llvm::StringRef backendName) {
  for (DeviceInfo &devInfo : deviceInfo_) {
    if (devInfo.backendName == backendName)
      return devInfo;
  }
  llvm_unreachable("Each backend should have at least one device");
}

Expected<DAGListTy>
NestPartitioner::partitionFromConfig(const PartitionConfig &partitionConfig,
                                 CompilationContext &cctx) {
  std::cout << std::endl << "Partitioning Graph-IR.." << std::endl;

//  std::cout << "== [Start] partitionFromConfigForNest == " << std::endl;

  DAGListTy partitions;
  // Prepare the mapping between BackendName and BackendInfo.
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);
  Function *F = module_->getFunction(partitionConfig.funcName);

  std::string fname = module_->getFunctions().front()->getName();
//  std::cout << "function name = " << fname << std::endl;

  if (!F) {
    return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                    strFormat("Can't find function %s in current module.",
                              F->getName().str().data()));
  }
//  std::cout << "backendNames = " << partitionConfig.backendNames.size() << std::endl;
//  std::cout << "logicalIDs = " << partitionConfig.logicalIDs.size() << std::endl;
  DCHECK(
      partitionConfig.numOfPartitions == partitionConfig.backendNames.size() &&
      partitionConfig.numOfPartitions == partitionConfig.partitionNames.size())
  << "Invalid user-defined partition config.";

  if (partitionConfig.backendHints.size()) {
    DCHECK(partitionConfig.numOfPartitions ==
           partitionConfig.backendHints.size())
    << "Invalid user-defined partition config (backendHints).";
  }

  NodeToFunctionMap partitionMap;
  std::vector<Function *> funcList;
  std::unordered_set<size_t> unused;
  std::vector<NodesSet> nodesSets(partitionConfig.numOfPartitions);
  // Create partitions based on the given number and names.
  for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
    Function *newF = module_->createFunction(partitionConfig.partitionNames[i]);
    funcList.push_back(newF);
    partitionMap.createPartition(newF, partitionConfig.backendNames[i]);
    unused.insert(i);

    //std::cout << "f name = " << newF->getName().str() << std::endl;
    //std::cout << "f filename = " << newF->getFilename().c_str() << std::endl;
  }

  // Map the nodes the the partitions.
  std::vector<Node *> unMapped;
  for (auto &node : F->getNodes()) {
    auto iter = partitionConfig.nodeToPartition.find(node.getName());
    if (iter == partitionConfig.nodeToPartition.end()) {
      // If a node in F is not in the node to partition mapping, put it into
      // unMaped list.
      unMapped.push_back(&node);
      //std::cout << "Unmapped = " << node.getName().str() << std::endl;
    } else {
      size_t partitionID = iter->second;
      DCHECK(partitionID < partitionConfig.numOfPartitions)
      << "Invalid partition id :" << partitionID;
      partitionMap.add(&node, funcList[partitionID]);
      unused.erase(partitionID);
      nodesSets[partitionID].insert(&node);
      //std::cout << "Mapped = " << node.getName().str() << std::endl;
    }
  }

  // If there is unused partition and unmapped nodes, map those nodes to the
  // unused partition.
  if (unMapped.size()) {
    DCHECK(unused.size() == 1) << "There must be exactly 1 unused partition.";
    auto partitionID = *(unused.begin());
    for (auto &node : unMapped) {
      partitionMap.add(node, funcList[partitionID]);
      nodesSets[partitionID].insert(node);
    }
  }

  // Set backend hints if they exist
  if (partitionConfig.backendHints.size()) {
    for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
      auto func = funcList[i];
      partitionMap.setBackendHints(func, partitionConfig.backendHints[i]);
    }
  }

  // If logical device assignments are provided use them otherwise assign them.
  if (partitionConfig.logicalIDs.size()) {
    DCHECK(partitionConfig.numOfPartitions ==
           partitionConfig.logicalIDs.size());
    for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
      auto func = funcList[i];

      for (auto logicalDevice : partitionConfig.logicalIDs[i]) {
        partitionMap.appendLogicalDeviceID(func, logicalDevice);
      }
    }
  } else {
    // Logical device ID validation.
    logicalDeviceID_ = assignLogicalDeviceID(partitionMap, backendMap_);
  }
  // Add replication count to config if provided.
  for (auto &replicationAssignment : partitionConfig.replicationCount) {
    auto func = funcList.at(replicationAssignment.first);
    partitionMap.addReplicationCount(func, replicationAssignment.second);
  }

  RETURN_IF_ERR(logicalDevicesValidation(partitionMap, backendMap_));

  // Do partition.
  partitions = doPartitioning(F->getName(), {F}, module_, partitionMap,
      /* saveDAG */ true,
                              cctx.backendOpts.backendSpecificNodeInfo);
  module_->eraseFunction(F);

  std::cout << "Graph-IR partitioning is done." << std::endl << std::endl;

  // DAG validation.
  RETURN_IF_ERR(dagValidation(partitions[0]));

  // profileMode : 0 -> no time, 1 -> time add
  // main.cpp : yaml create logic add

 // Loader loader;

  if(appGen_->getProfileMode() == 0) {
    std::cout << "Generating linking code (main.cpp).." << std::endl;
  } else if (appGen_->getProfileMode() == 1) {
    std::cout << "Generating profiling code (main.cpp).." << std::endl;
  }

  appGen_->generateCodeFromModels(funcList.size(), funcList, partitionConfig, appGen_->getProfileMode(),  appGen_->getInputPartitionName());

  if(appGen_->getProfileMode() == 0) {
    std::cout << "Linking-code generation is done." << std::endl << std::endl;
  } else if (appGen_->getProfileMode() == 1) {
    std::cout << "Profiling-code generation is done." << std::endl << std::endl;
  }

  // Verify the function.
  std::cout << "Optimizing Graph-IR for CPU and VTA.." << std::endl;

  for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
    auto func = funcList[i];

    Backend* B = backendMap_[partitionConfig.backendNames[i]].backend;

    if (!optimized_) {
      if(B->getBackendName().compare("CPU")) {
        // Specific configurations.
        cctx.precisionConfig.quantMode = QuantizationMode::Quantize;
        cctx.precisionConfig.quantConfig.schema = quantization::Schema::SymmetricWithPower2Scale;
        llvm::StringRef filename = quantFileName_;
        // std::cout << "quant filename = " << filename.str() << std::endl;
        deserializeProfilingInfosFromYaml(
            filename, cctx.precisionConfig.quantConfig.graphPreLowerHash, cctx.precisionConfig.quantConfig.infos);
        cctx.precisionConfig.quantConfig.checkGraphPreLowerHash = false;
      }

      RETURN_IF_ERR(::glow::optimizeFunction(func, *B, cctx));
    }

    DCHECK(func->verify()) << "Conversion led to invalid function";
  }

  std::cout << "Graph-IR optimization is done.." << std::endl;

  RETURN_IF_ERR(finalize(partitions, partitionMap));

//  std::cout << "== [End] partitionFromConfigForNest == " << std::endl;
  return std::move(partitions);
}

Expected<DAGListTy>
NestPartitioner::setupPrepartitionedModule(CompilationContext &cctx) {
  const PrePartitionedConfig &config = *cctx.prepartitionedConfig;

  RETURN_ERR_IF_NOT(
      !multiBackendNames_,
      "Do not support multiple backend kinds in prepartitioned flow.");

  // Prepare the mapping between BackendName and BackendInfo.
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);

  const std::vector<Function *> &funcs = config.funcs;

  Backend *B = backends[0];
  auto backendName = B->getBackendName();

  // Optimize all Functions if necessary.
  if (!optimized_) {
    for (Function *F : funcs) {
      RETURN_IF_ERR(::glow::optimizeFunction(
          F, *B, cctx, &getDeviceInfoForBackend(backendName)));
    }
  }

  NodeToFunctionMap partitionMap;
  // Create partitions based on the given number and names.
  for (size_t i = 0, e = funcs.size(); i < e; i++) {
    partitionMap.createPartition(funcs[i], deviceInfo_[0].backendName);
  }

  // Map the nodes the the partitions.
  for (Function *F : funcs) {
    for (auto &node : F->getNodes()) {
      partitionMap.add(&node, F);
    }
  }

  // Validate memory usage.
  for (Function *F : funcs) {
    partitionMap.setGraphMemInfo(F, getFunctionMemory(F));
  }
  RETURN_IF_ERR(memoryUsageValidation(partitionMap, backendMap_));

  // If logical device assignments are provided use them otherwise assign them.
  DCHECK(funcs.size() == config.logicalIDs.size());
  for (size_t i = 0; i < funcs.size(); i++) {
    Function *F = funcs[i];
    for (auto logicalDevice : config.logicalIDs[i]) {
      partitionMap.appendLogicalDeviceID(F, logicalDevice);
    }
  }
  RETURN_IF_ERR(logicalDevicesValidation(partitionMap, backendMap_));

  // Copy in or validate all members of the PPC.
  RETURN_ERR_IF_NOT(
      funcs.size() == config.backendSpecificOpts.size(),
      "Number of Functions must equal number of backendSpecificOpts");
  RETURN_ERR_IF_NOT(funcs.size() == config.backendHints.size(),
                    "Number of Functions must equal number of backendHints");
  RETURN_ERR_IF_NOT(funcs.size() == config.replicationCounts.size(),
                    "Number of Functions must equal");
  RETURN_ERR_IF_NOT(
      funcs.size() == config.backendNames.size() || config.backendNames.empty(),
      "If there are backendNames specified, there must be one per Function");
  for (size_t i = 0, e = funcs.size(); i < e; i++) {
    Function *F = funcs[i];
    partitionMap.setBackendSpecificOpts(F, config.backendSpecificOpts[i]);
    partitionMap.setBackendHints(F, config.backendHints[i]);
    partitionMap.addReplicationCount(F, config.replicationCounts[i]);
    if (!config.backendNames.empty()) {
      RETURN_ERR_IF_NOT(backendName == config.backendNames[i],
                        "Mismatch on backendName for partition");
    }
  }

  // Do partition.
  DAGListTy partitions = doPartitioning(
      config.funcName, funcs, module_, partitionMap,
      /* saveDAG */ true, cctx.backendOpts.backendSpecificNodeInfo,
      /* skipCloning */ true);

  // DAG validation.
  RETURN_IF_ERR(dagValidation(partitions[0]));

  // Verify the function.
  for (Function *F : funcs) {
    DCHECK(F->verify()) << "Conversion led to invalid function";
  }

  RETURN_IF_ERR(finalize(partitions, partitionMap));

  return std::move(partitions);
}

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <dirent.h>
#include <vector>

void NestPartitioner::loadPartitionPlan(runtime::PartitionConfig* partitionConfig, std::string filename)
{
//  std::cout << "----------- loadPartitionPlan ---------------" << std::endl;
  std::string partitionPlanFile(filename);
  //string partitionPlanFile("/Users/msyu/Development/partitionPlanTest.yaml");

  std::map<std::string, std::string> pmap = deserializeStrStrMapFromYaml(partitionPlanFile);
  partitionConfig->funcName = pmap.find("funcName")->second;
  partitionConfig->numOfPartitions = atol(pmap.find("numOfPartitions")->second.c_str());
  boost::split(partitionConfig->backendNames, pmap.find("backendNames")->second, [](char c){return c == ':';});
  boost::split(partitionConfig->partitionNames, pmap.find("partitionNames")->second, [](char c){return c == ':';});

  // cout << "funcName = " <<  partitionConfig->funcName << endl;

  std::vector<std::string> results;
  boost::split(results, pmap.find("nodeToPartition")->second, [](char c){return c == ':';});
  for (auto &s : results) {
    std::vector<std::string> nodesStr;
    boost::split(nodesStr, s, [](char c){return c == '-';});

    partitionConfig->nodeToPartition[nodesStr[0]] = atoi(nodesStr[1].c_str());
    //std::cout << "node str = " <<  nodesStr[0] << std::endl;
    //std::cout << "node partition = " << nodesStr[1].c_str() << std::endl;
  }

  results.clear();
  boost::split(results, pmap.find("logicalIDs")->second, [](char c){return c == ':';});
  for (auto &s : results) {
    std::vector<std::string> idsStr;
    boost::split(idsStr, s, [](char c){return c == '-';});

    std::vector<unsigned int> ids;
    for (auto &s : idsStr) {
      ids.push_back(atoi(s.c_str()));
    }
    partitionConfig->logicalIDs.push_back(ids);
  }
}

void NestPartitioner::outPerformProfileForEachNode(Function* function, std::string filename, DeviceInfo device)
{
//  std::cout << "== outPerformProfile : " << device.backendName << " == " << std::endl;

  std::ofstream profileFile(filename);
  if (profileFile.is_open()) {

    BFSLevel bfs = getBFSLevel(function);
    size_t level = bfs.size();
   // std::set<std::string> nameSet;

    profileFile << "---" << std::endl;
    profileFile << "backendName: " << device.backendName << std::endl;
    profileFile << "commCost: " << "0.0" << std::endl;

    for (int i = level - 1; i >= 0; i--) {
      for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
        Node* node = bfs[i][j];
        std::string key = appGen_->getProfileKey(node);
        if(appGen_->getPartitionProfileInfo().profileKeySet_.find(key) == appGen_->getPartitionProfileInfo().profileKeySet_.end()) {
          appGen_->getPartitionProfileInfo().profileKeySet_.insert(key);
            profileFile << key << ": " << "1" << std::endl;
        }
      }
    }

    profileFile << "..." << std::endl;
    profileFile.close();
  }
}

void NestPartitioner::outPerformProfileForEachNode_1(std::string *keylist, std::string filename, std::string backendName, int array_size)
{
//  std::cout << "== outPerformProfileForEachNode_1 start : " << backendName << " == " << std::endl;
//  std::cout << "keylist size : " << array_size << std::endl;

  std::ofstream profileFile(filename);
  if (profileFile.is_open()) {

    int array_count = 0;

    profileFile << "---" << std::endl;
    profileFile << "backendName: " << backendName << std::endl;
    profileFile << "commCost: " << "0.0" << std::endl;

    for (int i = 0; i < array_size; i++) {
      if(keylist[i].compare("blank"))
        profileFile << keylist[i] << ": " << appGen_->getDelayTime(array_count++) << std::endl;
    }

    profileFile << "..." << std::endl;
    profileFile.close();
  }
}

Node* NestPartitioner::getFirstInputOpNode(Node* node) {

    if(node->getNumInputs() > 0) {
        for(int in = 0; in < node->getNumInputs(); in++) {
            Node* inNode = node->getNthInput(in);
            if(!(inNode->getKind() == Kinded::Kind::ConstantKind ||
                 inNode->getKind() == Kinded::Kind::PlaceholderKind)) {
                return inNode;
            }
        }
    }
    return nullptr;
}

CostNode* NestPartitioner::findNodeFromCNodeList(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node) {

    size_t level = (*cnodeBfs).size();
    for (int i = 0; i < level; i++) {
        for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {
            if(node == (&(*cnodeBfs)[i][j])->node_)
                return &((*cnodeBfs)[i][j]);
        }
    }
    return nullptr;
}

bool NestPartitioner::isFusable(Node* node) {
    if(node->getNumInputs() > 1) {
        size_t usedCnt = 0;
        for(int in = 0; in < node->getNumInputs(); in++) {
            Node* inNode = node->getNthInput(in);
            if(!(inNode->getKind() == Kinded::Kind::ConstantKind ||
                 inNode->getKind() == Kinded::Kind::PlaceholderKind)) {
                usedCnt++;
            }
        }

        if(usedCnt > 1) {
            //std::cout << "More than one inputs!" << std::endl;
            return false;
        }
    }

    if(!node->hasOneUse() && node->getKind() != Kinded::Kind::SaveNodeKind) {
        //std::cout << "More than one users!" << std::endl;
        return false;
    }

    if(node->getKind() == Kinded::Kind::SaveNodeKind) {
        //std::cout << "Save node!" << std::endl;
        return false;
    }

    return true;
}

bool NestPartitioner::isUserDefinedFusable(Node *node)
{
    Node * firstPrevNode = getFirstInputOpNode(node);
    if(firstPrevNode == nullptr || !isFusable(firstPrevNode))
      return false;

    std::string doubleFusedKindName = std::string(firstPrevNode->getKindName()) + "-" + node->getKindName();
    //std::cout << "doubleFusedKindName = " << doubleFusedKindName << std::endl;
    if(appGen_->getPartitionProfileInfo().fuseOperatorSet_.find(doubleFusedKindName) != appGen_->getPartitionProfileInfo().fuseOperatorSet_.end()) {
      std::string profileStr = appGen_->getProfileKey(firstPrevNode) + "+" + appGen_->getProfileKey(node);
      //if(pInfo_.profileKeySet_.find(profileStr) == pInfo_.profileKeySet_.end()) {
      //  pInfo_.profileKeySet_.insert(profileStr);
        //profileFile << profileStr << " : 1" << std::endl;
        // std::cout << "Added!" << std::endl;
      //}
        return true;
    }

    Node *secondPrevNode = getFirstInputOpNode(firstPrevNode);
    if(secondPrevNode == nullptr || !isFusable(secondPrevNode))
      return false;

    std::string tripleFusedKindName = std::string(secondPrevNode->getKindName()) + "-" +
                                      firstPrevNode->getKindName() + "-" +
                                      node->getKindName();

    //std::cout << "tripleFusedKindName = " << tripleFusedKindName << std::endl;
    if(appGen_->getPartitionProfileInfo().fuseOperatorSet_.find(tripleFusedKindName) != appGen_->getPartitionProfileInfo().fuseOperatorSet_.end()) {
      //std::string profileStr = getProfileKey(secondPrevNode) + "+" + getProfileKey(firstPrevNode) + "+" + getProfileKey(node);
      //if(pInfo_.profileKeySet_.find(profileStr) == pInfo_.profileKeySet_.end()) {
      //  pInfo_.profileKeySet_.insert(profileStr);
        //profileFile << profileStr << " : 1" << std::endl;
        //std::cout << "Added!" << std::endl;
      //}
      return true;
    }
    return false;
}

void NestPartitioner::outPerformProfileForFusedNode(Function* function, std::string filename, DeviceInfo device)
{
//    std::cout << "== outPerformProfileForFusedNode : " << device.backendName << " == " << std::endl;

    std::ofstream profileFile(filename);
    if (profileFile.is_open()) {

        BFSLevel bfs = getBFSLevel(function);
        size_t level = bfs.size();

        profileFile << "---" << std::endl;
        profileFile << "backendName: " << device.backendName << std::endl;
        profileFile << "commCost: " << "0.0" << std::endl;

        for (int i = level - 1; i >= 0; i--) {
            for (size_t j = 0; j < bfs[i].size(); j++) {
                Node *firstPrevNode;
                Node *secondPrevNode;
                Node *node = bfs[i][j];

                if(!isFusable(node))
                    continue;

                firstPrevNode = getFirstInputOpNode(node);
                if(firstPrevNode == nullptr || !isFusable(firstPrevNode))
                    continue;

                std::string doubleFusedKindName = std::string(firstPrevNode->getKindName()) + "-" + node->getKindName();
                //std::cout << "doubleFusedKindName = " << doubleFusedKindName << std::endl;
                if(appGen_->getPartitionProfileInfo().fuseOperatorSet_.find(doubleFusedKindName) != appGen_->getPartitionProfileInfo().fuseOperatorSet_.end()) {
                    std::string profileStr = appGen_->getProfileKey(firstPrevNode) + "+" + appGen_->getProfileKey(node);
                    if(appGen_->getPartitionProfileInfo().profileKeySet_.find(profileStr) == appGen_->getPartitionProfileInfo().profileKeySet_.end()) {
                      appGen_->getPartitionProfileInfo().profileKeySet_.insert(profileStr);
                        profileFile << profileStr << " : 1" << std::endl;
                       // std::cout << "Added!" << std::endl;
                    }
                }

                secondPrevNode = getFirstInputOpNode(firstPrevNode);
                if(secondPrevNode == nullptr || !isFusable(secondPrevNode))
                    continue;

                std::string tripleFusedKindName = std::string(secondPrevNode->getKindName()) + "-" +
                                                  firstPrevNode->getKindName() + "-" +
                                                  node->getKindName();

                //std::cout << "tripleFusedKindName = " << tripleFusedKindName << std::endl;
                if(appGen_->getPartitionProfileInfo().fuseOperatorSet_.find(tripleFusedKindName) != appGen_->getPartitionProfileInfo().fuseOperatorSet_.end()) {
                    std::string profileStr = appGen_->getProfileKey(secondPrevNode) + "+" + appGen_->getProfileKey(firstPrevNode) + "+" + appGen_->getProfileKey(node);
                    if(appGen_->getPartitionProfileInfo().profileKeySet_.find(profileStr) == appGen_->getPartitionProfileInfo().profileKeySet_.end()) {
                      appGen_->getPartitionProfileInfo().profileKeySet_.insert(profileStr);
                        profileFile << profileStr << " : 1" << std::endl;
                        //std::cout << "Added!" << std::endl;
                    }
                }
            }
        }

        profileFile << "..." << std::endl;
        profileFile.close();
    }
}

void NestPartitioner::outPartitionPlanForSingleNode(Function* function, DeviceInfo deviceInfo, std::string filename)
{
//  std::cout << "== outPartitionPlanForSingleNode : " << filename << " == " << std::endl;

  //if(!useFusing) {
    std::ofstream planFile(filename);
    if (planFile.is_open()) {

      planFile << "---" << std::endl;
      planFile << "funcName: " << function->getName().str() << std::endl;
      planFile << "numOfPartitions: " << function->getNodes().size() << std::endl;

      std::string backendNameStr;
      std::string partitionNameStr;
      std::string nodeToPartitionStr;
      std::string logicalIDStr;

      ///
      BFSLevel bfs = getBFSLevel(function);
      size_t level = bfs.size();
      size_t idx = 0;

      for (int i = level - 1; i >= 0; i--) {
        for (size_t j = 0; j < bfs[i].size(); j++) {

          Node *node = bfs[i][j];
          //    for (auto& node : function->getNodes()) {
          if (node->getKind() == Kinded::Kind::SaveNodeKind)
            continue;

          partitionNameStr += ("p" + std::to_string(idx));
          backendNameStr += (deviceInfo.backendName);
          logicalIDStr += std::to_string(deviceInfo.deviceID);
          nodeToPartitionStr +=
              (node->getName().str() + "-" + std::to_string(idx));

          partitionNameStr += ":";
          backendNameStr += ":";
          logicalIDStr += ":";

          if (idx < function->getNodes().size()-2) {
            nodeToPartitionStr += "\n  :";
          }

          idx++;
        }
      }
        partitionNameStr += ("p" + std::to_string(idx));
        backendNameStr += "CPU";
        logicalIDStr += "0";

        planFile << "backendNames: " << backendNameStr << std::endl;
        planFile << "partitionNames: " << partitionNameStr << std::endl;
        planFile << "nodeToPartition: " << nodeToPartitionStr << std::endl;
        planFile << "logicalIDs: " << logicalIDStr << std::endl;

        planFile << "..." << std::endl;
        planFile.close();

    }
  //}
//  std::cout << "----------- [end] outPartitionPlanForSingleNode ---------------" << std::endl;
}

int NestPartitioner::getFusedPartitionCount(Function* function)
{
  int cnt = 1;
  BFSLevel bfs = getBFSLevel(function);
  size_t level = bfs.size();

  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0; j < bfs[i].size(); j++) {

      Node *node = bfs[i][j];
      //    for (auto& node : function->getNodes()) {
      if (node->getKind() == Kinded::Kind::SaveNodeKind)
        continue;

      if(isUserDefinedFusable(node))
        cnt--;

      cnt++;
    }
  }
  return (cnt);
}

void NestPartitioner::outPartitionPlanForFusion(Function* function, DeviceInfo deviceInfo, std::string filename)
{
//  std::cout << "== outPartitionPlanForFusedNode : " << filename << " == " << std::endl;

  //if(!useFusing) {
  std::ofstream planFile(filename);
  if (planFile.is_open()) {

    int partitionSize = getFusedPartitionCount(function);

    planFile << "---" << std::endl;
    planFile << "funcName: " << function->getName().str() << std::endl;
    planFile << "numOfPartitions: " << partitionSize << std::endl;

    std::string backendNameStr;
    std::string partitionNameStr;
    std::string nodeToPartitionStr;
    std::string logicalIDStr;

    ///
    //std::cout << "partitionSize = " << partitionSize << std::endl;

    BFSLevel bfs = getBFSLevel(function);
    size_t level = bfs.size();
    size_t idx = 0;
    size_t pcnt = 0;

    for (int i = level - 1; i >= 0; i--) {
      for (size_t j = 0; j < bfs[i].size(); j++) {

        Node *node = bfs[i][j];
        //    for (auto& node : function->getNodes()) {
        if (node->getKind() == Kinded::Kind::SaveNodeKind)
          continue;

        if(isUserDefinedFusable(node))
          idx--;

        if(pcnt < partitionSize-1) {
          partitionNameStr += ("p" + std::to_string(pcnt) + ":");
          backendNameStr += (deviceInfo.backendName + ":");
          logicalIDStr += (std::to_string(deviceInfo.deviceID) + ":");
        }

        nodeToPartitionStr +=
            (node->getName().str() + "-" + std::to_string(idx));


        if (pcnt++ < function->getNodes().size()-2) {
          nodeToPartitionStr += "\n  :";
        }

        idx++;

      }
    }
    partitionNameStr += ("p" + std::to_string(partitionSize-1));
//    backendNameStr += deviceInfo.backendName;
//    logicalIDStr += std::to_string(deviceInfo.deviceID);

    backendNameStr += "CPU";
    logicalIDStr += "0";

    planFile << "backendNames: " << backendNameStr << std::endl;
    planFile << "partitionNames: " << partitionNameStr << std::endl;
    planFile << "nodeToPartition: " << nodeToPartitionStr << std::endl;
    planFile << "logicalIDs: " << logicalIDStr << std::endl;

    planFile << "..." << std::endl;
    planFile.close();

  }
  //}
//  std::cout << "----------- [end] outPartitionPlanForSingleNode ---------------" << std::endl;
}

void NestPartitioner::outPartitionPlan(Function* function, std::string filename)
{
//  std::cout << "\n=== Partition plan file: " + filename + " ===" << std::endl << std::endl;
//  //split a graph and group adjacent nodes with the same backend.
  std::ofstream planFile(filename);
  if (planFile.is_open()) {

    if(nodeGroups_.size() > 1) {
      int partitionSize = nodeGroups_.back()->partitionID + 2;
      bool isThereParallelBranch = false;

      planFile << "---" << std::endl;
      planFile << "funcName: " << function->getName().str() << std::endl;
      //planFile << "numOfPartitions: " << nodeGroups_.size() + 1<< std::endl;
      planFile << "numOfPartitions: " << partitionSize<< std::endl;

      std::string backendNameStr;
      std::string partitionNameStr;
      std::string nodeToPartitionStr;
      std::string logicalIDStr;

      std::cout << "== Parallel partitions: ";

      size_t idx = 0;
      for (auto partition : nodeGroups_) {

        if (partition->branchList_.size() > 1 && partition->isParalleBranch) {
          isThereParallelBranch = true;
          for (auto branch : partition->branchList_) {

            std::reverse(branch->nodeList_.begin(), branch->nodeList_.end());
            //std::cout << "branch backend = " << branch->backendName_ << ", " << branch->nodeList_.size() << std::endl;

            DeviceInfo dinfo = getDeviceInfoForBackend(branch->backendName_);
            backendNameStr += (branch->backendName_ + ":");
            logicalIDStr += (std::to_string(dinfo.deviceID) + ":");
            partitionNameStr +=
                ("p" + std::to_string(branch->partitionID) + ":");

            for (auto cnode : branch->nodeList_) {
              nodeToPartitionStr +=
                  (cnode->name_ + "-" + std::to_string(branch->partitionID));

              //std::cout << "idx = " << idx << " nodes = " << function->getNodes().size() << std::endl;

              //if (!(branch->partitionID == partitionSize - 2 && cnode == branch->nodeList_.back())) {
              nodeToPartitionStr += "\n  :";
              //}
            }
          }
        } else {
          DeviceInfo dinfo = getDeviceInfoForBackend(partition->backendName_);
          backendNameStr += (partition->backendName_ + ":");
          logicalIDStr += (std::to_string(dinfo.deviceID) + ":");
          partitionNameStr +=
              ("p" + std::to_string(partition->partitionID) + ":");

          for (auto cnode : partition->nodeList_) {
            nodeToPartitionStr +=
                (cnode->name_ + "-" + std::to_string(partition->partitionID));

            //std::cout << "idx = " << idx << " nodes = " << function->getNodes().size() << std::endl;
            //if (idx++ < function->getNodes().size() - 2) {
            //if (!(partition->partitionID == partitionSize - 2 && cnode == partition->nodeList_.back())) {
            nodeToPartitionStr += "\n  :";
            //}
          }
        }
      }

      nodeToPartitionStr = nodeToPartitionStr.substr(0, nodeToPartitionStr.length() - 4);

      std::string parallelPartitions = "";
      std::string parallelBranchPairStr = "";

      if(isThereParallelBranch) {
        for (auto partition : nodeGroups_) {
          if(partition->isParalleBranch) {
            parallelBranchPairStr = "(";
            for (auto branch : partition->branchList_) {
              parallelBranchPairStr += ("p" + std::to_string(branch->partitionID)+"||");
            }
            parallelBranchPairStr = parallelBranchPairStr.substr(0, parallelBranchPairStr.length()-2) + ")";
          }
          if(parallelBranchPairStr.length() > 0) {
            std::cout << parallelBranchPairStr << " ";
            parallelPartitions += (parallelBranchPairStr + ":");
            parallelBranchPairStr = "";
          }
        }
      }

      parallelPartitions = parallelPartitions.substr(0, parallelPartitions.length()-1);


      partitionNameStr += ("p" + std::to_string(nodeGroups_.back()->partitionID+1));
      backendNameStr += "CPU";
      logicalIDStr += "0";

      planFile << "backendNames: " << backendNameStr << std::endl;
      planFile << "partitionNames: " << partitionNameStr << std::endl;
      planFile << "nodeToPartition: " << nodeToPartitionStr<< std::endl;
      planFile << "logicalIDs: " << logicalIDStr<< std::endl;

      if(parallelPartitions.length() > 0)
        planFile << "parallelPartitions: " << parallelPartitions<< std::endl;

      planFile << "...\n" << std::endl;
    } else {
      std::cout << "Single partition!" << std::endl;
    }
    planFile.close();
  }
  std::cout << std::endl << std::endl;
//  std::cout << "----------- [end] outPartitionPlan ---------------" << std::endl;

}

void NestPartitioner::loadPerformProfileInfo(std::string pdir)
{
  //std::cout << "=== loadPerformProfileInfo  === " << std::endl;
  struct dirent *entry;
  DIR *dir = opendir(pdir.c_str());
  if (dir == NULL) {
    std::cout << "Partition profile directory is not found: " + pdir;
    return;
  } else {
    std::cout << "profile directory = " <<  pdir << std::endl;
  }

  while ((entry = readdir(dir)) != NULL) {
    std::string fn (entry->d_name);

    if(fn.substr(fn.find_last_of(".") + 1) == "yaml") {

      std::map<std::string, std::string> profileMap = deserializeStrStrMapFromYaml(pdir+"/"+fn);
      std::string backendName = profileMap["backendName"];
      backendSet_.insert(backendName);
      float commCost = atof(profileMap["commCost"].c_str());
      commCostMap_[backendName] = commCost;

      std::map<std::string, float> puvalues;
      for(auto it = profileMap.begin();it != profileMap.end();it++) {
        if(!it->first.compare("backendName") || !it->first.compare("commCost")) continue;
        puvalues[it->first] = atof(profileMap[it->first].c_str());
       }
      appGen_->getPartitionProfileInfo().backendProfileMap_[backendName].insert(puvalues.begin(), puvalues.end());
    }
  }
  closedir(dir);
}


//std::string NestPartitioner::getProfileKey(Node *node) {
//    std::string key;
//    std::string kindName = node->getKindName();
//    if(node->getKind() == Kinded::Kind::ConvolutionNodeKind) {
//        key = std::string(node->getKindName()) + "-" + getDimArrayStr(node->getNthInput(0).getType())
//              + "-" + getDataTypeStr(node->getNthInput(0).getElementType())
//              + "-" + std::to_string(((ConvolutionNode*)node)->getFilter().dims().front())
//              + "-" + std::to_string(((ConvolutionNode*)node)->getKernels().front())
//              + "-" + std::to_string(((ConvolutionNode*)node)->getStrides().front());
//    } else {
//        key = std::string(node->getKindName()) + "-" + getDimArrayStr(node->getNthInput(0).getType())
//              + "-" + getDataTypeStr(node->getNthInput(0).getElementType()) ;
//    }
//
//    return key;
//}

float getTotalCost(std::vector<std::vector<CostNode>>* cnodeBfs) {
    //std::cout << "--------- getTotalCost -----------" << std::endl;
    float totalCost = 0.0;

    size_t level = (*cnodeBfs).size();
    for (int i = 0; i < level; i++) {
        for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {
            if((i == level - 1) && (j == cnodeBfs[level- 1].size()- 1))
                continue;

//            std::cout << "level = " << i << std::endl;
//            std::cout << "node name = " << cnodeBfs[i][j].name << std::endl;
//            std::cout << "node backend = " << cnodeBfs[i][j].backendName << std::endl;
//            std::cout << "node cost = " << cnodeBfs[i][j].totalCost << std::endl;
            totalCost += (*cnodeBfs)[i][j].cost_;
        }
    }
    std::cout << "== totalCost: " << totalCost << std::endl<< std::endl;
    return totalCost;
}


float NestPartitioner::getCostOfSingleNode(CostNode *cnode, std::string backendName)
{
////  std::cout << "--------- getCostOfSingleNode -----------" << std::endl;
  std::string key = appGen_->getProfileKey(cnode->node_);

  std::vector<std::string> backendList(backendSet_.size());
  std::copy(backendSet_.begin(), backendSet_.end(), backendList.begin());

  cnode->cost_ = commCostMap_[backendList[0]] + appGen_->getPartitionProfileInfo().backendProfileMap_[backendList[0]][key];
  cnode->backendName_ = backendList[0];

  if(backendMap_[backendName].nonSupportedNodesKinds.find(cnode->node_->getKind())
     != backendMap_[backendName].nonSupportedNodesKinds.end()) {
    return INFINITY;
  } else {
    float commCost = commCostMap_[backendName];
    float compCost = appGen_->getPartitionProfileInfo().backendProfileMap_[backendName][key];

    return (commCost + compCost);
  }
}


void NestPartitioner::getMinCostOfSingleNode(CostNode *cnode)
{
//  std::cout << "--------- getMinCostOfSingleNode -----------" << std::endl;
  std::string key = appGen_->getProfileKey(cnode->node_);

  std::vector<std::string> backendList(backendSet_.size());
  std::copy(backendSet_.begin(), backendSet_.end(), backendList.begin());

  cnode->cost_ = commCostMap_[backendList[0]] + appGen_->getPartitionProfileInfo().backendProfileMap_[backendList[0]][key];
  cnode->backendName_ = backendList[0];

  //  std::cout << "costNode->totalCost = " << cnode->totalCost << std::endl;
  for(int i = 1; i < backendList.size(); i++) {
//      std::string backendName = peBackendMap_.find(puIDList[i])->second;
      std::string backendName = backendList[i];

      //non supported type
      if(backendMap_[backendName].nonSupportedNodesKinds.find(cnode->node_->getKind())
      != backendMap_[backendName].nonSupportedNodesKinds.end()) continue;

      float commCost = commCostMap_[backendName];
      float compCost = appGen_->getPartitionProfileInfo().backendProfileMap_[backendName][key];

      float cost = compCost + commCost;

    if(cost < cnode->cost_) {
      cnode->cost_ = cost;
      cnode->backendName_ = backendName;
    }
  }
//    std::cout << "costNode->totalCost = " << cnode->totalCost << std::endl;
//    std::cout << "node = " << cnode->name_<<  ", backendName = " << cnode->backendName_ << std::endl;
}


void NestPartitioner::getMinCostOfFusedNode(CostNode* prevCNode, CostNode* curCNode)
{
     //std::cout << "--------- getMinCostOfFusedNode -----------" << std::endl;
    // std::cout << "cur kind name = " << curCNode->node_->getKindName() << std::endl;
    // std::cout << "cur node name = " << curCNode->node_->getName().str() << std::endl;

    std::string compType = prevCNode->node_->getKindName() + std::string("-") + curCNode->node_->getKindName();
    if(appGen_->getPartitionProfileInfo().fuseOperatorSet_.find(compType) == appGen_->getPartitionProfileInfo().fuseOperatorSet_.end()) {
        return;
    }

    CostNode tmpcnode[2];
    tmpcnode[0] = *prevCNode;
    tmpcnode[1] = *curCNode;

    std::string key[2];
    for(int i = 0; i < 2; i++){
        key[i] = appGen_->getProfileKey(tmpcnode[i].node_);
    }
    std::string compKey = key[0] +"+"+ key[1];

//    std::vector<size_t> puIDList(puIDSet_.size());
    std::vector<std::string> backendList(backendSet_.size());

    std::copy(backendSet_.begin(), backendSet_.end(), backendList.begin());

    float originalCost = prevCNode->cost_ + curCNode->cost_;
    //std::cout << "original prev backend = " << prevCNode->backendName_ << std::endl;
    //std::cout << "original cur backend = " << curCNode->backendName_ << std::endl;

    // std::cout << "compKey = " << compKey << std::endl;
    //std::cout << "original cost = " << prevCNode->totalCost + curCNode->totalCost << std::endl;

    float prevCost = INFINITY;
    for(int i = 0; i < backendList.size(); i++) {

        std::string backendName = backendList[i];
        //non supported type
//        if(backendMap_[backendName].nonSupportedNodesKinds.find(curCNode->node_->getKind())
//           != backendMap_[backendName].nonSupportedNodesKinds.end()) {
//            continue;
//        }
//

        if(appGen_->getPartitionProfileInfo().backendProfileMap_[backendName].find(compKey) == appGen_->getPartitionProfileInfo().backendProfileMap_[backendName].end())
            continue;

        float commCost = commCostMap_[backendName];
        float compCost = appGen_->getPartitionProfileInfo().backendProfileMap_[backendName][compKey];
        float cost = compCost + commCost;

        if(cost < originalCost && cost < prevCost) {
//          std::cout << "First node: " << prevCNode->name_ << ", " << prevCNode->backendName_ << std::endl;
//          std::cout << "Second node: " << curCNode->name_ << ", " << curCNode->backendName_ << std::endl;
//          std::cout << "Original cost: " << prevCNode->cost_ + curCNode->cost_ << std::endl;;

                    prevCost = cost;
          prevCNode->cost_ = cost;
          prevCNode->backendName_ = backendName;

          curCNode->cost_ = 0.0;
          curCNode->backendName_ = backendName;
          curCNode->isFused_ = false;

//          std::cout << "=> First node: " << prevCNode->backendName_ << std::endl;
//          std::cout << "=> Second node: " << curCNode->backendName_ << std::endl;
//          std::cout << "=> Changed cost: " << prevCNode->cost_ + curCNode->cost_ << std::endl<< std::endl;

        }
    }

    //std::cout << "changed prev backend = " << prevCNode->backendName_ << std::endl;
    //std::cout << "changed cur backend = " << curCNode->backendName_ << std::endl;
    //std::cout << "fused cost prev = " << prevCNode->totalCost << std::endl;
    //std::cout << "fused cost cur = " << curCNode->totalCost << std::endl;
}

void NestPartitioner::getMinCostOfFusedNode(CostNode* secondPrevCNode, CostNode* firstPrevCNode, CostNode* curCNode)
{
    // std::cout << "--------- getMinCostOfFusedNode -----------" << std::endl;
    // std::cout << "cur kind name = " << curCNode->node_->getKindName() << std::endl;
    // std::cout << "cur node name = " << curCNode->node_->getName().str() << std::endl;

    std::string compType = secondPrevCNode->node_->getKindName() + std::string("-") + firstPrevCNode->node_->getKindName() + std::string("-") + curCNode->node_->getKindName();
    if(appGen_->getPartitionProfileInfo().fuseOperatorSet_.find(compType) == appGen_->getPartitionProfileInfo().fuseOperatorSet_.end()) {
        return;
    }

    CostNode tmpcnode[3];
    tmpcnode[0] = *secondPrevCNode;
    tmpcnode[1] = *firstPrevCNode;
    tmpcnode[2] = *curCNode;

    std::string key[3];
    for(int i = 0; i < 3; i++){
        key[i] = appGen_->getProfileKey(tmpcnode[i].node_);
    }
    std::string compKey = key[0] + "+" + key[1] +  "+" + key[2];

    std::vector<std::string> backendList(backendSet_.size());
    std::copy(backendSet_.begin(), backendSet_.end(), backendList.begin());

    float originalCost = secondPrevCNode->cost_ + firstPrevCNode->cost_ + curCNode->cost_;
    //std::cout << "original prev backend = " << prevCNode->backendName_ << std::endl;
    //std::cout << "original cur backend = " << curCNode->backendName_ << std::endl;

    //std::cout << "compKey = " << compKey << std::endl;

    float prevCost = INFINITY;
    for(int i = 0; i < backendList.size(); i++) {

        std::string backendName = backendList[i];
        //non supported type
//        if(backendMap_[backendName].nonSupportedNodesKinds.find(curCNode->node_->getKind())
//           != backendMap_[backendName].nonSupportedNodesKinds.end()) {
//            continue;
//        }

        if(appGen_->getPartitionProfileInfo().backendProfileMap_[backendName].find(compKey) == appGen_->getPartitionProfileInfo().backendProfileMap_[backendName].end())
            continue;

        float commCost = commCostMap_[backendName];
        float compCost = appGen_->getPartitionProfileInfo().backendProfileMap_[backendName][compKey];
        float cost = compCost + commCost;

        if(cost < originalCost && cost < prevCost) {
          std::cout << "[Original cost] = " << secondPrevCNode->cost_ + firstPrevCNode->cost_ + curCNode->cost_ << std::endl;

          prevCost = cost;
            secondPrevCNode->cost_ = cost;
            secondPrevCNode->backendName_ = backendName;

            firstPrevCNode->cost_ = 0.0;
            firstPrevCNode->backendName_ = backendName;
            firstPrevCNode->isFused_ = false;

            curCNode->cost_ = 0.0;
            curCNode->backendName_ = backendName;
            curCNode->isFused_ = false;

          std::cout << "[Changed cost] = " << secondPrevCNode->cost_ + firstPrevCNode->cost_ + curCNode->cost_ << std::endl;
        }
    }

    //std::cout << "changed prev backend = " << secondPrevCNode->backendName_ << std::endl;
    //std::cout << "changed cur backend = " << curCNode->backendName_ << std::endl;
    //std::cout << "fused cost prev = " << secondPrevCNode->totalCost_ << std::endl;
    //std::cout << "fused cost cur = " << curCNode->totalCost_ << std::endl;

  //std::cout << "\tChanged cost = " << secondPrevCNode->totalCost_ + firstPrevCNode->totalCost_ + curCNode->totalCost_ << std::endl;

}

//Get initial partition based on the cost of each node (operation)
void  NestPartitioner::partitionBfs(std::vector<std::vector<CostNode>>* cnodeBfs) {
    //std::cout << "------ partitionBfs --------" << std::endl;

    std::vector<CostNode*> group;
    CostNode* prevCNode = nullptr;
    NodeGroup* partition;
    size_t partitionID = 0;
    size_t level = cnodeBfs->size();

    for (int i = 0; i < level; i++) {
        for (size_t j = 0; j < cnodeBfs->at(i).size(); j++) {
          CostNode* cnode = &(*cnodeBfs)[i][j];

         // if(!cnode->node_->getName().str().compare("resnetv10_dense0_fwd")) {
//            std::cout << "bfs node name = " << cnode->node_->getName().str() << std::endl;
//            std::cout << "backend = " << cnode->backendName_
//                      << std::endl;
//            if(prevCNode != nullptr)
//              std::cout << "prev backend = " << prevCNode->backendName_
//                      << std::endl;

          //}

          if(i == level-1 && j == cnodeBfs->at(level-1).size()-1) {
//            std::cout << "node kind = " << cnode->node_->getKindName()
//                      << std::endl;

            if (cnode->node_->getKind() != Kinded::Kind::SaveNodeKind) {
                  std::cout << "The last node is not SaveNode!" << std::endl;
              } else {
                partition = new NodeGroup;
                partition->partitionID = partitionID;
                partition->nodeList_.insert(partition->nodeList_.begin(), group.begin(), group.end());
                partition->backendName_ = prevCNode->backendName_;
                nodeGroups_.push_back(partition);
                //break;
                return;
              }
          }

          //first node
          if (prevCNode == nullptr) {
            prevCNode = cnode;
              cnode->partitionID_ = partitionID;
              group.push_back(cnode);
          } else {

            if (cnode->backendName_.compare(prevCNode->backendName_)) {
              partition = new NodeGroup();
              partition->partitionID = partitionID;
              partition->nodeList_.insert(partition->nodeList_.begin(),
                                          group.begin(), group.end());
              partition->backendName_ = prevCNode->backendName_;
              nodeGroups_.push_back(partition);
              group.clear();
              partitionID++;
            }
            group.push_back(cnode);
          }

          cnode->partitionID_ = partitionID;
          prevCNode = cnode;
        }
    }
}

void NestPartitioner::allocOptimalPUSingleNode(std::vector<std::vector<CostNode>>* cnodeBfs)
{
  //std::cout << "--------- findOptimalPartitionsSingleNode -----------" << std::endl;
  std::cout << "\n[Step 2] Allocate PUs to operations considering the cost of a *Single Operation*." << std::endl;

  //allocate a PU for each node
  for (int i = 0; i < (*cnodeBfs).size(); i++) {
    for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {
      CostNode *cnode = &(*cnodeBfs)[i][j];
      getMinCostOfSingleNode(cnode);
    }
  }
  getTotalCost(cnodeBfs);

}

CostNode* NestPartitioner::getCostNode(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node)
{
  for (int i = 0; i < (*cnodeBfs).size(); i++) {
    for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {
      CostNode *cnode = &(*cnodeBfs)[i][j];
      if (cnode->node_ == node) {
        return cnode;
      }
    }
  }
  return nullptr;
}

std::set<CostNode*> NestPartitioner::getOutNodeSet(std::vector<std::vector<CostNode>>* cnodeBfs, NodeGroup* partition)
{
  //std::cout << "== getOutNodeSet ==" << std::endl;
  std::set<CostNode*> outSet;
  for(auto cnode: partition->nodeList_) {
    for(int i = 0; i < cnode->node_->getNumInputs(); i++) {
      CostNode* prevCNode = getCostNode(cnodeBfs, cnode->node_->getNthInput(i).getNode());
      if (prevCNode != nullptr && prevCNode->partitionID_== partition->partitionID-1) {
        outSet.insert(prevCNode);
      }
    }
  }

  return outSet;
}

float NestPartitioner::getPartitionCost(NodeGroup* partition){
  float cost = 0.0;

  for(auto cnode: partition->nodeList_) {
    cost += cnode->cost_;
  }

  return cost;
}

NodeGroup* NestPartitioner::getBranchNodes(std::vector<std::vector<CostNode>>* cnodeBfs, CostNode* cnode, NodeGroup* partition){
  //std::cout << "== getBranchNodes ==" << std::endl;
  NodeGroup* group = new NodeGroup;
  group->nodeList_.push_back(cnode);
//  std::cout << cnode->node_->getName().str() << " cost: " << cnode->cost_<< " backend: "<< cnode->backendName_ << std::endl;
  //for(auto cnode: partition->nodeList_) {
  CostNode* prevCNode = nullptr;
  bool isNext = true;
  while(isNext){
    isNext = false;
    for (int i = 0; i < cnode->node_->getNumInputs(); i++) {
      prevCNode =
          getCostNode(cnodeBfs, cnode->node_->getNthInput(i).getNode());

      if (prevCNode != nullptr &&
          prevCNode->partitionID_ == partition->partitionID) {
        group->nodeList_.push_back(prevCNode);
        //group->nodeList_.insert(group->nodeList_.begin(), prevCNode);
//        std::cout << prevCNode->node_->getName().str()
//                  << " cost: " << prevCNode->cost_ << " backend: "<< prevCNode->backendName_ <<std::endl;
        isNext = true;
        cnode = prevCNode;
      }
    }
  };
 // }

  return group;
}

void NestPartitioner::makeCostNode(Function* function, std::vector<std::vector<CostNode>>* cnodeBfs) {
    //std::cout << "--------- makeCostNode -----------" << std::endl;
    BFSLevel bfs = getBFSLevel(function);
    size_t level = bfs.size();
//    size_t branchDepth = 0, branchLevel = 0;

    for (int i = level - 1; i >= 0; i--) {
    std::vector<CostNode> cnodeGroup;
        for (size_t j = 0; j < bfs[i].size(); j++) {
            Node *node = bfs[i][j];
            CostNode cnode;
            cnode.node_ = node;
            cnode.name_ = node->getName();
            cnodeGroup.push_back(cnode);
        }
        cnodeBfs->push_back(cnodeGroup);
    }
}

void NestPartitioner::allocOptimalPUFusedNodes(std::vector<std::vector<CostNode>>* cnodeBfs)
{
    //std::cout << "--------- allocOptimalPUFusedNodes -----------" << std::endl;
    allocOptimalPUSingleNode(cnodeBfs);

    std::cout << "[Step 3] Allocate PUs to operations considering the cost of *Fused Operations*." << std::endl;

    for (int i = 0; i < (*cnodeBfs).size(); i++) {
        for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {

            CostNode *firstPrevCNode;
            CostNode *secondPrevCNode;
            CostNode *cnode = &((*cnodeBfs)[i][j]);

            if(!isFusable(cnode->node_))
                continue;

            firstPrevCNode = findNodeFromCNodeList(cnodeBfs, getFirstInputOpNode(cnode->node_));
            if(firstPrevCNode == nullptr || !isFusable(firstPrevCNode->node_))
                continue;

            getMinCostOfFusedNode(firstPrevCNode, cnode);

            secondPrevCNode = findNodeFromCNodeList(cnodeBfs, getFirstInputOpNode(firstPrevCNode->node_));
            if(secondPrevCNode == nullptr || !isFusable(secondPrevCNode->node_))
                continue;

            getMinCostOfFusedNode(secondPrevCNode, firstPrevCNode, cnode);

        }
    }

    getTotalCost(cnodeBfs);
    //std::cout << "(Fused Node 2) Total network cost = " << getTotalCost(cnodeBfs) << std::endl;
}

#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>

typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
void NestPartitioner::loadFuseOperatorsConfig(std::string fname) {

//    std::cout << "== loadFuseOperatorsConfig ==" << std::endl;

    std::map<std::string, std::string> foMap = deserializeStrStrMapFromYaml(fname);
    std::string fuseOperators = foMap["fuseOperators"];

    boost::char_separator<char> sep(":");
    tokenizer tokens(fuseOperators, sep);

    BOOST_FOREACH(std::string token, tokens)
    {
        boost::trim_left(token);
        boost::trim_right(token);
//        std::cout << token << std::endl;
            appGen_->getPartitionProfileInfo().fuseOperatorSet_.insert(token);
    }
}

void NestPartitioner::findAndSetParalleCPUBranch(NodeGroup* partition) {
  // PUs: CPU1, VTA1
//  std::cout << "== findAndSetParalleCPUBranch ==" << std::endl;
  // find a branch having the lowest cost
  NodeGroup *lowestCostBranch = nullptr;
  for (auto branch : partition->branchList_) {
    if (lowestCostBranch == nullptr ||
        branch->totalCost_ < lowestCostBranch->totalCost_) {
      lowestCostBranch = branch;
    }
  }

  float totalCpuCost = 0;
  for (auto cnode : lowestCostBranch->nodeList_) {
    totalCpuCost = getCostOfSingleNode(cnode, "CPU");
  }

  if(totalCpuCost < partition->totalCost_) {
    partition->isParalleBranch = true;
    for (auto cnode : lowestCostBranch->nodeList_) {
      cnode->cost_ = getCostOfSingleNode(cnode, "CPU");
      cnode->backendName_ = "CPU";
    }
    lowestCostBranch->backendName_ = "CPU";
  }

//  std::cout << "----------- CPU cost = " << totalCpuCost << "---------------"
//            << std::endl;
}

void NestPartitioner::allocatePUToParallelBranch(Function* function, std::vector<std::vector<CostNode>>* cnodeBfs) {
//  std::cout << "----------- allocatePUToParallelBranch ---------------" << std::endl;
  //std::cout << "----------- current partition ---------------" << std::endl;

  //no nested branch, 1-NPU + 1-CPU
  //Get the number of out nodes. If the number is more than 1, there is a parallel branch.

  for (auto partition : nodeGroups_) {
//    std::cout << "-- partition: " << partition->partitionID << std::endl;
    partition->totalCost_ = getPartitionCost(partition);
//    std::cout << "=> Total partition cost = " << partition->totalCost_ << std::endl;
    if(partition->partitionID > 0) {

      NodeGroup* prevPartition = nodeGroups_.at(partition->partitionID - 1);
      prevPartition->outNodeSet_ = getOutNodeSet(cnodeBfs, partition);

      if(prevPartition->outNodeSet_.size() > 1) {
        for(auto cnode : prevPartition->outNodeSet_) {
//          std::cout << "\tout node (" << partition->partitionID - 1 << + "): " << cnode->name_ << std::endl;
          NodeGroup* branch = getBranchNodes(cnodeBfs, cnode, prevPartition);
          branch->backendName_ = prevPartition->backendName_;
          prevPartition->branchList_.push_back(branch);
          branch->totalCost_ = getPartitionCost(branch);
//          std::cout << prevPartition->branchList_.size() << " branch node # = " << branch->nodeList_.size() << " cost = " << branch->totalCost_ << std::endl;
        }

        if(prevPartition->branchList_.size() > 1) {
          findAndSetParalleCPUBranch(prevPartition);
        }
      }
    }
  }

  float cost = 0.0;
  //reset partitionIDs
  int newPartitionID = 0;
  for (auto partition : nodeGroups_) {
    if(partition->branchList_.size() > 1 && partition->isParalleBranch) {
      float highestCost = 0.0;
      for(auto branch: partition->branchList_) {
        highestCost = 0.0;
        branch->partitionID = newPartitionID;
        partition->partitionID = newPartitionID;
        newPartitionID++;

        if(highestCost < branch->totalCost_)
          highestCost = branch->totalCost_;
      }
      cost += highestCost;
    } else {
      partition->partitionID = newPartitionID++;
      cost += partition->totalCost_;
    }
  }

  std::cout << "== totalCost: " << cost << std::endl;
//  std::cout << "3) Allocate the original backend to the highest-cost branch. Find another backend for other branches "
//               "that mininizes the cost of a section" << std::endl;

}

Expected<DAGListTy> NestPartitioner::partition(CompilationContext &cctx) {
    std::string dir = "/home/msyu/Dropbox/Project/development/configs/partition_perform_profile";
    partition(cctx, 3, "", dir + "/NESTOptimalPlan.yaml", 0);
}

Expected<DAGListTy> NestPartitioner::partition(CompilationContext &cctx, size_t exeType, std::string profilePath, std::string partitionPlanFile, int profileMode) {
//  std::cout << "--------- partitionForNest exeType: " << exeType << "-----------" << std::endl;

  std::string inputPartitionName;
  if (partitionPlanFile.find('/') != std::string::npos) {
    inputPartitionName =
        partitionPlanFile.substr(partitionPlanFile.find_last_of("/") + 1);
    inputPartitionName = inputPartitionName.substr(0, inputPartitionName.size() - 5);
  }

  appGen_->setProfileMode(profileMode);
  appGen_->setInputPartitionName(inputPartitionName);
  appGen_->setProfilePath(profilePath);
  appGen_->setPartitionPlanFile(partitionPlanFile);

  Function *function = module_->getFunctions().front();

//  appGen_->setProfilingInfo_backendName(deviceInfo_.front().backendName);

  BFSLevel bfs = getBFSLevel(function);
  size_t level = bfs.size();
  int insertCount = 0;

  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node* node = bfs[i][j];
      std::string key = appGen_->getProfileKey(node);
      if(appGen_->getPartitionProfileInfo().profileKeySet_.find(key) == appGen_->getPartitionProfileInfo().profileKeySet_.end()) {
        appGen_->setKeyName(key, insertCount++);
      }
      else
      {
        appGen_->setKeyName("blank", insertCount++);
      }
    }
  }

  //Generate profiles of each node and fused nodes for each device
  if(exeType == 1) {

//    std::cout << "1. Generating partition plan for profiling.." << std::endl;
//    std::cout << "Linking-code generation is done." << std::endl << std::endl;

    std::cout << "1. Loading fusing options for partition-plan generation." << std::endl;
    loadFuseOperatorsConfig(profilePath + "/profileConfigs.yaml");

    std::vector<Backend *> backends;
    genBackendMap(backendMap_, backendHolder_, backends);
    // generate initial partition

    //std::string filename;
    //std::string pfilename;

    std::cout << "2. Partition-plan generation for the performance profiling of each PU" << std::endl;
    for (auto device : deviceInfo_) {

      std::cout << "  => Partition-plan generation for " << device.backendName << std::endl;

//      //separate each node
      //=============== test: save performance file

      std::string profileName = profilePath + "/data/initialProfile" + device.backendName +
                     std::to_string(device.deviceID) + ".yaml";
      outPerformProfileForEachNode(function, profileName , device);

      std::string planName = profilePath + "/initialSingleNodePartition" + device.backendName +
                                                  std::to_string(device.deviceID) + ".yaml";
      outPartitionPlanForSingleNode(function, device, planName);

      std::cout << "    . Partition plan for each NN operation: " << planName << std::endl;

      profileName = profilePath + "/data/fusedProfile" + device.backendName +
                  std::to_string(device.deviceID) + ".yaml";
      //fusing depth: 1
      outPerformProfileForFusedNode(function, profileName , device);

      planName = profilePath + "/initialFusedNodePartition" + device.backendName +
                             std::to_string(device.deviceID) + ".yaml";
      outPartitionPlanForFusion(function, device, planName);

      std::cout << "    . Partition plan for fused NN operation: " << planName << std::endl;


      nodeGroups_.clear();
      appGen_->getPartitionProfileInfo().profileKeySet_.clear();
      //std::cout <<"Use imagenet-partition-profiler to generate partition bundles based on the generated partition plan!" << std::endl;
    }

    exit(0);
    //partitionPlanFile = "/home/msyu/Dropbox/Project/development/configs/partition_perform_profile/initialFusedNodePartitionCPU0.yaml";
    //exeType = 0;
  } else if (exeType == 2) {

    //loadFuseOperatorsConfig(dir + "/profileConfigs.yaml");

    //Generate partitioning plans
    loadPerformProfileInfo(profilePath+"/test");
    //Simple partition plan only considering the performance of each node

    std::vector<std::vector<CostNode>> cnodeBfs;
    makeCostNode(function, &cnodeBfs);
    allocOptimalPUSingleNode(&cnodeBfs);
    partitionBfs(&cnodeBfs);

    partitionPlanFile = profilePath + "/SingleOpOptimalPlan.yaml";
    outPartitionPlan(function, partitionPlanFile);

    nodeGroups_.clear();
    exit(0);
    //exeType = 0;
    //partitionPlanFile = "/home/msyu/Dropbox/Project/development/configs/partition_perform_profile/firstOptimalPlan.yaml";
    //std::cout <<"Use imagenet-partition-profiler to generate partition bundles based on the generated partition plan!" << std::endl;
  } else if (exeType == 3) {

    // Our partitioning plan considers the followings by using the performance profile
    // - Performance of fused (grouped) and quantized nodes that is efficiently executed in a specific device
    // - Unsupported node
    // - Node (or node groups) that can be executed in parallel.

    //std::cout << std::endl << "[1] Load Fusion Configuration." << std::endl;
    loadFuseOperatorsConfig(profilePath + "/profileConfigs.yaml");

    std::vector<std::vector<CostNode>> cnodeBfs;
    makeCostNode(function, &cnodeBfs);

    loadPerformProfileInfo(profilePath+"/test");
//    std::cout << std::endl << "Allocate the PU ID to a node (an operation) that minimize the total cost (execution time)." << std::endl;
    allocOptimalPUFusedNodes(&cnodeBfs);

    partitionBfs(&cnodeBfs);
 //   partitionPlanFile = profilePath + "/NESTOptimalPlan.yaml";
    outPartitionPlan(function, profilePath + "/NESTOptimalPlan.yaml");
    nodeGroups_.clear();

    //exit(0);
    partitionPlanFile = profilePath + "/NESTOptimalPlan.yaml";
    exeType = 0;
    //std::cout <<"Use imagenet-partition-profiler to generate partition bundles based on the generated partition plan!" << std::endl;
  }  else if (exeType == 4) {

    // Our partitioning plan considers the followings by using the performance profile
    // - Performance of fused (grouped) and quantized nodes that is efficiently executed in a specific device
    // - Unsupported node
    // - Node (or node groups) that can be executed in parallel.

    loadFuseOperatorsConfig(profilePath + "/profileConfigs.yaml");

    std::vector<std::vector<CostNode>> cnodeBfs;
    makeCostNode(function, &cnodeBfs);

    //step 1. load performance profile
    std::cout << std::endl << "[Step 1] Load the Performance Profile of PUs" << std::endl;
    loadPerformProfileInfo(profilePath+"/test");
    //step 2. Allocate the PU ID to a node (an operation) that minimize the total cost (execution time).
    allocOptimalPUFusedNodes(&cnodeBfs);

    partitionBfs(&cnodeBfs);

//    std::cout << "Sequential partition # = " << nodeGroups_.size() << std::endl;

    //step 3. Adjust the allocated PU to increase parallelism
    std::cout << "[Step 4] Divide the partitions into parallel branches (sub-partitions) and reallocate PUs to the branches." << std::endl;
    allocatePUToParallelBranch(function, &cnodeBfs);

//    partitionPlanFile = profilePath + "/NESTOptimalPlan-thread.yaml";
    outPartitionPlan(function, profilePath + "/NESTOptimalPlan-thread.yaml");

    for(auto partition: nodeGroups_)
      delete partition;

    nodeGroups_.clear();
    exit(0);

    //partitionPlanFile = profilePath + "/NESTOptimalPlan-thread.yaml";;
    //exeType = 0;
    //std::cout <<"Use imagenet-partition-profiler to generate partition bundles based on the generated partition plan!" << std::endl;
  }

  //======== Generate partitioned graphs ==========

  runtime::PartitionConfig partitionConfig;

  if(exeType == 0) {//compile
    //load
    //std::string filename = dir + "initialPartitionCPU0.yaml";
    //std::string filename = dir + "firstOptimalPlan.yaml";

    cctx.partitionConfig = &partitionConfig;
    loadPartitionPlan(cctx.partitionConfig, partitionPlanFile);

  } else {
    DAGListTy partitions;
    return std::move(partitions);
  }



  if (cctx.prepartitionedConfig &&
      cctx.prepartitionedConfig->funcs.size() != 0) {
    return setupPrepartitionedModule(cctx);
  }

  if (cctx.partitionConfig) {
    partitionConfig_ = *cctx.partitionConfig;

  }

  if (partitionConfig_.enabled()) {
    // Call user-defined partition flow.
    cctx.precisionConfig.quantConfig.checkGraphPreLowerHash = false;
    return partitionFromConfig(partitionConfig_, cctx);
  }

    DAGListTy partitions;
    return std::move(partitions);
}

using namespace std;
string NestPartitioner::getDataArrayStr(const Type *value) {
  string str ="";
  if(value->numSizes_ <= 0) return str;

  for(int i = 0; i < value->numSizes_; i++) {
    string dimstr = to_string(value->sizes_[i]);

    str = str + "[" + dimstr + "]";
  }
  return str;
}

//string NestPartitioner::replaceAll(std::string &str, std::string pattern, std::string replace) {
//  string result = str;
//  string::size_type pos = 0;
//  string::size_type offset = 0;
//  while ((pos = result.find(pattern, offset)) != std::string::npos)
//  {
//    result.replace(result.begin() + pos, result.begin() + pos + pattern.size(), replace);
//    offset = pos + replace.size();
//  }
//  return result;
//}
