/*****************************************************************************
 *
 * Copyright Next-Generation System Software Research Group, All rights
 *reserved. Future Computing Research Division, Artificial Intelligence Reserch
 *Laboratory Electronics and Telecommunications Research Institute (ETRI)
 *
 * THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
 * WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
 * TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE"
 * REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL"
 * SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
 * IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
 * COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
 *
 * LICENSE file : LICENSE_ETRI located in the top directory
 *
 *****************************************************************************/

#include "glow/Partitioner/NestPartitioner.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Partitioner/PartitionerOptimizer.h"
#include "glow/Partitioner/PartitionerUtils.h"
#include "glow/Partitioner/PartitionerValidation.h"
#include "glow/Support/Support.h"

#include "glow/Quantization/Serialization.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
namespace glow {
// bool GlowEnableLoadBalancedPartitioning = true;
bool GlowLogNestPartition = false;
bool GlowDumpNestPartition = false;
// static llvm::cl::opt<bool, /* ExternalStorage */ true>
//    GlowEnableLoadBalancedPartitioningOpt(
//        "glow_partitioner_enable_load_balance",
//        llvm::cl::desc(
//            "Enable a partitioner pass to optimize for "
//            "load balance in addition to memory capacity constraints"),
//        llvm::cl::location(GlowEnableLoadBalancedPartitioning));
} // namespace glow
// -log-partition - Command line option to dump Partitioner logs.
static llvm::cl::OptionCategory PartitionerCat("Glow Partitioner Options");
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    logPartition("nest-log-partition",
                 llvm::cl::desc("Enable logging partition info"),
                 llvm::cl::location(glow::GlowLogNestPartition),
                 llvm::cl::cat(PartitionerCat));

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

  appGen_.setProfileSet( &profileKeySet_);
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
          // std::cout << "ph name = " << ph->getName().str() << std::endl;
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

NestPartitioner::NestPartitioner(Module *parent,
                                 const std::vector<DeviceInfo> &devices,
                                 bool optimized,
                                 PartitionConfig partitionConfig,
                                 std::string quantFileName)
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

    if (deviceInfo_[i].partitionDefaultDevice) {
      defaultBackend_ = backendName;
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
      backendInfo.fuseOperators =
          generateStrSet(deviceInfo_[i].fuseOperators);

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
  //  std::cout << "backendNames = " << partitionConfig.backendNames.size() <<
  //  std::endl; std::cout << "logicalIDs = " <<
  //  partitionConfig.logicalIDs.size() << std::endl;
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

    // std::cout << "f name = " << newF->getName().str() << std::endl;
    // std::cout << "f filename = " << newF->getFilename().c_str() << std::endl;
  }

  // Map the nodes the the partitions.
  std::vector<Node *> unMapped;
  for (auto &node : F->getNodes()) {
    auto iter = partitionConfig.nodeToPartition.find(node.getName());
    if (iter == partitionConfig.nodeToPartition.end()) {
      // If a node in F is not in the node to partition mapping, put it into
      // unMaped list.
      unMapped.push_back(&node);
      // std::cout << "Unmapped = " << node.getName().str() << std::endl;
    } else {
      size_t partitionID = iter->second;
      DCHECK(partitionID < partitionConfig.numOfPartitions)
          << "Invalid partition id :" << partitionID;
      partitionMap.add(&node, funcList[partitionID]);
      unused.erase(partitionID);
      nodesSets[partitionID].insert(&node);
      // std::cout << "Mapped = " << node.getName().str() << std::endl;
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

  if (appGen_.getProfileMode() == 0) {
    std::cout << "Generating linking code (main.cpp).." << std::endl;
  } else if (appGen_.getProfileMode() == 1) {
    std::cout << "Generating profiling code (main.cpp).." << std::endl;
  }

  appGen_.generateCodeFromModels(funcList.size(), funcList, partitionConfig,
                                 appGen_.getInputPartitionName());

  if (appGen_.getProfileMode() == 0) {
    std::cout << "Linking-code generation is done." << std::endl << std::endl;
  } else if (appGen_.getProfileMode() == 1) {
    std::cout << "Profiling-code generation is done." << std::endl << std::endl;
  }

  // Verify the function.
  std::cout << "Optimizing Graph-IR for CPU and VTA.." << std::endl;

  for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
    auto func = funcList[i];

    Backend *B = backendMap_[partitionConfig.backendNames[i]].backend;

    if (!optimized_) {
      if (B->getBackendName().compare("CPU") &&
          B->getBackendName().compare("Relay")) {
        // Specific configurations.
        cctx.precisionConfig.quantMode = QuantizationMode::Quantize;
        cctx.precisionConfig.quantConfig.schema =
            quantization::Schema::SymmetricWithPower2Scale;
        llvm::StringRef filename = quantFileName_;
        // std::cout << "quant filename = " << filename.str() << std::endl;
        deserializeProfilingInfosFromYaml(
            filename, cctx.precisionConfig.quantConfig.graphPreLowerHash,
            cctx.precisionConfig.quantConfig.infos);
        cctx.precisionConfig.quantConfig.checkGraphPreLowerHash = false;
      } else {
        cctx.precisionConfig.quantMode = QuantizationMode::None;
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
#include <dirent.h>
#include <iostream>
#include <vector>

void NestPartitioner::loadPartitionPlan(
    runtime::PartitionConfig *partitionConfig, std::string filename) {
  //  std::cout << "----------- loadPartitionPlan ---------------" << std::endl;
  std::string partitionPlanFile(filename);
  // string partitionPlanFile("/Users/msyu/Development/partitionPlanTest.yaml");

  std::map<std::string, std::string> pmap =
      deserializeStrStrMapFromYaml(partitionPlanFile);
  partitionConfig->funcName = pmap.find("funcName")->second;
  partitionConfig->numOfPartitions =
      atol(pmap.find("numOfPartitions")->second.c_str());
  boost::split(partitionConfig->backendNames, pmap.find("backendNames")->second,
               [](char c) { return c == ':'; });
  boost::split(partitionConfig->partitionNames,
               pmap.find("partitionNames")->second,
               [](char c) { return c == ':'; });

  for (int i = 0; i < partitionConfig->backendNames.size(); i++) {
    int puIdx = 0;
    if (partitionConfig->backendNames[i].find("-") != std::string::npos) {
      int tmpidx = partitionConfig->backendNames[i].find("-");
      puIdx = atoi(partitionConfig->backendNames[i]
                       .substr(tmpidx + 1,
                               partitionConfig->backendNames[i].size() - tmpidx)
                       .c_str());
      partitionConfig->backendNames[i] =
          partitionConfig->backendNames[i].substr(0, tmpidx);
    }
    puIdxMap_->insert({partitionConfig->partitionNames[i], puIdx});
  }

  std::vector<std::string> results;
  boost::split(results, pmap.find("nodeToPartition")->second,
               [](char c) { return c == ':'; });
  for (auto &s : results) {
    std::vector<std::string> nodesStr;
    boost::split(nodesStr, s, [](char c) { return c == '-'; });

    partitionConfig->nodeToPartition[nodesStr[0]] = atoi(nodesStr[1].c_str());
    // std::cout << "node str = " <<  nodesStr[0] << std::endl;
    // std::cout << "node partition = " << nodesStr[1].c_str() << std::endl;
  }

  results.clear();
  boost::split(results, pmap.find("logicalIDs")->second,
               [](char c) { return c == ':'; });
  for (auto &s : results) {
    std::vector<std::string> idsStr;
    boost::split(idsStr, s, [](char c) { return c == '-'; });

    std::vector<unsigned int> ids;
    for (auto &id : idsStr) {
      ids.push_back(atoi(id.c_str()));
    }
    partitionConfig->logicalIDs.push_back(ids);
  }

  results.clear();
  if (pmap.find("parallelPartitions") != pmap.end()) {
    boost::split(results, pmap.find("parallelPartitions")->second,
                 [](char c) { return c == ':'; });
    for (auto &s : results) {
      // std::cout << s << std::endl;
      boost::erase_all(s, "(");
      boost::erase_all(s, ")");
      // std::cout << s << std::endl;
      boost::replace_all(s, "||", "|");

      std::vector<std::string> pgroupStr;
      boost::split(pgroupStr, s, [](char c) { return c == '|'; });

      std::set<std::string> pnameSet;
      for (auto &pname : pgroupStr) {
        pnameSet.insert(pname.c_str());
      }
      partitionConfig->parallelPartitions.push_back(pnameSet);
    }
  }
}

Node *getFirstInputOpNode(Node *node) {

  if (node->getNumInputs() > 0) {
    for (int in = 0; in < node->getNumInputs(); in++) {
      Node *inNode = node->getNthInput(in);
      if (!(inNode->getKind() == Kinded::Kind::ConstantKind ||
            inNode->getKind() == Kinded::Kind::PlaceholderKind)) {
        return inNode;
      }
    }
  }
  return nullptr;
}

CostNode *findNodeFromCNodeList(std::vector<std::vector<CostNode>> *cnodeBfs,
                                Node *node) {

  size_t level = (*cnodeBfs).size();
  for (int i = 0; i < level; i++) {
    for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {
      if (node == (&(*cnodeBfs)[i][j])->node_) {
        return &((*cnodeBfs)[i][j]);
      }
    }
  }
  return nullptr;
}

bool isFusable(Node *node) {
  if (node->getNumInputs() > 1) {
    size_t usedCnt = 0;
    for (int in = 0; in < node->getNumInputs(); in++) {
      Node *inNode = node->getNthInput(in);
      if ((inNode->getKind() != Kinded::Kind::ConstantKind) &&
          inNode->getKind() != Kinded::Kind::PlaceholderKind) {
        usedCnt++;
      }
    }

    if (usedCnt > 1) {
      // std::cout << "More than one inputs!" << std::endl;
      return false;
    }
  }

  if (!node->hasOneUse() && node->getKind() != Kinded::Kind::SaveNodeKind) {
    // std::cout << "More than one users!" << std::endl;
    return false;
  }

  if (node->getKind() == Kinded::Kind::SaveNodeKind) {
    // std::cout << "Save node!" << std::endl;
    return false;
  }

  return true;
}

bool NestPartitioner::isUserDefinedFusable(Node *node, DeviceInfo deviceInfo) {
  Node *firstPrevNode = getFirstInputOpNode(node);
  if (firstPrevNode == nullptr || !isFusable(firstPrevNode))
    return false;

  std::string doubleFusedKindName =
      std::string(firstPrevNode->getKindName()) + "-" + node->getKindName();
  if (backendMap_[deviceInfo.backendName].fuseOperators.find(doubleFusedKindName) != backendMap_[deviceInfo.backendName].fuseOperators.end()) {
    std::string profileStr = appGen_.getProfileKey(firstPrevNode) + "+" +
                             appGen_.getProfileKey(node);
    return true;
  }

  Node *secondPrevNode = getFirstInputOpNode(firstPrevNode);
  if (secondPrevNode == nullptr || !isFusable(secondPrevNode))
    return false;

  std::string tripleFusedKindName = std::string(secondPrevNode->getKindName()) +
                                    "-" + firstPrevNode->getKindName() + "-" +
                                    node->getKindName();

  // std::cout << "tripleFusedKindName = " << tripleFusedKindName << std::endl;
  if (backendMap_[deviceInfo.backendName].fuseOperators.find(tripleFusedKindName) != backendMap_[deviceInfo.backendName].fuseOperators.end()) {
    return true;
  }
  return false;
}

void NestPartitioner::outPerformProfileForFusedNode(
    std::vector<NodeGroup *> *branchList, std::string filename,
    DeviceInfo device) {
  //    std::cout << "== outPerformProfileForFusedNode : " << device.backendName
  //    << " == " << std::endl;

  std::ofstream profileFile(filename);
  if (profileFile.is_open()) {

    profileFile << "---" << std::endl;
    profileFile << "backendName: " << device.backendName << std::endl;

    for (auto partition : *branchList) {
      for (auto cnode : partition->nodeList_) {
        Node *node = cnode->node_;

        Node *firstPrevNode;
        Node *secondPrevNode;

        firstPrevNode = getFirstInputOpNode(node);

        if (firstPrevNode == nullptr || !isFusable(firstPrevNode))
          continue;

        std::string doubleFusedKindName =
            std::string(firstPrevNode->getKindName()) + "-" +
            node->getKindName();
        // std::cout << "doubleFusedKindName = " << doubleFusedKindName <<
        // std::endl;
        if (backendMap_[device.backendName].fuseOperators.find(doubleFusedKindName) !=
                backendMap_[device.backendName].fuseOperators.end()) {
          std::string profileStr = appGen_.getProfileKey(firstPrevNode) + "+" +
                                   appGen_.getProfileKey(node);
          std::pair<std::set<std::string>::iterator, bool> ret =
              profileKeySet_.insert(profileStr);
          if (ret.second) {
            profileFile << profileStr << " : 1" << std::endl;
          }
        }

        secondPrevNode = getFirstInputOpNode(firstPrevNode);
        if (secondPrevNode == nullptr || !isFusable(secondPrevNode))
          continue;

        std::string tripleFusedKindName =
            std::string(secondPrevNode->getKindName()) + "-" +
            firstPrevNode->getKindName() + "-" + node->getKindName();

        if (backendMap_[device.backendName].fuseOperators.find(tripleFusedKindName) !=
                backendMap_[device.backendName].fuseOperators.end()) {
          std::string profileStr = appGen_.getProfileKey(secondPrevNode) + "+" +
                                   appGen_.getProfileKey(firstPrevNode) + "+" +
                                   appGen_.getProfileKey(node);
          std::pair<std::set<std::string>::iterator, bool> ret =
              profileKeySet_.insert(profileStr);
          if (ret.second) {
            profileFile << profileStr << " : 1" << std::endl;
          }
        }
      }
    }

    profileFile << "..." << std::endl;
    profileFile.close();
  }
}

void NestPartitioner::outPartitionPlanForFusion(
    std::string funcName, std::vector<NodeGroup *> *branchList,
    DeviceInfo deviceInfo, std::string filename, bool mergePartition) {
  //  std::cout << "== outPartitionPlanForFusedNode : " << filename << " == " <<
  //  std::endl;

  std::ofstream planFile(filename);
  if (planFile.is_open()) {

    int partitionSize = getFusedPartitionCount(branchList, deviceInfo);

    planFile << "---" << std::endl;
    planFile << "funcName: " << funcName << std::endl;
//    planFile << "numOfPartitions: " << partitionSize << std::endl;

    std::string backendNameStr;
    std::string partitionNameStr;
    std::string nodeToPartitionStr;
    std::string logicalIDStr;
    int mergedPartitionCount = 0;
    std::string curBackendName = defaultBackend_, prevBackendName = "";

    std::string curPartitionName = "", curLogicalID = "";
    // std::cout << "partitionSize = " << partitionSize << std::endl;

    size_t idx = 0;
    for (auto partition : *branchList) {
      for (auto cnode : partition->nodeList_) {
        Node *node = cnode->node_;

        if (node->getKind() == Kinded::Kind::SaveNodeKind)
          continue;

        bool isOffloading = false;
        if (!isSupportedOpType(cnode->node_->getKind(), deviceInfo.backendName, cnode)) {
          isOffloading = true;
        }

        if (!isOffloading) {
          curBackendName = (deviceInfo.backendName);
          curLogicalID = std::to_string(deviceInfo.deviceID);
        } else { // default
          curBackendName = defaultBackend_;
          curLogicalID = std::to_string(getDeviceInfoForBackend(defaultBackend_).deviceID);
        }

        bool isFused = isUserDefinedFusable(node, deviceInfo);
        if (isFused) {
          curBackendName = prevBackendName;
        }

        if(mergePartition) {
          if(curBackendName.compare(prevBackendName)) {
            mergedPartitionCount++;
            partitionNameStr += ("p" + std::to_string(mergedPartitionCount - 1) + ":");
            backendNameStr += (curBackendName + + ":");
            logicalIDStr += (curLogicalID+ ":");
          }
          nodeToPartitionStr += ((node->getName().str() + "-" + std::to_string(mergedPartitionCount-1) + "-" + curBackendName));
          prevBackendName = curBackendName;

        } else {
          if(!isFused) {
            idx++;
            partitionNameStr += ("p" + std::to_string(idx) + ":");
            backendNameStr += (curBackendName + ":");
            logicalIDStr +=
                    std::to_string(getDeviceInfoForBackend(curBackendName).deviceID) +
                    ":";
          }
          nodeToPartitionStr +=
                  (node->getName().str() + "-" + std::to_string(idx - 1));
        }
        nodeToPartitionStr += "\n  :";
      }
    }
    nodeToPartitionStr.pop_back();

    backendNameStr += defaultBackend_;
    logicalIDStr += std::to_string(getDeviceInfoForBackend(defaultBackend_).deviceID);

    if(mergePartition) {
      partitionNameStr += ("p" + std::to_string(mergedPartitionCount));
      planFile << "numOfPartitions: " << mergedPartitionCount + 1 << std::endl;
    } else {
      partitionNameStr += ("p" + std::to_string(partitionSize - 1));
      planFile << "numOfPartitions: " << partitionSize << std::endl;
    }
    planFile << "backendNames: " << backendNameStr << std::endl;
    planFile << "partitionNames: " << partitionNameStr << std::endl;
    planFile << "nodeToPartition: " << nodeToPartitionStr << std::endl;;
    planFile << "logicalIDs: " << logicalIDStr << std::endl;

    planFile << "..." << std::endl;
    planFile.close();
  }

  //  std::cout << "----------- [end] outPartitionPlanForSingleNode
  //  ---------------" << std::endl;
}

void NestPartitioner::outPartitionPlanForSingleNode(
    Function *function, std::vector<NodeGroup *> *branchList,
    DeviceInfo deviceInfo, std::string filename) {
  //  std::cout << "== outPartitionPlanForSingleNode : " << filename << " == "
  //  << std::endl; std::cout << "backend: " << deviceInfo.backendName <<
  //  std::endl;
  std::ofstream planFile(filename);
  if (planFile.is_open()) {

    planFile << "---" << std::endl;
    planFile << "funcName: " << function->getName().str() << std::endl;

    std::string backendNameStr;
    std::string partitionNameStr;
    std::string nodeToPartitionStr;
    std::string logicalIDStr;
    size_t idx = 0;


    for (auto partition : *branchList) {
      for (auto cnode : partition->nodeList_) {
        Node *node = cnode->node_;

        if (node->getKind() == Kinded::Kind::SaveNodeKind)
          continue;

        partitionNameStr += ("p" + std::to_string(idx));

        bool isOffloading = false;
        if (!isSupportedOpType(cnode->node_->getKind(), deviceInfo.backendName,
                               cnode)) {
          isOffloading = true;
        }

        if (!isOffloading) {
          backendNameStr += deviceInfo.backendName;
          logicalIDStr += std::to_string(deviceInfo.deviceID);
        } else { // default
            backendNameStr += defaultBackend_;
            logicalIDStr +=
                  std::to_string(getDeviceInfoForBackend(defaultBackend_).deviceID);
        }

        nodeToPartitionStr +=
            (node->getName().str() + "-" + std::to_string(idx));

        partitionNameStr += ":";
        backendNameStr += ":";
        logicalIDStr += ":";

        if (idx < function->getNodes().size() - 2) {
          nodeToPartitionStr += "\n  :";
        }
        idx++;
      }
    }

    partitionNameStr += ("p" + std::to_string(idx));
    backendNameStr += defaultBackend_;
    logicalIDStr += std::to_string(getDeviceInfoForBackend(defaultBackend_).deviceID);

    planFile << "numOfPartitions: " << function->getNodes().size() << std::endl;
    planFile << "backendNames: " << backendNameStr << std::endl;
    planFile << "partitionNames: " << partitionNameStr << std::endl;
    planFile << "nodeToPartition: " << nodeToPartitionStr << std::endl;
    planFile << "logicalIDs: " << logicalIDStr << std::endl;

    planFile << "..." << std::endl;
    planFile.close();
  }
  //  std::cout << "----------- [end] outPartitionPlanForSingleNode
  //  ---------------" << std::endl;
}

int NestPartitioner::getFusedPartitionCount(
    std::vector<NodeGroup *> *branchList, DeviceInfo deviceInfo) {
  int cnt = 1;
  for (auto partition : *branchList) {
    for (auto cnode : partition->nodeList_) {
      Node *node = cnode->node_;

      if (node->getKind() == Kinded::Kind::SaveNodeKind)
        continue;

      if (isUserDefinedFusable(node, deviceInfo))
        cnt--;

      cnt++;
    }
  }
  return (cnt);
}

int getBranchesWithSameLevel(std::vector<NodeGroup *> *branchList,
                             std::vector<NodeGroup *> *result, int level) {
  //    std::cout << "----------- getBranchesWithSameLevel level (" << level <<
  //    ")---------------" << std::endl;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    //        std::cout << "branch id: " << (*i)->ID_ << std::endl;
    //        std::cout << "branch level: " << (*i)->level_ << std::endl;
    //        std::cout << "partOfBranch_: " << (*i)->partOfBranch_ <<
    //        std::endl; if((*i)->level_ == level && !(*i)->partOfBranch_) {
    if ((*i)->level_ == level) {
      result->insert(result->end(), *i);
    }
  }
  return result->size();
}

int getParallelVTAPartitions(std::vector<NodeGroup *> *partitionList,
                             std::vector<NodeGroup *> *parallelPartitions,
                             std::vector<NodeGroup *> *nonParallelPartitions,
                             int level) {
  //    std::cout << "----------- getBranchesWithSameLevel level (" << level <<
  //    ")---------------" << std::endl;
  std::vector<std::string> addedVTAs;
  for (auto i = partitionList->begin(); i != partitionList->end(); i++) {
    //        std::cout << "partition id: " << (*i)->ID_ << std::endl;
    //        std::cout << "partition level: " << (*i)->level_ << std::endl;
    //        std::cout << "partOfBranch_: " << (*i)->partOfBranch_ <<
    //        std::endl; if((*i)->level_ == level && !(*i)->partOfBranch_) {
    if ((*i)->level_ == level && ((*i)->backendName_.rfind("VTA", 0) == 0) &&
        std::find(addedVTAs.begin(), addedVTAs.end(), (*i)->backendName_) ==
            addedVTAs.end()) {
      parallelPartitions->push_back(*i);
      addedVTAs.push_back((*i)->backendName_);
    } else {
      nonParallelPartitions->push_back(*i);
    }
  }
  return parallelPartitions->size();
}

int getParallelDeployBranches(std::vector<NodeGroup *> *branchList,
                              std::vector<NodeGroup *> *result) {
  //    std::cout << "----------- getParallelBranches level ---------------" <<
  //    std::endl;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    std::string bBackend =
        (*i)->backendName_.substr(0, (*i)->backendName_.find_first_of('-'));
    if ((*i)->parallelDeploy_) {
      result->insert(result->end(), *i);
    }
  }
  return result->size();
}

int getParallelDeployBranches(std::vector<NodeGroup *> *branchList,
                              std::string backendName,
                              std::vector<NodeGroup *> *result) {
  //    std::cout << "----------- getParallelBranches level ---------------" <<
  //    std::endl;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    std::string bBackend =
        (*i)->backendName_.substr(0, (*i)->backendName_.find_first_of('-'));
    if ((*i)->parallelDeploy_ && !bBackend.compare(backendName)) {
      result->insert(result->end(), *i);
    }
  }
  return result->size();
}

int getNonParallelDeployBranches(std::vector<NodeGroup *> *branchList,
                                 std::vector<NodeGroup *> *result) {
  //    std::cout << "----------- getNonParallelBranches level ---------------"
  //    << std::endl;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    if (!(*i)->parallelDeploy_) {
      result->insert(result->end(), *i);
    }
  }
  return result->size();
}

void NestPartitioner::outPartitionPlanCPUVTA(std::string funcName,
                                             std::string filename,
                                             bool isParallel) {
  //    std::cout << "----------- outPartitionPlan ---------------" <<
  //    std::endl;
  std::ofstream planFile(filename);
  if (planFile.is_open()) {

    std::string info = "";
    info.append("---\n");
    info.append("funcName: " + funcName + "\n");

    std::string backendNameStr;
    std::string partitionNameStr;
    std::string nodeToPartitionStr;
    std::string logicalIDStr;
    std::string parallelPartitions = "";
    std::set<int> addedPartitions;

    uint convNum = 0;
    uint vtaConvNum = 0;
    uint vtaReluNum = 0;
    uint parallelPartitionGroups = 0;

    for (auto partition : partitionList_) {
      //            std::cout << "partition ID: " << partition->ID_ <<
      //            std::endl; std::cout << "partition backend: " <<
      //            partition->backendName_ << std::endl;

      std::string deviceName = partition->backendName_;
      if (partition->backendName_.rfind("VTA", 0) == 0) {
        deviceName = "VTA";
      }

      DeviceInfo dinfo = getDeviceInfoForBackend(deviceName);
      backendNameStr += (partition->backendName_ + ":");
      logicalIDStr += (std::to_string(dinfo.deviceID) + ":");
      partitionNameStr += ("p" + std::to_string(partition->ID_) + ":");

      // std::cout << "partition ID: " << partition->ID_ << "
      // isParallelPartition_ "<< partition->isParallelPartition_<<std::endl;

      std::vector<NodeGroup *> parallelBranches;
      if (isParallel && !partition->partOfBranch_) {
        if (addedPartitions.find(partition->ID_) == addedPartitions.end()) {
          int pn = getBranchesWithSameLevel(&partitionList_, &parallelBranches,
                                            partition->level_);
          if (pn > 1 && parallelBranches.at(0)->parallelDeploy_) {

            parallelPartitionGroups++;
            parallelPartitions = parallelPartitions + ("(");
            addedPartitions.insert(partition->ID_);
            for (auto sameLevelPartition : parallelBranches) {
              // for (auto sameLevelPartition: parallelPartitions) {
              //                            std::cout <<
              //                            sameLevelPartition->backendName_ <<
              //                            std::endl;
              //                            if(sameLevelPartition->backendName_.rfind("VTA")
              //                            == 0) {
              parallelPartitions =
                  parallelPartitions +
                  ("p" + std::to_string(sameLevelPartition->ID_) + "||");
              //                            } else {
              //                                std::cout <<
              //                                "sameLevelPartition->backendName:
              //                                " <<
              //                                sameLevelPartition->backendName_
              //                                << std::endl;
              //                            }
              addedPartitions.insert(sameLevelPartition->ID_);
            }
            parallelPartitions.pop_back();
            parallelPartitions.pop_back();
            parallelPartitions = parallelPartitions + "):";
          }
        }
      }

      for (auto cnode : partition->nodeList_) {
        //          std::cout << "node name: " << cnode->node_->getName().str()
        //          << std::endl; std::cout << "node type: " <<
        //          cnode->node_->getKindName() << std::endl; std::cout <<
        //          "partition ID: " << partition->ID_ << std::endl;
        //                if(cnode->node_->getKind() ==
        //                Kinded::Kind::ConvolutionNodeKind) {
        //                    convNum++;
        //                    if(!cnode->backendName_.compare("VTA")) {
        //                        vtaConvNum++;
        //                    }
        //                } else if (cnode->node_->getKind() ==
        //                Kinded::Kind::ReluNodeKind) {
        //                    if(!cnode->backendName_.compare("VTA")) {
        //                        vtaReluNum++;
        //                    }
        //                }
        nodeToPartitionStr +=
            (cnode->name_ + "-" + std::to_string(partition->ID_));
        nodeToPartitionStr += "\n  :";
      }
    }
    nodeToPartitionStr =
        nodeToPartitionStr.substr(0, nodeToPartitionStr.length() - 4);

    partitionNameStr += ("p" + std::to_string(partitionList_.back()->ID_ + 1));

    if (backendMap_.find("Relay") == backendMap_.end()) {
      backendNameStr += "CPU";
      logicalIDStr += std::to_string(getDeviceInfoForBackend("CPU").deviceID);
    } else {
      backendNameStr += "Relay";
      logicalIDStr += std::to_string(getDeviceInfoForBackend("Relay").deviceID);
    }

    info.append(
        "numOfPartitions: " + std::to_string(partitionList_.size() + 1) + "\n");

    planFile << info;
    planFile << "backendNames: " << backendNameStr << std::endl;
    planFile << "partitionNames: " << partitionNameStr << std::endl;
    planFile << "nodeToPartition: " << nodeToPartitionStr << std::endl;
    planFile << "logicalIDs: " << logicalIDStr << std::endl;

    if (parallelPartitions.length() > 0) {
      parallelPartitions.pop_back();
      planFile << "parallelPartitions: " << parallelPartitions << std::endl;
    }
    planFile << "...\n" << std::endl;

    planFile.close();

    std::cout << "convNum: " << convNum << " vtaConvNum: " << vtaConvNum
              << " vtaReluNum: " << vtaReluNum
              << " parallelPartitionNum: " << parallelPartitionGroups
              << std::endl;
  }
}

void NestPartitioner::outPartitionPlanMultiVTA(std::string funcName,
                                               std::string filename,
                                               bool isParallel) {
  //    std::cout << "----------- outPartitionPlan ---------------" <<
  //    std::endl;
  std::ofstream planFile(filename);
  if (planFile.is_open()) {

    std::string info = "";
    info.append("---\n");
    info.append("funcName: " + funcName + "\n");

    std::string backendNameStr;
    std::string partitionNameStr;
    std::string nodeToPartitionStr;
    std::string logicalIDStr;
    std::string parallelPartitionStr = "";
    std::set<int> addedPartitions;

    uint convNum = 0;
    uint vtaConvNum = 0;
    uint vtaReluNum = 0;
    uint parallelPartitionGroups = 0;

    for (auto partition : partitionList_) {
      //            std::cout << "partition ID: " << partition->ID_ <<
      //            std::endl; std::cout << "partition backend: " <<
      //            partition->backendName_ << std::endl;

      std::string deviceName = partition->backendName_;
      if (partition->backendName_.rfind("VTA", 0) == 0) {
        deviceName = "VTA";
      }

      DeviceInfo dinfo = getDeviceInfoForBackend(deviceName);
      backendNameStr += (partition->backendName_ + ":");
      logicalIDStr += (std::to_string(dinfo.deviceID) + ":");
      partitionNameStr += ("p" + std::to_string(partition->ID_) + ":");

      // std::cout << "partition ID: " << partition->ID_ << "
      // isParallelPartition_ "<< partition->isParallelPartition_<<std::endl;
      std::vector<NodeGroup *> parallelPartitions, nonParallelPartitions;
      if (isParallel && deviceName.compare("VTA")) {
        if (addedPartitions.find(partition->ID_) == addedPartitions.end()) {
          getParallelVTAPartitions(&partitionList_, &parallelPartitions,
                                   &nonParallelPartitions, partition->level_);
          if (parallelPartitions.size() > 1) {
            parallelPartitionGroups++;
            parallelPartitionStr = parallelPartitionStr + ("(");
            addedPartitions.insert(partition->ID_);
            for (auto parallelVtaPartition : parallelPartitions) {
              //                            std::cout <<
              //                            sameLevelPartition->backendName_ <<
              //                            std::endl;
              parallelPartitionStr =
                  parallelPartitionStr +
                  ("p" + std::to_string(parallelVtaPartition->ID_) + "||");
              addedPartitions.insert(parallelVtaPartition->ID_);
            }
            parallelPartitionStr.pop_back();
            parallelPartitionStr.pop_back();
            parallelPartitionStr = parallelPartitionStr + "):";
          }
        }
      }

      for (auto cnode : partition->nodeList_) {
        //          std::cout << "node name: " << cnode->node_->getName().str()
        //          << std::endl; std::cout << "node type: " <<
        //          cnode->node_->getKindName() << std::endl; std::cout <<
        //          "partition ID: " << partition->ID_ << std::endl;

        //                if(cnode->node_->getKind() ==
        //                Kinded::Kind::ConvolutionNodeKind) {
        //                    convNum++;
        //                    if(!cnode->backendName_.compare("VTA")) {
        //                        vtaConvNum++;
        //                    }
        //                } else if (cnode->node_->getKind() ==
        //                Kinded::Kind::ReluNodeKind) {
        //                    if(!cnode->backendName_.compare("VTA")) {
        //                        vtaReluNum++;
        //                    }
        //                }

        nodeToPartitionStr +=
            (cnode->name_ + "-" + std::to_string(partition->ID_)) + "-" +
            partition->backendName_;
        nodeToPartitionStr += "\n  :";
      }
    }
    nodeToPartitionStr =
        nodeToPartitionStr.substr(0, nodeToPartitionStr.length() - 4);

    partitionNameStr += ("p" + std::to_string(partitionList_.back()->ID_ + 1));

    if (backendMap_.find("Relay") == backendMap_.end()) {
      backendNameStr += "CPU";
      logicalIDStr += std::to_string(getDeviceInfoForBackend("CPU").deviceID);
    } else {
      backendNameStr += "Relay";
      logicalIDStr += std::to_string(getDeviceInfoForBackend("Relay").deviceID);
    }

    info.append(
        "numOfPartitions: " + std::to_string(partitionList_.size() + 1) + "\n");

    planFile << info;
    planFile << "backendNames: " << backendNameStr << std::endl;
    planFile << "partitionNames: " << partitionNameStr << std::endl;
    planFile << "nodeToPartition: " << nodeToPartitionStr << std::endl;
    planFile << "logicalIDs: " << logicalIDStr << std::endl;

    if (parallelPartitionStr.length() > 0) {
      parallelPartitionStr.pop_back();
      planFile << "parallelPartitions: " << parallelPartitionStr << std::endl;
    }
    planFile << "...\n" << std::endl;

    planFile.close();

    std::cout << "convNum: " << convNum << " vtaConvNum: " << vtaConvNum
              << " vtaReluNum: " << vtaReluNum
              << " parallelPartitionNum: " << parallelPartitionGroups
              << std::endl;
  }
}
void NestPartitioner::outPartitionPlan(std::string funcName,
                                       std::string filename, bool isParallel) {
  //    std::cout << "----------- outPartitionPlan ---------------" <<
  //    std::endl;
  std::ofstream planFile(filename);
  if (planFile.is_open()) {

    std::string info = "";
    info.append("---\n");
    info.append("funcName: " + funcName + "\n");

    std::string backendNameStr;
    std::string partitionNameStr;
    std::string nodeToPartitionStr;
    std::string logicalIDStr;
    std::string parallelPartitions = "";
    std::set<int> addedPartitions;

    //        uint convNum = 0;
    //        uint vtaConvNum = 0;
    //        uint vtaReluNum = 0;
    uint parallelPartitionGroups = 0;

    for (auto partition : partitionList_) {
      //            std::cout << "partition ID: " << partition->ID_ <<
      //            std::endl; std::cout << "partition backend: " <<
      //            partition->backendName_ << std::endl;

      std::string deviceName = partition->backendName_;
      if (partition->backendName_.rfind("VTA", 0) == 0) {
        deviceName = "VTA";
      }

      DeviceInfo dinfo = getDeviceInfoForBackend(deviceName);
      backendNameStr += (partition->backendName_ + ":");
      logicalIDStr += (std::to_string(dinfo.deviceID) + ":");
      partitionNameStr += ("p" + std::to_string(partition->ID_) + ":");

      // std::cout << "partition ID: " << partition->ID_ << "
      // isParallelPartition_ "<< partition->isParallelPartition_<<std::endl;

      std::vector<NodeGroup *> parallelBranches;
      if (isParallel && !partition->partOfBranch_) {
        if (addedPartitions.find(partition->ID_) == addedPartitions.end()) {
          int pn = getBranchesWithSameLevel(&partitionList_, &parallelBranches,
                                            partition->level_);
          if (pn > 1 && parallelBranches.at(0)->parallelDeploy_) {

            parallelPartitionGroups++;
            parallelPartitions = parallelPartitions + ("(");
            addedPartitions.insert(partition->ID_);
            for (auto sameLevelPartition : parallelBranches) {
              std::cout << sameLevelPartition->backendName_ << std::endl;
              if (sameLevelPartition->backendName_.rfind("VTA") == 0) {
                parallelPartitions =
                    parallelPartitions +
                    ("p" + std::to_string(sameLevelPartition->ID_) + "||");
              } else {
                std::cout << "sameLevelPartition->backendName: "
                          << sameLevelPartition->backendName_ << std::endl;
              }
              addedPartitions.insert(sameLevelPartition->ID_);
            }
            parallelPartitions.pop_back();
            parallelPartitions.pop_back();
            parallelPartitions = parallelPartitions + "):";
          }
        }
      }

      for (auto cnode : partition->nodeList_) {
        //          std::cout << "node name: " << cnode->node_->getName().str()
        //          << std::endl; std::cout << "node type: " <<
        //          cnode->node_->getKindName() << std::endl; std::cout <<
        //          "partition ID: " << partition->ID_ << std::endl;

        //                if(cnode->node_->getKind() ==
        //                Kinded::Kind::ConvolutionNodeKind) {
        //                    convNum++;
        //                    if(!cnode->backendName_.compare("VTA")) {
        //                        vtaConvNum++;
        //                    }
        //                } else if (cnode->node_->getKind() ==
        //                Kinded::Kind::ReluNodeKind) {
        //                    if(!cnode->backendName_.compare("VTA")) {
        //                        vtaReluNum++;
        //                    }
        //                }

        nodeToPartitionStr +=
            (cnode->name_ + "-" + std::to_string(partition->ID_)) + "-" +
            partition->backendName_;
        nodeToPartitionStr += "\n  :";
      }
    }
    nodeToPartitionStr =
        nodeToPartitionStr.substr(0, nodeToPartitionStr.length() - 4);

    partitionNameStr += ("p" + std::to_string(partitionList_.back()->ID_ + 1));

    if (backendMap_.find("Relay") == backendMap_.end()) {
      backendNameStr += "CPU";
      logicalIDStr += std::to_string(getDeviceInfoForBackend("CPU").deviceID);
    } else {
      backendNameStr += "Relay";
      logicalIDStr += std::to_string(getDeviceInfoForBackend("Relay").deviceID);
    }

    info.append(
        "numOfPartitions: " + std::to_string(partitionList_.size() + 1) + "\n");

    planFile << info;
    planFile << "backendNames: " << backendNameStr << std::endl;
    planFile << "partitionNames: " << partitionNameStr << std::endl;
    planFile << "nodeToPartition: " << nodeToPartitionStr << std::endl;
    planFile << "logicalIDs: " << logicalIDStr << std::endl;

    if (parallelPartitions.length() > 0) {
      parallelPartitions.pop_back();
      planFile << "parallelPartitions: " << parallelPartitions << std::endl;
    }
    planFile << "...\n" << std::endl;

    planFile.close();

    //        std::cout << "convNum: " << convNum << " vtaConvNum: "<<
    //        vtaConvNum << " vtaReluNum: " << vtaReluNum << "
    //        parallelPartitionNum: " << parallelPartitionGroups << std::endl;
  }
}

NodeGroup *partitionSingleBranch(std::vector<NodeGroup *> *partitionList,
                                 NodeGroup *branch, unsigned int *partitionID,
                                 NodeGroup *lastPartition) {
  // std::cout << "---- partitionSingleBranch ----" << std::endl;

  CostNode *prevNode = nullptr;
  NodeGroup *partition = nullptr;

  if (lastPartition != nullptr && lastPartition->nodeList_.size() > 0 &&
      !lastPartition->backendName_.compare(
          branch->nodeList_.at(0)->backendName_)) {
    partition = lastPartition;
    prevNode = lastPartition->nodeList_.back();
    partition->backendName_ = lastPartition->backendName_;
  } else {
    partition = new NodeGroup;
    partition->ID_ = (*partitionID)++;
    partitionList->push_back(partition);
    partition->level_ = branch->level_;
    partition->branchID_ = branch->ID_;
  }

  //  std::cout << "branch nodes: " << branch->nodeList_.size() << std::endl;
  for (auto ni = branch->nodeList_.begin(); ni != branch->nodeList_.end();
       ni++) {
    CostNode *cn = *ni;

    if (prevNode != nullptr &&
        cn->backendName_.compare(prevNode->backendName_)) {
      partition->backendName_ = prevNode->backendName_;
      partition->partOfBranch_ = true;

      //     std::cout << "2 backend: " << partition->backendName_ << std::endl;
      partition = new NodeGroup;
      partition->ID_ = (*partitionID)++;
      partitionList->push_back(partition);
      partition->level_ = branch->level_;
      partition->branchID_ = branch->ID_;
      partition->partOfBranch_ = true;
    }

    if (branch->parallelDeploy_) {
      partition->backendName_ = branch->backendName_;
      partition->parallelDeploy_ = true;
    } else {
      partition->backendName_ = cn->backendName_;
    }

    partition->nodeList_.push_back(cn);
    prevNode = cn;
  }

  //  std::cout << "partition nodes: " << partition->nodeList_.size() <<
  //  std::endl; std::cout << "partition backend: " << partition->backendName_
  //  << ", ID: " << partition->ID_ << std::endl;
  return partition;
}

void makeSingleList(std::vector<NodeGroup *> *branchList,
                    NodeGroup *singleBranch, NodeGroup *exception) {

  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = *i;

    if (branch != nullptr && branch == exception)
      continue;

    for (auto ii = branch->nodeList_.begin(); ii != branch->nodeList_.end();
         ii++) {
      singleBranch->nodeList_.push_back((*ii));
    }
  }
}

void NestPartitioner::partitionBranches(std::vector<NodeGroup *> *branchList) {
  //  std::cout << "--------- partitionBranches -----------" << std::endl;

  unsigned int partitionID = 0;
  NodeGroup *lastPartition = nullptr;

  std::set<int> processedLevels;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = *i;
    lastPartition = partitionSingleBranch(&partitionList_, branch, &partitionID,
                                          lastPartition);
  }

  //    for (auto partition : partitionList_) {
  //        std::cout << "partition backend: " << partition->backendName_ <<
  //        std::endl;
  //    }
  std::cout << "--------- # partition: " << partitionList_.size()
            << "-----------" << std::endl;
}

float NestPartitioner::getInCommCost(CostNode *cnode) {
  float commCost = 0.0;

  std::string inKey = appGen_.getProfileKey(cnode->node_);
  commCost = backendCommProfileMap_["CPU"][inKey];

  return commCost;
}

void NestPartitioner::partitionBranches(std::vector<NodeGroup *> *branchList,
                                        bool isParallelDeploy) {
  //  std::cout << "--------- partitionBranches -----------" << std::endl;

  unsigned int partitionID = 0;
  NodeGroup *lastPartition = nullptr;

  std::set<int> processedLevels;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = *i;
    if (processedLevels.find(branch->level_) != processedLevels.end())
      continue;

    std::vector<NodeGroup *> sameLevelBranches, parallelBranches,
        nonParallelBranches;

    int bn = getBranchesWithSameLevel(branchList, &sameLevelBranches,
                                      branch->level_);
    processedLevels.insert(branch->level_);
    //        std::cout << "bn: " << bn << " level = " << branch->level_ <<
    //        std::endl;
    if (isParallelDeploy && bn > 1) {
      // get parallel branches
      int npn = getNonParallelDeployBranches(&sameLevelBranches,
                                             &nonParallelBranches);
      int pn = getParallelDeployBranches(&sameLevelBranches, "VTA",
                                         &parallelBranches);
      //            std::cout << "pn: " << pn << " npn = " << npn << std::endl;

      if (pn > 1) {

        // calculate communication overhead for spliting multiple branches
        for (auto pb : parallelBranches) {
          //                    std::cout << "* partitionID " << partitionID  <<
          //                    std::endl;
          lastPartition =
              partitionSingleBranch(&partitionList_, pb, &partitionID, nullptr);
          lastPartition = nullptr;
        }
      }
      if (npn > 0) {
        //                std::cout << "partitionID " << partitionID  <<
        //                std::endl;
        NodeGroup *nb = new NodeGroup;
        nb->level_ = branch->level_;
        nb->backendName_ = branch->backendName_;
        nb->branchID_ = branch->ID_;

        makeSingleList(&nonParallelBranches, nb, nullptr);
        lastPartition =
            partitionSingleBranch(&partitionList_, nb, &partitionID, nullptr);
      }

    } else if (bn == 1) {
      //            std::cout << "--> " <<
      //            branch->nodeList_.at(0)->node_->getName().str()  <<
      //            std::endl;
      lastPartition = partitionSingleBranch(&partitionList_, branch,
                                            &partitionID, lastPartition);
    } else {
      std::cout << "No branch in branchList." << std::endl;
      continue;
    }
  }

//  for (auto partition : partitionList_) {
//    std::cout << "partition backend: " << partition->backendName_ << std::endl;
//  }
  std::cout << "--------- # partition: " << partitionList_.size()
            << "-----------" << std::endl;
}

void NestPartitioner::partitionBranchesCPUVTA(
    std::vector<NodeGroup *> *branchList, bool isParallelDeploy) {
  //    std::cout << "--------- partitionBranches -----------" << std::endl;

  unsigned int partitionID = 0;
  NodeGroup *lastPartition = nullptr;

  std::set<int> processedLevels;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = *i;

    //        if(!branch->nodeList_.at(0)->node_->getName().compare("resnetv10_stage2_conv0_fwd__2"))
    //        {
    //            std::cout << "--> " <<
    //            branch->nodeList_.at(0)->node_->getName().str()  << std::endl;
    //        }

    if (processedLevels.find(branch->level_) != processedLevels.end())
      continue;

    std::vector<NodeGroup *> sameLevelBranches, parallelBranches,
        nonParallelBranches;

    int bn = getBranchesWithSameLevel(branchList, &sameLevelBranches,
                                      branch->level_);
    processedLevels.insert(branch->level_);
    //        std::cout << "bn: " << bn << " level = " << branch->level_ <<
    //        std::endl;
    if (isParallelDeploy && bn > 1) {
      // get parallel branches
      int npn = getNonParallelDeployBranches(&sameLevelBranches,
                                             &nonParallelBranches);
      int pn = getParallelDeployBranches(&sameLevelBranches, &parallelBranches);
      //            std::cout << "pn: " << pn << " npn = " << npn << std::endl;

      if (pn > 1) {

        // calculate communication overhead for spliting multiple branches

        for (auto pb : parallelBranches) {
          //                    std::cout << "* partitionID " << partitionID  <<
          //                    std::endl;
          lastPartition =
              partitionSingleBranch(&partitionList_, pb, &partitionID, nullptr);
          lastPartition = nullptr;
        }
      }
      if (npn > 0) {
        //                std::cout << "partitionID " << partitionID  <<
        //                std::endl;
        NodeGroup *nb = new NodeGroup;
        nb->level_ = branch->level_;
        nb->backendName_ = branch->backendName_;
        nb->branchID_ = branch->ID_;

        makeSingleList(&nonParallelBranches, nb, nullptr);
        lastPartition =
            partitionSingleBranch(&partitionList_, nb, &partitionID, nullptr);
      }

    } else if (bn == 1) {
      //            std::cout << "--> " <<
      //            branch->nodeList_.at(0)->node_->getName().str()  <<
      //            std::endl;
      lastPartition = partitionSingleBranch(&partitionList_, branch,
                                            &partitionID, lastPartition);
    } else {
      std::cout << "No branch in branchList." << std::endl;
      continue;
    }
  }

  //    for (auto partition : partitionList_) {
  //        std::cout << "partition backend: " << partition->backendName_ <<
  //        std::endl;
  //    }
  //    std::cout << "--------- # partition: " << partitionList_.size() <<
  //    "-----------" << std::endl;
}

void NestPartitioner::loadPerformProfileInfo(std::string pdir) {
  //    std::cout << "=== loadPerformProfileInfo start === " << std::endl;

  struct dirent *entry;
  DIR *dir = opendir(pdir.c_str());
  if (dir == NULL) {
    std::cout << "Partition profile directory is not found: " + pdir;
    return;
  } else {
    std::cout << "profile directory = " << pdir << std::endl;
  }

  while ((entry = readdir(dir)) != NULL) {
    std::string fn(entry->d_name);

    if (fn.substr(fn.find_last_of(".") + 1) == "yaml") {

      std::map<std::string, std::string> profileMap =
          deserializeStrStrMapFromYaml(pdir + "/" + fn);
      std::map<std::string, std::map<std::string, float>> *performMap;
      std::string performType = "ExeTime";
      if (profileMap.find("Type") != profileMap.end()) {
        performType = profileMap["Type"];
        //                std::cout <<"perform fn = " << fn << std::endl;
        //                std::cout <<"type = " << performType << std::endl;
      }

      if (!performType.compare("ExeTime")) {
        performMap = &backendExeProfileMap_;
      } else {
        performMap = &backendCommProfileMap_;
      }

      std::string backendName = profileMap["backendName"];
      backendSet_.insert(backendName);

      std::map<std::string, float> puValues;
      (*performMap).insert(std::make_pair(backendName, puValues));

      for (auto it = profileMap.begin(); it != profileMap.end(); it++) {
        if (!it->first.compare("backendName"))
          continue;

        std::string costValueStr = profileMap[it->first].c_str();

        //                std::cout <<"name = " << it->first.c_str() <<
        //                std::endl; std::cout <<"value = " <<
        //                costValueStr.c_str() << std::endl;

        if (!costValueStr.compare("INFINITY")) {
          puValues.insert(std::make_pair(it->first.c_str(), INFINITY));
        } else {
          puValues.insert(
              std::make_pair(it->first.c_str(), atof(costValueStr.c_str())));
        }
      }
      (*performMap)[backendName].insert(puValues.begin(), puValues.end());
    }
  }

  closedir(dir);

  //    std::cout << "=== loadPerformProfileInfo end === " << std::endl;
}

void NestPartitioner::getMinCostOfSingleNode(CostNode *cnode,
                                             std::vector<Backend *> backends,
                                             CostNode *prevCNode) {
  // std::cout << "--------- getMinCostOfSingleNode -----------" << std::endl;
  // std::cout << "node type: " << cnode->node_->getKindName() << std::endl;

  std::string key = appGen_.getProfileKey(cnode->node_);
  //    std::cout << "key: " << key << std::endl;

  //    cnode->cost_ = INFINITY;
  cnode->initCost();

  for (int i = 0; i < backends.size(); i++) {
    std::string backendName = backends[i]->getBackendName();
    //        std::cout << "backendName = " << backendName << std::endl;
    // non supported type
    if (!isSupportedOpType(cnode->node_->getKind(), backendName, cnode)) {
      continue;
    }

    float exeCost = backendExeProfileMap_[backendName][key];
    float commCost = 0.0;

    if (prevCNode != nullptr) {
      if (prevCNode->backendName_.compare(backendName)) { // different backend
        commCost = getInCommCost(cnode);
      }
    }

    //        std::cout << "exeCost = " << exeCost << std::endl;
    //        std::cout << "commCost = " << commCost << std::endl;

    float cost = exeCost + commCost;
    if (cost <= cnode->getCost()) {
      cnode->commCost_ = commCost;
      cnode->exeCost_ = exeCost;
      cnode->backendName_ = backendName;
    }
  }
  //    std::cout << "node = " << cnode->name_ << ", type = " <<
  //    cnode->node_->getKindName() <<", selected backendName = " <<
  //    cnode->backendName_ << std::endl;
}

void setChecked(std::vector<std::vector<CostNode>> *cnodeBfs, Node *node) {
  for (int i = 0; i < (*cnodeBfs).size(); i++) {
    for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {
      CostNode *cnode = &(*cnodeBfs)[i][j];
      if (cnode->node_ == node) {
        cnode->isChecked_ = true;
      }
    }
  }
}

bool isChecked(std::vector<std::vector<CostNode>> *cnodeBfs, Node *node) {
  for (int i = 0; i < (*cnodeBfs).size(); i++) {
    for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {
      CostNode *cnode = &(*cnodeBfs)[i][j];
      if (cnode->node_ == node) {
        return cnode->isChecked_;
      }
    }
  }
  return false;
}

CostNode *getCostNode(std::vector<std::vector<CostNode>> *cnodeBfs,
                      Node *node) {
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

void NestPartitioner::makeCostNode(
    Function *function, std::vector<std::vector<CostNode>> *cnodeBfs) {
  // std::cout << "--------- makeCostNode -----------" << std::endl;
  BFSLevel bfs = getBFSLevel(function);
  size_t level = bfs.size();

  for (int i = level - 1; i >= 0; i--) {
    std::vector<CostNode> cnodeGroup;
    for (size_t j = 0; j < bfs[i].size(); j++) {
      Node *node = bfs[i][j];
      CostNode cnode;
      cnode.node_ = node;
      cnode.name_ = node->getName();
      cnodeGroup.push_back(cnode);

      for (size_t i = 0, e = deviceInfo_.size(); i < e; i++) {

        std::string backendName = deviceInfo_[i].backendName;
        cnode.minCostMap_[backendName] = INFINITY;
      }
    }
    cnodeBfs->push_back(cnodeGroup);
  }
}

float getCost(std::vector<NodeGroup *> *branchList) {
  // std::cout << "----------- getCost ---------------" << std::endl;
  // std::cout << "branch size: " << branchList->size() << std::endl;

  float totalCost = 0.0;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = *i;

    // std::cout << "branch id: " << branch->ID_ << std::endl;
    for (auto ii = branch->nodeList_.begin(); ii != branch->nodeList_.end();
         ii++) {
      totalCost += (*ii)->getCost();
      // std::cout << "totalCost: " << totalCost << std::endl;
    }
  }
  return totalCost;
}

float getCost(NodeGroup *branch) {
  float totalCost = 0.0;
  for (auto ii = branch->nodeList_.rbegin(); ii != branch->nodeList_.rend();
       ++ii) {
    // std::cout << "name: " << (*ii)->name_<< ", backend = " <<
    // (*ii)->backendName_ << ", cost = " << (*ii)->cost_ << std::endl;
    totalCost += (*ii)->getCost();
  }
  return totalCost;
}

bool NestPartitioner::isSupportedOpType(Kinded::Kind opType,
                                        std::string backendName,
                                        CostNode *cnode) {

  if (backendMap_[backendName].nonSupportedNodesKinds.size() > 0 &&
      (backendMap_[backendName].nonSupportedNodesKinds.find(opType) !=
       backendMap_[backendName].nonSupportedNodesKinds.end()))
    return false;

  if (backendMap_[backendName].supportedNodesKinds.size() > 0) {
    if (backendMap_[backendName].supportedNodesKinds.find(opType) ==
        backendMap_[backendName].supportedNodesKinds.end()) {
      return false;
    }
    if (opType == Kinded::Kind::ConvolutionNodeKind &&
        !isVTAConv((ConvolutionNode *)cnode->node_)) {
      return false;
    }
  }
  return true;
}

void NestPartitioner::setCost(std::vector<Backend *> *backends, CostNode *pnode,
                              CostNode *cnode) {
  //    std::cout << "----------- getCost() start ---------------" << std::endl;
  std::string curKey = appGen_.getProfileKey(cnode->node_);

  bool noBackend = true;
  for (Backend *backend : *backends) {
    if (!backend->getBackendName().compare(defaultBackend_))
      continue;

    if (isSupportedOpType(cnode->node_->getKind(), backend->getBackendName(),
                          cnode)) {
      noBackend = false;
    }
  }

  if (pnode == nullptr) {
    for (Backend *backend : *backends) {
      float exeCost = INFINITY;
      if (isSupportedOpType(cnode->node_->getKind(), backend->getBackendName(),
                            cnode) &&
          backend->getBackendName().compare(defaultBackend_)) {
        exeCost = backendExeProfileMap_[backend->getBackendName()][curKey];
      }

      if (noBackend && !backend->getBackendName().compare(defaultBackend_)) {
        exeCost = 0.0;
      }

      //            std::cout << "backend = " << backend->getBackendName() << ",
      //            exeCost = " << exeCost << std::endl;
      cnode->minCostMap_[backend->getBackendName()] = exeCost;
    }

  } else {

    std::string compKey = appGen_.getProfileKey(pnode->node_) + "+" + curKey;
    //        std::cout << "curKey = " << curKey << std::endl;

    // check supported backends
    for (Backend *curBackend : *backends) {
      float exeCost = INFINITY;
      float minCost = INFINITY;
      float totalCost = INFINITY;
      std::string minBackend = backends->front()->getBackendName();

      if (isSupportedOpType(cnode->node_->getKind(),
                            curBackend->getBackendName(), cnode) &&
          curBackend->getBackendName().compare(defaultBackend_)) {
        exeCost = backendExeProfileMap_[curBackend->getBackendName()][curKey];
      }

      if (noBackend && !curBackend->getBackendName().compare(defaultBackend_)) {
        exeCost = 0.0;
      }
      //            std::cout << "<curBackend> => " <<
      //            curBackend->getBackendName() << ", exeCost = " << exeCost <<
      //            std::endl;
      //
      float commCost = getInCommCost(cnode);
      for (Backend *prevBackend : *backends) {
        bool fused = false;
        //                std::cout << "commCost = " << commCost << std::endl;
        if (!curBackend->getBackendName().compare(
                prevBackend->getBackendName())) { // same backend

          float compCost = INFINITY;
          if (backendExeProfileMap_[curBackend->getBackendName()].find(
                  compKey) !=
              backendExeProfileMap_[prevBackend->getBackendName()].end()) {
            // fusible
            compCost =
                backendExeProfileMap_[curBackend->getBackendName()][compKey];
          }

          float prevExeCost =
              backendExeProfileMap_[prevBackend->getBackendName()]
                                   [appGen_.getProfileKey(pnode->node_)];
          if (compCost < (prevExeCost + exeCost)) {
            totalCost = pnode->minCostMap_[prevBackend->getBackendName()] -
                        prevExeCost + compCost - commCost;
            fused = true;
          } else {
            // same backend, do not fuse operators
            totalCost = pnode->minCostMap_[prevBackend->getBackendName()] +
                        exeCost - commCost;
          }
        } else {
          totalCost =
              pnode->minCostMap_[prevBackend->getBackendName()] + commCost * 2 +
              exeCost +
              2000; // 2000: penalty: function call overhead, cache conflict, ..
          //  totalCost = pnode->minCostMap_[prevBackend->getBackendName()] +
          //  commCost*2 + exeCost; //2000: penalty: function call overhead,
          //  cache conflict, ..
        }
        //                std::cout << "prevBackend = " <<
        //                prevBackend->getBackendName() << std::endl; std::cout
        //                << "totalCost = " << totalCost << std::endl; std::cout
        //                << "minCost = " << minCost << std::endl;

        if (totalCost <= minCost) {
          minCost = totalCost;
          minBackend = prevBackend->getBackendName();
        }
        //                std::cout << "minBackend = " << minBackend <<
        //                std::endl; std::cout << "minCost = " << minCost <<
        //                std::endl;
      }
      cnode->minCostMap_[curBackend->getBackendName()] = minCost;
      cnode->prevMinBackendMap_[curBackend->getBackendName()] = minBackend;
    }
  }

  //    std::cout << "----------- result ---------------" << std::endl;
  //    for(Backend* curBackend: *backends) {
  //        std::cout << "backend => " << curBackend->getBackendName() <<
  //        std::endl; std::cout << "cost => " <<
  //        cnode->minCostMap_[curBackend->getBackendName()] << std::endl;
  //        std::cout << "prev backend => " <<
  //        cnode->prevMinBackendMap_[curBackend->getBackendName()] <<
  //        std::endl;
  //    }
  //
  //    std::cout << "----------- getCost() end ---------------" << std::endl;
}

void NestPartitioner::allocMinBranchCost(std::vector<NodeGroup *> *branchList) {
  std::cout << "--------- allocMinBranchCost -----------" << std::endl;

  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);

  // calculate costs for nodes
  float totalCost = 0.0;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    //        std::cout << " ================== new branch =================="<<
    //        std::endl;
    NodeGroup *branch = (*i);

    int size = branch->nodeList_.size();
    for (int bi = 0; bi < size; bi++) {
      CostNode *cnode = branch->nodeList_.at(bi);
      CostNode *pnode = nullptr;

      if (bi > 0) {
        pnode = branch->nodeList_.at(bi - 1);
      }
      //            if(pnode != nullptr)
      //                std::cout << "pnode = " << pnode->node_->getName().str()
      //                << std::endl;
      //            std::cout << "cnode = " << cnode->node_->getName().str() <<
      //            std::endl;
      setCost(&backends, pnode, cnode);
      pnode = cnode;
    }
    // allocate backends requiring the minimum total cost
    CostNode *lastNode = branch->nodeList_.at(size - 1);
    //        std::cout << "lastNode = " << lastNode->node_->getName().str() <<
    //        std::endl;
    float minCost = INFINITY;
    std::string minBackend = "";
    for (Backend *backend : backends) {
      std::string backendName = backend->getBackendName();
      float cost = lastNode->minCostMap_[backendName];

      //            std::cout << "backend = " << backendName << std::endl;
      //            std::cout << "cost = " << cost << std::endl;

      if (cost <= minCost) {
        minCost = cost;
        minBackend = backendName;
      }
    }
    lastNode->backendName_ = minBackend;
    totalCost += lastNode->minCostMap_[minBackend];

    //        std::cout << "Last node = " << lastNode->node_->getName().str() <<
    //        std::endl; std::cout << "Backend = " << minBackend << std::endl;

    for (int bi = size - 2; bi >= 0; bi--) {
      CostNode *cnode = branch->nodeList_.at(bi);
      cnode->backendName_ = lastNode->prevMinBackendMap_[minBackend];
      minBackend = cnode->backendName_;
      lastNode = cnode;

      //            std::cout << "node = " << cnode->node_->getName().str() <<
      //            std::endl; std::cout << "Backend = " << cnode->backendName_
      //            << std::endl;
    }

    //      exit(0);
  }

  std::cout << "Total Cost = " << totalCost << std::endl;
}

bool NestPartitioner::isVTAConv(ConvolutionNode *convNode) {
  //    std::cout << "(Conv) name: " << convNode->getName().str() << ", filter
  //    size: " << convNode->getFilter().dims()[0] << ", input channel: " <<
  //    convNode->getInput().dims()[3] << std::endl;
  if (convNode->getFilter().dims()[0] % 16 == 0 &&
      convNode->getInput().dims()[3] % 16 == 0 && convNode->getGroup() == 1) {
    return true;
  }

  return false;
}

void NestPartitioner::allocateVTAOps(std::vector<NodeGroup *> *branchList) {
  std::cout << "--------- allocateVTAOps -----------" << std::endl;
  // std::cout << "\n[allocateVTAOps] Allocate VTA to VTA-computable
  // operations*." << std::endl;
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);

  bool isRelayExist = false;
  if (backendMap_.find("Relay") != backendMap_.end()) {
    isRelayExist = true;
  }

  CostNode *prevCnode = nullptr;
  std::string key, compKey;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = (*i);
    int size = branch->nodeList_.size();
    for (int bi = 0; bi < size; bi++) {
      CostNode *cnode = branch->nodeList_.at(bi);
      bool isAlloc = false;

      key = appGen_.getProfileKey(cnode->node_);
      cnode->commCost_ = 0.0;

      if (isSupportedOpType(cnode->node_->getKind(), "VTA", cnode)) {
        cnode->backendName_ = "VTA";
        isAlloc = true;

        cnode->exeCost_ = backendExeProfileMap_[cnode->backendName_][key];

      } else {
        if (prevCnode != nullptr &&
            cnode->node_->getKind() == Kinded::Kind::ReluNodeKind) {
          if (prevCnode->node_->getKind() ==
              Kinded::Kind::ConvolutionNodeKind) {
            cnode->backendName_ = prevCnode->backendName_;
            isAlloc = true;

            compKey = appGen_.getProfileKey(prevCnode->node_) + "+" + key;
            prevCnode->exeCost_ =
                backendExeProfileMap_[prevCnode->backendName_][compKey];
            cnode->exeCost_ = 0.0;
          }
        }
      }
      if (!isAlloc) {
        if (isRelayExist &&
            isSupportedOpType(cnode->node_->getKind(), "Relay", cnode)) {
          cnode->backendName_ = "Relay";
        } else {
          cnode->backendName_ = "CPU";
          //                std::cout << "node = " <<
          //                cnode->node_->getName().str() << std::endl;
          //                std::cout << "Backend = " << cnode->backendName_ <<
          //                std::endl;
        }

        cnode->exeCost_ = backendExeProfileMap_[cnode->backendName_][key];
      }

      if (prevCnode != nullptr &&
          cnode->backendName_.compare(prevCnode->backendName_)) {
        cnode->commCost_ = backendCommProfileMap_[cnode->backendName_][key];
        ;
      }

      prevCnode = cnode;
    }
  }
}

void NestPartitioner::allocateRelayOps(std::vector<NodeGroup *> *branchList) {
  std::cout << "--------- allocateRelayOps -----------" << std::endl;
  // std::cout << "\n[allocateVTAOps] Allocate VTA to VTA-computable
  // operations*." << std::endl;
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);

  if (backendMap_.find("Relay") == backendMap_.end()) {
    std::cout << "No Relay backend!!" << std::endl;
    return;
  }

  CostNode *prevCnode = nullptr;
  std::string key, compKey;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    for (auto bi = (*i)->nodeList_.begin(); bi != (*i)->nodeList_.end(); bi++) {
      CostNode *cnode = *bi;

      key = appGen_.getProfileKey(cnode->node_);
      cnode->commCost_ = 0.0;

      if (isSupportedOpType(cnode->node_->getKind(), "Relay", cnode)) {
        cnode->backendName_ = "Relay";
      } else {
        cnode->backendName_ = "CPU";
        //                std::cout << "node = " <<
        //                cnode->node_->getName().str() << std::endl; std::cout
        //                << "Backend = " << cnode->backendName_ << std::endl;
      }

      cnode->exeCost_ = backendExeProfileMap_[cnode->backendName_][key];

      if (prevCnode != nullptr &&
          cnode->backendName_.compare(prevCnode->backendName_)) {
        cnode->commCost_ = backendCommProfileMap_[cnode->backendName_][key];
        ;
      }

      prevCnode = cnode;
    }
  }
}

void NestPartitioner::allocateOptimalPUSingleNode(
    std::vector<NodeGroup *> *branchList) {
  // std::cout << "--------- findOptimalPartitionsSingleNode -----------" <<
  // std::endl;
  // std::cout << "\n[allocateOptimalPUSingleNode] Allocate PUs to operations
  // considering the cost of a *Single Operation*." << std::endl;
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);

  CostNode *prevCNode = nullptr;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    for (auto bi = (*i)->nodeList_.begin(); bi != (*i)->nodeList_.end(); bi++) {
      CostNode *cnode = *bi;
      getMinCostOfSingleNode(cnode, backends, prevCNode);
      prevCNode = cnode;
    }
  }
}

void getBackendSet(std::set<std::string> *bset, NodeGroup *branch) {
  for (auto bii = branch->nodeList_.begin(); bii != branch->nodeList_.end();
       bii++) {
    bset->insert((*bii)->backendName_);
  }
}

bool NestPartitioner::setBranchPU(NodeGroup *branch) {

  std::string backend = "";
  bool singlePU = true;
  for (int i = 0; i < branch->nodeList_.size(); i++) {
    CostNode *node = branch->nodeList_.at(i);
    if (!node->backendName_.compare("VTA")) {
      branch->vtaCount_++;
    }
    if (i == 0) {
      backend = node->backendName_;
      continue;
    } else {
      if (backend.compare(node->backendName_)) {
        singlePU = false;
      }
    }
  }
  // std::cout << "[useSinglePU] backend = " << backend << std::endl;
  //    branch->branchBackend_ = backend;
  if (singlePU) {
    branch->backendName_ = backend;
  }
  branch->isSinglePU = singlePU;

  return singlePU;
}

void NestPartitioner::findParallelDeployBranchesForMultiVTA(
    std::vector<NodeGroup *> *branchList, int cpuNum, int vtaNum) {
  std::cout << "--------- [findParallelBranches] -----------" << std::endl;
  // std::cout << "Original total cost =  " << originalTotalCost << std::endl;

  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = (*i);
    setBranchPU(branch);
    //        std::cout << "branch backend =  " << branch->backendName_ <<
    //        std::endl;
  }

  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    // bool existParallelBranch = true;
    if ((*i)->inBranchList_.size() > 1) { // parallel branches
      int vtaCnt = 0;
      std::vector<NodeGroup *> pBranchList;
      for (auto bi = (*i)->inBranchList_.begin();
           bi != (*i)->inBranchList_.end(); bi++) {
        NodeGroup *branch = (*bi);

        std::cout << branch->nodeList_.at(0)->name_
                  << " backend = " << branch->backendName_
                  << " vtaCount = " << branch->vtaCount_ << std::endl;

        if ((branch->isSinglePU && !branch->backendName_.compare("VTA")) ||
            branch->vtaCount_ > 0) {
          // if (!branch->backendName_.compare("VTA")) {
          vtaCnt++;
          pBranchList.push_back(branch);
        }
      }

      // parallel vta brancvtaNumh
      if (pBranchList.size() >= 2) {
        if (vtaCnt > vtaNum) {
          std::cout << "Too many parallel branches for NPU!!" << std::endl;
          continue;
        }

        for (auto bi = pBranchList.begin(); bi != pBranchList.end(); bi++) {
          NodeGroup *branch = (*bi);
          // std::cout << "branch backend = " << branch->backendName_ <<
          // std::endl;

          std::cout << "1) VTA branch = " << branch->nodeList_.at(0)->name_
                    << "backend = " << branch->backendName_
                    << "parallelDeploy_ = " << branch->parallelDeploy_
                    << std::endl;
          if (vtaNum > 0 && vtaCnt > 0) {
            // if (!branch->backendName_.compare("VTA") && vtaCnt > 0) {
            //                        std::cout << "branch vtacnt = " <<
            //                        branch->vtaCount_ << std::endl;
            if (((branch->isSinglePU &&
                  !branch->backendName_.compare("VTA")))) {
              branch->parallelDeploy_ = true;
              if (vtaNum == 1) {
                branch->backendName_ = "VTA";
              } else {
                branch->backendName_ = "VTA-" + std::to_string(vtaCnt - 1);
              }
              vtaCnt--;
            } else if (!branch->isSinglePU && branch->vtaCount_ > 0) {
              // replace vta#
              branch->parallelDeploy_ = true;
              if (vtaNum > 1) {
                for (CostNode *node : branch->nodeList_) {
                  if (!node->backendName_.compare("VTA")) {
                    node->backendName_ = "VTA-" + std::to_string(vtaCnt - 1);
                  }
                }
                branch->backendName_ = "VTA-" + std::to_string(vtaCnt - 1);
              }
              vtaCnt--;
            }
            // std::cout << "branch backend = " << branch->backendName_ << ", "
            // << branch->parallelDeploy_ << std::endl;
          }
          std::cout << "2) VTA branch = " << branch->nodeList_.at(0)->name_
                    << "backend = " << branch->backendName_
                    << "parallelDeploy_ = " << branch->parallelDeploy_
                    << std::endl;
        }
        pBranchList.clear();
      }
    }
  }
}

void NestPartitioner::findParallelDeployBranchesForCPUVTA(
    std::vector<NodeGroup *> *branchList, int cpuNum, int vtaNum) {
  std::cout << "--------- [findParallelBranches] -----------" << std::endl;
  // std::cout << "Original total cost =  " << originalTotalCost << std::endl;

  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = (*i);
    setBranchPU(branch);
    //        std::cout << "branch backend =  " << branch->backendName_ <<
    //        std::endl;
  }

  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    if ((*i)->inBranchList_.size() > 1) { // parallel branches
      //            std::cout << "--" << std::endl;

      int vtaCnt = 0, cpuCnt = 0;
      std::vector<NodeGroup *> pBranchList;
      for (auto bi = (*i)->inBranchList_.begin();
           bi != (*i)->inBranchList_.end(); bi++) {
        NodeGroup *branch = (*bi);
        //                pBranchList.push_back(branch);
        if (!branch->backendName_.compare("VTA")) {
          vtaCnt++;
          pBranchList.push_back(branch);
        } else if (!branch->backendName_.compare("CPU")) {
          cpuCnt++;
          pBranchList.push_back(branch);
        } else {
          //                    std::cout << "branch backend = " <<
          //                    branch->backendName_ << std::endl; for(auto n:
          //                    branch->nodeList_) {
          //                        std::cout << " " << n->backendName_;
          //                    }
          //                    std::cout << std::endl;
        }
      }

      // parallel vta branch
      if (pBranchList.size() >= 2) {
        if (vtaCnt > vtaNum) {
          std::cout << "Too many parallel branches for NPU!!" << std::endl;
          continue;
        }

        if (cpuCnt > cpuNum) {
          std::cout << "Too many parallel branches for CPU!!" << std::endl;
          continue;
        }

        for (auto bi = pBranchList.begin(); bi != pBranchList.end(); bi++) {
          NodeGroup *branch = (*bi);
          //                    std::cout << "branch backend = " <<
          //                    branch->backendName_ << std::endl;

          if (vtaNum > 0) {
            if (!branch->backendName_.compare("VTA") && vtaCnt > 0) {
              //                            std::cout << "VTA branch = " <<
              //                            branch->nodeList_.at(0)->name_ <<
              //                            std::endl;
              branch->parallelDeploy_ = true;
              if (vtaNum == 1) {
                branch->backendName_ = "VTA";
              } else {
                branch->backendName_ = "VTA-" + std::to_string(vtaCnt - 1);
              }
              vtaCnt--;
              //                            std::cout << "branch backend = " <<
              //                            branch->backendName_ << ", " <<
              //                            branch->isParallelBranch_ <<
              //                            std::endl;
            }
          }

          if (cpuNum > 0) {
            if (!branch->backendName_.compare("CPU") && cpuCnt > 0) {
              //                            std::cout << "CPU branch = " <<
              //                            branch->nodeList_.at(0)->name_ <<
              //                            std::endl;
              branch->parallelDeploy_ = true;
              branch->backendName_ = "CPU";
              cpuCnt--;
              //                          std::cout << "branch backend = " <<
              //                          branch->backendName_ << ", " <<
              //                          branch->isParallelBranch_ <<
              //                          std::endl;
            }
          }
        }
        pBranchList.clear();
      }
    } else {
      //            std::cout << "> branch backend = " << (*i)->backendName_ <<
      //            ", " << (*i)->isParallelBranch_ << std::endl; std::cout <<
      //            "> branch backend = " <<
      //            (*i)->nodeList_.at(0)->node_->getName().str()<< std::endl;
    }
  }
}

Expected<DAGListTy> NestPartitioner::partition(CompilationContext &cctx) {
  std::string dir = "/home/msyu/Dropbox/Project/development/configs/"
                    "partition_perform_profile";
  return partition(cctx, 3, "", dir + "/NESTOptimalPlan.yaml", 0, 0, nullptr);
}

std::vector<Node *> getParallelNodes(BFSLevel dag, Node *node) {

  std::vector<Node *> parentNodes;

  if (node->getKind() == Kinded::Kind::PlaceholderKind ||
      node->getKind() == Kinded::Kind::ConstantKind) {
    return parentNodes;
  }

  for (int i = 0; i < node->getNumInputs(); i++) {
    auto inNode = node->getNthInput(i).getNode();

    if (inNode->getKind() == Kinded::Kind::SaveNodeKind ||
        inNode->getKind() == Kinded::Kind::ConstantKind ||
        inNode->getKind() == Kinded::Kind::PlaceholderKind) {
      continue;
    }

    if (!inNode->hasOneUse()) {
      continue;
    }
    parentNodes.push_back(inNode);
  }
  return parentNodes;
}

std::vector<Node *> getParentNodes(BFSLevel dag, Node *node) {
  std::vector<Node *> parentNodes;

  if (node->getKind() == Kinded::Kind::PlaceholderKind ||
      node->getKind() == Kinded::Kind::ConstantKind) {
    return parentNodes;
  }

  for (int i = 0; i < node->getNumInputs(); i++) {
    auto inNode = node->getNthInput(i).getNode();
    if ((inNode->getKind() != Kinded::Kind::ConstantKind) &&
        inNode->getKind() != Kinded::Kind::PlaceholderKind) {
      parentNodes.push_back(inNode);
    }
  }
  return parentNodes;
}

void addToBranchList(std::vector<NodeGroup *> *branchList, NodeGroup *branch) {
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    if ((*i)->level_ > branch->level_) {
      branchList->insert(i, branch);
      return;
    }
  }
  branchList->push_back(branch);
}

void NestPartitioner::getBranch(std::vector<std::vector<CostNode>> *cnodebfs,
                                BFSLevel dag, std::vector<Node *> parallelNodes,
                                Node *lastNode, NodeGroup *prevBranch,
                                unsigned int level, unsigned int branchID,
                                std::vector<NodeGroup *> *branchList) {
  //  std::cout << "[start getBranch]" << " level = " << level << std::endl;

  if (lastNode->getKind() == Kinded::Kind::PlaceholderKind)
    return;

  //  std::cout << "[getBranch]" << "lastNode = " << lastNode->getKindName() <<
  //  " name = "<< lastNode->getName().str() << std::endl; std::cout <<
  //  "parallelNodes.size() = " << parallelNodes.size() << " level = " << level
  //  << std::endl;

  for (int i = 0; i < parallelNodes.size(); i++) {

    NodeGroup *branch = new NodeGroup;
    branch->level_ = level;
    branch->ID_ = branchID;
    // branchList->push_back(branch);
    addToBranchList(branchList, branch);
    if (prevBranch != nullptr) {
      prevBranch->inBranchList_.push_back(branch);
    }

    Node *curNode = parallelNodes.at(i);
    std::vector<Node *> pNodes = getParallelNodes(dag, curNode);

    //    std::cout <<  "node = " << curNode->getKindName() << " name = "<<
    //    lastNode->getName().str() << " level = " << level << std::endl;
    //    std::cout << "pNodes size = " << pNodes.size() << std::endl;

    if (pNodes.size() > 1) {
      // std::cout << "[getBranch 1]" << std::endl;
      getBranch(cnodebfs, dag, pNodes, curNode, branch, level + 1, ++branchID,
                branchList);

    } else if (pNodes.size() == 1) {
      // std::cout << "[getBranch 11]" << std::endl;
      if (!isChecked(cnodebfs, lastNode)) {
        //        std::cout << "[getBranch]"
        //                  << "add node = " << curNode->getKindName()
        //                  << " name = " << curNode->getName().str()
        //                  << " level = " << level << " branch = " <<
        //                  branch->ID_
        //                  << std::endl;
        branch->nodeList_.push_back(findNodeFromCNodeList(cnodebfs, lastNode));
        setChecked(cnodebfs, lastNode);
      }

      backtrack(cnodebfs, dag, curNode, branch, level, branchID, branchList);
      // std::cout << "[getBranch 2] branch size = " << branch->nodeList_.size()
      // << " level = " << level << std::endl;
    } else {
      if (!isChecked(cnodebfs, curNode)) {
        //        std::cout << "[getBranch]"
        //                  << "add node = " << curNode->getKindName()
        //                  << " name = " << curNode->getName().str()
        //                  << " level = " << level << " branch = " <<
        //                  branch->ID_
        //                  << std::endl;
        branch->nodeList_.push_back(findNodeFromCNodeList(cnodebfs, curNode));
        setChecked(cnodebfs, curNode);

        pNodes = getParentNodes(dag, curNode);
        getBranch(cnodebfs, dag, pNodes, curNode, branch, level + 1, ++branchID,
                  branchList);
        //      } else {
        //        std::cout << "HERE 1." << std::endl;
      }
    }

    if (branch->nodeList_.size() == 0) {
      branchList->erase(find(branchList->begin(), branchList->end(), branch));
      if (prevBranch != nullptr) {
        prevBranch->inBranchList_.erase(find(prevBranch->inBranchList_.begin(),
                                             prevBranch->inBranchList_.end(),
                                             branch));
      }
      delete branch;
    }
  }

  //  std::cout << "[finish getBranch]" << " level = " << level << std::endl;
}

void NestPartitioner::backtrack(std::vector<std::vector<CostNode>> *cnodebfs,
                                BFSLevel dag, Node *lastNode, NodeGroup *branch,
                                unsigned int level, unsigned int branchID,
                                std::vector<NodeGroup *> *branchList) {
  //  std::cout << "[start backtrack]"
  //            << "add node = " << lastNode->getKindName()
  //            << " name = " << lastNode->getName().str() << " level = " <<
  //            level
  //            << " branch = " << branch->ID_
  //            << std::endl;

  if (lastNode->getKind() == Kinded::Kind::PlaceholderKind)
    return;

  if (lastNode->getKind() == Kinded::Kind::SaveNodeKind)
    return;

  if (isChecked(cnodebfs, lastNode))
    return;

  //  std::cout << "[backtrack]"
  //            << "add node = " << lastNode->getKindName()
  //            << " name = " << lastNode->getName().str()
  //            << " level = " << level << " branch = " << branch->ID_
  //            << std::endl;
  branch->nodeList_.push_back(findNodeFromCNodeList(cnodebfs, lastNode));
  setChecked(cnodebfs, lastNode);

  std::vector<Node *> pNodes = getParallelNodes(dag, lastNode);
  if (pNodes.size() > 1) {
    // std::cout << "[getBranch 2]" << std::endl;
    getBranch(cnodebfs, dag, pNodes, lastNode, branch, level + 1, branchID + 1,
              branchList);
  } else if (pNodes.size() == 1) {
    // std::cout << "[getBranch 22]" << std::endl;
    backtrack(cnodebfs, dag, pNodes.at(0), branch, level, branchID, branchList);
  } else if (pNodes.size() == 0) {
    pNodes = getParentNodes(dag, lastNode);
    // if(pNodes.size() == 1) {
    //      std::cout << "[backtrack] parent nodes size() == " << pNodes.size()
    //      << std::endl;
    getBranch(cnodebfs, dag, pNodes, lastNode, branch, level + 1, branchID + 1,
              branchList);
    //} else {
    //  std::cout << "[backtrack] parent nodes size() == 0" << std::endl;
    // return;
    //}
  }
  // std::cout << "[finish backtrack]" << " branch = " << branch->ID_<<
  // std::endl;
}

void NestPartitioner::analyzeDAG(std::vector<std::vector<CostNode>> *cnodebfs,
                                 BFSLevel dag,
                                 std::vector<NodeGroup *> *branchList) {
  unsigned int level = 0;
  //    std::cout << "[analyzeDAG]"<< std::endl;

  size_t saveNodeSize = dag[0].size();
  for (int i = 0; i < saveNodeSize; i++) {
    setChecked(cnodebfs, dag[0][i]);

    std::vector<Node *> pNodes = getParallelNodes(dag, dag[0][i]);

    //        std::cout << " curNode = " << dag[0][i]->getKindName() << ", "<<
    //        dag[0][i]->getName().str() << std::endl;
    //    std::cout << "[analyzeDAG]"
    //              << " pNodes size = " << pNodes.size() << std::endl;
    //  std::cout << "[getBranch 0]" << std::endl;
    getBranch(cnodebfs, dag, pNodes, dag[0][i], nullptr, level, 0, branchList);
    //    std::cout << "- finish analyzeDAG -" << std::endl;
    //    std::cout << "# branch: " << branchList->size() << std::endl;

    std::reverse(branchList->begin(), branchList->end());
    for (auto i = branchList->begin(); i != branchList->end(); i++) {
      std::reverse((*i)->nodeList_.begin(), (*i)->nodeList_.end());
    }
  }
  //
  //  for (auto i = branchList->begin(); i != branchList->end(); i++) {
  //      NodeGroup* branch = (*i);
  //
  //      std::vector<NodeGroup*> parallelBranches;
  //      int pn = getBranchesWithSameLevel(&partitionList_, &parallelBranches,
  //      branch->level_); if(pn > 1) {
  //          for (auto sameLevelBranch: parallelBranches) {
  //            sameLevelBranch->parallelBranch_ = true;
  //          }
  //      }
  ////    std::cout << "branchID: " << (*i)->ID_ << ", level: " << (*i)->level_
  ///<< ", size: " << (*i)->nodeList_.size() << std::endl; /    std::cout <<
  ///"level : " << (*i)->level_ << std::endl; /    std::cout << "# inBranch: "
  ///<< (*i)->inBranchList_.size() << std::endl;
  ////
  ////    for (auto ii = (*i)->nodeList_.begin(); ii != (*i)->nodeList_.end();
  /// ii++ ) { /      std::cout << "name: " << (*ii)->name_<< std::endl; /
  /// std::cout << "type: " << (*ii)->node_->getKindName()<< std::endl;
  ////
  ////    }
  //  }
}

void NestPartitioner::generateApplicationCode(std::string profilePath,
                                              std::string partitionPlanFile,
                                              int profileMode,
                                              int partitionExe) {
  //  std::cout << "--------- generateApplicationCode -----------" << std::endl;

  std::string inputPartitionName;
  if (partitionPlanFile.find('/') != std::string::npos) {
    inputPartitionName =
        partitionPlanFile.substr(partitionPlanFile.find_last_of("/") + 1);
    inputPartitionName =
        inputPartitionName.substr(0, inputPartitionName.size() - 5);
  }

  appGen_.setProfileMode(profileMode);
  appGen_.setPartitionExeMode(partitionExe);
  appGen_.setInputPartitionName(inputPartitionName);
  appGen_.setProfilePath(profilePath);
  appGen_.setPartitionPlanFile(partitionPlanFile);

  Function *function = module_->getFunctions().front();
  BFSLevel bfs = getBFSLevel(function);
  size_t level = bfs.size();
  int insertCount = 0;
  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node *node = bfs[i][j];
      std::string key = appGen_.getProfileKey(node);
      if (profileKeySet_.find(key) == profileKeySet_.end()) {
        appGen_.setKeyName(key, insertCount++);
      } else {
        appGen_.setKeyName("blank", insertCount++);
      }
    }
  }
}

void NestPartitioner::generatePlanForProfiling(Function *function,
                                               std::string profilePath,
                                               int profileMode) {

  BFSLevel bfs = getBFSLevel(function);

  std::vector<std::vector<CostNode>> cnodeBfs;
  makeCostNode(function, &cnodeBfs);
  std::vector<NodeGroup *> branchList;

  analyzeDAG(&cnodeBfs, bfs, &branchList);

  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);

  std::cout
      << "<Partition-plan generation for the performance profiling of each PU>"
      << std::endl;
  for (auto device : deviceInfo_) {
    std::cout << "  => Partition-plan generation for " << device.backendName
              << std::endl;
    // separate each node
    std::string planName = profilePath + "/initialSingleNodePartition" +
                           device.backendName +
                           std::to_string(device.deviceID) + ".yaml";
    outPartitionPlanForSingleNode(function, &branchList, device, planName);
    std::cout << "    . Partition plan for each NN operation: " << planName
              << std::endl;
    std::string profileName = profilePath + "/data/fusedProfile" +
                              device.backendName +
                              std::to_string(device.deviceID) + ".yaml";

    // fusing depth: 1
    outPerformProfileForFusedNode(&branchList, profileName, device);
    planName = profilePath + "/initialFusedNodePartition" + device.backendName +
               std::to_string(device.deviceID) + ".yaml";
    outPartitionPlanForFusion(function->getName(), &branchList, device,
                              planName, false);

    std::cout << "    . Partition plan for fused NN operation: " << planName
              << std::endl;
  }

  for (auto branch : partitionList_)
    delete branch;

  partitionList_.clear();
}

//#include <filesystem>
//
//void NestPartitioner::generatePlanForRelay(Function *function,
//                                           std::string profilePath,
//                                           std::string partitionPlanFile,
//                                           int profileMode) {
//
//  BFSLevel bfs = getBFSLevel(function);
//  std::vector<std::vector<CostNode>> cnodeBfs;
//  std::vector<NodeGroup *> branchList;
//
//  makeCostNode(function, &cnodeBfs);
//  analyzeDAG(&cnodeBfs, bfs, &branchList);
//  allocateRelayOps(&branchList);
//  partitionBranches(&branchList);
//  outPartitionPlan(function->getName().str(),
//                   profilePath + "/RelayOPsPlan.yaml", false);
//
//  //    std::cout << "[generatePlanForRelay] Total cost = " <<
//  //    getCost(&branchList) << std::endl;
//
//  for (auto branch : partitionList_)
//    delete branch;
//
//  partitionList_.clear();
//}

void NestPartitioner::generatePlansForDevices(Function *function,
                                            std::string profilePath,
                                            int profileMode, int VTANum) {
    BFSLevel bfs = getBFSLevel(function);

    std::vector<std::vector<CostNode>> cnodeBfs;
    makeCostNode(function, &cnodeBfs);
    std::vector<NodeGroup *> branchList;

    analyzeDAG(&cnodeBfs, bfs, &branchList);

    std::vector<Backend *> backends;
    genBackendMap(backendMap_, backendHolder_, backends);

    std::cout
            << "<Partition-plan generation for each PU according to their supported operators.>"
            << std::endl;
    for (auto device : deviceInfo_) {
        std::string planName = profilePath + "/" +
                               device.backendName +
                               std::to_string(device.deviceID) + "OPsPlan.yaml";
      std::cout << "  => Partition-plan generation for " << device.backendName
                << ": " << planName << std::endl;

        outPartitionPlanForFusion(function->getName(), &branchList, device, planName, true);
    }

    for (auto branch : partitionList_)
        delete branch;

    partitionList_.clear();

}

void NestPartitioner::generatePlanForVTAOps(Function *function,
                                            std::string profilePath,
                                            int profileMode, int VTANum) {

  BFSLevel bfs = getBFSLevel(function);
  std::vector<std::vector<CostNode>> cnodeBfs;
  std::vector<NodeGroup *> branchList;

  makeCostNode(function, &cnodeBfs);
  analyzeDAG(&cnodeBfs, bfs, &branchList);
  allocateVTAOps(&branchList);

  std::string planName = "";

  if (VTANum > 1) {
    findParallelDeployBranchesForMultiVTA(&branchList, 0, VTANum);
    planName = "/VTAOPsPlan-multiVTA.yaml";
  } else {
    planName = "/VTAOPsPlan.yaml";
  }

  partitionBranches(&branchList, true);

  if (VTANum > 1) {
    outPartitionPlanMultiVTA(function->getName().str(), profilePath + planName,
                             true);
  } else {
    outPartitionPlan(function->getName().str(), profilePath + planName, false);
  }
  //    std::cout << "[generatePlanForVTAOps] Total cost = " <<
  //    getCost(&branchList) << std::endl;

  for (auto branch : partitionList_)
    delete branch;

  partitionList_.clear();
}

#include <boost/filesystem.hpp>
void NestPartitioner::generateMinCostPlan(Function *function,
                                          std::string profilePath,
                                          int profileMod0e, int cpuNum,
                                          int VTANum) {

  std::vector<std::vector<CostNode>> cnodeBfs;
  BFSLevel bfs = getBFSLevel(function);
  std::vector<NodeGroup *> branchList;

//  std::string profileConfigFile = profilePath;
//  if (profilePath.empty()) {
//    profileConfigFile = "./profileConfigs.yaml";
//  } else {
//    profileConfigFile = profileConfigFile + "/profileConfigs.yaml";
//  }
//  loadFuseOperatorsConfig(profileConfigFile);
  makeCostNode(function, &cnodeBfs);

  if (profilePath.empty())
    profilePath = ".";
  std::string profileDataDir = profilePath + "/test";
  if (!boost::filesystem::exists(profileDataDir)) {
    boost::filesystem::create_directories(profilePath + "/test");
    std::cout << "There is no profiling files. Profiling folder => "
              << profileDataDir << std::endl;
    exit(0);
  }

  loadPerformProfileInfo(profileDataDir);
  analyzeDAG(&cnodeBfs, bfs, &branchList);
  allocMinBranchCost(&branchList);

  std::string planName = "";

  if (VTANum > 1) {
    findParallelDeployBranchesForMultiVTA(&branchList, cpuNum, VTANum);
    planName = "/NESTMinCostOptimalPlan-multiVTA.yaml";
  } else {
    planName = "/NESTMinCostOptimalPlan.yaml";
  }

  partitionBranches(&branchList, true);

  if (VTANum > 1) {
    outPartitionPlanMultiVTA(function->getName().str(), profilePath + planName,
                             true);
  } else {
    outPartitionPlan(function->getName().str(), profilePath + planName, true);
  }
  //    std::cout << "[generateMinCostPlan] Total cost = " <<
  //    getCost(&branchList) << std::endl;

  for (auto branch : partitionList_)
    delete branch;

  partitionList_.clear();
}

void NestPartitioner::generateMinCostPlanCPUVTA(Function *function,
                                                std::string profilePath,
                                                int profileMod0e, int cpuNum,
                                                int VTANum) {

  std::vector<std::vector<CostNode>> cnodeBfs;
  BFSLevel bfs = getBFSLevel(function);
  std::vector<NodeGroup *> branchList;

//  std::string profileConfigFile = profilePath;
//  if (profilePath.empty()) {
//    profileConfigFile = "./profileConfigs.yaml";
//  } else {
//    profileConfigFile = profileConfigFile + "/profileConfigs.yaml";
//  }
//  loadFuseOperatorsConfig(profileConfigFile);
  makeCostNode(function, &cnodeBfs);

  if (profilePath.empty())
    profilePath = ".";
  std::string profileDataDir = profilePath + "/test";
  if (!boost::filesystem::exists(profileDataDir)) {
    boost::filesystem::create_directories(profilePath + "/test");
    std::cout << "There is no profiling files. Profiling folder => "
              << profileDataDir << std::endl;
    exit(0);
  }

  loadPerformProfileInfo(profileDataDir);

  analyzeDAG(&cnodeBfs, bfs, &branchList);
  allocMinBranchCost(&branchList);

  std::string planName = "/NESTMinCostOptimalPlan.yaml";

  if (VTANum == 1 && cpuNum == 1) {
    findParallelDeployBranchesForCPUVTA(&branchList, cpuNum, VTANum);
    planName = "/NESTMinCostOptimalPlan-parallel-etrij.yaml";
  }

  partitionBranchesCPUVTA(&branchList, true);
  outPartitionPlanCPUVTA(function->getName().str(), profilePath + planName,
                         true);

  //    std::cout << "[generateMinCostPlan] Total cost = " <<
  //    getCoprofileDataDirst(&branchList) << std::endl;

  for (auto branch : partitionList_)
    delete branch;

  partitionList_.clear();

}

Expected<DAGListTy> NestPartitioner::generatePartitionCode(
    CompilationContext &cctx, std::string profilePath,
    std::string partitionPlanFile, int profileMode, int partitionExe) {

  // load partition plan
  runtime::PartitionConfig partitionConfig;
  cctx.partitionConfig = &partitionConfig;
  loadPartitionPlan(cctx.partitionConfig, partitionPlanFile);

  // generate main.cpp
  generateApplicationCode(profilePath, partitionPlanFile, profileMode,
                          partitionExe);

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
}

Expected<DAGListTy> NestPartitioner::partition(
    CompilationContext &cctx, size_t exeType, std::string profilePath,
    std::string partitionPlanFile, int profileMode, int partitionExe,
    std::map<std::string, int> *puIdxMap) {
  //  std::cout << "--------- partitionForNest exeType: " << exeType <<
  //  "-----------" << std::endl;
  puIdxMap_ = puIdxMap;
  Function *function = module_->getFunctions().front();

  if(exeType == 1) {
    //generate partition plans without profiling data

    //profiling code
    generatePlanForProfiling(function, profilePath, profileMode);

    //Partition plans for each PU in a device configuration file
    generatePlansForDevices(function, profilePath, profileMode, 1);

    exit(0);
  } else if (exeType == 2){
    //VTA-style backend allocation (4 EVTAs)
    generatePlanForVTAOps(function, profilePath, profileMode, 4);
    exit(0);

  }else if (exeType == 3) { //CPU-single-thread + Multi-VTA, parallel execution on 4 VTAs

    generatePlanForVTAOps(function, profilePath, profileMode, 1);
    exit(0);

  }else if (exeType == 4) { //CPU-single-thread + Multi-VTA, parallel execution on 4 VTAs

    generateMinCostPlan(function, profilePath, profileMode, 0, 4);
    exit(0);

  }else if (exeType == 5) { //CPU-single-thread + Multi-VTA, parallel execution on 4 VTAs

    generateMinCostPlan(function, profilePath, profileMode, 0, 4);
    exit(0);

  } else if (exeType == 6) { //CPU-single-thread + single-VTA, parallel execution on 1 CPU and 1 VTA (etri journal)

    generateMinCostPlanCPUVTA(function, profilePath, profileMode, 1, 1);
    exit(0);
  }


  if (exeType == 0) { // compile
    return generatePartitionCode(cctx, profilePath, partitionPlanFile,
                                 profileMode, partitionExe);
  } else {
    DAGListTy partitions;
    return std::move(partitions);
  }
}
