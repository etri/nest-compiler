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

bool comparePartitionOrder(std::pair<std::string, Function *> e1, std::pair<std::string, Function *> e2) {
    size_t pid1 = std::stoi(e1.first.substr(1, e1.first.length()));
    size_t pid2 = std::stoi(e2.first.substr(1, e2.first.length()));

    return (pid1 < pid2);
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
    std::map<std::string, Function *> funcMap;
    std::unordered_set<size_t> unused;
    std::map<size_t, NodesSet> nodesSetMap;

    // Create partitions based on the given number and names.
    for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
        std::string pname = std::string(partitionConfig.partitionNames[i].c_str());
        size_t partitionID = std::stoi(pname.substr(1, pname.length()));

        Function *newF = module_->createFunction(partitionConfig.partitionNames[i]);
        funcMap.insert(std::make_pair(partitionConfig.partitionNames[i], newF));
        partitionMap.createPartition(newF, partitionConfig.backendNames[i]);
        unused.insert(partitionID);

        nodesSetMap.insert(std::make_pair(partitionID, NodesSet()));
    }

    // Map the nodes the partitions.
    std::vector<Node *> unMapped;
    for (auto &node : F->getNodes()) {
        auto iter = partitionConfig.nodeToPartition.find(node.getName());
        if (iter == partitionConfig.nodeToPartition.end()) {
            // If a node in F is not in the node to partition mapping, put it into
            // unMaped list.
            unMapped.push_back(&node);
        } else {
            size_t partitionID = iter->second;
//            DCHECK(partitionID < partitionConfig.numOfPartitions)
//                            << "Invalid partition id :" << partitionID;
            partitionMap.add(&node, funcMap["p"+std::to_string(partitionID)]);
            unused.erase(partitionID);
            nodesSetMap[partitionID].insert(&node);
        }
    }

    // If there is unused partition and unmapped nodes, map those nodes to the
    // unused partition.
    if (unMapped.size()) {
        DCHECK(unused.size() == 1) << "There must be exactly 1 unused partition.";
        auto partitionID = *(unused.begin());
        for (auto &node : unMapped) {
            partitionMap.add(node, funcMap["p" + std::to_string(partitionID)]);
            nodesSetMap[partitionID].insert(node);
        }
    }

    // Set backend hints if they exist
    if (partitionConfig.backendHints.size()) {
        for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
            std::string pname = std::string(partitionConfig.partitionNames[i].c_str());
            auto func = funcMap[pname];

            partitionMap.setBackendHints(func, partitionConfig.backendHints[i]);
        }
    }

    // If logical device assignments are provided use them otherwise assign them.
    if (partitionConfig.logicalIDs.size()) {
        DCHECK(partitionConfig.numOfPartitions ==
               partitionConfig.logicalIDs.size());
        for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
            std::string pname = std::string(partitionConfig.partitionNames[i].c_str());
            auto func = funcMap[pname];

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
        auto func = funcMap[partitionConfig.partitionNames[replicationAssignment.first]];
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

    if (appGen_.getProfileMode() == 0) {
        std::cout << "Generating linking code (main.cpp).." << std::endl;
    } else if (appGen_.getProfileMode() == 1) {
        std::cout << "Generating profiling code (main.cpp).." << std::endl;
    }

    appGen_.setFuncMap(&funcMap);
    appGen_.generateCodeFromModels(partitionConfig,
                                 appGen_.getInputPartitionName());

    if (appGen_.getProfileMode() == 0) {
        std::cout << "Linking-code generation is done." << std::endl << std::endl;
    } else if (appGen_.getProfileMode() == 1) {
        std::cout << "Profiling-code generation is done." << std::endl << std::endl;
    }

    // Verify the function.
    std::cout << "Optimizing Graph-IR for CPU and VTA.." << std::endl;

    for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
        std::string pname = std::string(partitionConfig.partitionNames[i].c_str());
        auto func = funcMap[pname];
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

  // Map the nodes the partitions.
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
//    std::cout << "partition name = " << partitionConfig->partitionNames[i].c_str() << std::endl;
//    std::cout << "puIdx = " << puIdx << std::endl;
  }

  std::vector<std::string> results;
  boost::split(results, pmap.find("nodeToPartition")->second,
               [](char c) { return c == ':'; });
  for (auto &s : results) {
    std::vector<std::string> nodesStr;
    boost::split(nodesStr, s, [](char c) { return c == '-'; });

    partitionConfig->nodeToPartition[nodesStr[0]] = atoi(nodesStr[1].c_str());
//     std::cout << "node str = " <<  nodesStr[0] << std::endl;
//     std::cout << "node partition = " << nodesStr[1].c_str() << std::endl;
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

bool sortNodePartitionNames(std::string p1, std::string p2) {
//    std::cout << "------- sortNodePartitionNames ------ " << std::endl;

    std::vector<std::string> p1Str, p2Str;
    boost::split(p1Str, p1, boost::is_any_of("-"));
    boost::split(p2Str, p2, boost::is_any_of("-"));

    std::cout << "p1 = " << p1 << std::endl;
    std::cout << "p2 = " << p2 << std::endl;

    return (std::stoi(p1Str[1]) <= std::stoi(p2Str[1]));
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
//    std::vector<std::string> nodeStrList;
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
          if(isSupportedOpType(cnode->node_->getKind(), defaultBackend_, cnode)) {
            curBackendName = defaultBackend_;
          } else {
            curBackendName = "CPU";
          }
          curLogicalID = std::to_string(getDeviceInfoForBackend(curBackendName).deviceID);
        }

        bool isFused = isUserDefinedFusable(node, deviceInfo);
        if (isFused) {
          curBackendName = prevBackendName;
        }

        if(mergePartition) {
          if(curBackendName.compare(prevBackendName)) {
            mergedPartitionCount++;
            backendNameStr += (curBackendName + + ":");
            logicalIDStr += (curLogicalID+ ":");
            partitionNameStr += ("p" + std::to_string(mergedPartitionCount-1) + ":");

          }
          nodeToPartitionStr += ((node->getName().str() + "-" + std::to_string(mergedPartitionCount-1) + "-" + curBackendName));

        } else {
          if(!isFused) {
            idx++;
            backendNameStr += (curBackendName + ":");
            logicalIDStr +=
                    std::to_string(getDeviceInfoForBackend(curBackendName).deviceID) +
                    ":";
            partitionNameStr += ("p" + std::to_string(idx-1) + ":");
          }
          nodeToPartitionStr +=
                  (node->getName().str() + "-" + std::to_string(idx - 1) + "-" + curBackendName);
        }
        nodeToPartitionStr += "\n  :";
        prevBackendName = curBackendName;
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

        std::string backend;
        if (!isOffloading) {
          backend = deviceInfo.backendName;
          backendNameStr += backend;
          logicalIDStr += std::to_string(deviceInfo.deviceID);
        } else { // default
            if(isSupportedOpType(cnode->node_->getKind(), defaultBackend_, cnode)) {
                backend = defaultBackend_;
            } else {
                backend = "CPU";
            }
            backendNameStr += backend;
            logicalIDStr +=
                  std::to_string(getDeviceInfoForBackend(backend).deviceID);
        }

        nodeToPartitionStr +=
            (node->getName().str() + "-" + std::to_string(idx) + "-" + backend);

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


int getParallelVTAPartitions(NodeGroup *partition,
                             std::vector<NodeGroup *> *partitionList,
                             std::vector<NodeGroup *> *parallelPartitions) {
//      std::cout << "----------- getParallelVTAPartitions()---------------" << std::endl;
//  std::cout << "partition id: " << partition->ID_ << std::endl;
//  std::cout << "partition level: " << partition->level_ << std::endl;
//  std::cout << "partition branchID: " << partition->branchID_ << std::endl;
//  std::cout << "partition backend: " << partition->backendName_ << std::endl;

  std::vector<int> processedBranches;
  for (auto i = partitionList->begin(); i != partitionList->end(); i++) {

    if(((*i)->level_ == partition->level_ && (*i)->branchID_ != partition->branchID_) && !(*i)->backendName_.compare("VTA") &&
            (std::find(processedBranches.begin(), processedBranches.end(), (*i)->branchID_) == processedBranches.end())){
      processedBranches.push_back((*i)->branchID_);
      parallelPartitions->push_back(*i);
    }
  }
  return parallelPartitions->size();
}

void scheduleParallelVTAPartitions(std::vector<NodeGroup *> *parallelVTAPartitions, std::map<std::string, std::vector<NodeGroup *>> *scheduledParallelVTAPartitions, int VTANum) {
//    std::cout << "-- scheduleParallelVTAPartitions --" << std::endl;
    for (auto partition: *parallelVTAPartitions) {
        (*scheduledParallelVTAPartitions)[partition->backendName_].push_back(partition);
    }
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
  //    std::cout << "----------- outPartitionPlanCPUVTA ---------------" <<
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

//    uint convNum = 0;
//    uint vtaConvNum = 0;
//    uint vtaReluNum = 0;

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

      std::vector<NodeGroup *> parallelBranches;
      if (isParallel && !partition->partOfBranch_) {
        if (addedPartitions.find(partition->ID_) == addedPartitions.end()) {
          int pn = getBranchesWithSameLevel(&partitionList_, &parallelBranches,
                                            partition->level_);
          if (pn > 1 && parallelBranches.at(0)->parallelDeploy_) {

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
    std::cout << "--------- # partition: " << partitionList_.size() + 1
              << "-----------" << std::endl;

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

//    std::cout << "convNum: " << convNum << " vtaConvNum: " << vtaConvNum
//              << " vtaReluNum: " << vtaReluNum
//              << " parallelPartitionNum: " << parallelPartitionGroups
//              << std::endl;
  }
}

bool compareID(NodeGroup* p1, NodeGroup* p2) {
    return (p1->ID_ <= p2->ID_);
}

bool compareMemUsage(NodeGroup* p1, NodeGroup* p2) {
  return (p1->memoryUsage_ >= p2->memoryUsage_);
}

int getMinMemVTAIndex(long vtaMem[], int vtaNum) {
  int minMem = vtaMem[0], idx = 0;

  for(int i = 1; i < vtaNum; i++) {
    if(minMem > vtaMem[i]) {
      minMem = vtaMem[i];
      idx = i;
    }
  }
  return idx;
}

void NestPartitioner::matchPartitionToVTA(std::vector<NodeGroup*>* parallelVTAPartitions, int vtaNum) {
//  std::cout << "----------- matchPartitionToVTA ---------------" << std::endl;
  for(auto partition : *parallelVTAPartitions) {
    for(auto node: partition->nodeList_) { //memory usage of each partition
      auto mem = getNodeMemUsage(node->node_);
      partition->memoryUsage_ += mem;
    }
  }
  sort(parallelVTAPartitions->begin(), parallelVTAPartitions->end(), compareMemUsage);

  long vtaMem[vtaNum];
  for(int i = 0; i < vtaNum; i++) {
    vtaMem[i] = 0;
  }

  if((*parallelVTAPartitions).size() > 1 && vtaNum > 1) { //multiple parallel partitions, multivta
    for(auto partition : *parallelVTAPartitions) {
      int vIdx = getMinMemVTAIndex(vtaMem, vtaNum);
      vtaMem[vIdx] += partition->memoryUsage_;
      partition->backendName_ = partition->backendName_ + "-" + std::to_string(vIdx);
    }
  }
}

void NestPartitioner::mergePartitions(std::vector<NodeGroup *>* mergePartitionList) {

    sort(mergePartitionList->begin(), mergePartitionList->end(), compareID);

    for (int i = 1; i < mergePartitionList->size(); i++) {
        mergePartitionList->at(i)->ID_ = mergePartitionList->at(0)->ID_;
    }
}

void NestPartitioner::outPartitionPlanMultiVTA(std::string funcName,
                                               std::string filename,
                                               int VTANum, bool isParallel) {
  //    std::cout << "----------- outPartitionPlanMultiVTA ---------------" <<
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
    std::set<NodeGroup*> addedPartitions;
    std::map<int, int> mergedPartitionMap;

//    uint convNum = 0;
//    uint vtaConvNum = 0;
//    uint vtaReluNum = 0;

    std::vector<std::string>nodeStrList;
    for (auto partition : partitionList_) {
//      std::cout << "partition ID: " << partition->ID_ <<
//      std::endl; std::cout << "partition backend: " <<
//      partition->backendName_ << std::endl;

      std::string deviceName = partition->backendName_;
      if (partition->backendName_.rfind("VTA", 0) == 0) {
        deviceName = "VTA";
      }

      std::vector<NodeGroup *> parallelVTAPartitions;
      std::map<std::string, std::vector<NodeGroup *>> scheduledParallelVTAPartitions;
      std::vector<NodeGroup*> sameVTAPartitions[VTANum];
      for(int i = 0; i < VTANum; i++) {
        scheduledParallelVTAPartitions.insert(std::make_pair("VTA-" + std::to_string(i), sameVTAPartitions[i]));
      }

      if (isParallel) {
        if(!deviceName.compare("VTA")) {
          if (addedPartitions.find(partition) == addedPartitions.end()) {

            parallelVTAPartitions.push_back(partition);
            getParallelVTAPartitions(partition, &partitionList_, &parallelVTAPartitions);
            matchPartitionToVTA(&parallelVTAPartitions, VTANum);

            if (parallelVTAPartitions.size() > 1 && VTANum > 1) {
              scheduleParallelVTAPartitions(&parallelVTAPartitions, &scheduledParallelVTAPartitions, VTANum);

              parallelPartitionStr = parallelPartitionStr + ("(");
              for (auto pp: scheduledParallelVTAPartitions) {

                for (int pcnt = 0; pcnt < pp.second.size(); pcnt++) {
                  auto sp = pp.second.at(pcnt);
                  if(pcnt == 0) {
                    parallelPartitionStr = parallelPartitionStr + ("p" + std::to_string(sp->ID_));
                  } else {
                      mergedPartitionMap.insert(std::make_pair(sp->ID_, pp.second.at(0)->ID_));
                  }
                }

                parallelPartitionStr = parallelPartitionStr + ("||");
                addedPartitions.insert(pp.second.begin(), pp.second.end());
              }

              parallelPartitionStr.pop_back();
              parallelPartitionStr.pop_back();

              parallelPartitionStr = parallelPartitionStr + "):";
            }
          }
        }
      }

      int partitionID = partition->ID_;
      if(mergedPartitionMap.find(partition->ID_) == mergedPartitionMap.end()) {
        DeviceInfo dinfo = getDeviceInfoForBackend(deviceName);
        logicalIDStr += (std::to_string(dinfo.deviceID) + ":");
        partitionNameStr += ("p" + std::to_string(partition->ID_) + ":");
        backendNameStr += (partition->backendName_ + ":");
      } else {
          partitionID = mergedPartitionMap[partition->ID_];
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

        nodeStrList.push_back((cnode->name_ + "-" + std::to_string(partitionID)) + "-" + partition->backendName_);
      }
    }
    if(mergedPartitionMap.size() > 0) {
        std::sort(nodeStrList.begin(), nodeStrList.end(), sortNodePartitionNames);
    }

    for(auto str: nodeStrList) {
      nodeToPartitionStr += (str + "\n  :");
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
            "numOfPartitions: " + std::to_string(partitionList_.size() + 1 - mergedPartitionMap.size()) + "\n");

//      std::cout << "--------- # partition: " << partitionList_.size() + 1 - mergedPartitionMap.size() << "-----------" << std::endl;

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

//    std::cout << "convNum: " << convNum << " vtaConvNum: " << vtaConvNum
//              << " vtaReluNum: " << vtaReluNum
//              << " parallelPartitionNum: " << parallelPartitionGroups
//              << std::endl;
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

    std::cout << "--------- # partition: " << partitionList_.size() + 1 << "-----------" << std::endl;

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

  for (auto ni = branch->nodeList_.begin(); ni != branch->nodeList_.end();
       ni++) {
    CostNode *cn = *ni;

    if (prevNode != nullptr &&
        cn->backendName_.compare(prevNode->backendName_)) {
      partition->backendName_ = prevNode->backendName_;
      partition->partOfBranch_ = true;

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
      partition->branchID_ = branch->ID_;
    } else {
      partition->backendName_ = cn->backendName_;
    }

    partition->nodeList_.push_back(cn);
    prevNode = cn;
  }

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
    if (isParallelDeploy && bn > 1) {
      // get parallel branches
      int npn = getNonParallelDeployBranches(&sameLevelBranches,
                                             &nonParallelBranches);
      int pn = getParallelDeployBranches(&sameLevelBranches, "VTA",
                                         &parallelBranches);

      if (pn > 1) {

        // calculate communication overhead for spliting multiple branches
        for (auto pb : parallelBranches) {
          lastPartition =
              partitionSingleBranch(&partitionList_, pb, &partitionID, nullptr);
          lastPartition = nullptr;
        }
      }
      if (npn > 0) {
        NodeGroup *nb = new NodeGroup;
        nb->level_ = branch->level_;
        nb->backendName_ = branch->backendName_;
        nb->branchID_ = branch->ID_;

        makeSingleList(&nonParallelBranches, nb, nullptr);
        lastPartition =
            partitionSingleBranch(&partitionList_, nb, &partitionID, nullptr);
      }

    } else if (bn == 1) {
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
}

void NestPartitioner::partitionBranchesCPUVTA(
    std::vector<NodeGroup *> *branchList, bool isParallelDeploy) {
  //    std::cout << "--------- partitionBranches -----------" << std::endl;

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
    if (isParallelDeploy && bn > 1) {
      // get parallel branches
      int npn = getNonParallelDeployBranches(&sameLevelBranches,
                                             &nonParallelBranches);
      int pn = getParallelDeployBranches(&sameLevelBranches, &parallelBranches);

      if (pn > 1) {

        // calculate communication overhead for spliting multiple branches
        for (auto pb : parallelBranches) {
          lastPartition =
              partitionSingleBranch(&partitionList_, pb, &partitionID, nullptr);
          lastPartition = nullptr;
        }
      }
      if (npn > 0) {
        NodeGroup *nb = new NodeGroup;
        nb->level_ = branch->level_;
        nb->backendName_ = branch->backendName_;
        nb->branchID_ = branch->ID_;

        makeSingleList(&nonParallelBranches, nb, nullptr);
        lastPartition =
            partitionSingleBranch(&partitionList_, nb, &partitionID, nullptr);
      }

    } else if (bn == 1) {
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
      cnode->minCostMap_[backend->getBackendName()] = exeCost;
    }

  } else {

    std::string compKey = appGen_.getProfileKey(pnode->node_) + "+" + curKey;

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
              exeCost + 2000;
          // 2000: penalty: function call overhead, cache conflict, ..

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
//  std::cout << "--------- allocMinBranchCost -----------" << std::endl;

  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);

  // calculate costs for nodes
  float totalCost = 0.0;
  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = (*i);

    int size = branch->nodeList_.size();
    for (int bi = 0; bi < size; bi++) {
      CostNode *cnode = branch->nodeList_.at(bi);
      CostNode *pnode = nullptr;

      if (bi > 0) {
        pnode = branch->nodeList_.at(bi - 1);
      }
      setCost(&backends, pnode, cnode);
      pnode = cnode;
    }
    // allocate backends requiring the minimum total cost
    CostNode *lastNode = branch->nodeList_.at(size - 1);
    float minCost = INFINITY;
    std::string minBackend = "";
    for (Backend *backend : backends) {
      std::string backendName = backend->getBackendName();
      float cost = lastNode->minCostMap_[backendName];

      if (cost <= minCost) {
        minCost = cost;
        minBackend = backendName;
      }
    }
    lastNode->backendName_ = minBackend;
    totalCost += lastNode->minCostMap_[minBackend];

    for (int bi = size - 2; bi >= 0; bi--) {
      CostNode *cnode = branch->nodeList_.at(bi);
      cnode->backendName_ = lastNode->prevMinBackendMap_[minBackend];
      minBackend = cnode->backendName_;
      lastNode = cnode;
    }
  }

//  std::cout << "Total Cost = " << totalCost << std::endl;
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
bool NestPartitioner::setBranchPU(NodeGroup *branch) {

  std::string backend = "";
  bool singlePU = true;
  for (int i = 0; i < branch->nodeList_.size(); i++) {
    CostNode *node = branch->nodeList_.at(i);
    if (!node->backendName_.compare("VTA")) {
//      branch->vtaCount_++;
        branch->containVTA_ = true;
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

  if (singlePU) {
    branch->backendName_ = backend;
  }
  branch->isSinglePU_ = singlePU;

  return singlePU;
}


void NestPartitioner::findParallelDeployBranchesForMultiVTA(
        std::vector<NodeGroup *> *branchList, int vtaNum) {
//  std::cout << "--------- [findParallelBranches] -----------" << std::endl;

  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    NodeGroup *branch = (*i);
    setBranchPU(branch);
  }

  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    if ((*i)->inBranchList_.size() > 1) { // parallel branches
      std::vector<NodeGroup *> pVTABranchList;
      for (auto bi = (*i)->inBranchList_.begin();
           bi != (*i)->inBranchList_.end(); bi++) {
        NodeGroup *branch = (*bi);

        if (!branch->backendName_.compare("VTA")) {
          pVTABranchList.push_back(branch);
        }
      }

      if (pVTABranchList.size() >= 2) {
        for (auto bi = pVTABranchList.begin(); bi != pVTABranchList.end(); bi++) {
          NodeGroup *branch = (*bi);
            if(branch->containVTA_ > 0) {
              branch->parallelDeploy_ = true;
            }
        }
        pVTABranchList.clear();
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
  }

  for (auto i = branchList->begin(); i != branchList->end(); i++) {
    if ((*i)->inBranchList_.size() > 1) { // parallel branches

      int vtaCnt = 0, cpuCnt = 0;
      std::vector<NodeGroup *> pBranchList;
      for (auto bi = (*i)->inBranchList_.begin();
           bi != (*i)->inBranchList_.end(); bi++) {
        NodeGroup *branch = (*bi);
        //                pBranchList.push_back(branch);
        if (!branch->backendName_.compare("VTA")) {
          vtaCnt++;
          pBranchList.push_back(branch);
        } else if (!branch->backendName_.compare("CPU") || !branch->backendName_.compare("Relay")) {
          cpuCnt++;
          pBranchList.push_back(branch);
        } else {
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
          if (vtaNum > 0) {
            if (!branch->backendName_.compare("VTA") && vtaCnt > 0) {
              branch->parallelDeploy_ = true;
              if (vtaNum == 1) {
                branch->backendName_ = "VTA";
              } else {
                branch->backendName_ = "VTA-" + std::to_string(vtaCnt - 1);
              }
              vtaCnt--;
            }
          }

          if (cpuNum > 0) {
            if (!branch->backendName_.compare("CPU") && cpuCnt > 0) {
              branch->parallelDeploy_ = true;
              branch->backendName_ = "CPU";
              cpuCnt--;
            }
          }
        }
        pBranchList.clear();
      }
    } else {
    }
  }
}

Expected<DAGListTy> NestPartitioner::partition(CompilationContext &cctx) {
  std::string dir = "/home/msyu/Dropbox/Project/development/configs/"
                    "partition_perform_profile";
  return partition(cctx, 3, "", dir + "/NESTOptimalPlan.yaml", 0, 0, nullptr, 1);
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

unsigned int branchID_ = 0;
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
    branch->ID_ = ++branchID_;
    addToBranchList(branchList, branch);
    if (prevBranch != nullptr) {
      prevBranch->inBranchList_.push_back(branch);
    }

    Node *curNode = parallelNodes.at(i);
    std::vector<Node *> pNodes = getParallelNodes(dag, curNode);

    if (pNodes.size() > 1) {
      getBranch(cnodebfs, dag, pNodes, curNode, branch, level + 1, ++branchID_,
                branchList);

    } else if (pNodes.size() == 1) {
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

      backtrack(cnodebfs, dag, curNode, branch, level, branchID_, branchList);
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
        getBranch(cnodebfs, dag, pNodes, curNode, branch, level + 1, ++branchID_,
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
    getBranch(cnodebfs, dag, pNodes, lastNode, branch, level + 1, branchID + 1,
              branchList);
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


void NestPartitioner::generateTVMPlan(Function *function,
                                              std::string profilePath,
                                              int profileMode, DeviceInfo device, int VTANum) {
    BFSLevel bfs = getBFSLevel(function);

    std::vector<std::vector<CostNode>> cnodeBfs;
    makeCostNode(function, &cnodeBfs);
    std::vector<NodeGroup *> branchList;

    analyzeDAG(&cnodeBfs, bfs, &branchList);

    std::vector<Backend *> backends;
    genBackendMap(backendMap_, backendHolder_, backends);

    defaultBackend_ = "Relay";
    std::string planName = profilePath + "/" +
                           device.backendName +
                           std::to_string(device.deviceID) + "OPsPlan-TVM.yaml";
    std::cout << "  => TVM partition-plan generation for " << device.backendName
              << ": " << planName << std::endl;

    outPartitionPlanForFusion(function->getName(), &branchList, device, planName, true);


    for (auto branch : partitionList_)
        delete branch;

    partitionList_.clear();

}

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
                             std::to_string(device.deviceID) + "OPsPlan"  + ".yaml";
    std::cout << "  => Partition-plan generation for " << device.backendName
              << ": " << planName << std::endl;

    outPartitionPlanForFusion(function->getName(), &branchList, device, planName, true);
  }

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
    findParallelDeployBranchesForMultiVTA(&branchList, VTANum);
    planName = "/NESTMinCostOptimalPlan-multiVTA"+ std::to_string(VTANum) + ".yaml";
  } else {
    planName = "/NESTMinCostOptimalPlan.yaml";
  }

  partitionBranches(&branchList, true);

  if (VTANum > 1) {
    outPartitionPlanMultiVTA(function->getName().str(), profilePath + planName, VTANum, true);
  } else {
    outPartitionPlan(function->getName().str(), profilePath + planName, true);
  }

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
    std::map<std::string, int> *puIdxMap, int evtaNum) {
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

    if (backendMap_.find("Relay") != backendMap_.end() && defaultBackend_.compare("Relay") && backendMap_.find("VTA") != backendMap_.end()) {
      generateTVMPlan(function, profilePath, profileMode, getDeviceInfoForBackend("VTA"), 1);
    }
//    generatePlanForVTAOps(function, profilePath, profileMode, 1);
    exit(0);
  } else if (exeType == 2){ //CPU-single-thread + Multi-VTA, parallel execution on 4 VTAs
    std::cout << "# EVTA: " << evtaNum << std::endl;
    generateMinCostPlan(function, profilePath, profileMode, 0, evtaNum);
    exit(0);
  }else if (exeType == 3) {
    std::cout << "Not supported exeType." << std::endl;
  }else if (exeType == 4) {
    std::cout << "Not supported exeType." << std::endl;
  }else if (exeType == 5) {
    std::cout << "Not supported exeType." << std::endl;
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
