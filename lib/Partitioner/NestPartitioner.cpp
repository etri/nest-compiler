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

    appGen_.setProfileSet(&fuseOperatorSet_, &profileKeySet_);
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

    if(appGen_.getProfileMode() == 0) {
        std::cout << "Generating linking code (main.cpp).." << std::endl;
    } else if (appGen_.getProfileMode() == 1) {
        std::cout << "Generating profiling code (main.cpp).." << std::endl;
    }

    appGen_.generateCodeFromModels(funcList.size(), funcList, partitionConfig, appGen_.getInputPartitionName());

    if(appGen_.getProfileMode() == 0) {
        std::cout << "Linking-code generation is done." << std::endl << std::endl;
    } else if (appGen_.getProfileMode() == 1) {
        std::cout << "Profiling-code generation is done." << std::endl << std::endl;
    }

    // Verify the function.
    std::cout << "Optimizing Graph-IR for CPU and VTA.." << std::endl;

    for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
        auto func = funcList[i];

        Backend* B = backendMap_[partitionConfig.backendNames[i]].backend;

        if (!optimized_) {
            if(B->getBackendName().compare("CPU") && B->getBackendName().compare("Relay")) {
                // Specific configurations.
                cctx.precisionConfig.quantMode = QuantizationMode::Quantize;
                cctx.precisionConfig.quantConfig.schema = quantization::Schema::SymmetricWithPower2Scale;
                llvm::StringRef filename = quantFileName_;
                // std::cout << "quant filename = " << filename.str() << std::endl;
                deserializeProfilingInfosFromYaml(
                        filename, cctx.precisionConfig.quantConfig.graphPreLowerHash, cctx.precisionConfig.quantConfig.infos);
                cctx.precisionConfig.quantConfig.checkGraphPreLowerHash = false;
            }
            else{
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

    for(int i = 0; i < partitionConfig->backendNames.size(); i++) {
        int puIdx = 0;
        if(partitionConfig->backendNames[i].find("-") != std::string::npos) {
            int tmpidx = partitionConfig->backendNames[i].find("-");
            puIdx = atoi(partitionConfig->backendNames[i].substr(tmpidx+1, partitionConfig->backendNames[i].size() - tmpidx).c_str());
            partitionConfig->backendNames[i] = partitionConfig->backendNames[i].substr(0, tmpidx);
        }
        puIdxMap_->insert({partitionConfig->partitionNames[i], puIdx});
    }

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
        for (auto &id : idsStr) {
            ids.push_back(atoi(id.c_str()));
        }
        partitionConfig->logicalIDs.push_back(ids);
    }

    results.clear();
    if(pmap.find("parallelPartitions") != pmap.end()) {
        boost::split(results, pmap.find("parallelPartitions")->second, [](char c) { return c == ':'; });
        for (auto &s: results) {
            //std::cout << s << std::endl;
            boost::erase_all(s, "(");
            boost::erase_all(s, ")");
            //std::cout << s << std::endl;
            boost::replace_all(s, "||", "|");

            std::vector<std::string> pgroupStr;
            boost::split(pgroupStr, s, [](char c) { return c == '|'; });

            std::set<std::string> pnameSet;
            for (auto &pname: pgroupStr) {
                pnameSet.insert(pname.c_str());
            }
            partitionConfig->parallelPartitions.push_back(pnameSet);
        }
    }

}

void NestPartitioner::outPerformProfileForEachNode(std::vector<NodeGroup*>* branchList, std::string filename, DeviceInfo device)
{
    std::cout << "== outPerformProfileForEachNode : " << device.backendName << ", filename: "  << filename << " == " << std::endl;
    std::vector<Backend *> backends;
    genBackendMap(backendMap_, backendHolder_, backends);

    std::ofstream profileFile(filename);
    if (profileFile.is_open()) {

        profileFile << "---" << std::endl;
        profileFile << "backendName: " << device.backendName << std::endl;

        for (auto partition : *branchList) {
            for (auto cnode : partition->nodeList_) {
//        std::cout << "name = " << node->getName().str() << std::endl;
                Node* node = cnode->node_;
                std::string key = appGen_.getProfileKey(node);
//        std::cout << "key = " << key << std::endl;
                std::pair<std::set<std::string>::iterator, bool> ret = profileKeySet_.insert(key);
                if(ret.second) {
                    if(backendMap_[device.backendName].supportedNodesKinds.size() == 0 ||
                       (backendMap_[device.backendName].supportedNodesKinds.size() > 0 && (backendMap_[device.backendName].supportedNodesKinds.find(node->getKind())
                                                                                           != backendMap_[device.backendName].supportedNodesKinds.end()))) {
                        profileFile << key << ": "
                                    << "1" << std::endl;
                    } else {
                        profileFile << key << ": " << "INFINITY" << std::endl;
                    }
                }
            }
        }

        profileFile << "..." << std::endl;
        profileFile.close();
    }
}

Node* getFirstInputOpNode(Node* node) {

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

CostNode* findNodeFromCNodeList(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node) {

    size_t level = (*cnodeBfs).size();
    for (int i = 0; i < level; i++) {
        for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {
            if(node == (&(*cnodeBfs)[i][j])->node_) {
                return &((*cnodeBfs)[i][j]);
            }
        }
    }
    return nullptr;
}

bool isFusable(Node* node) {
    if(node->getNumInputs() > 1) {
        size_t usedCnt = 0;
        for(int in = 0; in < node->getNumInputs(); in++) {
            Node* inNode = node->getNthInput(in);
            if((inNode->getKind() != Kinded::Kind::ConstantKind) &&
               inNode->getKind() != Kinded::Kind::PlaceholderKind) {
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
    if(fuseOperatorSet_.find(doubleFusedKindName) != fuseOperatorSet_.end()) {
        std::string profileStr = appGen_.getProfileKey(firstPrevNode) + "+" + appGen_.getProfileKey(node);
        return true;
    }

    Node *secondPrevNode = getFirstInputOpNode(firstPrevNode);
    if(secondPrevNode == nullptr || !isFusable(secondPrevNode))
        return false;

    std::string tripleFusedKindName = std::string(secondPrevNode->getKindName()) + "-" +
                                      firstPrevNode->getKindName() + "-" +
                                      node->getKindName();

    //std::cout << "tripleFusedKindName = " << tripleFusedKindName << std::endl;
    if(fuseOperatorSet_.find(tripleFusedKindName) != fuseOperatorSet_.end()) {
        return true;
    }
    return false;
}

void NestPartitioner::outPerformProfileForFusedNode(std::vector<NodeGroup*>* branchList, std::string filename, DeviceInfo device)
{
//    std::cout << "== outPerformProfileForFusedNode : " << device.backendName << " == " << std::endl;

    std::ofstream profileFile(filename);
    if (profileFile.is_open()) {

//        BFSLevel bfs = getBFSLevel(function);
//        size_t level = bfs.size();

        profileFile << "---" << std::endl;
        profileFile << "backendName: " << device.backendName << std::endl;

//        for (int i = level - 1; i >= 0; i--) {
//            for (size_t j = 0; j < bfs[i].size(); j++) {
        for (auto partition : *branchList) {
            for (auto cnode : partition->nodeList_) {
                Node *node = cnode->node_;

                Node *firstPrevNode;
                Node *secondPrevNode;
//                Node *node = bfs[i][j];
/*
                if(!isFusable(node))
                    continue;
*/
                firstPrevNode = getFirstInputOpNode(node);
                if(firstPrevNode == nullptr || !isFusable(firstPrevNode))
                    continue;

                std::string doubleFusedKindName = std::string(firstPrevNode->getKindName()) + "-" + node->getKindName();
                //std::cout << "doubleFusedKindName = " << doubleFusedKindName << std::endl;
                if(fuseOperatorSet_.find(doubleFusedKindName) != fuseOperatorSet_.end()) {
                    std::string profileStr = appGen_.getProfileKey(firstPrevNode) + "+" + appGen_.getProfileKey(node);
                    std::pair<std::set<std::string>::iterator, bool> ret = profileKeySet_.insert(profileStr);
                    if(ret.second) {
                        profileFile << profileStr << " : 1" << std::endl;
                    }
                }

                secondPrevNode = getFirstInputOpNode(firstPrevNode);
                if(secondPrevNode == nullptr || !isFusable(secondPrevNode))
                    continue;

                std::string tripleFusedKindName = std::string(secondPrevNode->getKindName()) + "-" +
                                                  firstPrevNode->getKindName() + "-" +
                                                  node->getKindName();

                if(fuseOperatorSet_.find(tripleFusedKindName) != fuseOperatorSet_.end()) {
                    std::string profileStr = appGen_.getProfileKey(secondPrevNode) + "+" + appGen_.getProfileKey(firstPrevNode) + "+" + appGen_.getProfileKey(node);
                    std::pair<std::set<std::string>::iterator, bool> ret = profileKeySet_.insert(profileStr);
                    if(ret.second) {
                        profileFile << profileStr << " : 1" << std::endl;
                    }
                }
            }
        }

        profileFile << "..." << std::endl;
        profileFile.close();
    }
}

void NestPartitioner::outPartitionPlanForSingleNode(Function *function, std::vector<NodeGroup*>* branchList, DeviceInfo deviceInfo, std::string filename)
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
//      BFSLevel bfs = getBFSLevel(function);
//      size_t level = bfs.size();
        size_t idx = 0;
//
//      for (int i = level - 1; i >= 0; i--) {
//        for (size_t j = 0; j < bfs[i].size(); j++) {
//
//          Node *node = bfs[i][j];
        for (auto partition : *branchList) {
            for (auto cnode : partition->nodeList_) {
                Node *node = cnode->node_;

                if (node->getKind() == Kinded::Kind::SaveNodeKind)
                    continue;

                partitionNameStr += ("p" + std::to_string(idx));

                if(backendMap_[deviceInfo.backendName].supportedNodesKinds.size() > 0 &&
                   (backendMap_[deviceInfo.backendName].supportedNodesKinds.find(node->getKind()) != backendMap_[deviceInfo.backendName].supportedNodesKinds.end())) {
                    backendNameStr += (deviceInfo.backendName);
                    logicalIDStr += std::to_string(deviceInfo.deviceID);
                } else {
                    backendNameStr += "CPU";
                    logicalIDStr += "0";
                }

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
//  std::cout << "----------- [end] outPartitionPlanForSingleNode ---------------" << std::endl;
}


int NestPartitioner::getFusedPartitionCount(std::vector<NodeGroup*>* branchList)
{
    int cnt = 1;
//  BFSLevel bfs = getBFSLevel(function);
//  size_t level = bfs.size();
//
//  for (int i = level - 1; i >= 0; i--) {
//    for (size_t j = 0; j < bfs[i].size(); j++) {
//
//      Node *node = bfs[i][j];
    //    for (auto& node : function->getNodes()) {

    for (auto partition : *branchList) {
        for (auto cnode : partition->nodeList_) {
            Node *node = cnode->node_;

            if (node->getKind() == Kinded::Kind::SaveNodeKind)
                continue;

            if(isUserDefinedFusable(node))
                cnt--;

            cnt++;
        }
    }
    return (cnt);
}

void NestPartitioner::outPartitionPlanForFusion(std::string funcName, std::vector<NodeGroup*>* branchList, DeviceInfo deviceInfo, std::string filename)
{
//  std::cout << "== outPartitionPlanForFusedNode : " << filename << " == " << std::endl;

    std::ofstream planFile(filename);
    if (planFile.is_open()) {

        int partitionSize = getFusedPartitionCount(branchList);

        planFile << "---" << std::endl;
        planFile << "funcName: " << funcName << std::endl;
        planFile << "numOfPartitions: " << partitionSize << std::endl;

        std::string backendNameStr;
        std::string partitionNameStr;
        std::string nodeToPartitionStr;
        std::string logicalIDStr;

        ///
        //std::cout << "partitionSize = " << partitionSize << std::endl;

//    BFSLevel bfs = getBFSLevel(function);
//    size_t level = bfs.size();
        size_t idx = 0;
        //size_t pcnt = 0;

//    for (int i = level - 1; i >= 0; i--) {
//      for (size_t j = 0; j < bfs[i].size(); j++) {
//
//        Node *node = bfs[i][j];
        for (auto partition : *branchList) {
            for (auto cnode : partition->nodeList_) {
                Node *node = cnode->node_;

                if (node->getKind() == Kinded::Kind::SaveNodeKind)
                    continue;

                bool isFused = isUserDefinedFusable(node);

                if(!isFused) {

                    nodeToPartitionStr +=
                            (node->getName().str() + "-" + std::to_string(idx));
                    nodeToPartitionStr += "\n  :";

                    partitionNameStr += ("p" + std::to_string(idx) + ":");

                    if (backendMap_[deviceInfo.backendName].supportedNodesKinds.size() > 0 &&
                        (backendMap_[deviceInfo.backendName].supportedNodesKinds.find(node->getKind()) !=
                         backendMap_[deviceInfo.backendName].supportedNodesKinds.end())) {
                        backendNameStr += (deviceInfo.backendName + ":");
                        logicalIDStr += (std::to_string(deviceInfo.deviceID) + ":");

                    } else {
                        backendNameStr += "CPU:";
                        logicalIDStr += "0:";
                    }
                    idx++;
                } else {
                    nodeToPartitionStr +=
                            (node->getName().str() + "-" + std::to_string(idx-1));
                    nodeToPartitionStr += "\n  :";
                }
            }
        }
        nodeToPartitionStr.pop_back();

        partitionNameStr += ("p" + std::to_string(partitionSize-1));

        backendNameStr += "CPU";
        logicalIDStr += "0";

        planFile << "backendNames: " << backendNameStr << std::endl;
        planFile << "partitionNames: " << partitionNameStr << std::endl;
        planFile << "nodeToPartition: " << nodeToPartitionStr << std::endl;
        planFile << "logicalIDs: " << logicalIDStr << std::endl;

        planFile << "..." << std::endl;
        planFile.close();

    }
//  std::cout << "----------- [end] outPartitionPlanForSingleNode ---------------" << std::endl;
}


int getBranchesWithSameLevel(std::vector<NodeGroup*>* branchList, std::vector<NodeGroup*>* result, int level) {
//    std::cout << "----------- getBranchesWithSameLevel level (" << level << ")---------------" << std::endl;
    for (auto i = branchList->begin(); i != branchList->end(); i++) {
//        std::cout << "branch id: " << (*i)->ID_ << std::endl;
//        std::cout << "branch level: " << (*i)->level_ << std::endl;
//        std::cout << "partOfBranch_: " << (*i)->partOfBranch_ << std::endl;

        if((*i)->level_ == level && !(*i)->partOfBranch_) {
            result->insert(result->end(), *i);
        }
    }
    return result->size();
}

void NestPartitioner::outPartitionPlan(std::string funcName, std::string filename, bool isParallel) {
    std::cout << "----------- outPartitionPlan ---------------" << std::endl;
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

        for (auto partition : nodeGroups_) {
//            std::cout << "partition ID: " << partition->ID_ << std::endl;
//            std::cout << "partition backend: " << partition->backendName_ << std::endl;

            std::string deviceName = partition->backendName_;
            if(partition->backendName_.rfind("VTA", 0) == 0) {
                deviceName = "VTA";
            }

            DeviceInfo dinfo = getDeviceInfoForBackend(deviceName);
            backendNameStr += (partition->backendName_ + ":");
            logicalIDStr += (std::to_string(dinfo.deviceID) + ":");
            partitionNameStr +=
                    ("p" + std::to_string(partition->ID_) + ":");

            //std::cout << "partition ID: " << partition->ID_ << " isParallelPartition_ "<< partition->isParallelPartition_<<std::endl;

            std::vector<NodeGroup*> parallelBranches;
            if(isParallel && !partition->partOfBranch_) {
                if(addedPartitions.find(partition->ID_) == addedPartitions.end()) {
                    int pn = getBranchesWithSameLevel(&nodeGroups_, &parallelBranches, partition->level_);
                    if (pn > 1 && parallelBranches.at(0)->isParallelBranch_) {

                        parallelPartitionGroups++;
                        parallelPartitions = parallelPartitions + ("(");
                        addedPartitions.insert(partition->ID_);
                        for (auto sameLevelPartition: parallelBranches) {
                            parallelPartitions = parallelPartitions + ("p" + std::to_string(sameLevelPartition->ID_) + "||");
                            addedPartitions.insert(sameLevelPartition->ID_);
                        }
                        parallelPartitions.pop_back();
                        parallelPartitions.pop_back();
                        parallelPartitions = parallelPartitions + "):";
                    }
                }
            }

            for (auto cnode : partition->nodeList_) {
//          std::cout << "node name: " << cnode->node_->getName().str() << std::endl;
//          std::cout << "node type: " << cnode->node_->getKindName() << std::endl;
//          std::cout << "partition ID: " << partition->ID_ << std::endl;
//
//                if(!std::string(cnode->node_->getKindName()).compare("Convolution")) {
//                    convNum++;
//                    if (!cnode->backendName_.compare("VTA"))
//                        vtaConvNum++;
//                }else if(!std::string(cnode->node_->getKindName()).compare("Relu")) {
//                    if (!cnode->backendName_.compare("VTA"))
//                        vtaReluNum++;
//                }
//
                nodeToPartitionStr +=
                        (cnode->name_ + "-" + std::to_string(partition->ID_));
                nodeToPartitionStr += "\n  :";
            }
        }
        nodeToPartitionStr = nodeToPartitionStr.substr(0, nodeToPartitionStr.length() - 4);

        partitionNameStr += ("p" + std::to_string(nodeGroups_.back()->ID_+1));
        backendNameStr += "CPU";
        logicalIDStr += "0";

        info.append("numOfPartitions: " + std::to_string(nodeGroups_.size() + 1) + "\n");

        planFile << info;
        planFile << "backendNames: " << backendNameStr << std::endl;
        planFile << "partitionNames: " << partitionNameStr << std::endl;
        planFile << "nodeToPartition: " << nodeToPartitionStr<< std::endl;
        planFile << "logicalIDs: " << logicalIDStr<< std::endl;

        if(parallelPartitions.length() > 0) {
            parallelPartitions.pop_back();
            planFile << "parallelPartitions: " << parallelPartitions << std::endl;
        }
        planFile << "...\n" << std::endl;

        planFile.close();

//        std::cout << "convNum: " << convNum << " vtaConvNum: "<< vtaConvNum << " vtaReluNum: " << vtaReluNum << " parallelPartitionNum: " << parallelPartitionGroups << std::endl;

    }

}

NodeGroup* partitionSingleBranch(std::vector<NodeGroup*>* partitionList, NodeGroup* branch, unsigned int *partitionID, NodeGroup* lastPartition) {
    //std::cout << "---- partitionSingleBranch ----" << std::endl;

    CostNode *prevNode = nullptr;
    NodeGroup* partition = nullptr;

    if(lastPartition != nullptr && lastPartition->nodeList_.size() > 0 && !lastPartition->backendName_.compare(branch->nodeList_.at(0)->backendName_)) {
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
    for (auto ni = branch->nodeList_.begin(); ni != branch->nodeList_.end(); ni++) {
        CostNode *cn = *ni;

        if (prevNode != nullptr && cn->backendName_.compare(prevNode->backendName_)) {
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

        if(branch->isParallelBranch_) {
            partition->backendName_ = branch->backendName_;
            partition->isParallelBranch_ = true;
        } else {
            partition->backendName_ = cn->backendName_;
        }

        partition->nodeList_.push_back(cn);
        prevNode = cn;
    }

//  std::cout << "partition nodes: " << partition->nodeList_.size() << std::endl;
//  std::cout << "partition backend: " << partition->backendName_ << ", ID: " << partition->ID_ << std::endl;
    return partition;
}

NodeGroup* findParallelBranch(std::vector<NodeGroup*>* branchList) {

    for (auto i = branchList->begin(); i != branchList->end(); i++) {
        NodeGroup *branch = *i;
        if(branch->isParallelBranch_)
            return branch;
    }
    return nullptr;
}

void makeSingleList(std::vector<NodeGroup*>* branchList, NodeGroup* singleBranch, NodeGroup* exception) {

    for (auto i = branchList->begin(); i != branchList->end(); i++) {
        NodeGroup *branch = *i;

        if (branch != nullptr && branch == exception)
            continue;

        for (auto ii = branch->nodeList_.begin(); ii != branch->nodeList_.end(); ii++) {
            singleBranch->nodeList_.push_back((*ii));
        }
    }
}


void NestPartitioner::partitionBranches(std::vector<NodeGroup*>* branchList, bool isParallel) {
//  std::cout << "--------- partitionBranches -----------" << std::endl;

    unsigned int partitionID = 0;
    NodeGroup* lastPartition = nullptr;

    std::set<int> processedLevels;
    for (auto i = branchList->begin(); i != branchList->end(); i++) {
        NodeGroup* branch = *i;
        if(processedLevels.find(branch->level_) != processedLevels.end())
            continue;

        std::vector<NodeGroup*> parallelBranches;
        int bn = getBranchesWithSameLevel(branchList, &parallelBranches, branch->level_);
        processedLevels.insert(branch->level_);
//        std::cout << "bn: " << bn << " level = " << branch->level_ << std::endl;
        if(isParallel && bn > 1 ) {
//            std::cout << "(*i)->isParallelBranch_: " << ", " << (*i)->isParallelBranch_ << std::endl;
            if((*i)->isParallelBranch_) {
                for(auto pb: parallelBranches) {
                    std::cout << "partitionID " << partitionID  << std::endl;
                    lastPartition = partitionSingleBranch(&nodeGroups_, pb, &partitionID, nullptr);
                    lastPartition = nullptr;
                }
            } else {
                NodeGroup* nb = new NodeGroup;
                nb->level_ = branch->level_;
                nb->backendName_ = branch->backendName_;
                nb->branchID_ = branch->ID_;

                makeSingleList(&parallelBranches, nb, nullptr);
                lastPartition = partitionSingleBranch(&nodeGroups_, nb, &partitionID, nullptr);
            }

        } else if(bn == 1) {
            //std::cout << "4 branch --->" << partitionID << std::endl;
            lastPartition = partitionSingleBranch(&nodeGroups_, branch, &partitionID, lastPartition);
            //std::cout << "backend: " << lastPartition->backendName_ << std::endl;
        } else {
            std::cout << "No branch in branchList." << std::endl;
            continue;
        }

        //      std::cout << "lastPartition size = " <<lastPartition->nodeList_.size() << std::endl;
    }

//  for (auto partition : nodeGroups_) {
//    std::cout << "partition backend: " << partition->backendName_ << std::endl;
//  }
    std::cout << "--------- # partition: " << nodeGroups_.size() << "-----------" << std::endl;

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

            std::map<std::string, float> puvalues;
            backendProfileMap_.insert(std::make_pair(backendName, puvalues));

            for(auto it = profileMap.begin();it != profileMap.end();it++) {
                if(!it->first.compare("backendName")) continue;

                std::string costValueStr = profileMap[it->first].c_str();
                if(!costValueStr.compare("INFINITY")) {
                    puvalues.insert(std::make_pair(it->first.c_str(), INFINITY));
                } else {
                    puvalues.insert(std::make_pair(it->first.c_str(), atof(costValueStr.c_str())));
                }
            }
            backendProfileMap_[backendName].insert(puvalues.begin(), puvalues.end());
        }
    }
    closedir(dir);
}

float NestPartitioner::getCostOfSingleNode(CostNode *cnode, std::string backendName)
{
////  std::cout << "--------- getCostOfSingleNode -----------" << std::endl;
    std::string key = appGen_.getProfileKey(cnode->node_);

    std::vector<std::string> backendList(backendSet_.size());
    std::copy(backendSet_.begin(), backendSet_.end(), backendList.begin());

    backendProfileMap_[backendList[0]][key];

    if(backendMap_[backendName].nonSupportedNodesKinds.find(cnode->node_->getKind())
       != backendMap_[backendName].nonSupportedNodesKinds.end()) {
        return INFINITY;
    } else {
        float compCost = backendProfileMap_[backendName][key];
        return compCost;
    }
}


void NestPartitioner::getMinCostOfSingleNode(CostNode *cnode, std::vector<Backend *> backends)
{
    //std::cout << "--------- getMinCostOfSingleNode -----------" << std::endl;
    //std::cout << "node type: " << cnode->node_->getKindName() << std::endl;

    std::string key = appGen_.getProfileKey(cnode->node_);
    //std::cout << "key: " << key << std::endl;

    //auto iter = backendSet_.begin();
    //cnode->cost_ = backendProfileMap_[backends[0]->getBackendName()][key];
    //cnode->backendName_ = backends[0]->getBackendName();
    cnode->cost_ = INFINITY;

    for(int i = 0; i < backends.size(); i++) {
        std::string backendName = backends[i]->getBackendName();
        //std::cout << "backendName = " << backendName << std::endl;
        //non supported type
        if(backendMap_[backendName].nonSupportedNodesKinds.size() > 0 &&
           (backendMap_[backendName].nonSupportedNodesKinds.find(cnode->node_->getKind())
            != backendMap_[backendName].nonSupportedNodesKinds.end())) continue;

        if(backendMap_[backendName].supportedNodesKinds.size() > 0) {
            if(backendMap_[backendName].supportedNodesKinds.find(cnode->node_->getKind()) == backendMap_[backendName].supportedNodesKinds.end()) {
                continue;
            }
            if (cnode->node_->getKind() == Kinded::Kind::ConvolutionNodeKind && !isVTAConv((ConvolutionNode *) cnode->node_)) {
                continue;
            }
        }

        float compCost = backendProfileMap_[backendName][key];
//    std::cout << "compCost = " << compCost << std::endl;

        float cost = compCost;
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
    if(fuseOperatorSet_.find(compType) == fuseOperatorSet_.end()) {
        return;
    }

    CostNode tmpcnode[2];
    tmpcnode[0] = *prevCNode;
    tmpcnode[1] = *curCNode;

    std::string key[2];
    for(int i = 0; i < 2; i++){
        key[i] = appGen_.getProfileKey(tmpcnode[i].node_);
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

        if(backendProfileMap_[backendName].find(compKey) == backendProfileMap_[backendName].end())
            continue;

        float cost = backendProfileMap_[backendName][compKey];
        if(cost < originalCost && cost < prevCost) {
//          std::cout << "First node: " << prevCNode->name_ << ", " << prevCNode->backendName_ << std::endl;
//          std::cout << "Second node: " << curCNode->name_ << ", " << curCNode->backendName_ << std::endl;
//          std::cout << "Original cost: " << prevCNode->cost_ + curCNode->cost_ << std::endl;;

            prevCost = cost;
            prevCNode->cost_ = cost;
            prevCNode->backendName_ = backendName;

            curCNode->cost_ = 0.0;
            curCNode->backendName_ = backendName;
//          curCNode->isFused_ = false;

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
    if(fuseOperatorSet_.find(compType) == fuseOperatorSet_.end()) {
        return;
    }

    CostNode tmpcnode[3];
    tmpcnode[0] = *secondPrevCNode;
    tmpcnode[1] = *firstPrevCNode;
    tmpcnode[2] = *curCNode;

    std::string key[3];
    for(int i = 0; i < 3; i++){
        key[i] = appGen_.getProfileKey(tmpcnode[i].node_);
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

        if(backendProfileMap_[backendName].find(compKey) == backendProfileMap_[backendName].end())
            continue;
        float cost = backendProfileMap_[backendName][compKey];
        if(cost < originalCost && cost < prevCost) {
//          std::cout << "[Original cost] = " << secondPrevCNode->cost_ + firstPrevCNode->cost_ + curCNode->cost_ << std::endl;

            prevCost = cost;
            secondPrevCNode->cost_ = cost;
            secondPrevCNode->backendName_ = backendName;

            firstPrevCNode->cost_ = 0.0;
            firstPrevCNode->backendName_ = backendName;
//            firstPrevCNode->isFused_ = false;

            curCNode->cost_ = 0.0;
            curCNode->backendName_ = backendName;
//            curCNode->isFused_ = false;
//          std::cout << "[Changed cost] = " << secondPrevCNode->cost_ + firstPrevCNode->cost_ + curCNode->cost_ << std::endl;
        }
    }

    //std::cout << "changed prev backend = " << secondPrevCNode->backendName_ << std::endl;
    //std::cout << "changed cur backend = " << curCNode->backendName_ << std::endl;
    //std::cout << "fused cost prev = " << secondPrevCNode->totalCost_ << std::endl;
    //std::cout << "fused cost cur = " << curCNode->totalCost_ << std::endl;
    //std::cout << "\tChanged cost = " << secondPrevCNode->totalCost_ + firstPrevCNode->totalCost_ + curCNode->totalCost_ << std::endl;

}

void setChecked(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node)
{
    for (int i = 0; i < (*cnodeBfs).size(); i++) {
        for (size_t j = 0; j < (*cnodeBfs)[i].size(); j++) {
            CostNode *cnode = &(*cnodeBfs)[i][j];
            if (cnode->node_ == node) {
                cnode->isChecked_ = true;
            }
        }
    }
}

bool isChecked(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node)
{
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

CostNode* getCostNode(std::vector<std::vector<CostNode>>* cnodeBfs, Node* node)
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

void makeCostNode(Function* function, std::vector<std::vector<CostNode>>* cnodeBfs) {
    //std::cout << "--------- makeCostNode -----------" << std::endl;
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
        }
        cnodeBfs->push_back(cnodeGroup);
    }
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
                    fuseOperatorSet_.insert(token);
                }
}


float getCostRemain(std::vector<NodeGroup*>* branchList, std::set<CostNode*> skip){

    float totalCost = 0.0;
    for (auto i = branchList->begin(); i != branchList->end(); i++) {
        NodeGroup* branch = *i;

        for (auto ii = branch->nodeList_.begin(); ii != branch->nodeList_.end(); ii++) {
            //std::cout << "name: " << (*ii)->name_<< ", backend = " << (*ii)->backendName_ << ", cost = " << (*ii)->cost_ << std::endl;
            if(skip.find((*ii)) != skip.end())
                totalCost+=(*ii)->cost_;
        }
    }
    return totalCost;
}


float getCostWithConstraint(std::vector<NodeGroup*>* branchList, NodeGroup* exception){
    //std::cout << "----------- getCostWithConstraint ---------------" << std::endl;
    //std::cout << "branch size: " << branchList->size() << std::endl;

    float totalCost = 0.0;
    for (auto i = branchList->begin(); i != branchList->end(); i++) {
        NodeGroup* branch = *i;

        if(branch != nullptr && branch == exception)
            continue;

        //std::cout << "branch id: " << branch->ID_ << std::endl;
        for (auto ii = branch->nodeList_.begin(); ii != branch->nodeList_.end(); ii++) {
            //std::cout << "name: " << (*ii)->name_<< ", backend = " << (*ii)->backendName_ << ", cost = " << (*ii)->cost_ << std::endl;

            if(exception != nullptr) {
                if((*ii)->backendName_.compare("VTA")){
                    std::cout << "** Node backend must be VTA! **" << std::endl;
                    exit(0);
                }
            }

            totalCost+=(*ii)->cost_;
            //std::cout << "totalCost: " << totalCost << std::endl;
        }
    }
    return totalCost;
}


float getCost(NodeGroup* branch){
    float totalCost = 0.0;
    for (auto ii = branch->nodeList_.rbegin(); ii != branch->nodeList_.rend(); ++ii ) {
        //std::cout << "name: " << (*ii)->name_<< ", backend = " << (*ii)->backendName_ << ", cost = " << (*ii)->cost_ << std::endl;
        totalCost+=(*ii)->cost_;
    }
    return totalCost;
}

float NestPartitioner::getCPUCost(NodeGroup* branch){
    float totalCost = 0.0;
    for (auto ii = branch->nodeList_.rbegin(); ii != branch->nodeList_.rend(); ++ii ) {
        totalCost += getCostOfSingleNode(*ii, "CPU");
    }
    return totalCost;
}

void NestPartitioner::allocOptimalPUFusedNodes(std::vector<NodeGroup*>* branchList)
{
//  std::cout << "--------- allocOptimalPUFusedNodes -----------" << std::endl;
    allocateOptimalPUSingleNode(branchList);

    for (auto i = branchList->begin(); i != branchList->end(); i++) {
        NodeGroup* branch = (*i);
        CostNode *firstPrevCNode = nullptr;
        CostNode *secondPrevCNode = nullptr;

        int size = branch->nodeList_.size();
        for (int bi = 0; bi < size; bi++) {
            CostNode *cnode = branch->nodeList_.at(bi);

            if(bi > 0) {
                firstPrevCNode = branch->nodeList_.at(bi-1);

                if(firstPrevCNode == nullptr)
                    continue;

                if(firstPrevCNode->node_->getKind() == Kinded::Kind::ConvolutionNodeKind && !isVTAConv((ConvolutionNode *)firstPrevCNode->node_)) {
                        continue;
                }

                if(!firstPrevCNode->backendName_.compare(cnode->backendName_))
                    continue;

                getMinCostOfFusedNode(firstPrevCNode, cnode);

                if(bi > 1) {
                    secondPrevCNode = branch->nodeList_.at(bi-2);
                    if(secondPrevCNode == nullptr /*|| !isFusable(secondPrevCNode->node_)*/)
                        continue;

                    if(!firstPrevCNode->backendName_.compare(secondPrevCNode->backendName_) &&
                       !secondPrevCNode->backendName_.compare(cnode->backendName_))
                        continue;

                    getMinCostOfFusedNode(secondPrevCNode, firstPrevCNode, cnode);
                }
            }
        }
    }
}


bool NestPartitioner::isVTAConv(ConvolutionNode* convNode)
{
   // std::cout << "filter size: " << convNode->getFilter().dims()[0] << std::endl;
   // std::cout << "input channel: " << convNode->getInput().dims()[3] << std::endl;
    if(convNode->getFilter().dims()[0]%16 == 0 && convNode->getInput().dims()[3]%16 == 0 && convNode->getGroup() == 1) {
        return true;
    }

    return false;
}
void NestPartitioner::allocateVTAOps(std::vector<NodeGroup*>* branchList) {
    // std::cout << "--------- allocateVTAOps -----------" << std::endl;
    //std::cout << "\n[allocateVTAOps] Allocate VTA to VTA-computable operations*." << std::endl;
    std::vector<Backend *> backends;
    genBackendMap(backendMap_, backendHolder_, backends);

    CostNode* prevCnode = nullptr;
    for (auto i = branchList->begin(); i != branchList->end(); i++) {
        for (auto bi = (*i)->nodeList_.begin(); bi != (*i)->nodeList_.end(); bi++) {
            CostNode *cnode = *bi;

            if (cnode->node_->getKind() == Kinded::Kind::ReluNodeKind) {
                cnode->backendName_ = prevCnode->backendName_;
            } else {
                if (backendMap_["VTA"].supportedNodesKinds.size() > 0 &&
                    (backendMap_["VTA"].supportedNodesKinds.find(cnode->node_->getKind()) !=
                     backendMap_["VTA"].supportedNodesKinds.end())) {
                    if (cnode->node_->getKind() == Kinded::Kind::ConvolutionNodeKind) {
                        if (isVTAConv((ConvolutionNode *) cnode->node_)) {
                            cnode->backendName_ = "VTA";
                        } else {
                            std::cout << "** nonVTA Convolution." << std::endl;
                            cnode->backendName_ = "CPU";
                        }
                    } else {
                        cnode->backendName_ = "VTA";
                    }

                } else {
                    cnode->backendName_ = "CPU";
                }
                prevCnode = cnode;
            }
        }
    }
}

void NestPartitioner::allocateOptimalPUSingleNode(std::vector<NodeGroup*>* branchList) {
    // std::cout << "--------- findOptimalPartitionsSingleNode -----------" << std::endl;
    //std::cout << "\n[allocateOptimalPUSingleNode] Allocate PUs to operations considering the cost of a *Single Operation*." << std::endl;
    std::vector<Backend *> backends;
    genBackendMap(backendMap_, backendHolder_, backends);

    for (auto i = branchList->begin(); i != branchList->end(); i++) {
        for (auto bi = (*i)->nodeList_.begin(); bi != (*i)->nodeList_.end(); bi++) {
            CostNode *cnode = *bi;
            getMinCostOfSingleNode(cnode, backends);
        }
    }
}

void getBackendSet(std::set<std::string>* bset, NodeGroup* branch){
    for (auto bii = branch->nodeList_.begin(); bii != branch->nodeList_.end(); bii++) {
        bset->insert((*bii)->backendName_);
    }
}

float getMaxCost(std::vector<NodeGroup*>* branchList) {
    float maxCost = 0;
    for(auto branch: *branchList) {
        float cost = getCost(branch);
        if(maxCost < cost)
            maxCost = cost;
    }

    return maxCost;
}

bool NestPartitioner::setBranchPU(NodeGroup *branch) {

    std::string backend = "";
    for(int i = 0; i < branch->nodeList_.size(); i++) {
        CostNode* node = branch->nodeList_.at(i);
        if(i == 0) {
            backend = node->backendName_;
            continue;
        } else {
            if(backend.compare(node->backendName_))
                return false;
        }
    }
    //std::cout << "[useSinglePU] backend = " << backend << std::endl;
//    branch->branchBackend_ = backend;
    branch->backendName_ = backend;

    return true;
}

void NestPartitioner::findParallelBranchesForMultiVTA(std::vector<NodeGroup *> *branchList, int cpuNum, int vtaNum) {
    std::cout << "--------- [findParallelBranches] -----------" << std::endl;
    float originalTotalCost = getCostWithConstraint(branchList, nullptr);
    std::cout << "Original total cost =  " << originalTotalCost << std::endl;

    for (auto i = branchList->begin(); i != branchList->end(); i++) {
        NodeGroup* branch = (*i);
        setBranchPU(branch);
    }

    for (auto i = branchList->begin(); i != branchList->end(); i++) {
       // bool existParallelBranch = true;
        if((*i)->inBranchList_.size() > 1) {//parallel branches
            int cpuCnt = 0, vtaCnt = 0;
            std::vector<NodeGroup *> pBranchList;
            for (auto bi = (*i)->inBranchList_.begin(); bi != (*i)->inBranchList_.end(); bi++) {
                NodeGroup* branch = (*bi);
                if(!branch->backendName_.compare("CPU")) {
                    pBranchList.push_back(branch);
                    cpuCnt++;
                } else if (!branch->backendName_.compare("VTA")) {
                    pBranchList.push_back(branch);
                    vtaCnt++;
                }
            }

            if(pBranchList.size() >= 2) {

//                std::cout << "cpuCnt = " << cpuCnt << ", " << "vtaCnt = " << vtaCnt << std::endl;
                if(cpuCnt > cpuNum || vtaCnt > vtaNum) {
                    std::cout << "Too many parallel branches!!" << std::endl;
                    continue;
                }

                for (auto bi = pBranchList.begin(); bi != pBranchList.end(); bi++) {
                    NodeGroup *branch = (*bi);
                    //std::cout << "branch backend = " << branch->backendName_ << std::endl;

                    if(vtaNum > 0) {
                        if (!branch->backendName_.compare("CPU") && cpuCnt > 0) {
//                            std::cout << "CPU branch = " << branch->nodeList_.at(0)->name_ << std::endl;
                            branch->isParallelBranch_ = true;
                            cpuCnt--;
                        } else if (!branch->backendName_.compare("VTA") && vtaCnt > 0) {
//                            std::cout << "VTA branch = " << branch->nodeList_.at(0)->name_ << std::endl;
                            branch->isParallelBranch_ = true;
                            if (vtaNum == 1) {
                                branch->backendName_ = "VTA";
                            } else {
                                branch->backendName_ = "VTA-" + std::to_string(vtaCnt-1);
                            }
                            vtaCnt--;
//                            std::cout << "branch backend = " << branch->backendName_ << ", " << branch->isParallelBranch_ << std::endl;
                        }
                    }
                }
                pBranchList.clear();
            }
        }
    }
}

Expected<DAGListTy> NestPartitioner::partition(CompilationContext &cctx) {
    std::string dir = "/home/msyu/Dropbox/Project/development/configs/partition_perform_profile";
    return partition(cctx, 3, "", dir + "/NESTOptimalPlan.yaml", 0, 0, nullptr);
}

std::vector<Node*> getParallelNodes(BFSLevel  dag, Node* node) {

    std::vector<Node *> parentNodes;

    if (node->getKind() == Kinded::Kind::PlaceholderKind || node->getKind() == Kinded::Kind::ConstantKind) {
        return parentNodes;
    }

    for(int i = 0; i < node->getNumInputs(); i++) {
        auto inNode = node->getNthInput(i).getNode();

        if (inNode->getKind() == Kinded::Kind::SaveNodeKind || inNode->getKind() == Kinded::Kind::ConstantKind || inNode->getKind() == Kinded::Kind::PlaceholderKind) {
            continue;
        }

        if (!inNode->hasOneUse()) {
            continue;
        }
        parentNodes.push_back(inNode);
    }
    return parentNodes;
}


std::vector<Node*> getParentNodes(BFSLevel  dag, Node* node) {
    std::vector<Node *> parentNodes;

    if (node->getKind() == Kinded::Kind::PlaceholderKind || node->getKind() == Kinded::Kind::ConstantKind) {
        return parentNodes;
    }

    for(int i = 0; i < node->getNumInputs(); i++) {
        auto inNode = node->getNthInput(i).getNode();
        if ((inNode->getKind() != Kinded::Kind::ConstantKind) &&
            inNode->getKind() != Kinded::Kind::PlaceholderKind) {
            parentNodes.push_back(inNode);
        }
    }
    return parentNodes;
}


void addToBranchList(std::vector<NodeGroup*>* branchList, NodeGroup* branch) {
    for(auto i = branchList->begin(); i != branchList->end(); i++) {
        if((*i)->level_ > branch->level_) {
            branchList->insert(i, branch);
            return;
        }
    }
    branchList->push_back(branch);
}

void NestPartitioner::getBranch(std::vector<std::vector<CostNode>>* cnodebfs, BFSLevel dag, std::vector<Node *> parallelNodes, Node* lastNode,
                                NodeGroup* prevBranch, unsigned int level, unsigned int branchID, std::vector<NodeGroup*>* branchList)
{
//  std::cout << "[start getBranch]" << " level = " << level << std::endl;

    if(lastNode->getKind() == Kinded::Kind::PlaceholderKind)
        return;

//  std::cout << "[getBranch]" << "lastNode = " << lastNode->getKindName() << " name = "<< lastNode->getName().str() << std::endl;
//  std::cout << "parallelNodes.size() = " << parallelNodes.size() << " level = " << level << std::endl;

    for(int i = 0; i < parallelNodes.size(); i++) {

        NodeGroup* branch = new NodeGroup;
        branch->level_ = level;
        branch->ID_ = branchID;
        //branchList->push_back(branch);
        addToBranchList(branchList, branch);
        if(prevBranch != nullptr) {
            prevBranch->inBranchList_.push_back(branch);
        }

        Node* curNode = parallelNodes.at(i);
        std::vector<Node*> pNodes = getParallelNodes(dag, curNode);

//    std::cout <<  "node = " << curNode->getKindName() << " name = "<< lastNode->getName().str() << " level = " << level << std::endl;
//    std::cout << "pNodes size = " << pNodes.size() << std::endl;

        if(pNodes.size() > 1) {
            //std::cout << "[getBranch 1]" << std::endl;
            getBranch(cnodebfs, dag, pNodes, curNode, branch, level+1, ++branchID, branchList);

        } else if (pNodes.size() == 1) {
            //std::cout << "[getBranch 11]" << std::endl;
            if(!isChecked(cnodebfs, lastNode)) {
//        std::cout << "[getBranch]"
//                  << "add node = " << curNode->getKindName()
//                  << " name = " << curNode->getName().str()
//                  << " level = " << level << " branch = " << branch->ID_
//                  << std::endl;
                branch->nodeList_.push_back(findNodeFromCNodeList(cnodebfs, lastNode));
                setChecked(cnodebfs, lastNode);
            }

            backtrack(cnodebfs, dag, curNode, branch, level, branchID, branchList);
            //std::cout << "[getBranch 2] branch size = " << branch->nodeList_.size() << " level = " << level << std::endl;
        } else {
            if(!isChecked(cnodebfs, curNode)) {
//        std::cout << "[getBranch]"
//                  << "add node = " << curNode->getKindName()
//                  << " name = " << curNode->getName().str()
//                  << " level = " << level << " branch = " << branch->ID_
//                  << std::endl;
                branch->nodeList_.push_back(findNodeFromCNodeList(cnodebfs, curNode));
                setChecked(cnodebfs, curNode);

                pNodes = getParentNodes(dag, curNode);
                //       if(pNodes.size() == 1) {
                //std::cout << "[getBranch 12]" << std::endl;
                getBranch(cnodebfs, dag, pNodes, curNode, branch, level+1, ++branchID, branchList);
//        }
//      } else {
//        std::cout << "HERE 1." << std::endl;
            }
        }

        if(branch->nodeList_.size() == 0) {
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


void NestPartitioner::backtrack(std::vector<std::vector<CostNode>>* cnodebfs, BFSLevel dag, Node *lastNode,
                                NodeGroup *branch, unsigned int level, unsigned int branchID, std::vector<NodeGroup*>* branchList)
{
//  std::cout << "[start backtrack]"
//            << "add node = " << lastNode->getKindName()
//            << " name = " << lastNode->getName().str() << " level = " << level
//            << " branch = " << branch->ID_
//            << std::endl;

    if(lastNode->getKind() == Kinded::Kind::PlaceholderKind)
        return;

    if(lastNode->getKind() == Kinded::Kind::SaveNodeKind)
        return;

    if(isChecked(cnodebfs, lastNode))
        return;

//  std::cout << "[backtrack]"
//            << "add node = " << lastNode->getKindName()
//            << " name = " << lastNode->getName().str()
//            << " level = " << level << " branch = " << branch->ID_
//            << std::endl;
    branch->nodeList_.push_back(findNodeFromCNodeList(cnodebfs, lastNode));
    setChecked(cnodebfs, lastNode);

    std::vector<Node*> pNodes = getParallelNodes(dag, lastNode);
    if(pNodes.size() > 1) {
        //std::cout << "[getBranch 2]" << std::endl;
        getBranch(cnodebfs, dag, pNodes, lastNode, branch, level+1, branchID+1, branchList);
    } else if (pNodes.size() == 1) {
        //std::cout << "[getBranch 22]" << std::endl;
        backtrack(cnodebfs, dag, pNodes.at(0), branch, level, branchID, branchList);
    } else if (pNodes.size() == 0){
        pNodes = getParentNodes(dag, lastNode);
        //if(pNodes.size() == 1) {
//      std::cout << "[backtrack] parent nodes size() == " << pNodes.size() << std::endl;
        getBranch(cnodebfs, dag, pNodes, lastNode, branch, level+1, branchID+1, branchList);
        //} else {
        //  std::cout << "[backtrack] parent nodes size() == 0" << std::endl;
        //return;
        //}
    }
    //std::cout << "[finish backtrack]" << " branch = " << branch->ID_<< std::endl;
}

void NestPartitioner::analyzeDAG(std::vector<std::vector<CostNode>>* cnodebfs, BFSLevel dag, std::vector<NodeGroup*>* branchList)
{
    unsigned int level = 0;
    std::cout << "[analyzeDAG]"
              << "save size = " << dag[0].size() << std::endl;

    size_t saveNodeSize = dag[0].size();
    for(int i = 0; i < saveNodeSize; i++) {
        setChecked(cnodebfs, dag[0][i]);

        std::vector<Node *> pNodes = getParallelNodes(dag, dag[0][i]);

        std::cout << " curNode = " << dag[0][i]->getKindName() << ", "<< dag[0][i]->getName().str() << std::endl;
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

//  for (auto i = branchList->begin(); i != branchList->end(); i++) {
//    std::cout << "branchID: " << (*i)->ID_ << ", level: " << (*i)->level_ << ", size: " << (*i)->nodeList_.size() << std::endl;
//    std::cout << "level : " << (*i)->level_ << std::endl;
//    std::cout << "# inBranch: " << (*i)->inBranchList_.size() << std::endl;
//
//    for (auto ii = (*i)->nodeList_.begin(); ii != (*i)->nodeList_.end(); ii++ ) {
//      std::cout << "name: " << (*ii)->name_<< std::endl;
//      std::cout << "type: " << (*ii)->node_->getKindName()<< std::endl;
//
//    }
//  }
}

void NestPartitioner::generateApplicationCode(std::string profilePath, std::string partitionPlanFile, int profileMode, int partitionExe) {
//  std::cout << "--------- generateApplicationCode -----------" << std::endl;

    std::string inputPartitionName;
    if (partitionPlanFile.find('/') != std::string::npos) {
        inputPartitionName =
                partitionPlanFile.substr(partitionPlanFile.find_last_of("/") + 1);
        inputPartitionName = inputPartitionName.substr(0, inputPartitionName.size() - 5);
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
            Node* node = bfs[i][j];
            std::string key = appGen_.getProfileKey(node);
            if(profileKeySet_.find(key) == profileKeySet_.end()) {
                appGen_.setKeyName(key, insertCount++);
            }
            else {
                appGen_.setKeyName("blank", insertCount++);
            }
        }
    }
}

void NestPartitioner::generateTestProfile(Function *function, std::string profilePath, std::string partitionPlanFile, int profileMode) {

    BFSLevel bfs = getBFSLevel(function);

    //loadPerformProfileInfo(profilePath+"/test");
    std::vector<std::vector<CostNode>> cnodeBfs;
    makeCostNode(function, &cnodeBfs);
    std::vector<NodeGroup*> branchList;

    analyzeDAG(&cnodeBfs, bfs, &branchList);

    std::cout << "1. Loading fusing options for partition-plan generation." << std::endl;
    loadFuseOperatorsConfig(profilePath + "/profileConfigs.yaml");

    std::cout << "2. Partition-plan generation for the performance profiling of each PU" << std::endl;
    for (auto device : deviceInfo_) {
        std::cout << "  => Partition-plan generation for " << device.backendName << std::endl;
        //separate each node

        std::string profileName = profilePath + "/data/initialProfile" + device.backendName +
                                  std::to_string(device.deviceID) + ".yaml";
        outPerformProfileForEachNode(&branchList, profileName, device);
        std::string planName = profilePath + "/initialSingleNodePartition" + device.backendName +
                               std::to_string(device.deviceID) + ".yaml";
        outPartitionPlanForSingleNode(function, &branchList, device, planName);
        std::cout << "    . Partition plan for each NN operation: " << planName << std::endl;
        profileName = profilePath + "/data/fusedProfile" + device.backendName +
                      std::to_string(device.deviceID) + ".yaml";

        //fusing depth: 1
        outPerformProfileForFusedNode(&branchList, profileName , device);
        planName = profilePath + "/initialFusedNodePartition" + device.backendName +
                   std::to_string(device.deviceID) + ".yaml";
        outPartitionPlanForFusion(function->getName(), &branchList, device, planName);

        std::cout << "    . Partition plan for fused NN operation: " << planName << std::endl;
    }

    for(auto branch: nodeGroups_)
        delete branch;

    nodeGroups_.clear();
}

void NestPartitioner::generateOptimalPlanForSingleNodes(Function *function, std::string profilePath, std::string partitionPlanFile, int profileMode) {

    BFSLevel bfs = getBFSLevel(function);

    loadPerformProfileInfo(profilePath+"/test");
    std::vector<std::vector<CostNode>> cnodeBfs;
    makeCostNode(function, &cnodeBfs);
    std::vector<NodeGroup*> branchList;

    analyzeDAG(&cnodeBfs, bfs, &branchList);
    allocateOptimalPUSingleNode(&branchList);
    partitionBranches(&branchList, false);
    outPartitionPlan(function->getName().str(), profilePath + "/SingleOpOptimalPlan.yaml", false);

    std::cout << "Total cost = " << getCostWithConstraint(&branchList, nullptr) << std::endl;

    for(auto branch: nodeGroups_)
        delete branch;

    nodeGroups_.clear();
}

void NestPartitioner::generatePlanForVTAOps(Function *function, std::string profilePath, std::string partitionPlanFile, int profileMode) {

    BFSLevel bfs = getBFSLevel(function);

    //loadPerformProfileInfo(profilePath+"/test");
    std::vector<std::vector<CostNode>> cnodeBfs;
    makeCostNode(function, &cnodeBfs);
    std::vector<NodeGroup*> branchList;

    analyzeDAG(&cnodeBfs, bfs, &branchList);
    allocateVTAOps(&branchList);
    partitionBranches(&branchList, false);
    //partitionBranches(&branchList, true);
    outPartitionPlan(function->getName().str(), profilePath + "/VTAOPsPlan.yaml", false);

    for(auto branch: nodeGroups_)
        delete branch;

    nodeGroups_.clear();

}


//void NestPartitioner::changeBackendAndGeneratePlanForParallelBranches(Function *function, std::string profilePath, std::string partitionPlanFile, int profileMode) {
//
//    std::vector<std::vector<CostNode>> cnodeBfs;
//    BFSLevel bfs = getBFSLevel(function);
//    std::vector<NodeGroup*> branchList;
//
//    loadFuseOperatorsConfig(profilePath + "/profileConfigs.yaml");
//    makeCostNode(function, &cnodeBfs);
//    loadPerformProfileInfo(profilePath+"/test");
//
//    analyzeDAG(&cnodeBfs, bfs, &branchList);
//    allocOptimalPUFusedNodes(&branchList);
//    allocateCPUToParallelBranch(&branchList);
//
//    partitionBranchesReplaceCPU(&branchList, true);
//    outPartitionPlanReplaceCPU(function->getName().str(), profilePath + "/NESTOptimalPlan-branch.yaml");
//
//    for(auto branch: nodeGroups_)
//        delete branch;
//
//    nodeGroups_.clear();
//}

void NestPartitioner::generateOptimalPlanForParallelBranches(Function *function, std::string profilePath, std::string partitionPlanFile, int profileMode) {

    std::vector<std::vector<CostNode>> cnodeBfs;
    BFSLevel bfs = getBFSLevel(function);
    std::vector<NodeGroup*> branchList;

    loadFuseOperatorsConfig(profilePath + "/profileConfigs.yaml");
    makeCostNode(function, &cnodeBfs);
    loadPerformProfileInfo(profilePath+"/test");

    analyzeDAG(&cnodeBfs, bfs, &branchList);
    allocOptimalPUFusedNodes(&branchList);
//    findParallelBranches(&branchList);
//
//    partitionBranches(&branchList, true);

    findParallelBranchesForMultiVTA(&branchList, 1, 1);
    partitionBranches(&branchList, true);

    outPartitionPlan(function->getName().str(), profilePath + "/NESTOptimalPlan-parallel.yaml", true);

    for(auto branch: nodeGroups_)
        delete branch;

    nodeGroups_.clear();
}

void NestPartitioner::generateOptimalPlanForMultiVTA(Function *function, std::string profilePath, std::string partitionPlanFile, int profileMode) {


    std::vector<std::vector<CostNode>> cnodeBfs;
    BFSLevel bfs = getBFSLevel(function);
    std::vector<NodeGroup*> branchList;

    loadFuseOperatorsConfig(profilePath + "/profileConfigs.yaml");
    makeCostNode(function, &cnodeBfs);
    loadPerformProfileInfo(profilePath+"/test");

    analyzeDAG(&cnodeBfs, bfs, &branchList);
    allocOptimalPUFusedNodes(&branchList);

    findParallelBranchesForMultiVTA(&branchList, 1, 4);

    partitionBranches(&branchList, true);
    outPartitionPlan(function->getName().str(), profilePath + "/NESTOptimalPlan-multivta.yaml", true);

    for(auto branch: nodeGroups_)
        delete branch;

    nodeGroups_.clear();

}

void NestPartitioner::generateDAGStatistics(Function *function) {

    std::vector<std::vector<CostNode>> cnodeBfs;
    BFSLevel bfs = getBFSLevel(function);
    std::vector<NodeGroup*> branchList;

    makeCostNode(function, &cnodeBfs);
    analyzeDAG(&cnodeBfs, bfs, &branchList);

    std::cout << std::endl << "== Statistics == " << std::endl << std::endl;

    int groups = 0;
    int pbranches = 0;
//  std::cout << "# branches: " << branchList.size() << std::endl << std::endl;
    for (auto i = branchList.begin(); i != branchList.end(); i++) {
//    std::cout << "branchID: " << (*i)->ID_ << ", level: " << (*i)->level_ << ", # nodes: " << (*i)->nodeList_.size() << " first node: " << (*i)->nodeList_[0]->node_->getName().str() << std::endl;
//    std::cout << "# inBranch: " << (*i)->inBranchList_.size() << std::endl;

        int bcnt = (*i)->inBranchList_.size();
        if( bcnt > 1) {
//      std::cout << "Parallel branch!!: " << std::endl;
            groups += 1;
            pbranches += bcnt;
        }
        //    for (auto ii = (*i)->nodeList_.begin(); ii != (*i)->nodeList_.end(); ii++ ) {
        //      std::cout << "name: " << (*ii)->name_<< std::endl;
        //    }
    }

    std::cout << "# parallel branches: " << pbranches << std::endl;
    std::cout << "# groups of parallel branches: " << groups << std::endl;

}

Expected<DAGListTy> NestPartitioner::generatePartitionCode(CompilationContext &cctx, std::string profilePath, std::string partitionPlanFile, int profileMode, int partitionExe) {

    //load partition plan
    runtime::PartitionConfig partitionConfig;
    cctx.partitionConfig = &partitionConfig;
    loadPartitionPlan(cctx.partitionConfig, partitionPlanFile);

    //generate main.cpp
    generateApplicationCode(profilePath, partitionPlanFile, profileMode, partitionExe);

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

Expected<DAGListTy> NestPartitioner::partition(CompilationContext &cctx, size_t exeType, std::string profilePath, std::string partitionPlanFile, int profileMode, int partitionExe, std::map<std::string, int>* puIdxMap) {
//  std::cout << "--------- partitionForNest exeType: " << exeType << "-----------" << std::endl;
    puIdxMap_ = puIdxMap;

    Function *function = module_->getFunctions().front();

    if(exeType == 1) {
        generateTestProfile(function, profilePath, partitionPlanFile, profileMode);
        exit(0);
    } else if (exeType == 2) {
        generateOptimalPlanForSingleNodes(function, profilePath, partitionPlanFile, profileMode);
        exit(0);
    } else if (exeType == 3) {
        generatePlanForVTAOps(function, profilePath, partitionPlanFile, profileMode);
        exit(0);

    } else if (exeType == 4) {//1 CPU, 1 VTA
        generateOptimalPlanForParallelBranches(function, profilePath, partitionPlanFile, profileMode);
        exit(0);
    }else if (exeType == 5){
        //   generateDAGStatistics(function);
        generateOptimalPlanForMultiVTA(function, profilePath, partitionPlanFile, profileMode);
        exit(0);
    }else if (exeType == 6){
        generateDAGStatistics(function);
        exit(0);
    } /*else if (exeType == 7){
      changeBackendAndGeneratePlanForParallelBranches(function, profilePath, partitionPlanFile, profileMode);
      exit(0);
  }*/

    if(exeType == 0) {//compile
        return generatePartitionCode(cctx, profilePath, partitionPlanFile, profileMode, partitionExe);
    } else {
        DAGListTy partitions;
        return std::move(partitions);
    }
}


