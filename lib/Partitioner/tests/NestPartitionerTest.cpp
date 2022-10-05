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
#include "glow/Support/Random.h"
#include "gtest/gtest.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/IR/Instrs.h"
#include "llvm/Support/FileSystem.h"

#include "glow/Partitioner/NestPartitionerSchedule.h"

using namespace glow;
TEST(NestPartitionerTest, ParallelBranchInputTest) {

  Module M;
  PlaceholderBindings bindings;

  auto *F = M.createFunction("main");
  Node *K1 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input1", false);
  Node *K2 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input2", false);

  Node *relu1 = F->createRELU("relu1", K1);
  Node *relu2 = F->createRELU("relu2", K2);

  Node *add = F->createAdd("add", relu1, relu2);
  Placeholder *output =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "out", false);

  F->createSave("return", add, output);

  NestPartitionerSchedule appGen;
  Function *function = M.getFunctions().front();

  std::vector<std::string> inputList;
  std::vector<std::string> outputList;

  appGen.setPartitionInputOutputList(function, &inputList, &outputList, 0);

  EXPECT_TRUE(inputList.size() == 2);
}
