#include "gtest/gtest.h"
#include "glow/Support/Random.h"

#include "glow/Graph/Graph.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Node.h"
#include "glow/IR/Instrs.h"
#include "llvm/Support/FileSystem.h"

#include "glow/Partitioner/NestPartitionerSchedule.h"

using namespace glow;
TEST(NestPartitionerTest, ParallelBranchInputTest) {


    Module M;
    PlaceholderBindings bindings;

    auto *F = M.createFunction("main");
    Node *K1 =M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input1", false);
    Node *K2 =M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input2", false);

    Node *relu1 = F->createRELU("relu1", K1);
    Node *relu2 = F->createRELU("relu2", K2);

    Node *add = F->createAdd("add", relu1, relu2);
    Placeholder *output =M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "out", false);

    F->createSave("return", add, output);

    NestPartitionerSchedule appGen;
    Function* function = M.getFunctions().front();

    std::string inputList[maxInOut];
    std::string outputList[maxInOut];
    int inputCount = 0;
    int outputCount = 0;

    appGen.setPartitionInputOutputList(function, inputList, outputList, &inputCount, &outputCount);

    EXPECT_TRUE(inputCount == 2);
}
