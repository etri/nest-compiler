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

#include "CCodeGenerator.h"
#include "glow/Importer/ONNXModelLoader.h"
#include <iostream>
#include <stdio.h>

using namespace glow;
using namespace std;

string CCodeGenerator::getTotalSizeStr(const NodeValue &value) {
  string str = "";

  llvm::ArrayRef<dim_t> dims = value.getType()->dims();
  if (dims.size() <= 0)
    return str;

  str = str + to_string(value.getType()->getElementSize());
  for (int i = 0; i < dims.size(); i++) {

    // size = size*dims.at(i);
    string dimstr = to_string(dims[i]);
    str = str + "*" + dimstr;
  }

  return str;
}

string CCodeGenerator::getDataArrayStr(TypeRef value) {
  string str = "";
  if (value->numSizes_ <= 0)
    return str;

  for (int i = 0; i < value->numSizes_; i++) {
    string dimstr = to_string(value->sizes_[i]);

    str = str + "[" + dimstr + "]";
  }
  return str;
}

string CCodeGenerator::getDataCountStr(const TypeRef value) {

  string str = "";

  llvm::ArrayRef<dim_t> dims = value->dims();
  if (dims.size() <= 0)
    return str;

  string dimstr = to_string(dims[0]);

  // size_t size =
  // (size_t)tensor.getType().getElementSize(tensor.getElementType());
  size_t size = 4; // check!
  str = str + to_string(size) + "*" + dimstr;

  for (int i = 1; i < dims.size(); i++) {

    dimstr = to_string(dims[i]);
    str = str + "*" + dimstr;
  }

  return str;
}

string splatValueVar;
string shuffleGroupVar;
string shuffleKernelVar;
string concatAxisVar;

string filterVecVar;
string kernelVecVar;
string strideVecVar;
string padVecVar;
string groupVar;

string channelVar;
string epsilonVar;
string momentumVar;

set<string> constVars;

unsigned int nodeIdx = 0;

void CCodeGenerator::declareConstVar(string &wfilec, Constant *node) {
  if (varSet_.find(node->getVarName()) == varSet_.end()) {
    wfilec.append(node->getType()->getElementTypeName().str() + " (*" +
                  node->getVarName() + ")" + getDataArrayStr(node->getType()) +
                  ";\n");
    ;
    varSet_.insert(node->getVarName());
  }
}

void CCodeGenerator::generateDeclaration(Loader &loader, string &wfilec,
                                         Node &n) {
  cout << "generateDeclaration: " << n.getVarName() << endl;

  //    NodesList& nodeList = loader.getFunction()->getNodes();
  string kindName = n.getKindName();
  //    cout << kindName << endl;

  // for (auto &n : nodeList) {
  //    if (n == nodeList.back() || !kindName.compare("Save")) {
  if (&n == lastNode) {
    cout << "===> 1. Variable declaration has been generated." << endl;
    return;
  }

  for (int i = 0; i < n.getNumInputs(); i++) {
    string ikname = n.getNthInput(i).getNode()->getKindName();
    Constant *inode = (Constant *)n.getNthInput(i).getNode();

    if (!ikname.compare("Constant")) {

      if (!inode->getVarName().compare("varselected")) {
        continue;
      }

      declareConstVar(wfilec, inode);
    }
  }

  string outvarname = n.getNthResult(0).getNode()->getVarName();

  string in0varname = "";
  if (n.getNumInputs() > 0) {
    in0varname = n.getNthInput(0).getNode()->getVarName();
  }

  if (varSet_.find(outvarname) == varSet_.end()) {
    wfilec.append(n.getNthResult(0).getType()->getElementTypeName().str() +
                  " (*" + outvarname + ")" +
                  getDataArrayStr(n.getNthResult(0).getType()) + ";\n");
    varSet_.insert(outvarname);
  }

  if (n.getNumInputs() > 0 && varSet_.find(in0varname) == varSet_.end()) {
    wfilec.append(n.getNthInput(0).getType()->getElementTypeName().str() +
                  " (*" + in0varname + ")" +
                  getDataArrayStr(n.getNthInput(0).getType()) + ";\n");
    varSet_.insert(in0varname);
  }

  if (!kindName.compare("Add")) {

  } else if (!kindName.compare("Concat")) {

    wfilec.append("//Concat\n");
    // ConcatNode* node = (ConcatNode *)&n;

    concatAxisVar = string("concatAxis").append(to_string(nodeIdx));
    wfilec.append("int " + concatAxisVar + ";\n");
    nodeIdx++;

  } else if (!kindName.compare("ChannelShuffle")) {

    wfilec.append("//ChannelShuffle\n");
    shuffleGroupVar = string("shuffleGroup").append(to_string(nodeIdx));
    wfilec.append("int " + shuffleGroupVar + ";\n");
    shuffleKernelVar = string("shuffleKernel").append(to_string(nodeIdx));
    wfilec.append("int " + shuffleKernelVar + ";\n");
    nodeIdx++;

  } else if (!kindName.compare("Transpose")) {
    wfilec.append("//Transpose\n");
    TransposeNode *node = (TransposeNode *)&n;

    size_t id = (size_t)node->getShuffle().data();
    string shuffleVar = string("shuffle").append(to_string(id));

    wfilec.append("vector<size_t> " + shuffleVar + ";\n");

  } else if (!kindName.compare("Convolution")) {

    ConvolutionNode *node = (ConvolutionNode *)&n;

    wfilec.append("//Convolution\n");

    filterVecVar = string("filter").append(to_string((size_t)node));
    wfilec.append("vector<size_t> " + filterVecVar + ";\n");

    kernelVecVar = string("kernel").append(to_string((size_t)node));
    wfilec.append("vector<size_t> " + kernelVecVar + ";\n");

    strideVecVar = string("stride").append(to_string((size_t)node));
    wfilec.append("vector<size_t> " + strideVecVar + ";\n");

    padVecVar = string("pad").append(to_string((size_t)node));
    wfilec.append("vector<size_t> " + padVecVar + ";\n");

    groupVar = string("group").append(to_string((size_t)node));
    wfilec.append("size_t " + groupVar + ";\n");

  } else if (!kindName.compare("Relu")) {
    //            ReluNode* node = (ReluNode *)&n;
    //            wfilec.append( "//Relu\n");

  } else if (!kindName.compare("MaxPool")) {
    MaxPoolNode *node = (MaxPoolNode *)&n;

    wfilec.append("//MaxPool\n");

    kernelVecVar = string("kernel").append(to_string((size_t)node));
    wfilec.append("vector<size_t> " + kernelVecVar + ";\n");

    strideVecVar = string("stride").append(to_string((size_t)node));
    wfilec.append("vector<size_t> " + strideVecVar + ";\n");

    padVecVar = string("pad").append(to_string((size_t)node));
    wfilec.append("vector<size_t> " + padVecVar + ";\n");

  } else if (!kindName.compare("Reshape")) {

  } else if (!kindName.compare("Gemm")) {

  } else if (!kindName.compare("MatMul")) {
    // MatMulNode* node = (MatMulNode *)&n;
    // string in1varname =
    // string("var").append(node->getNthInput(1).getNode()->getName());

    wfilec.append("//MatMul\n");

  } else if (!kindName.compare("FullyConnected")) {
    // FullyConnectedNode* node = (FullyConnectedNode *)&n;
    // string in1varname =
    // string("var").append(node->getInput().getNode()->getName()); string
    // biasname = string("var").append(node->getBias().getNode()->getName());

    wfilec.append("//FullyConnected\n");

  } else if (!kindName.compare("SoftMax")) {
    // SoftMaxNode* node = (SoftMaxNode *)&n;
    wfilec.append("//SoftMax\n");

  } else if (!kindName.compare("BatchNormalization")) {
    //        BatchNormalizationNode* node = (BatchNormalizationNode *)&n;

    wfilec.append("//BatchNormalization\n");

    channelVar = string("channelVar").append(to_string((size_t)nodeIdx));
    epsilonVar = string("epsilonVar").append(to_string((size_t)nodeIdx));
    momentumVar = string("momentum").append(to_string((size_t)nodeIdx));

    wfilec.append("unsigned int " + channelVar + ";\n");
    wfilec.append("float " + epsilonVar + ";\n");
    wfilec.append("float " + momentumVar + ";\n");

    nodeIdx++;

  } else if (!kindName.compare("AvgPool")) {
    AvgPoolNode *node = (AvgPoolNode *)&n;

    wfilec.append("//AvgPool\n");

    kernelVecVar =
        string("kernel").append(to_string((size_t)node->getKernels().data()));
    wfilec.append("vector<size_t> " + kernelVecVar + ";\n");

    strideVecVar =
        string("stride").append(to_string((size_t)node->getStrides().data()));
    wfilec.append("vector<size_t> " + strideVecVar + ";\n");

    padVecVar = string("pad").append(to_string((size_t)node->getPads().data()));
    wfilec.append("vector<size_t> " + padVecVar + ";\n");

  } else if (!kindName.compare("Splat")) {
    // SplatNode* node = (SplatNode *)&n;
    wfilec.append("//Splat\n");

    splatValueVar = string("splat").append(to_string(nodeIdx));
    wfilec.append("float " + splatValueVar + ";\n");
    nodeIdx++;
    // std::cout << "value = " << node->getValue() << std::endl;
  } else if (!kindName.compare("ChannelShuffle")) {
    // ChannelShuffleNode* node = (ChannelShuffleNode *)&n;
    wfilec.append("//ChannelShuffle\n");

  } else if (!kindName.compare("Concat")) {
    // ConcatNode* node = (ConcatNode *)&n;
    wfilec.append("//Concat\n");
  } else if (!kindName.compare("Gemm")) {
    // SplatNode* node = (SplatNode *)&n;
    wfilec.append("//Gemm\n");
  } else {
    std::cout << "Unknown node type!!" << std::endl;
  }
}

void CCodeGenerator::allocConstVar(string &wfilec, ofstream *wfileb,
                                   Constant *node) {
  // wfilec.append("\t//Constant\n");

  string varname = node->getVarName();

  if (allocVarList_.find(varname) == allocVarList_.end()) {
    allocVarList_.insert(varname);
    wfilec.append("\t" + varname + " = (" +
                  node->getType()->getElementTypeName().str() + "(*)" +
                  getDataArrayStr(node->getType()) + ")malloc(sizeof(" +
                  node->getType()->getElementTypeName().str() +
                  getDataArrayStr(node->getType()) + "));\n");

    wfileb->write(node->getPayload().getUnsafePtr(),
                  node->getPayload().getUnpaddedSizeInBytes());
    wfilec.append("\t" + string("readFileBin.read((char *)") + varname + "," +
                  getDataCountStr(node->getType()) + ");\n\n");

    constVars.insert(varname);
  }
}

void CCodeGenerator::generateInitialization(Loader &loader, string &wfilec,
                                            ofstream *wfileb, Node &n) {

  cout << "generateInitialization: " << n.getVarName() << endl;
  //    NodesList &nodeList = loader.getFunction()->getNodes();
  cout << n.getKindName() << endl;
  string kindName = n.getKindName();

  //    if (n == nodeList.front()) {
  if (&n == firstNode) {

    string binOpenStr = "\tstring binfile = dataFilePath;\n"
                        "\tifstream readFileBin;\n"
                        "\treadFileBin.open(binfile.data(), ios::binary);\n";

    wfilec.append("\n");
    wfilec.append("void initializeVariables(string dataFilePath)\n {\n");
    wfilec.append(binOpenStr + "\n");

    //    }else if (n == nodeList.back() ||!kindName.compare("Save")) {
  } else if (&n == lastNode) {
    wfilec.append("\n}\n");
    cout << "===> 2. Memory allocation & variable initialization part has been "
            "generated.\n";
    return;
  }

  for (int i = 0; i < n.getNumInputs(); i++) {
    string ikname = n.getNthInput(i).getNode()->getKindName();
    if (!ikname.compare("Constant")) {

      Constant *inode = (Constant *)n.getNthInput(i).getNode();
      if (!inode->getVarName().compare("varselected")) {
        continue;
      }
      // cout << "== Constant input == " << endl;
      //            if(!inode->getVarName().compare("varconv_bias")) {
      //                cout << "here....." << endl;
      //            }
      wfilec.append("\t//Constant input\n");
      allocConstVar(wfilec, wfileb, inode);
    }
  }

  string outvarname = n.getNthResult(0).getNode()->getVarName();

  string in0varname = "";
  if (n.getNumInputs() > 0) {
    in0varname = n.getNthInput(0).getNode()->getVarName();
  }

  if (!kindName.compare("Add")) {
    //    AddNode* node = (AddNode *)&n;
    //     wfilec.append("\t//Add\n");

  } else if (!kindName.compare("Transpose")) {

    TransposeNode *node = (TransposeNode *)&n;
    wfilec.append("\t//Transpose\n");

    int slen = node->getShuffle().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getShuffle()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");

    string shuffleVar =
        string("shuffle").append(to_string((size_t)node->getShuffle().data()));

    wfilec.append("\tread_vector(&" + shuffleVar + ", &readFileBin);\n");

  } else if (!kindName.compare("Convolution")) {
    ConvolutionNode *node = (ConvolutionNode *)&n;

    wfilec.append("\t//Convolution\n");

    int slen = node->getFilter().dims().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getFilter().dims()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + filterVecVar + ", &readFileBin);\n");

    slen = node->getKernels().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getKernels()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + kernelVecVar + ", &readFileBin);\n");

    slen = node->getStrides().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getStrides()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + strideVecVar + ", &readFileBin);\n");

    slen = node->getPads().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getPads()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + padVecVar + ", &readFileBin);\n");

    size_t group = (size_t)node->getGroup();
    wfileb->write((char *)&group, sizeof(size_t));
    //        wfilec.append("\t//group: " + std::to_string(group) +  ";\n");
    wfilec.append("\treadFileBin.read((char *)&" + groupVar +
                  ", sizeof(size_t));\n");
    // cout << "bias: " << node->getBias().getNode()->getVarName() << endl;

  } else if (!kindName.compare("Relu")) {

  } else if (!kindName.compare("MaxPool")) {
    MaxPoolNode *node = (MaxPoolNode *)&n;

    wfilec.append("\t//MaxPool\n");

    int slen = node->getKernels().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getKernels()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + kernelVecVar + ", &readFileBin);\n");

    slen = node->getStrides().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getStrides()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + strideVecVar + ", &readFileBin);\n");

    slen = node->getPads().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getPads()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + padVecVar + ", &readFileBin);\n");

  } else if (!kindName.compare("Reshape")) {

    wfilec.append("\t//Reshape\n");

  } else if (!kindName.compare("MatMul")) {

  } else if (!kindName.compare("Gemm")) {

  } else if (!kindName.compare("SoftMax")) {
    SoftMaxNode *node = (SoftMaxNode *)&n;
    wfilec.append("\t//SoftMax\n");

    wfilec.append("\t" + outvarname + " = (" +
                  node->getResult().getType()->getElementTypeName().str() +
                  "(*)" + getDataArrayStr(node->getResult().getType()) +
                  ")malloc(sizeof(" +
                  node->getResult().getType()->getElementTypeName().str() +
                  getDataArrayStr(node->getResult().getType()) + "));\n");
    allocVarList_.insert(outvarname);

  } else if (!kindName.compare("BatchNormalization")) {
    BatchNormalizationNode *node = (BatchNormalizationNode *)&n;

    wfilec.append("\t//BatchNormalization\n");

    unsigned int channel = node->getChannelIdx();
    wfileb->write((char *)(&channel), sizeof(unsigned int));
    wfilec.append("\treadFileBin.read((char *)&" + channelVar +
                  ", sizeof(unsigned int));\n");

    float epsilon = node->getEpsilon();
    wfileb->write((char *)(&epsilon), sizeof(float));
    wfilec.append("\treadFileBin.read((char *)&" + epsilonVar +
                  ", sizeof(float));\n");

    float momentum = node->getMomentum();
    wfileb->write((char *)(&momentum), sizeof(float));
    wfilec.append("\treadFileBin.read((char *)&" + momentumVar +
                  ", sizeof(float));\n");

    //         wfilec.append("\t" + outvarname + " = (" + node->getResult().getType()->getElementTypeName().str() + "(*)" + getDataArrayStr(node->getResult().getType()) + ")malloc(sizeof(" \
//        + node->getResult().getType()->getElementTypeName().str() + getDataArrayStr(node->getResult().getType()) + "));\n");
    //         wfilec.append("\tmemset("+ outvarname +", 0x00, sizeof(" +
    //         node->getResult().getType()->getElementTypeName().str() +
    //         getDataArrayStr(node->getResult().getType()) +"));\n");

  } else if (!kindName.compare("AvgPool")) {
    AvgPoolNode *node = (AvgPoolNode *)&n;

    wfilec.append("\t//AvgPool\n");

    int slen = node->getKernels().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getKernels()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + kernelVecVar + ", &readFileBin);\n");

    slen = node->getStrides().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getStrides()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + strideVecVar + ", &readFileBin);\n");

    slen = node->getPads().size();
    wfileb->write((char *)(&slen), sizeof(int));
    wfilec.append("\t//");
    for (int i = 0; i < slen; i++) {
      size_t val = node->getPads()[i];
      wfileb->write((char *)(&val), sizeof(size_t));
      wfilec.append(to_string(val) + " ");
    }
    wfilec.append("\n");
    wfilec.append("\tread_vector(&" + padVecVar + ", &readFileBin);\n");

  } else if (!kindName.compare("Splat")) {
    SplatNode *node = (SplatNode *)&n;
    wfilec.append("\t//Splat\n");
    int value = node->getValue();
    wfileb->write((char *)(&value), sizeof(float));
    wfilec.append("\treadFileBin.read((char *)&" + splatValueVar +
                  ", sizeof(float));\n");
    // std::cout << "value = " << node->getValue() << std::endl;
  } else if (!kindName.compare("ChannelShuffle")) {
    ChannelShuffleNode *node = (ChannelShuffleNode *)&n;
    wfilec.append("\t//ChannelShuffle\n");
    int group = node->getGroup();
    int kernel = node->getKernel();
    wfileb->write((char *)(&group), sizeof(int));
    wfilec.append("\treadFileBin.read((char *)&" + shuffleGroupVar +
                  ", sizeof(int));\n");
    wfileb->write((char *)(&kernel), sizeof(int));
    wfilec.append("\treadFileBin.read((char *)&" + shuffleKernelVar +
                  ", sizeof(int));\n");
    // std::cout << "value = " << node->getValue() << std::endl;
  } else if (!kindName.compare("Concat")) {
    ConcatNode *node = (ConcatNode *)&n;
    wfilec.append("\t//Concat\n");

    int axis = node->getDim();
    wfileb->write((char *)(&axis), sizeof(int));
    wfilec.append("\treadFileBin.read((char *)&" + concatAxisVar +
                  ", sizeof(int));\n");
    // std::cout << "axis value = " << axis << std::endl;

  } else if (!kindName.compare("Gemm")) {
    // SplatNode* node = (SplatNode *)&n;
    wfilec.append("//Gemm\n");
  } else {
    std::cout << "kindName = " << kindName << std::endl;
    std::cout << "Unknown node type!!" << std::endl;
  }

  if (constVars.find(outvarname) == constVars.end()) {
    wfilec.append("\t" + outvarname + " = (" +
                  n.getType(0)->getElementTypeName().str() + "(*)" +
                  getDataArrayStr(n.getNthResult(0).getType()) +
                  ")malloc(sizeof(" +
                  n.getNthResult(0).getType()->getElementTypeName().str() +
                  getDataArrayStr(n.getNthResult(0).getType()) + "));\n");
    wfilec.append("\tmemset(" + outvarname + ", 0x00, sizeof(" +
                  n.getNthResult(0).getType()->getElementTypeName().str() +
                  getDataArrayStr(n.getNthResult(0).getType()) + "));\n");
  }
  wfilec.append("\n");
}

void CCodeGenerator::generateInference(Loader &loader, string inputName,
                                       string &wfilec, string inAryStr,
                                       Node &n) {
  // cout << "generateDInference" << endl;

  //    NodesList &nodeList = loader.getFunction()->getNodes();

  //    Node &lastNode = nodeList.back();
  bool isLast = false;
  string kindName = n.getKindName();

  // if (n == nodeList.front()) {
  if (&n == firstNode) {
    // vector<TypeRef> inputTypes = loader.getInputTypes();

    wfilec.append("\n");
    wfilec.append("void inference(" + n.getType(0)->getElementTypeName().str() +
                  "(*" + "var" + inputName + ")" + inAryStr + ", float result" +
                  getDataArrayStr(lastNode->getNthInput(0).getType()) +
                  ")\n {\n");

    //    } else if (n == nodeList.back() ||!kindName.compare("Save")) {
  } else if (&n == lastNode) {

    wfilec.append("\n}\n");
    cout << "===> 3. Inference part has been generated." << endl;
    return;

    //    } else if(n == *nodeList.getPrevNode(lastNode)) {
  } else if (&n == lastNode->getNthInput(0).getNode()) {
    isLast = true;
  }

  string outvarname =
      string("var").append(n.getNthResult(0).getNode()->getName());

  string in0varname = "";
  if (n.getNumInputs() > 0) {
    in0varname = string("var").append(n.getNthInput(0).getNode()->getName());
  }

  if (!kindName.compare("Add")) {

    AddNode *node = (AddNode *)&n;

    wfilec.append("\t//Add\n");
    string in1varname =
        string("var").append(node->getNthInput(1).getNode()->getName());

    wfilec.append("\tadd(*" + in0varname + ", *" + in1varname + ");\n");
    wfilec.append("\t" + outvarname + " = " + in0varname + ";\n");

    freeVarList_.insert(outvarname);

  } else if (!kindName.compare("Transpose")) {
    TransposeNode *node = (TransposeNode *)&n;

    size_t id = (size_t)node->getShuffle().data();
    string shuffleVar = string("shuffle").append(to_string(id));

    wfilec.append("\t//Transpose\n");
    wfilec.append("\ttranspose(*" + in0varname + ", " + shuffleVar + ", *" +
                  outvarname + ");\n\n");

  } else if (!kindName.compare("Convolution")) {
    ConvolutionNode *node = (ConvolutionNode *)&n;

    wfilec.append("\t//Convolution\n");

    string biasname = node->getBias().getNode()->getVarName();
    string filtername = node->getFilter().getNode()->getVarName();
    unsigned_t group = node->getGroup();

    //        std::cout << "group = " << group << std::endl;
    wfilec.append("\t//group: " + std::to_string(group) + "\n");
    if (group == 1) {
      wfilec.append("\tconv(*" + in0varname + ", *" + filtername + ", " +
                    filterVecVar + ", *" + biasname + ", " + strideVecVar +
                    ", " + padVecVar + ", " + groupVar + ", *" + outvarname +
                    ");\n\n");
    } else { // group > 1
      wfilec.append("\tconv_group(*" + in0varname + ", *" + filtername + ", " +
                    filterVecVar + ", *" + biasname + ", " + strideVecVar +
                    ", " + padVecVar + ", " + groupVar + ", *" + outvarname +
                    ");\n\n");
    }

  } else if (!kindName.compare("Relu")) {

    wfilec.append("\t//Relu\n");
    wfilec.append("\trelu(*" + in0varname + ");\n");
    wfilec.append("\t" + outvarname + " = " + in0varname + ";\n");
    freeVarList_.insert(outvarname);

  } else if (!kindName.compare("MaxPool")) {

    wfilec.append("\t//MaxPool\n");

    wfilec.append("\tmaxpool(*" + in0varname + ", " + kernelVecVar + ", " +
                  strideVecVar + ", " + padVecVar + ", *" + outvarname +
                  ");\n\n");

  } else if (!kindName.compare("Reshape")) {
    ReshapeNode *node = (ReshapeNode *)&n;
    wfilec.append("\t//Reshape\n");

    string inAryStr2(getDataArrayStr(node->getInput().getType()));
    string outAryStr(getDataArrayStr(node->getResult().getType()));

    if (!outAryStr.compare(inAryStr2)) { // same
      wfilec.append("\t" + outvarname + " = " + in0varname + ";\n");
    } else { // different

      if (!outAryStr.compare("[1]" + inAryStr2)) {
        wfilec.append("\tmemcpy((*" + outvarname + ")[0]," + in0varname + ", " +
                      "sizeof(" +
                      node->getResult().getType()->getElementTypeName().str() +
                      getDataArrayStr(node->getResult().getType()) + "));\n");
      } else {
        wfilec.append("\treshape(*" + in0varname + ", *" + outvarname +
                      ");\n\n");
      }
    }

  } else if (!kindName.compare("MatMul")) {
    // MatMulNode* node = (MatMulNode *)&n;

    wfilec.append("\t//Matmul\n");
    string in1varname =
        string("var").append(n.getNthInput(1).getNode()->getName());
    wfilec.append("\tmatmul(*" + in0varname + ", *" + in1varname + ", *" +
                  outvarname + ");\n\n");

  } else if (!kindName.compare("Gemm")) {
    GemmNode *node = (GemmNode *)&n;
    // cout << "Gemm inference !!" << endl;
    wfilec.append("\t//Gemm\n");

    string Avarname = string("var").append(node->getA().getNode()->getName());

    string Bvarname = string("var").append(node->getB().getNode()->getName());

    string Cvarname = string("var").append(node->getC().getNode()->getName());

    float alpha = node->getAlpha();
    float beta = node->getBeta();
    int transA = node->getTransposeA();
    int transB = node->getTransposeB();

    //          cout << Avarname <<endl;
    //          cout << Bvarname <<endl;
    //          cout << Cvarname <<endl;
    //          cout << to_string(alpha) <<endl;
    //          cout << to_string(beta) <<endl;
    //          cout << to_string(transA) <<endl;
    //          cout << to_string(transB) <<endl;

    string funcStr = "\tgemm(*" + Avarname + ", *" + Bvarname + ", *" +
                     Cvarname + ", " + to_string(alpha) + ", " +
                     to_string(beta) + ", " + to_string(transA) + ", " +
                     to_string(transB) + ", *" + outvarname + ");\n\n";

    //          cout << funcStr << endl;
    //          cout << getDataArrayStr(node->getA().getType()) << endl;
    //          cout << getDataArrayStr(node->getB().getType()) << endl;
    //          cout << getDataArrayStr(node->getC().getType()) << endl;
    wfilec.append(funcStr);
  } else if (!kindName.compare("FullyConnected")) {
    FullyConnectedNode *node = (FullyConnectedNode *)&n;

    wfilec.append("\t//FullyConnected\n");
    string weightvarname =
        string("var").append(node->getWeights().getNode()->getName());

    string biasname =
        string("var").append(node->getBias().getNode()->getName());

    wfilec.append("\tfullyconnected(*" + in0varname + ", *" + weightvarname +
                  ", *" + biasname + ", *" + outvarname + ");\n\n");

  } else if (!kindName.compare("SoftMax")) {
    // SoftMaxNode* node = (SoftMaxNode *)&n;

    wfilec.append("\t//SoftMax\n");
    // selected <== 1
    wfilec.append("\tsoftmax(*" + in0varname + ", 1, *" + outvarname +
                  ");\n\n");

  } else if (!kindName.compare("BatchNormalization")) {
    BatchNormalizationNode *node = (BatchNormalizationNode *)&n;

    wfilec.append("\t//BatchNormalization\n");
    wfilec.append("\tbatchNorm(*" + in0varname + ", *" +
                  node->getScale().getNode()->getVarName() + ", *" +
                  node->getBias().getNode()->getVarName() + ", *" +
                  node->getMean().getNode()->getVarName() + ", *" +
                  node->getVar().getNode()->getVarName() + ", " + channelVar +
                  ", " + epsilonVar + ", " + momentumVar + ", *" + outvarname +
                  ");\n\n");

  } else if (!kindName.compare("AvgPool")) {

    wfilec.append("\t//AvgPool\n");
    wfilec.append("\tavgpool(*" + in0varname + ", " + kernelVecVar + ", " +
                  strideVecVar + ", " + padVecVar + ", *" + outvarname +
                  ");\n\n");

  } else if (!kindName.compare("Splat")) {
    // SplatNode* node = (SplatNode *)&n;
    wfilec.append("//Splat\n");
    wfilec.append("\tsplat(" + splatValueVar + ", *" + outvarname + ");\n\n");

  } else if (!kindName.compare("Save")) {
    wfilec.append("//Save\n");

    // std::cout << "value = " << node->getValue() << std::endl;
  } else if (!kindName.compare("ChannelShuffle")) {
    // SplatNode* node = (SplatNode *)&n;
    wfilec.append("//ChannelShuffle\n");
    wfilec.append("\tchannel_shuffle(*" + in0varname + ", " + shuffleGroupVar +
                  ", " + shuffleKernelVar + ", *" + outvarname + ");\n\n");

    // std::cout << "value = " << node->getValue() << std::endl;
  } else if (!kindName.compare("Concat")) {
    ConcatNode *node = (ConcatNode *)&n;
    wfilec.append("\t//Concat\n");

    std::vector<NodeValue> concatInputs = node->getInputs();
    std::string inputParamStr = "";
    for (int i = 0; i < concatInputs.size(); i++) {
      NodeValue v = concatInputs.at(i);
      inputParamStr += (", *" + v.getNode()->getVarName());
      //            std::cout << v.getNode()->getVarName() << std::endl;
      //            std::cout << getTotalSizeStr(v) << std::endl;
    }
    wfilec.append("\tconcat(" + concatAxisVar + inputParamStr + ", *" +
                  outvarname + ");\n\n");
  } else {
    std::cout << "kindName = " << kindName << std::endl;
    std::cout << "Unknown node type!!" << std::endl;
  }

  if (isLast == 1) {
    wfilec.append("\tmemcpy(result," + outvarname + ", " +
                  getTotalSizeStr(n.getNthResult(0)) + ");\n");
  }
}

void CCodeGenerator::generateFree(Loader &loader, string inputName,
                                  string &wfilec, string inAryStr, Node &n) {
  // cout << "generateFree" << endl;

  //    NodesList& nodeList = loader.getFunction()->getNodes();
  string kindName = n.getKindName();
  int isLast = 0;

  //    const auto &lastNode = nodeList.back();

  //    if(n == nodeList.front()) {
  if (&n == firstNode) {
    // vector<const char *> inputNames = loader.getInputNames();
    // vector<TypeRef> inputTypes = loader.getInputTypes();

    wfilec.append("\n");
    wfilec.append("void memFree(" + n.getType(0)->getElementTypeName().str() +
                  "(*" + "var" + inputName + ")" + inAryStr + ", float result" +
                  getDataArrayStr(lastNode->getNthInput(0).getType()) +
                  ")\n{\n");
    //    } else if(n == nodeList.back() || !kindName.compare("Save")) {
  } else if (&n == lastNode) {
    wfilec.append("\n}\n");
    cout << "===> 4. Memory deallocation part has been generated." << endl;

    return;
    //    } else if(n == *nodeList.getPrevNode(lastNode)) {
  } else if (&n == lastNode->getNthInput(0).getNode()) {
    isLast = true;
  }

  string outvarname =
      string("var").append(n.getNthResult(0).getNode()->getName());

  string in0varname = "";
  if (n.getNumInputs() > 0) {
    in0varname = string("var").append(n.getNthInput(0).getNode()->getName());

    if (freeVarList_.find(in0varname) == freeVarList_.end()) {
      wfilec.append("\tfree(" + in0varname + ");\n");
      freeVarList_.insert(in0varname);
    }
  }

  if (!kindName.compare("Add")) {
    AddNode *node = (AddNode *)&n;

    string in1varname =
        string("var").append(node->getNthInput(1).getNode()->getName());

    if (freeVarList_.find(in1varname) == freeVarList_.end()) {
      wfilec.append("\t//Add\n");
      wfilec.append("\tfree(" + in1varname + ");\n");
      freeVarList_.insert(in1varname);
    }

  } else if (!kindName.compare("Transpose")) {

  } else if (!kindName.compare("Convolution")) {

    ConvolutionNode *node = (ConvolutionNode *)&n;
    wfilec.append("\t//Convolution\n");

    string tmpvarname = node->getFilter().getNode()->getVarName();
    if (freeVarList_.find(tmpvarname) == freeVarList_.end()) {
      wfilec.append("\tfree(" + tmpvarname + ");\n");
      freeVarList_.insert(tmpvarname);
    }
    tmpvarname = node->getBias().getNode()->getVarName();
    if (freeVarList_.find(tmpvarname) == freeVarList_.end()) {
      wfilec.append("\tfree(" + tmpvarname + ");\n");
      freeVarList_.insert(tmpvarname);
    }

  } else if (!kindName.compare("Relu")) {
    wfilec.append("\t//Relu\n");
  } else if (!kindName.compare("MaxPool")) {
    wfilec.append("\t//MaxPool\n");
  } else if (!kindName.compare("Reshape")) {
    wfilec.append("\t//Reshape\n");
  } else if (!kindName.compare("MatMul")) {
    // MatMulNode* node = (MatMulNode *)&n;

    wfilec.append("\t//MatMul\n");

    string in1varname =
        string("var").append(n.getNthInput(1).getNode()->getName());
    if (freeVarList_.find(in1varname) == freeVarList_.end()) {
      wfilec.append("\tfree(" + in1varname + ");\n");
      freeVarList_.insert(in1varname);
    }
  } else if (!kindName.compare("FullyConnected")) {
    FullyConnectedNode *node = (FullyConnectedNode *)&n;
    wfilec.append("\t//FullyConnected\n");

    string weightvarname =
        string("var").append(node->getWeights().getNode()->getName());
    if (freeVarList_.find(weightvarname) == freeVarList_.end()) {
      wfilec.append("\tfree(" + weightvarname + ");\n");
      freeVarList_.insert(weightvarname);
    }

    string biasname =
        string("var").append(node->getBias().getNode()->getName());
    if (freeVarList_.find(biasname) == freeVarList_.end()) {
      wfilec.append("\tfree(" + biasname + ");\n");
      freeVarList_.insert(biasname);
    }

  } else if (!kindName.compare("SoftMax")) {
    wfilec.append("\t//SoftMax\n");

  } else if (!kindName.compare("BatchNormalization")) {
    wfilec.append("\t//BatchNormalization\n");
  } else if (!kindName.compare("AvgPool")) {
    wfilec.append("\t//AvgPool\n");
  } else if (!kindName.compare("Splat")) {
    wfilec.append("\t//Splat\n");
  } else if (!kindName.compare("ChannelShuffle")) {
    wfilec.append("\t//ChannelShuffle\n");
  } else if (!kindName.compare("Concat")) {
    // SplatNode* node = (SplatNode *)&n;
    wfilec.append("//Concat\n");
  } else if (!kindName.compare("Gemm")) {
    // SplatNode* node = (SplatNode *)&n;
    wfilec.append("//Gemm\n");
  } else {
    std::cout << "kindName = " << kindName << std::endl;
    std::cout << "Unknown node type!!" << std::endl;
  }

  if (isLast == 1) {

    if (freeVarList_.find(in0varname) == freeVarList_.end()) {
      wfilec.append("\tfree(" + in0varname + ");\n");
      freeVarList_.insert(in0varname);
    }

    if (freeVarList_.find(outvarname) == freeVarList_.end()) {
      wfilec.append("\tfree(" + outvarname + ");\n");
      freeVarList_.insert(outvarname);
    }
  }
}

string CCodeGenerator::replaceAll(std::string &str, std::string pattern,
                                  std::string replace) {
  string result = str;
  string::size_type pos = 0;
  string::size_type offset = 0;
  while ((pos = result.find(pattern, offset)) != std::string::npos) {
    result.replace(result.begin() + pos, result.begin() + pos + pattern.size(),
                   replace);
    offset = pos + replace.size();
  }
  return result;
}

#include "glow/Partitioner/PartitionerUtils.h"

void CCodeGenerator::generateCodeFromModels(Loader &loader) {

  BFSLevel bfs = getBFSLevel(loader.getFunction());
  size_t level = bfs.size();

  firstNode = bfs[level - 1][bfs[level - 1].size() - 1];
  lastNode = bfs[0][0];

  cout << "first node size = "
       << getDataArrayStr(firstNode->getNthInput(0).getType()) << endl;
  cout << "last node size = "
       << getDataArrayStr(lastNode->getNthInput(0).getType()) << endl;

  cout << "Loader::generateCodeFromModels()" << endl;
  cout << "Model = " + loader.getOnnxModelFilename().str() << endl;

  string outfile_bin = "data.bin";
  string outfile_c = "network.cpp";
  string outfile_h = "network.h";

  ofstream writeFileC;
  writeFileC.open(outfile_c.data());

  ofstream writeFileH;
  writeFileH.open(outfile_h.data());

  ofstream writeFileBin;
  writeFileBin.open(outfile_bin.data(), ios::binary);

  //  NodesList& nodes = loader.getFunction()->getNodes();
  //  const auto &lastNode = nodes.back();

  vector<Type> inputTypes = loader.getInputTypes();
  vector<string> inputNames = loader.getInputNames();

  string inputName = replaceAll(inputNames[0], "/", "_");
  string inAryStr = getDataArrayStr(&inputTypes[0]);

  cout << "input name = " << inputName << endl << endl;
  cout << "input type = " << inAryStr << endl;

  if (writeFileH.is_open()) {
    writeFileH << "#ifndef NETWORK_H_" << endl << "#define NETWORK_H_" << endl;
    writeFileH << "#include <vector>" << endl
               << "#include <iostream>" << endl
               << "#include <fstream>" << endl
               << "#include \"library.h\"" << endl
               << endl;
    writeFileH << "//-------" + loader.getOnnxModelFilename().str() << endl
               << endl;
    writeFileH << "void initializeVariables(string dataFilePath);" << endl;

    writeFileH
        << "void inference(" + inputTypes[0].getElementTypeName().str() +
               " (*var"
        << inputName << ")" << inAryStr
        << ", " +
               lastNode->getNthInput(0).getType()->getElementTypeName().str() +
               " result" + getDataArrayStr(lastNode->getNthInput(0).getType())
        << ");" << endl;

    writeFileH
        << "void memFree(" + inputTypes[0].getElementTypeName().str() + " (*var"
        << inputName << ")" << inAryStr
        << ", " +
               lastNode->getNthInput(0).getType()->getElementTypeName().str() +
               " result" + getDataArrayStr(lastNode->getNthInput(0).getType())
        << ");" << endl;

    writeFileH << "#endif /* NETWORK_H_ */\n" << endl;
    writeFileH.close();
  }

  if (writeFileC.is_open() && writeFileBin.is_open()) {

    writeFileC << "#include \"network.h\"" << endl << endl;

    // NodesList& nodeList = loader.getFunction()->getNodes();
    string declare;
    string initialize;
    string inference;
    string memfree;

    //    for (auto &n : nodeList) {
    //      generateDeclaration(loader, declare, n);
    //      generateInitialization(loader, initialize, &writeFileBin, n);
    //      generateInference(loader, inputName, inference, inAryStr, n);
    //      generateFree(loader, inputName, memfree, inAryStr, n);
    //    }

    for (int i = level - 1; i >= 0; i--) {
      for (size_t j = 0; j < bfs[i].size(); j++) {
        //            if((i == 0 && j == 0) || (i == level-1 && j ==
        //            bfs[i].size()-2)) continue;

        Node *node = bfs[i][j];
        cout << "cur node type = " << node->getKindName() << endl;
        cout << "cur node name = " << node->getName().str() << endl;

        generateDeclaration(loader, declare, *node);
        generateInitialization(loader, initialize, &writeFileBin, *node);
        generateInference(loader, inputName, inference, inAryStr, *node);
        generateFree(loader, inputName, memfree, inAryStr, *node);
      }
    }

    writeFileC << declare;
    writeFileC << initialize;
    writeFileC << inference;
    writeFileC << memfree;

    writeFileC.close();
  }

  writeFileBin.close();
}

int main(int argc, char **argv) {

  parseCommandLine(argc, argv);

  // Initialize the loader object.
  Loader loader;
  loader.loadModel();

  CompilationContext cctx = loader.getCompilationContext();
  ::glow::fold(loader.getFunction(), cctx);
  ::glow::optimize(loader.getFunction(), cctx);

  CCodeGenerator codeGenerator;
  codeGenerator.generateCodeFromModels(loader);

  return 0;
}
