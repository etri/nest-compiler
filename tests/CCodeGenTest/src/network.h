#ifndef NETWORK_H_
#define NETWORK_H_
#include <vector>
#include <iostream>
#include <fstream>
#include "library.h"

//-------/home/etri/eclipse-workspace/models/resnet18v2.onnx

void initializeVariables(std::string dataFilePath);
void inference(float (*vardata)[1][3][224][224], float result[1][1000]);
void memFree(float (*vardata)[1][3][224][224], float result[1][1000]);
#endif /* NETWORK_H_ */

