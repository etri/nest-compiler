/**
 * Copyright (c) 2021, Etri.
 *
 * This program or software including the accompanying associated documentation
 * ("Software") is the proprietary software of Etri and/or its licensors.
 * Etri reserves all rights in and to the Software and all intellectual
 * property therein. The Software is distributed on an "AS IS" basis, without
 * warranties or conditions of any kind, unless required by applicable law or a
 * written agreement.
 *
 * @file: NMPONNXModelWriter.cpp
 * @brief description: This file writes specific NMP nodes to Graph
 * @date: 11 17, 2021
 */

Error ONNXModelWriter::writeNMPFullyConnected(const NMPFullyConnectedNode *node,
                                        GraphType &graph) {
  auto *proto = graph.add_node();
  return writeAllWithNode("NMPFullyConnected", node, graph, proto);
}
