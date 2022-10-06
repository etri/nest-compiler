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
 * @file: NMPSpecificNodesVerification.h
 * @brief description: Define verify function to NMP' specific Nodes
 * @date: 11 17, 2021
 */

#ifdef GLOW_WITH_NMP

bool NMPFullyConnectedNode::verify() const { return true; }

#endif // GLOW_WITH_NMP
