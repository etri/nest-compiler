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

#ifndef GLOW_TOOLS_PARTITION_TUNER_H
#define GLOW_TOOLS_PARTITION_TUNER_H

namespace glow {

class CostNode {
public:
  // Node* node_;
  size_t peID = -1;
  size_t partitionID = -1;
  std::string name;
  std::string backendName;
  size_t totalCost = INFINITY;
};

} // namespace glow

#endif // GLOW_TOOLS_PARTITION_TUNER_H
