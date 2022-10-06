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
 * @file: NMPTensorLayout.h
 * @brief description: Define the default layout for tensors ion the NMP model.
 * @date: 11 18, 2021
 */

#ifndef GLOW_BACKENDS_NMP_TENSORLAYOUT_H
#define GLOW_BACKENDS_NMP_TENSORLAYOUT_H

#include "glow/Graph/TensorLayout.h"

namespace glow {

class NMPTensorLayout final
    : public TensorLayoutCommon,
      public TensorLayoutSingleton<NMPTensorLayout> {
public:
  NMPTensorLayout(token_) { enabled_ = true; }

  // returns layout requirements of the Nth input n of a Node.
  std::string getNthInputLayoutRequirements(const Node *node,
                                            size_t n) override;

  /// \returns layout requirements of the Nth result n of a Node.
  std::string getNthResultLayoutRequirements(const Node *node,
                                             size_t n) override;
};

} // end namespace glow

#endif // GLOW_BACKENDS_NMP_TENSORLAYOUT_H
