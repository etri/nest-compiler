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
 * @file: NMPFactory.cpp
 * @brief description: Register the NMP Backend.
 * @date: 11 18, 2021
 */

#include "NMPBackend.h"

namespace glow {

REGISTER_GLOW_BACKEND_FACTORY(NMPFactory, NMPBackend);

} // namespace glow
