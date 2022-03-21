/*****************************************************************************
	*
	* Copyright Next-Generation System Software Research Group, All rights reserved.
	* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
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
	* LICENSE file : README_LICENSE_ETRI located in the top directory
	*
*****************************************************************************/

#ifndef VTALIB_EXAMPLE_VTAELEMADDTESTBUNDLE_H
#define VTALIB_EXAMPLE_VTAELEMADDTESTBUNDLE_H

#include <stdint.h>

// ---------------------------------------------------------------
//                       Common definitions
// ---------------------------------------------------------------
#ifndef _GLOW_BUNDLE_COMMON_DEFS
#define _GLOW_BUNDLE_COMMON_DEFS

// Glow bundle error code for correct execution.
#define GLOW_SUCCESS 0

// Type describing a symbol table entry of a generated bundle.
struct SymbolTableEntry {
    // Name of a variable.
    const char *name;
    // Offset of the variable inside the memory area.
    uint64_t offset;
    // The number of elements inside this variable.
    uint64_t size;
    // Variable kind: 1 if it is a mutable variable, 0 otherwise.
    char kind;
};

// Type describing the config of a generated bundle.
struct BundleConfig {
    // Size of the constant weight variables memory area.
    uint64_t constantWeightVarsMemSize;
    // Size of the mutable weight variables memory area.
    uint64_t mutableWeightVarsMemSize;
    // Size of the activations memory area.
    uint64_t activationsMemSize;
    // Alignment to be used for weights and activations.
    uint64_t alignment;
    // Number of symbols in the symbol table.
    uint64_t numSymbols;
    // Symbol table.
    const SymbolTableEntry *symbolTable;
};

#endif

#ifdef __cplusplus
extern "C" {
#endif

// Bundle memory configuration (memory layout).
extern BundleConfig vtaElemAddTestBundle_config;

// Bundle entry point (inference function). Returns 0
// for correct execution or some error code otherwise.
int vtaElemAddTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations);

#ifdef __cplusplus
}
#endif
#endif //VTALIB_EXAMPLE_VTAELEMADDTESTBUNDLE_H
