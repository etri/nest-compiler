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

#include "Tensor.h"
#include "../../glow/lib/Base/Tensor.cpp"
#include <fstream>

template <class ElemTy>
static void toBinImpl(Handle<ElemTy> handle, llvm::raw_ostream &os,
                      const char *name) {
  char filename[100];
  assert(handle.size() % 2 == 0);
  strcpy(filename, name);
  strcat(filename, ".bin");

  std::ofstream fos(filename, std::ios::binary);

  int16_t data16 = 0;
  for (size_t i = 0, e = handle.size(); i < e; i++) {
    auto data = handle.raw(i);
    if (data > 127)
      data = 127.0;
    if (data < -128)
      data = -128.0;
    int8_t clip_data = std::floor(data);
    if (i % 2 == 0) {
      data16 = 0xff & clip_data;
    } else {
      data16 = data16 | clip_data << 8;
      fos.write((const char *)&data16, 2);
    }
  }
  fos.close();
}

static void toBinImpl_int32(Handle<int32_t> handle, llvm::raw_ostream &os,
                            const char *name) {
  char filename[100];
  assert(handle.size() % 2 == 0);
  strcpy(filename, name);
  strcat(filename, ".bin");

  std::ofstream fos(filename, std::ios::binary);

  for (size_t i = 0, e = handle.size(); i < e; i++) {
    auto data = handle.raw(i);
    fos.write((const char *)&data, 4);
  }
  fos.close();
}

static void toBinImpl_float(Handle<float> handle, llvm::raw_ostream &os,
                            const char *name) {
  char filename[100];
  strcpy(filename, name);
  strcat(filename, ".bin");

  std::ofstream fos(filename, std::ios::binary);

  for (size_t i = 0, e = handle.size(); i < e; i++) {
    auto data = handle.raw(i);
    fos.write((const char *)&data, 4);
  }
  fos.close();
}

void Tensor::toBin(const char *name) const {
  auto T = this;
  // assert(T->getElementType() == ElemKind::Int8QTy ||T->getElementType() ==
  // ElemKind::Int32QTy );
  switch (T->getElementType()) {
  case ElemKind::Int8QTy:
    return toBinImpl(T->getHandle<int8_t>(), llvm::outs(), name);
  case ElemKind::Int32QTy:
    return toBinImpl_int32(T->getHandle<int32_t>(), llvm::outs(), name);
  case ElemKind::FloatTy:
    return toBinImpl_float(T->getHandle<float>(), llvm::outs(), name);
  case ElemKind::Float16Ty:
  case ElemKind::UInt8QTy:
  case ElemKind::Int16QTy:
  case ElemKind::Int32ITy:
  case ElemKind::Int64ITy:
  case ElemKind::UInt8FusedQTy:
  case ElemKind::UInt8FusedFP16QTy:
  case ElemKind::UInt4FusedFP16QTy:
  case ElemKind::BoolTy:
    return;
  }
}
