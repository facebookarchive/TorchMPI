/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "torch_mpi_cuda.h"

#include <iostream>

/**********************************************************************
 ************************* Torch CPP Wrappers *************************
 **********************************************************************/
// We don't want an explicit boost dependency just for preprocessor
// concatenation, just use 2 levels of .cpp.in inclusion.
#include "generic/cutorch.cpp.in"

/**********************************************************************
 ******************************** MPI *********************************
 **********************************************************************/
using namespace std;
using namespace torch::mpi;
using namespace torch::mpi::resources;

namespace torch { namespace mpi { namespace thc {

  template<typename THStoragePtrType>
  std::unordered_map<THStoragePtrType, bool>& retainedStorages() {
    static std::unordered_map<THStoragePtrType, bool> storages;
    static std::thread::id tid = std::this_thread::get_id();
    if (tid != std::this_thread::get_id()) {
      THError("Collective resource can only be acquired from the main thread");
    }
    return storages;
  }

  template<typename THTensorType>
  void retainStorage(THCState* state, THTensorType* tensor) {
    auto& retained = retainedStorages<decltype(tensor->storage)>();
    if (retained.find(tensor->storage) == retained.end()) {
      torch::thc::retain(state, tensor->storage);
      retained[tensor->storage] = true;
    }
  }

  template<typename THStoragePtrType>
  void freeRetainedStorage(THCState* state) {
    freeCollectiveResourcesCuda();
    auto& retained = retainedStorages<THStoragePtrType>();
    for (auto it : retained) {
      torch::thc::free(state, it.first);
    }
    retained.clear();
  }

} // ns thc

}} // ns torch::mpi

template
void torch::mpi::thc::retainStorage<THCudaByteTensor>(
  THCState*,
  THCudaByteTensor* tensor);
template
void torch::mpi::thc::retainStorage<THCudaShortTensor>(
  THCState*,
  THCudaShortTensor* tensor);
template
void torch::mpi::thc::retainStorage<THCudaIntTensor>(
  THCState*,
  THCudaIntTensor* tensor);
template
void torch::mpi::thc::retainStorage<THCudaLongTensor>(
  THCState*,
  THCudaLongTensor* tensor);
template
void torch::mpi::thc::retainStorage<THCudaTensor>(
  THCState*,
  THCudaTensor* tensor);
template
void torch::mpi::thc::retainStorage<THCudaDoubleTensor>(
  THCState*,
  THCudaDoubleTensor* tensor);
