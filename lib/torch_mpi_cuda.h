/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include "torch_mpi.h"

#include <THC.h>

namespace torch {

// Having all processes in a communicator agree on which communicator to use
// in a dynamic multithreaded environment is not easy.
//
// First, note that one can't just allocate a threadlocal communicator
// because processes would still have to agree on the same thread to perform
// the collective. This does not play well with hierarchical communicators.
//
// The easy way is to always alloc a new communicator from the main thread.
// This works fine but inserts synchronization points at each collective and
// performance suffers quickly.
//
// One possible way is to obtain a resource object tied to the underlying
// pointer.
// This does not play well with realloc in a distributed setting:
//   Some process may decide to realloc to a pointer for which
//   distributed resources have been created. This may not be the case for
//   other processes and communicator would then deadlock.
//
// A potential solution would be to have a THAllocator with callbacks for
// alloc / realloc / free.
// Still that would force all frees to become collectives which is highly
// improbable due to the garbage collector (how do we know storages will be
// freed at the same time on different processes??)
//
// In the absence of this, we require that all storages on which collectives
// are applied be retained and provide a function to free them all.
// This function is now a collective.
//
namespace mpi { namespace thc {

template<typename THStoragePtrType>
std::unordered_map<THStoragePtrType, bool>& retainedStorages();

template<typename THTensorType>
void retainStorage(THCState*, THTensorType* tensor);

template<typename THStoragePtrType> void freeRetainedStorage(THCState*);

}} // ns thc

namespace thc {

// Minimal set of THC operations needed for collectives and parameterserver
template<typename Scalar, class THType>
Scalar* data(THCState* state, THType* t);

template<class THType> bool isContiguous(THCState* state, const THType* t);
template<class THType> long nElement(THCState* state, const THType* t);
template<class THType> THType* newWithTensor(THCState* state, THType* t);
template<class THType> void resize1d(THCState* state, THType* t, long nElement);
template<class THType> THType* newClone(THCState* state, THType* t);
template<class THType> void free(THCState* state, THType* t);
template<class THType> void retain(THCState* state, THType* t);
template<class THType> void copy(THCState* state, THType* src, THType* dst);
template<class THType>
void narrow(THCState* state,
            THType* t,
            int dimension,
            long firstIndex,
            long size);

template<typename Scalar, class THType>
void fill(THCState* state, THType* t, Scalar val);
template<typename Scalar, class THType>
void div(THCState* state, THType* res, THType* t, Scalar val);
template<typename Scalar, class THType>
void cadd(THCState* state,
          THType* res,
          THType* t,
          Scalar val = 1,
          THType* src = nullptr);

}} // ns torch::thc
