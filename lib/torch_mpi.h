/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include <mpi.h>
#include <TH.h>

#include <future>
#include <mutex>
#include <string>
#include <vector>

#include "resources.h"

namespace torch { namespace mpi {

// Forward declarations
namespace resources {

struct Barrier;
enum struct CommunicatorType;
struct Communicator;

}

// Make all collectives transit through pinned CPU buffers or not.
// On TCP, it is likely you want to set this to true.
// On IB + RDMA it is likely you want to set this to false.
extern bool kCPUStagedCollectives;

//////////////////////////////////////////////////////////////////////////////
// Returns the cloned, memoized, MPI::Intracomm for the current level and type
// that is safe for this thread to use
//////////////////////////////////////////////////////////////////////////////
const resources::Communicator& getMainThreadCommunicator();

//////////////////////////////////////////////////////////////////////////////
// Convenience functions
//////////////////////////////////////////////////////////////////////////////

int getCommunicatorLevel();
int getMaxCommunicatorLevel();
std::string getCommunicatorNames();
std::pair<int, int> getCollectiveSpan();
resources::CommunicatorType getCommunicatorType();

int commRank(resources::Barrier& b);
int commRank(const MPI::Intracomm& comm);
int commSize(resources::Barrier& b);
int commSize(const MPI::Intracomm& comm);
void barrier(resources::Barrier& b);
void barrier(const MPI::Intracomm& comm);

void setCollectiveSpan(int begin, int end);
void setCommunicator(resources::CommunicatorType type, int level);


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
namespace th {

template<typename THStoragePtrType>
std::unordered_map<THStoragePtrType, bool>& retainedStorages();

template<typename THTensorType> void retainStorage(THTensorType* tensor);

template<typename THStoragePtrType> void freeRetainedStorage();

}} // ns mpi::th

namespace th {

// Minimal set of TH operations needed for collectives and parameterserver
template<typename Scalar, class THType> Scalar* data(THType* t);

template<class THType> bool isContiguous(const THType* t);
template<class THType> long nElement(const THType* t);
template<class THType> THType* newWithTensor(THType* t);
template<class THType> void resize1d(THType* t, long nElement);
template<class THType> THType* newClone(THType* t);
template<class THType> void free(THType* t);
template<class THType> void retain(THType* t);
template<class THType> void copy(THType* src, THType* dst);
template<class THType>
void narrow(THType* t,
            int dimension,
            long firstIndex,
            long size);

template<typename Scalar, class THType> void fill(THType* t, Scalar val);
template<typename Scalar, class THType>
void div(THType* res, THType* t, Scalar val);
template<typename Scalar, class THType>
void cadd(THType* res,
          THType* t,
          Scalar val = 1,
          THType* src = nullptr);

template<class THStorageType> void retain(THStorageType* storage);

} // ns th

} // ns torch
