/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "torch_mpi.h"

#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <list>
#include <sstream>
#include <vector>

#include "parameterserver.h"
#include "resources.h"

/**********************************************************************
 ************************* Torch CPP Wrappers *************************
 **********************************************************************/
// We don't want an explicit boost dependency just for preprocessor
// concatenation, just use 2 levels of .cpp.in inclusion.
#include "generic/torch.cpp.in"

/**********************************************************************
 ******************************** MPI *********************************
 **********************************************************************/
using namespace std;
using namespace torch::mpi::resources;

namespace torch { namespace mpi {

int communicatorLevel = 0;
CommunicatorType communicatorType = CommunicatorType::intra;
std::vector<Communicator> mainThreadCommunicators;
std::pair<int, int>  collectiveSpans = pair<int, int>(0, 0);

///////////////////////////////////////////////////////////////////////////
// Helper functions
///////////////////////////////////////////////////////////////////////////
const Communicator& getMainThreadCommunicatorAtLevel(int level) {
  // Main thread guard, avoids deadlocks
  static std::thread::id tid = std::this_thread::get_id();
  if (tid != std::this_thread::get_id()) {
    THError("Collective resource can only be acquired from the main thread");
  }

  if (level < 0 && level > getMaxCommunicatorLevel()) {
    THError("Expected level %d in range [%d-%d]\n",
            level, 0, getMaxCommunicatorLevel());
  }
  return mainThreadCommunicators[level];
}

const MPI::Intracomm& getMainThreadMPICommunicator() {
  if (communicatorLevel == 0) {
    // No such thing as global intercomm
    return getMainThreadCommunicatorAtLevel(0).intraComm;
  }
  if (communicatorType == CommunicatorType::inter) {
    return getMainThreadCommunicatorAtLevel(communicatorLevel).interComm;
  }
  return getMainThreadCommunicatorAtLevel(communicatorLevel).intraComm;
}

const Communicator& getMainThreadCommunicator() {
  return getMainThreadCommunicatorAtLevel(communicatorLevel);
}

void pushCommunicator(string& str) {
  mainThreadCommunicators.emplace_back(
    getMainThreadMPICommunicator(),
    CommunicatorKey::fromString(str));
}

//////////////////////////////////////////////////////////////////////////////
// Convenience functions
//////////////////////////////////////////////////////////////////////////////
void setCollectiveSpan(int begin, int end) {
  if (begin < 0 || end > getMaxCommunicatorLevel()) {
    THError("Expected begin-end [%d, %d] in range [%d-%d]\n",
            begin, end, 0, getMaxCommunicatorLevel());
  }
  collectiveSpans = std::make_pair(begin, end);
}

void setCommunicator(CommunicatorType type,  int level) {
  communicatorType = type;
  communicatorLevel = level;
}

int getCommunicatorLevel() {
  return communicatorLevel;
}

int getMaxCommunicatorLevel() {
  return mainThreadCommunicators.size() - 1;
}

string getCommunicatorNames() {
  stringstream ss;
  auto t = (communicatorType == resources::CommunicatorType::inter) ?
    "inter" : "intra";
  ss << "current: " << getCommunicatorLevel() << " " << t << "\n";
  for (int l = 0; l < mainThreadCommunicators.size(); ++l) {
    auto &c = getMainThreadCommunicatorAtLevel(l);
    ss << "l : " << l << " -> key("
       << c.toString()
       << ") (" << commSize(c.interComm) << " in intercomm, I am rank "
       << commRank(c.interComm) << ") ";
    if (l == getCommunicatorLevel() &&
        getCommunicatorType() == CommunicatorType::inter) {
      ss << "* ";
    }
    ss << " (" << commSize(c.intraComm) << " in intracomm, I am rank "
       << commRank(c.intraComm) << ") ";
    ss << "cartesian: " << (c.cartesian ? "true" : "false");
    if (l == getCommunicatorLevel()) { ss << " * "; }
    ss << endl;
  }
  return ss.str();
}

std::pair<int, int> getCollectiveSpan() {
  THAssert(collectiveSpans.second <= getMaxCommunicatorLevel());
  return collectiveSpans;
}

CommunicatorType getCommunicatorType() {
  return communicatorType;
}

int commRank(const MPI::Intracomm& comm) {
  return comm.Get_rank();
}

int commRank(Barrier& b) {
  return b.rank;
}

int commSize(const MPI::Intracomm& comm) {
  return comm.Get_size();
}

int commSize(Barrier& b) {
  return b.size;
}

void barrier(const MPI::Intracomm& comm) {
  VLOG_1("Enter barrier on comm@" << &comm << endl);
  comm.Barrier();
  VLOG_1("Exit barrier on comm@" << &comm << endl);
}

void barrier(Barrier& b) {
  VLOG_1("Enter barrier @" << &b << endl);
  b.barrier();
  VLOG_1("Exit barrier @" << &b << endl);
}

namespace th {

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
  void retainStorage(THTensorType* tensor) {
    auto& retained = retainedStorages<decltype(tensor->storage)>();
    if (retained.find(tensor->storage) == retained.end()) {
      torch::th::retain(tensor->storage);
      retained[tensor->storage] = true;
    }
  }

  void freeRetainedStorages() {
    freeCollectiveResources();

    freeRetainedStorage<THByteStorage*>();
    freeRetainedStorage<THShortStorage*>();
    freeRetainedStorage<THIntStorage*>();
    freeRetainedStorage<THLongStorage*>();
    freeRetainedStorage<THFloatStorage*>();
    freeRetainedStorage<THDoubleStorage*>();
  }

  template<typename THStoragePtrType>
  void freeRetainedStorage() {
    auto& retained = retainedStorages<THStoragePtrType>();
    for (auto it : retained) {
      torch::th::free(it.first);
    }
    retained.clear();
  }

} // ns torch::mpi::th

}} // ns torch

template void
torch::mpi::th::retainStorage<THByteTensor>(THByteTensor* tensor);
template void
torch::mpi::th::retainStorage<THShortTensor>(THShortTensor* tensor);
template void
torch::mpi::th::retainStorage<THIntTensor>(THIntTensor* tensor);
template void
torch::mpi::th::retainStorage<THLongTensor>(THLongTensor* tensor);
template void
torch::mpi::th::retainStorage<THFloatTensor>(THFloatTensor* tensor);
template void
torch::mpi::th::retainStorage<THDoubleTensor>(THDoubleTensor* tensor);

/************************* General ************************************/
using namespace torch::mpi;


extern "C" {

void torchmpi_start() {
  int kMPIProvidedLevel = 0;
  auto req = MPI_THREAD_MULTIPLE;
  // MVAPICH2 deprecates the cpp binding here
  // auto provided = MPI::Init_thread(req);
  MPI_Init_thread(nullptr,
                  nullptr,
                  req,
                  &kMPIProvidedLevel);
  if (kMPIProvidedLevel < MPI_THREAD_MULTIPLE) {
    THError("[ERROR] MPI_Init_Thread only provides %d, we need at least %d\n",
            kMPIProvidedLevel, MPI_THREAD_MULTIPLE);
  }

  mainThreadCommunicators.emplace_back(
    MPI::COMM_WORLD, CommunicatorKey::fromString("global"));
}

int torchmpi_push_communicator(const char* key) {
  string keyStr(key, strlen(key));
  pushCommunicator(keyStr);
  return mainThreadCommunicators.size() - 1;
}

void torchmpi_set_communicator(int level) {
  if (level < 0 || level >= mainThreadCommunicators.size()) {
    THError("Valid communicator levels [%d, %d[ (0 for global communicator):"
            " got %d\n",
            0, mainThreadCommunicators.size() - 1, level);
  }
  setCommunicator(resources::CommunicatorType::intra, level);
}

bool torchmpi_is_cartesian_communicator() {
  return getMainThreadCommunicator().cartesian;
}

int torchmpi_rank() {
  return commRank(getMainThreadMPICommunicator());
}

int torchmpi_size() {
  return commSize(getMainThreadMPICommunicator());
}

void torchmpi_barrier() {
  barrier(getMainThreadMPICommunicator());
}

void torchmpi_stop() {
  barrier(getMainThreadMPICommunicator());

  // Sync
  syncAll();
  setTerminateParameterServerThread(); // make server thread joinable
  auto& server = torch::mpi::parameterServerThread();
  if (server.joinable()) {
    server.join();
  }

  // Free
  torch::mpi::freeParameterServers();
  torch::mpi::th::freeRetainedStorages();

  barrier(getMainThreadMPICommunicator());
  freeCollectiveResources();
#if TORCH_MPI_CUDA
  freeCollectiveResourcesCuda();
#endif
  mainThreadCommunicators.clear();
  if (!getenv("FB_INFRA")) {
    MPI::Finalize();
  }
}

const char* torchmpi_communicator_names() {
  return getCommunicatorNames().c_str();
}

void torchmpi_set_collective_span(int begin, int end) {
  setCollectiveSpan(begin, end);
}

SynchronizationHandle* torchmpi_synchronize_handle(SynchronizationHandle* h) {
  torch::mpi::resources::wait(h);
  return nullptr;
}

int torchmpi_num_nodes_in_communicator() {
  auto comm = getMainThreadMPICommunicator();

  constexpr int kNameLen = 1024;
  struct Name {
    char name[kNameLen];
  } name;
  memset(name.name, 0, kNameLen);

  gethostname(name.name, kNameLen);
  vector<Name> names(commSize(comm));
  names[commRank(comm)] = name;

  comm.Allgather(
    name.name, kNameLen, MPI_BYTE,
    names[0].name, kNameLen, MPI_BYTE);

  std::vector<std::string> uniqueNames;
  for (auto n : names) {
    uniqueNames.push_back(std::string(n.name));
  }
  auto end = std::unique(uniqueNames.begin(), uniqueNames.end());

  int count = 0;
  for (auto it = uniqueNames.begin(); it != end; it++) {
    ++count;
  }

  return count;
}

}
