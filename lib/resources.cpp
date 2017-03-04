/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "resources.h"

#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef TORCH_MPI_CUDA
#include <THC.h>
#endif

#ifdef TORCH_MPI_GLOO
#include <gloo/transport/tcp/device.h>
#endif

using namespace std;

#include <sstream>

#ifdef TORCH_MPI_NCCL

ncclComm_t* makeNCCLCommunicator(const MPI::Intracomm& localComm) {
  torch::mpi::barrier(localComm);

  auto size = torch::mpi::commSize(localComm);
  auto rank = torch::mpi::commRank(localComm);

  VLOG_1("Init NCCL communicator with size " << size << std::endl);
  // NCCL Communicator creation using the local MPI communicator
  ncclUniqueId commId;
  NCCLCHECK(ncclGetUniqueId(&commId));
  localComm.Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0);
  auto res = new ncclComm_t();
  ncclResult_t ret = ncclCommInitRank(res, size, commId, rank);
  if (ret != ncclSuccess) {
    THError("NCCL Init failed (%d) '%s'\n", ret, ncclGetErrorString(ret));
  }
  VLOG_1("Rank " << rank <<
         " NCCL inited with commId " << commId.internal << std::endl);

  return res;
}

#endif

#ifdef TORCH_MPI_GLOO
std::shared_ptr<gloo::mpi::Context> makeGlooContext(const MPI::Intracomm &localComm) {
  // TODO: ibverbs support.
  auto dev = gloo::transport::tcp::CreateDevice("");

  // Create Gloo context from MPI communicator
  auto context = std::make_shared<gloo::mpi::Context>(localComm);
  context->connectFullMesh(dev);

  return context;
}
#endif

namespace torch { namespace mpi { namespace resources {

#define MAIN_THREAD_GUARD()                                             \
  static std::thread::id tid = std::this_thread::get_id();              \
  if (tid != std::this_thread::get_id()) {                              \
    THError("Collective resource can only be acquired from the main thread"); \
  }

// TODO: activate this but there is a chicken and egg problem between init.lua
// ffi and freezing constants
// constants::immutableConstants = true;

CollectiveResources::CollectiveResources(void* p, const Communicator* pc) :
    inUse(false),
    comm(new Communicator(*pc)),
    ptr(p)
#ifdef TORCH_MPI_GLOO
    , glooContext(nullptr)
#endif
{
  MAIN_THREAD_GUARD();
}

CollectiveResources::~CollectiveResources() {
  delete comm;
}

CollectiveResources* acquireCollectiveResources(
  void* dataPtr,
  Spin spin,
  WithNCCLComm nccl,
  WithGlooContext gloo,
  WithEvents e)
{
  auto& resources = collectiveResources();
  auto pComm = &getMainThreadCommunicator();
  CollectiveResourcesKey rk(dataPtr, pComm);
  auto it = resources.find(rk);
  if (it == resources.end()) {
    auto r = new CollectiveResources(dataPtr, pComm);
    barrier(r->comm->intraComm);
    resources.emplace(rk, r);
    it = resources.find(rk);
  } else {
    ;
  }
  if (!spin.spin) {
    THAssert(!it->second->inUse);
  } else {
    int nIterations = 0;
    while (it->second->inUse) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      nIterations++;
      if (nIterations >= 100000) {
        THError("Main thread spinned for 10 seconds waiting for thread to"
                "release collective resource, this looks like a deadlock!");
      }
    }
  }

#ifdef TORCH_MPI_GLOO
  if (gloo.with && !it->second->glooContext) {
    it->second->glooContext =
      makeGlooContext(it->second->comm->intraComm);
  }
#endif

  it->second->inUse = true;
  return it->second;
}

CollectiveResourcesMap& collectiveResources() {
  MAIN_THREAD_GUARD();
  static CollectiveResourcesMap resources;
  return resources;
}

void releaseCollectiveResources(CollectiveResources* r) {
  THAssert(r->inUse);
  r->inUse = false;
}

void freeCollectiveResources() {
  auto& cr = collectiveResources();
  for (auto it : cr) {
    delete it.second;
  }
  cr.clear();
}

CommunicatorKey::CommunicatorKey() {
  memset(key, 0, kCommunicatorKeyLen);
}

CommunicatorKey::CommunicatorKey(const CommunicatorKey& other) {
  memset(key, 0, kCommunicatorKeyLen);
  memcpy(key, other.key, strlen(other.key));
}

CommunicatorKey& CommunicatorKey::operator=(const CommunicatorKey& other) {
  memset(key, 0, kCommunicatorKeyLen);
  memcpy(key, other.key, strlen(other.key));
  return *this;
}

CommunicatorKey CommunicatorKey::fromString(std::string s) {
  CommunicatorKey key;
  memset(key.key, 0, kCommunicatorKeyLen);
  memcpy(key.key, s.c_str(), s.size());
  return key;
}

Communicator::Communicator(const MPI::Intracomm& parent, CommunicatorKey key) :
    cartesian(true), key(key)
{
  MAIN_THREAD_GUARD();

  auto sizeInParent = commSize(parent);
  auto rankInParent = commRank(parent);

  // Need to register new communicators.
  // Must synchronize all outstanding MPI calls so deadlocks don't occur
  syncAll();

  // Allgather
  vector<CommunicatorKey> keys(sizeInParent);
  keys[rankInParent] = key;

  parent.Allgather(key.key,
                   kCommunicatorKeyLen,
                   MPI_CHAR,
                   const_cast<CommunicatorKey*>(&keys[0]),
                   kCommunicatorKeyLen,
                   MPI_CHAR);

  // Create a sorted (key, rank) vector from parent
  typedef pair<CommunicatorKey, int> KeyRank;
  vector<KeyRank> keyRanks;
  int rank = 0;
  for (auto &k : keys) {
    keyRanks.push_back(make_pair(k, rank++));
  }
  std::sort(
    keyRanks.begin(),
    keyRanks.end(),
    [](const KeyRank &a, const KeyRank &b) {
      auto res = string(a.first.key).compare(string(b.first.key));
      if (res < 0) { return true; }
      if (res > 0) { return false; }
      return a.second < b.second;
    });

  vector<KeyRank> myIntraCommParticipants;
  std::copy_if(
    keyRanks.begin(),
    keyRanks.end(),
    std::back_inserter(myIntraCommParticipants),
    [=](const KeyRank& a) {
      return string(a.first.key) == string(key.key);
    });

  int myRankInIntraComm = 0;
  for (auto &r : myIntraCommParticipants) {
    if (r.second == rankInParent) {
      break;
    }
    myRankInIntraComm++;
  }

  stringstream ss;
  ss << "\n\nSorted keys: ";
  for (auto &h : keyRanks) {
    ss << string(h.first.key) << ":" << h.second << " ";
  }
  ss << "myIntraCommParticipants: ";
  for (auto &h : myIntraCommParticipants) {
    ss << string(h.first.key) << ":" << h.second << " ";
  }
  ss << " my rank in myIntraCommParticipants is: " << myRankInIntraComm;


  vector<KeyRank> uniqueKeys;
  std::copy(keyRanks.begin(), keyRanks.end(), std::back_inserter(uniqueKeys));
  auto last = std::unique(
    uniqueKeys.begin(),
    uniqueKeys.end(),
    [=](const KeyRank& a, const KeyRank& b) {
      return std::string(a.first.key) == std::string(b.first.key);
    });

  ss << "\n--- uniqueKeys: ";
  int numPerIntercomm = -1;
  bool isCartesian = true;
  for (auto it = uniqueKeys.begin(); it != last; it++) {
    ss << string(it->first.key) << ":" << it->second << " ";

    int count = std::count_if(
      keyRanks.begin(),
      keyRanks.end(),
      [=](const KeyRank& a) {
        return string(a.first.key) == string(it->first.key);
      });
    if (numPerIntercomm == -1) { numPerIntercomm = count; }
    else if (numPerIntercomm != count) { isCartesian = false; }
  }
  cartesian = isCartesian && torch::mpi::constants::kUseCartesianCommunicator;

  // All ranks that are members of the same intraComm need to agree on the
  // same name. Name must be unique across intraComms.
  // By construction myIntraCommParticipants[0] is the perfect candidate for
  // such a name.
  intraComm = parent.Split(
    myIntraCommParticipants[0].second, myRankInIntraComm);

  // Now we build the interComm:
  // 1. if the communicator is not cartesian we only link the roots of each
  // interComm together (i.e. myRankInIntraComm == 0)
  // 2. otherwise we link each myRankInIntraComm
  // together.
  // In both cases we create a vector which maps rankInParent to
  // myRankInIntraComm
  MPI_Comm _interComm;
  vector<int> rankInParentToMyRankInIntraComm(sizeInParent);
  // Filter the group of processes to only the root of each intraComm
  parent.Allgather(
    &myRankInIntraComm,
    1,
    MPI_INT,
    const_cast<int*>(&rankInParentToMyRankInIntraComm[0]),
    1,
    MPI_INT);
  rankInParentToMyRankInIntraComm[rankInParent] = myRankInIntraComm;

  MPI::Group parentGroup = parent.Get_group();
  // Intercomm participant
  vector<int> interCommParticipants;
  int ind = 0;
  for (auto rIIC : rankInParentToMyRankInIntraComm) {
    if (rIIC == myRankInIntraComm) {
      interCommParticipants.push_back(ind);
    }
    ++ind;
  }

  int myRankInInterComm = -1;
  {
    int ind = 0;
    for (auto icp : interCommParticipants) {
      if (icp == rankInParent) {
        myRankInInterComm = ind;
      }
      ind++;
    }
  }

  if (cartesian || myRankInIntraComm == 0) {
    ss << "\n--- interCommParticipants: ";
    for (auto i : interCommParticipants) { ss << i << " "; }
    ss << "\n--- rankInParent: " << rankInParent;

    THAssert(myRankInInterComm >= 0);

    _interComm = parent.Split(
      interCommParticipants[0], myRankInInterComm);
  } else {
    // Not participating in interComm, just create a single element
    // communicator so we can ignore hierarchy.
    _interComm = parent.Split(rankInParent, 0);
  }

  interComm = MPI::Intracomm(_interComm);

  VLOG_1(ss.str());
}

Communicator::Communicator(const Communicator& orig) :
    cartesian(orig.cartesian),
    key(orig.key),
    interComm(orig.interComm.Clone()),
    intraComm(orig.intraComm.Clone())
{ }

Communicator::~Communicator() {
  interComm.Free();
  intraComm.Free();
}

bool Communicator::hasInterCollective() const {
  if (getCommunicatorLevel() <= 0) {
    THError("Communicator::hasInterCollective called from level 0!");
  }
  CommunicatorGuard cs(getCommunicatorLevel() - 1);
  return commSize(getMainThreadCommunicator().intraComm) != commSize(intraComm);
}

bool Communicator::hasIntraCollective() const {
  return commSize(intraComm) > 1;
}

string Communicator::toString() const {
  return std::string(key.key);
}

///////////////////////////////////////////////////////////////////////////
// RAII wrapper to set / unset a communicator level
///////////////////////////////////////////////////////////////////////////
CommunicatorGuard::CommunicatorGuard(int l) :
    origLevel(getCommunicatorLevel()), origType(getCommunicatorType()) {
  if (l < 0) {
    THError("Communicator level must be >= 0, requested %d\n", l);
  }
  setCommunicator(CommunicatorType::intra, l);
}

CommunicatorGuard::~CommunicatorGuard() {
  setCommunicator(origType, origLevel);
}

///////////////////////////////////////////////////////////////////////////////
// Single producer, multiple consumer lockfree thread pool and vector of
// futures to wait on
///////////////////////////////////////////////////////////////////////////////
ThreadPool& collectiveOffloadThreadPool() {
  static int poolSize = constants::kCollectiveOffloadThreadPoolSize;
  static ThreadPool collectiveOffloadThreadPool(poolSize);
  return collectiveOffloadThreadPool;
}

static std::vector<std::future<void>> collectiveFutures;

std::vector<std::future<void>> & getCollectiveFutures() {
  MAIN_THREAD_GUARD();

  if (collectiveFutures.size() >= constants::kNumAsyncCollectivesInFlight) {
    for (auto& f : collectiveFutures) { f.wait(); }
    collectiveFutures.clear();
  }
  if (collectiveFutures.size() == 0) {
    collectiveFutures.reserve(constants::kNumAsyncCollectivesInFlight);
  }
  return collectiveFutures;
}

// Given single threaded properties of lua, worst case that can happen here
// is the future queue has already been flushed. No big deal.
void syncCollectiveFuture(int futureIndex) {
  MAIN_THREAD_GUARD();

  if (futureIndex >= collectiveFutures.size()) { return ; }
  THAssert(futureIndex >= 0 && futureIndex < collectiveFutures.size());
  collectiveFutures[futureIndex].wait();
}

ThreadPool& parameterServerOffloadThreadPool() {
  static int poolSize = constants::kParameterServerOffloadThreadPoolSize;
  static ThreadPool parameterServerOffloadThreadPool(poolSize);
  return parameterServerOffloadThreadPool;
}

static std::vector<std::future<void>> parameterServerFutures;

std::vector<std::future<void>> & getParameterServerFutures() {
  MAIN_THREAD_GUARD();

  if (parameterServerFutures.size() >=
      constants::kNumAsyncParameterServersInFlight) {
    for (auto& f : parameterServerFutures) { f.wait(); }
    parameterServerFutures.clear();
  }
  if (parameterServerFutures.size() == 0) {
    parameterServerFutures.reserve(
      constants::kNumAsyncParameterServersInFlight);
  }
  return parameterServerFutures;
}

// Given single threaded properties of lua, worst case that can happen here
// is the future queue has already been flushed. No big deal.
void syncParameterServerFuture(int futureIndex) {
  MAIN_THREAD_GUARD();

  if (futureIndex >= parameterServerFutures.size()) { return ; }
  THAssert(futureIndex >= 0 && futureIndex < parameterServerFutures.size());
  parameterServerFutures[futureIndex].wait();
}

void syncAll() {
  MAIN_THREAD_GUARD();

  for (auto& f : collectiveFutures) {
    f.wait();
  }
  collectiveFutures.clear();

  for (auto& f : parameterServerFutures) {
    f.wait();
  }
  parameterServerFutures.clear();

  auto& reqs = getMPIRequests();
  for (auto& r : reqs) {
    r.Wait();
  }
  reqs.clear();
}

///////////////////////////////////////////////////////////////////////////////
// For local synchronization, avoid the cost of multithreaded MPI
///////////////////////////////////////////////////////////////////////////////
Barrier::Barrier(const MPI::Intracomm& comm) :
    ready(true), rank(commRank(comm)), size(commSize(comm))
{
  for (auto i = 0; i < size; ++i) {
    auto name = string("semaphore") + to_string(i);
    auto s = sem_open(name.c_str(),
                      O_CREAT,
                      0644,
                      0);
    THAssert(s);
    semaphores.push_back(s);
  }
}

Barrier::~Barrier() {
  for (auto i = 0; i < semaphores.size(); ++i) {
    THAssert(sem_close(semaphores[i]) == 0);
  }
  semaphores.clear();
}

Barrier* Barrier::acquire() {
  THError("Barrier unsupported atm") ;
  // static unordered_map<void*, std::vector<std::unique_ptr<Barrier>>>
  //   semaphores;
  // static mutex mut;
  // lock_guard<mutex> lg(mut);
  // int poolSize = collectiveOffloadThreadPool().size();
  // auto comm = torch::mpi::getMPICommunicator();
  // if (semaphores.find(comm) == semaphores.end()) {
  //   std::vector<std::unique_ptr<Barrier>> v;
  //   for (auto i = 0; i < poolSize; ++i) {
  //     std::unique_ptr<Barrier> b(new Barrier(comm));
  //     v.emplace_back(std::move(b));
  //   }
  //   semaphores.emplace(comm, std::move(v));
  // }
  // auto& sems = semaphores[comm];
  // for (auto& s : sems) {
  //   if (s->ready) {
  //     s->ready = false;
  //     return s.get();
  //   }
  // }
  // THAssert(sems.size() == poolSize);
  // THError("Should have found at least 1 free semaphore list!");
  // return sems.back().get();
  return 0;
}

void Barrier::release(Barrier* s) {
  THAssert(!s->ready);
  s->ready = true;
}

///////////////////////////////////////////////////////////////////////////////
// Vector of MPI::Request objects so we can offload MPI Ixyz calls directly
// from the main thread
///////////////////////////////////////////////////////////////////////////////
std::vector<MPI::Request>& getMPIRequests() {
  MAIN_THREAD_GUARD();

  static std::vector<MPI::Request> asyncRequests;
  if (asyncRequests.size() >= constants::kNumAsyncRequestsInFlight) {
    for (auto& req : asyncRequests) {
      req.Wait();
    }
    asyncRequests.clear();
  }
  if (asyncRequests.size() == 0) {
    asyncRequests.reserve(constants::kNumAsyncRequestsInFlight);
  }
  return asyncRequests;
}

// Worst case that can happen here is the  queue has already been flushed.
// No big deal.
void syncMPIRequest(int requestIndex) {
  MAIN_THREAD_GUARD();

  auto& reqs = getMPIRequests();
  if (requestIndex >= reqs.size()) { return ; }
  VLOG_1("sync request: " << requestIndex << "\n");
  reqs[requestIndex].Wait();
}

int enqueueMPIRequest(MPI::Request req) {
  MAIN_THREAD_GUARD();

  auto& reqs = getMPIRequests();
  reqs.push_back(std::move(req));
  return reqs.size() - 1;
}



///////////////////////////////////////////////////////////////////////////////
// Communication plans used in custom collectives
///////////////////////////////////////////////////////////////////////////////
// Step -> startChunk -> (sendingChunk, receivingChunk)
typedef std::vector<std::vector<std::pair<long, long>>> AllReducePlan;

template<typename PlanType> std::pair<AllReducePlan, AllReducePlan>
getPlan(size_t totalChunks,
        size_t currentIndex,
        size_t prevIndex,
        size_t commRank,
        size_t commSize) {
  static mutex mut;
  lock_guard<mutex> lg(mut);
  typedef std::tuple<size_t, size_t, size_t> Key;
  struct Hash : public std::unary_function<Key, size_t> {
    size_t operator()(const Key& k) const {
      return std::get<0>(k) ^ std::get<1>(k) ^ std::get<2>(k);
    }
  };
  struct Equal {
    bool operator()(const Key& k1, const Key& k2) const {
      return std::get<0>(k1) == std::get<0>(k2) &&
        std::get<1>(k1) == std::get<1>(k2) &&
        std::get<2>(k1) == std::get<2>(k2);
    }
  };
  static unordered_map<
    Key, std::pair<AllReducePlan, AllReducePlan>, Hash, Equal> plans;
  auto key = std::make_tuple(totalChunks, commRank, commSize);
  if (plans.find(key) == plans.end()) {
    plans[key] = std::pair<AllReducePlan, AllReducePlan>();
    auto& planReduce = plans[key].first;
    auto& planBroadcast = plans[key].second;

    for (long step = 0; step < commSize - 1; ++step) {
      THAssert(planReduce.size() == step);
      planReduce.push_back(std::vector<std::pair<long, long>>());
      auto& plan = planReduce[step];
      for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
           startChunk += (long)commSize, ++startChunkIndex) {
        long sendingChunk = -1; // Index of the sendingChunk *in prev*
        long receivingChunk = -1; // Index of the receivingChunk *in current*
        THAssert(plan.size() == startChunkIndex);
        long nChunks = std::min(commSize, totalChunks - startChunk);

        // 'Chunk + step' % commSize is the sending rank for 'chunk'
        // 'Chunk + step + 1' % commSize is the receiving rank for 'chunk'
        for (int chunk = 0; chunk < nChunks; ++chunk) {
          if ((chunk + step) % commSize == prevIndex) {
            THAssert(sendingChunk == -1);
            sendingChunk = startChunk + chunk;
         }
          if ((chunk + step + 1) % commSize == currentIndex) {
            THAssert(receivingChunk == -1);
            receivingChunk = startChunk + chunk;
          }
        }
        plan.push_back(std::pair<long, long>(sendingChunk, receivingChunk));
      }
    }

    for (long step = 0; step < commSize - 1; ++step) {
      THAssert(planBroadcast.size() == step);
      planBroadcast.push_back(std::vector<std::pair<long, long>>());
      auto& plan = planBroadcast[step];
      for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
           startChunk += (long)commSize, ++startChunkIndex) {
        long sendingChunk = -1; // Index of the sendingChunk *in prev*
        long receivingChunk = -1; // Index of the receivingChunk *in current*
        THAssert(plan.size() == startChunkIndex);
        long nChunks = std::min(commSize, totalChunks - startChunk);

        // 'Chunk + step - 1' % commSize is the sending rank for 'chunk'
        // 'Chunk + step' % commSize is the receiving rank for 'chunk'
        for (int chunk = 0; chunk < nChunks; ++chunk) {
          if ((chunk + step - 1 + commSize) % commSize == prevIndex) {
            THAssert(sendingChunk == -1);
            sendingChunk = startChunk + chunk;
          }
          if ((chunk + step) % commSize == currentIndex) {
            THAssert(receivingChunk == -1);
            receivingChunk = startChunk + chunk;
          }
        }
        plan.push_back(std::pair<long, long>(sendingChunk, receivingChunk));
      }
    }
  }
  return plans[key];
}

template std::pair<AllReducePlan, AllReducePlan>
getPlan<MpiPlan>(size_t, size_t, size_t, size_t, size_t);

template std::pair<AllReducePlan, AllReducePlan>
getPlan<CudaPlan>(size_t, size_t, size_t, size_t, size_t);

#if TORCH_MPI_CUDA

CollectiveResourcesCudaMap& collectiveResourcesCuda() {
  MAIN_THREAD_GUARD();
  static CollectiveResourcesCudaMap resources;
  return resources;
}

CollectiveResourcesCuda::CollectiveResourcesCuda(
  void* p, const Communicator* pc, const CollectiveIpcEvents &cEvents) :
    CollectiveResources(p, pc),
    events(cEvents)
#ifdef TORCH_MPI_NCCL
    , ncclComm(nullptr)
#endif
{
  MAIN_THREAD_GUARD();
}

CollectiveResourcesCuda::~CollectiveResourcesCuda() {
#ifdef TORCH_MPI_NCCL
  if (ncclComm) {
    ncclCommDestroy(*ncclComm);
    delete ncclComm;
  }
#endif
}

CollectiveResourcesCuda* acquireCollectiveResourcesCuda(
  void* dataPtr,
  Spin spin,
  WithNCCLComm nccl,
  WithGlooContext gloo,
  WithEvents e)
{
  auto& resources = collectiveResourcesCuda();
  auto pComm = &getMainThreadCommunicator();
  CollectiveResourcesKey rk(dataPtr, pComm);
  auto it = resources.find(rk);
  if (it == resources.end()) {
    auto events = e.with ?
      torch::mpi::resources::cuda::getCollectiveIPCEvents(
        constants::kNumBuffersPerCollectiveGPU, pComm->intraComm)
      :
      CollectiveIpcEvents();
    auto r = new CollectiveResourcesCuda(dataPtr, pComm, events);
    barrier(r->comm->intraComm);
    resources.emplace(rk, r);
    it = resources.find(rk);
  } else {
    ;
  }
  if (!spin.spin) {
    THAssert(!it->second->inUse);
  } else {
    int nIterations = 0;
    while (it->second->inUse) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      nIterations++;
      if (nIterations >= 100000) {
        THError("Main thread spinned for 10 seconds waiting for thread to"
                "release collective resource, this looks like a deadlock!");
      }
    }
  }

#ifdef TORCH_MPI_NCCL
  if (nccl.with && !it->second->ncclComm) {
    it->second->ncclComm =
      makeNCCLCommunicator(it->second->comm->intraComm);
  }
#endif

#ifdef TORCH_MPI_GLOO_CUDA
  if (gloo.with && !it->second->glooContext) {
    it->second->glooContext =
      makeGlooContext(it->second->comm->intraComm);
  }
#endif

  it->second->inUse = true;
  return it->second;
}

void freeCollectiveResourcesCuda() {
  auto& cr = collectiveResourcesCuda();
  for (auto it : cr) {
    delete it.second;
  }
  cr.clear();
}


namespace cuda {

///////////////////////////////////////////////////////////////////////////////
// Small collectives have high latency and are better performed copying to
// CPU memory and back
///////////////////////////////////////////////////////////////////////////////
SmallPinnedBufferProvider* SmallPinnedBufferProvider::acquire() {
  static vector<std::unique_ptr<SmallPinnedBufferProvider>> buffers;
  static mutex mut;
  lock_guard<mutex> lg(mut);
  if (buffers.size() == 0) {
    for (int i = 0; i < constants::kCollectiveOffloadThreadPoolSize; ++i) {
      std::unique_ptr<SmallPinnedBufferProvider> pb(
        new SmallPinnedBufferProvider());
      buffers.emplace_back(std::move(pb));
    }
  }
  for (auto& b : buffers) {
    if (b->ready) {
      b->ready = false;
      return b.get();
    }
  }
  THAssert(buffers.size() > 0);
  THError("Should have found at least 1 free buffer!");
  return buffers.back().get();
}

void SmallPinnedBufferProvider::release(SmallPinnedBufferProvider* buf) {
  THAssert(!buf->ready);
  buf->ready = true;
}

void* getPinnedBuffer(void* devPtr, size_t size) {
  struct Wrap {
    void* hostPtr;
    size_t size;
    Wrap(size_t s) : hostPtr(0), size(s) {
      THCudaCheck(cudaMallocHost(&hostPtr, size));
    }
    ~Wrap() {
      // THCudaCheck(cudaFreeHost(hostPtr));
    }
  };
  static std::unordered_map<void*, Wrap> map;
  static mutex mut;
  lock_guard<mutex> lg(mut);
  if (map.find(devPtr) != map.end() && map.at(devPtr).size < size) {
    // Trigger realloc
    map.erase(devPtr);
  }
  if (map.find(devPtr) == map.end()) {
    map.emplace(std::make_pair(devPtr, Wrap(size)));
  }
  return map.at(devPtr).hostPtr;
}

///////////////////////////////////////////////////////////////////////////////
// IPC descriptors are necessary for cross-process P2P transfers
///////////////////////////////////////////////////////////////////////////////
std::unordered_map<void*, std::unique_ptr<IPCDesc>>& getIPCDescs() {
  static std::unordered_map<void*, std::unique_ptr<IPCDesc>> descs;
  return descs;
}

// Sanity check that everyone is on the same machine or IPC communications
// will fail.
void IPCDesc::sanityCheck(const MPI::Intracomm& comm) {
  const int kNameLen = 1024;
  struct Name {
    char name[kNameLen];
  } name;
  memset(name.name, 0, kNameLen);
  gethostname(name.name, kNameLen);
  vector<Name> names(commSize(comm));
  names[rank_] = name;

  comm.Allgather(
    name.name, kNameLen, MPI_BYTE,
    names[0].name, kNameLen, MPI_BYTE);

  for (int i = 0; i < names.size() - 1; ++i) {
    if (string(names[i].name) != string(names[i + 1].name)) {
      THError("GPU P2P sanity check fails, not same machines %d:%s and %d:%s",
              i, names[i].name, i + 1, names[i + 1].name);
    }
  }
}

IPCDesc::IPCDesc(THCState* state, void* data, const MPI::Intracomm& comm) :
    rank_(commRank(comm)),
    size_(commSize(comm)),
    data_(data)
{
  static bool usingCachingAllocator =
      THCState_getDeviceAllocator(state) == THCCachingAllocator_get();
  VLOG_1("Rank " << rank_ << " IPCDesc ctor for data @ " << data_ << endl);
  sanityCheck(comm);

  // Prepare
  allDevices = std::vector<int>(size_);
  allDevicePointers = std::vector<void*>(size_);
  allMemHandles = std::vector<cudaIpcMemHandle_t>(size_);
  allMemHandlePtrs = std::vector<void*>(size_);
  allHostnames = std::vector<std::string>();
  for (int i = 0; i < size_; ++i) {
    std::string s(kHostnameLength, '.');
    allHostnames.push_back(s);
  }

  VLOG_1("@comm: " << &comm << " " << rank_ << "/" << size_
         << " start IPCDesc" << endl);

  // Get current device
  THCudaCheck(cudaGetDevice(&device));
  // Blocking Allgather
  comm.Allgather(&device,
                 1,
                 MPI_INT,
                 &allDevices[0],
                 1,
                 MPI_INT);
  // Place it properly for Allgather
  allDevices[rank_] = device;

  VLOG_1("@comm: " << &comm << " " << rank_ << "/" << size_
         << " got devices in IPCDesc ");
  for (auto d : allDevices) {
    VLOG_CONTINUE_1(d << " ");
  }
  VLOG_CONTINUE_1(endl);

  // Get a memhandle for local device pointer
  // This is required to be the base pointer if using THCCachingAllocator
  size_t base_size;
  void *base_ptr = !usingCachingAllocator ?
      data : THCCachingAllocator_getBaseAllocation(data, &base_size);
  std::ptrdiff_t ptr_offset = (char *)data - (char *)base_ptr;
  THCudaCheck(cudaIpcGetMemHandle(&memHandle, static_cast<void*>(base_ptr)));
  VLOG_1("@comm: " << &comm << " " <<
         rank_ << "/" << size_
         << " got memhandle " << string(memHandle.reserved)
         << " for data @" << data << endl);

  comm.Allgather(&memHandle,
                 sizeof(cudaIpcMemHandle_t),
                 MPI_BYTE,
                 &allMemHandles[0],
                 sizeof(cudaIpcMemHandle_t),
                 MPI_BYTE);
  // Place it properly for Allgather
  allMemHandles[rank_] = memHandle;

  VLOG_1("@comm: " << &comm << " " << rank_ << "/" << size_
         << " exchanged memhandles: ");
  for (auto m : allMemHandles) {
    VLOG_CONTINUE_1("str:" << string(m.reserved) << " ");
  }
  VLOG_CONTINUE_1(endl);

  int dev;
  THCudaCheck(cudaGetDevice(&dev));
  VLOG_1("@comm: " << &comm << " " << rank_ << "/" << size_
         << " on device " << dev << endl);

  // For each exported IPC mem handle, open it
  for (int i = 0; i < size_; i++) {
    if (i != rank_) {
      void* ptr;
      THCudaCheck(
        cudaIpcOpenMemHandle(&ptr,
                             allMemHandles[i],
                             cudaIpcMemLazyEnablePeerAccess));
      allMemHandlePtrs[i] = ptr;
      allDevicePointers[i] = (char *)ptr + ptr_offset;
    } else {
      // Can't open a device in the same process that exported
      allMemHandlePtrs[i] = NULL;
      allDevicePointers[i] = data;
    }
  }

  VLOG_1("@comm: " << &comm << " " << rank_ << "/" << size_
         << " opened memhandles in IPCDesc, device pointers are: ");
  for (auto p : allDevicePointers) {
    VLOG_CONTINUE_1(p << " ");
  }
  VLOG_CONTINUE_1(endl);

  // Exchange hostnames
  char allNames[kHostnameLength * size_];
  gethostname(&allNames[kHostnameLength * rank_], kHostnameLength);
  comm.Allgather(&allNames[kHostnameLength * rank_],
                 kHostnameLength * sizeof(char),
                 MPI_BYTE,
                 &allNames[0],
                 kHostnameLength * sizeof(char),
                 MPI_BYTE);

}

IPCDesc::~IPCDesc() {
  VLOG_1("Rank " << rank_ << " IPCDesc dtor for data @ " << data_ << endl);
  auto i = 0;
  for (auto dp : allMemHandlePtrs) {
    if (i != rank_) {
      VLOG_1(rank_ <<
             " close memhandle " << i << " @" << dp << " in ~IPCDesc" << endl);
      THCudaCheck(cudaIpcCloseMemHandle(dp));
    }
    ++i;
  }
}

// Threadsafe
IPCDesc* getIPCDesc(THCState* state, void* data, const MPI::Intracomm& comm) {
  static mutex mut;
  lock_guard<mutex> lg(mut);

  auto rank = commRank(comm);
  auto size = commSize(comm);
  int exportsNewMemHandle = 0;
  auto& descs = getIPCDescs();
  if (descs.find(data) == descs.end()) {
    exportsNewMemHandle = 1;
  }

  if (exportsNewMemHandle == 0) {
    return descs[data].get();
  }

  // Need to register new IPC descs, this is a collective operation.
  // Must synchronize all outstanding MPI calls so deadlocks don't occur in
  // MPI_SERIALIZED mode
  syncAll();

  VLOG_1("@comm: " << &comm << " " << "Get Desc for " << data << " found: "
         << !((bool)exportsNewMemHandle) << endl);
  VLOG_1("@comm: " << &comm << " " << "Allreduce exportsNewMemHandle\n");

  // Blocking collectives wait immediately.
  comm.Allreduce(MPI_IN_PLACE,
                 const_cast<int*>(&exportsNewMemHandle),
                 1,
                 MPI_INT,
                 MPI_SUM);
  VLOG_1("@comm: " << &comm << " " << rank << "/" << size
         << " exportsNewMemHandle count: " << exportsNewMemHandle << endl);

  if (exportsNewMemHandle < size) {
    // Need to erase descriptor in lockstep
    torch::mpi::barrier(comm);
    bool erase = false;
    if (descs.find(data) != descs.end()) {
      std::cerr
        << "WARNING: IPCDesc need to be recreated for data @" << data << endl;
      erase = true;
    }
    if (erase) { descs.erase(data); }
    torch::mpi::barrier(comm);
  }

  std::unique_ptr<IPCDesc> val(new IPCDesc(state, data, comm));
  descs.emplace(data, std::move(val));
  return descs[data].get();
}


///////////////////////////////////////////////////////////////////////////////
// High priority non-blocking stream/event pair for offloading collectives
///////////////////////////////////////////////////////////////////////////////
pair<cudaEvent_t, std::vector<cudaStream_t>> getCollectiveStreams() {
  struct Wrap {
    Wrap() {
      int hiPri, loPri;
      THCudaCheck(cudaDeviceGetStreamPriorityRange(&loPri, &hiPri));
      for (int i = 0; i < constants::kMaxNumBuffersPerCollectiveGPU; ++i) {
        streams.push_back(0);
        THCudaCheck(cudaStreamCreateWithPriority(
                      &streams.back(), cudaStreamNonBlocking, hiPri));
      }
      THCudaCheck(cudaEventCreate(&event));
    }
    ~Wrap() {
      for (auto& stream : streams) {
        THCudaCheck(cudaStreamDestroy(stream));
      }
      THCudaCheck(cudaEventDestroy(event));
    }
    cudaEvent_t event;
    std::vector<cudaStream_t> streams;
  };
  static mutex mut;
  lock_guard<mutex> lg(mut);
  static std::unordered_map<std::thread::id, Wrap> ws;
  auto tid = std::this_thread::get_id();
  auto it = ws.find(tid);
  if (it == ws.end()) {
    ws.emplace(std::piecewise_construct,
               std::forward_as_tuple(tid),
               std::forward_as_tuple());
    it = ws.find(tid);
  }
  // Allocate kMaxNumBuffersPerCollectiveGPU but only return the requested
  // kNumBuffersPerCollectiveGPU.
  return std::make_pair(
    it->second.event,
    std::vector<cudaStream_t>(
      it->second.streams.begin(),
      it->second.streams.begin() + constants::kNumBuffersPerCollectiveGPU));
}

CollectiveIpcEvents getCollectiveIPCEvents(
  size_t nEvents, const MPI::Intracomm& comm)
{
  struct Wrap {
    std::vector<std::vector<cudaEvent_t>> allEvents;
    std::vector<std::vector<cudaIpcEventHandle_t>> allEventHandles;

    Wrap(const MPI::Intracomm& comm)
    {
      int dev;
      THCudaCheck(cudaGetDevice(&dev));

      for (size_t i = 0; i < constants::kMaxNumBuffersPerCollectiveGPU; ++i) {
        allEvents.push_back(std::vector<cudaEvent_t>(commSize(comm)));
        allEventHandles.push_back(std::vector<cudaIpcEventHandle_t>(commSize(comm)));
        auto& events = allEvents.back();
        auto& eventHandles = allEventHandles.back();

        cudaEvent_t event;
        cudaIpcEventHandle_t eventHandle;
        THCudaCheck(cudaEventCreate(
           &event, cudaEventDisableTiming | cudaEventInterprocess));
        THCudaCheck(cudaIpcGetEventHandle(&eventHandle, event));
        comm.Allgather(&eventHandle,
                       sizeof(cudaIpcEventHandle_t),
                       MPI_BYTE,
                       &eventHandles[0],
                       sizeof(cudaIpcEventHandle_t),
                       MPI_BYTE);
        // For each exported IPC event handle, open it into a local event
        for (int i = 0; i < commSize(comm); i++) {
          if (i != commRank(comm)) {
            THCudaCheck(
              cudaIpcOpenEventHandle(&events[i], eventHandles[i]));
          } else {
            events[i] = event;
          }
          // Local sanity check on construction
          THCudaCheck(cudaStreamWaitEvent(0, events[i], 0));
        }
      }

    }
    ~Wrap() {
      for (auto& v : allEvents) {
        for (auto event : v) {
          THCudaCheck(cudaEventDestroy(event));
        }
      }
      // cudaIpcEventHandle_t do not get freed
    }
  };

  MAIN_THREAD_GUARD();
  static std::unordered_map<const MPI::Intracomm*, Wrap> ws;
  {
    static mutex mut;
    lock_guard<mutex> lg(mut);
    auto it = ws.find(&comm);
    if (it == ws.end()) {
      ws.emplace(&comm, comm);
      it = ws.find(&comm);
    }
    return CollectiveIpcEvents(
      it->second.allEvents.begin(),
      it->second.allEvents.begin() + constants::kNumBuffersPerCollectiveGPU);
  }
}

} // ns cuda

#endif // TORCH_MPI_CUDA


///////////////////////////////////////////////////////////////////////////////
// Synchronization
///////////////////////////////////////////////////////////////////////////////
SynchronizationHandle* synchronizationHandleFromMPIRequest(size_t reqIdx) {
  SynchronizationHandle* res = new SynchronizationHandle();
  res->hasMPIRequest = true;
  res->mpiRequestIndex = reqIdx;
  res->hasFuture = false;
  res->hasStream = false;
#ifdef TORCH_MPI_CUDA
  res->stream = reinterpret_cast<cudaStream_t>(-1);
#endif
  res->futureIndex = static_cast<size_t>(-1);
  return res;
}

SynchronizationHandle* synchronizationHandleFromFuture(size_t fIdx) {
  SynchronizationHandle* res = new SynchronizationHandle();
  res->hasFuture = true;
  res->futureIndex = fIdx;
  res->hasMPIRequest = false;
  res->hasStream = false;
#ifdef TORCH_MPI_CUDA
  res->stream = reinterpret_cast<cudaStream_t>(-1);
#endif
  res->mpiRequestIndex = static_cast<size_t>(-1);
  return res;
}

#ifdef TORCH_MPI_CUDA
SynchronizationHandle* synchronizationHandleFromStream(cudaStream_t s) {
  SynchronizationHandle* res = new SynchronizationHandle();
  res->hasStream = true;
  res->stream = s;
  res->hasFuture = false;
  res->hasMPIRequest = false;
  res->mpiRequestIndex = static_cast<size_t>(-1);;
  res->futureIndex = static_cast<size_t>(-1);
  return res;
}
#endif

SynchronizationHandle* wait(SynchronizationHandle* sh) {
  if (!sh) {
    return nullptr;
  }
  if (sh->hasMPIRequest) { syncMPIRequest(sh->mpiRequestIndex); }
  if (sh->hasFuture) { syncCollectiveFuture(sh->futureIndex); }
#ifdef TORCH_MPI_CUDA
  if (sh->hasStream) { THCudaCheck(cudaStreamSynchronize(sh->stream)); }
#endif
  delete sh;
  return nullptr;
}

ParameterServerSynchronizationHandle*
parameterServerSynchronizationHandleFromFuture(size_t fIdx) {
  ParameterServerSynchronizationHandle* res =
    new ParameterServerSynchronizationHandle();
  res->hasFuture = true;
  res->futureIndex = fIdx;
  return res;
}

ParameterServerSynchronizationHandle* wait(
    ParameterServerSynchronizationHandle* sh) {
  if (!sh) {
    return nullptr;
  }
  if (sh->hasFuture) { syncParameterServerFuture(sh->futureIndex); }
  delete sh;
  return nullptr;
}

}}} // ns torch::mpi::resources


extern "C" {

  int torchmpi_has_nccl() {
#ifdef TORCH_MPI_NCCL
    return 1;
#else
    return 0;
#endif
  }

  int torchmpi_has_gloo() {
#ifdef TORCH_MPI_GLOO
    return 1;
#else
    return 0;
#endif
  }

  int torchmpi_has_gloo_cuda() {
#ifdef TORCH_MPI_GLOO_CUDA
    return 1;
#else
    return 0;
#endif
  }
}
