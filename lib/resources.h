/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include <semaphore.h>

#include <future>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef TORCH_MPI_CUDA
#include <THC.h>
#endif

#include "constants.h"
#include "torch_mpi.h"
#include "thread_pool-in.h"

#ifdef TORCH_MPI_NCCL
#include "nccl.h"

#define NCCLCHECK(cmd) do {                             \
    ncclResult_t r = cmd;                               \
    if (r!= ncclSuccess) {                              \
      printf("NCCL failure %s:%d '%s'\n",               \
             __FILE__,__LINE__,ncclGetErrorString(r));  \
      exit(EXIT_FAILURE);                               \
    }                                                   \
  } while(0)
#endif

#ifdef TORCH_MPI_GLOO
#include <gloo/mpi/context.h>
#endif

#define DEBUG 0

#if DEBUG
#define VLOG_1(x) std::cout << std::this_thread::get_id() << " " << x;
#define VLOG_CONTINUE_1(x) std::cout << " " << x;
#define VLOG_2(x) ;
#else
#define VLOG_1(x) ;
#define VLOG_CONTINUE_1(x) ;
#define VLOG_2(x) ;
#endif

namespace torch { namespace mpi { namespace resources {

///////////////////////////////////////////////////////////////////////////
// Tags to separate between regular collectives, client/server exchanges //
// and signalling                                                        //
///////////////////////////////////////////////////////////////////////////
// Used for collectives, no notion of client or server here.
constexpr size_t kDefaultTag = 0;
// Need to distinguish between the two because clients should only
// talk to servers or pain will ensue.
constexpr size_t kServerChunkTag =
  1000 + // Just add some slack
  constants::kMaxNumBuffersPerCollectiveCPU +
  constants::kMaxNumBuffersPerCollectiveGPU + 1;
constexpr size_t kClientChunkTag = kServerChunkTag + 1;
// The following are always sent by client, consumed by server thread.
constexpr size_t kRuleTag = kClientChunkTag + 1;
constexpr size_t kTriggerServerSendShardTag = kRuleTag + 1;
constexpr size_t kSentinelTag = kTriggerServerSendShardTag + 1;

constexpr bool kDebugFlag = (bool)DEBUG;
constexpr int kCommunicatorKeyLen = 1024;

///////////////////////////////////////////////////////////////////////////////
// Main entry point to get resources for a collective
///////////////////////////////////////////////////////////////////////////////

struct Communicator;

struct CollectiveResources {
  bool inUse;
  Communicator* comm;
  void* ptr;
#ifdef TORCH_MPI_GLOO
  std::shared_ptr<gloo::mpi::Context> glooContext;
#endif
  CollectiveResources(void*, const Communicator*);
  ~CollectiveResources();
};

struct CollectiveResourcesKey {
  void* ptr;
  const Communicator* pComm;
  CollectiveResourcesKey(void* p, const Communicator* pc) :
      ptr(p), pComm(pc) {}
};

struct CollectiveResourcesHash :
      public std::unary_function<CollectiveResourcesKey, size_t> {
  size_t operator()(const CollectiveResourcesKey& k) const {
    return reinterpret_cast<size_t>(k.ptr);
  }
};

struct CollectiveResourcesEqual {
  bool operator()(const CollectiveResourcesKey& k1,
                  const CollectiveResourcesKey& k2) const {
    return k1.ptr == k2.ptr && k1.pComm == k2.pComm;
  }
};

typedef std::unordered_map<CollectiveResourcesKey,
                           CollectiveResources*,
                           CollectiveResourcesHash,
                           CollectiveResourcesEqual> CollectiveResourcesMap;

CollectiveResourcesMap& collectiveResources();

struct Spin {
  bool spin;
  explicit Spin(bool s = false) : spin(s) {}
};
struct WithNCCLComm {
  bool with;
  explicit WithNCCLComm(bool w = false) : with(w) {}
};
struct WithGlooContext {
  bool with;
  explicit WithGlooContext(bool w = false) : with(w) {}
};
struct WithEvents {
  bool with;
  explicit WithEvents(bool w = false) : with(w) {}
};
CollectiveResources* acquireCollectiveResources(
  void* dataPtr,
  Spin s = Spin(),
  WithNCCLComm n = WithNCCLComm(),
  WithGlooContext g = WithGlooContext(),
  WithEvents e = WithEvents());
void releaseCollectiveResources(CollectiveResources* r);
void freeCollectiveResources();

#ifdef TORCH_MPI_CUDA
typedef std::vector<std::vector<cudaEvent_t>> CollectiveIpcEvents;

struct CollectiveResourcesCuda : CollectiveResources {
  CollectiveIpcEvents events;
#ifdef TORCH_MPI_NCCL
  ncclComm_t* ncclComm;
#endif
  CollectiveResourcesCuda(void*, const Communicator*, const CollectiveIpcEvents &);
  ~CollectiveResourcesCuda();
};

typedef std::unordered_map<CollectiveResourcesKey,
                           CollectiveResourcesCuda*,
                           CollectiveResourcesHash,
                           CollectiveResourcesEqual> CollectiveResourcesCudaMap;

CollectiveResourcesCudaMap& collectiveResourcesCuda();

CollectiveResourcesCuda* acquireCollectiveResourcesCuda(
  void* dataPtr,
  Spin s = Spin(),
  WithNCCLComm n = WithNCCLComm(),
  WithGlooContext g = WithGlooContext(),
  WithEvents e = WithEvents());

void freeCollectiveResourcesCuda();
#endif

///////////////////////////////////////////////////////////////////////////////
// CommunicatorKey structure used to create communicators
// Any string can be used for key, 1024 should be ample enough room
// Communicators are based on ordered pair<CommunicatorKey, rank>
// Typically one would use strings based on the proper topology and create as
// many communicators as necessary for expressing multiLayer parameterserver
// with synchronous SGD as leaves.
///////////////////////////////////////////////////////////////////////////////
struct CommunicatorKey {
  char key[kCommunicatorKeyLen];

  CommunicatorKey();
  CommunicatorKey(const CommunicatorKey& other);
  CommunicatorKey& operator=(const CommunicatorKey& other);
  static CommunicatorKey fromString(std::string s);
};

///////////////////////////////////////////////////////////////////////////////
// We split a communicator along an intercomm and an intracomm arbitrarily
// based on equality of pair<string, parent rank> keys.
///////////////////////////////////////////////////////////////////////////////
struct Communicator {
  bool cartesian;
  CommunicatorKey key;

  MPI::Intracomm interComm;
  MPI::Intracomm intraComm;

  Communicator(const MPI::Intracomm& parent, CommunicatorKey key);
  Communicator(const Communicator& orig);
  ~Communicator();
  bool hasInterCollective() const;
  bool hasIntraCollective() const;
  std::string toString() const;
};

enum struct CommunicatorType { inter = 0, intra = 1};

///////////////////////////////////////////////////////////////////////////
// RAII wrapper to set / unset a communicator level
///////////////////////////////////////////////////////////////////////////
struct CommunicatorGuard {
  int origLevel;
  CommunicatorType origType;
  explicit CommunicatorGuard(int l);
  ~CommunicatorGuard();
};

///////////////////////////////////////////////////////////////////////////////
// Synchronization
///////////////////////////////////////////////////////////////////////////////
extern "C" {

  typedef struct SynchronizationHandle {
    bool hasMPIRequest;
    bool hasFuture;
    bool hasStream;
#ifdef TORCH_MPI_CUDA
    cudaStream_t stream;
#endif
    size_t mpiRequestIndex;
    size_t futureIndex;
  } SynchronizationHandle;

  typedef struct ParameterServerSynchronizationHandle {
    bool hasFuture;
    size_t futureIndex;
  } ParameterServerSynchronizationHandle;

}

SynchronizationHandle* synchronizationHandleFromMPIRequest(size_t);
SynchronizationHandle* synchronizationHandleFromFuture(size_t);
#ifdef TORCH_MPI_CUDA
SynchronizationHandle* synchronizationHandleFromStream(cudaStream_t);
#endif
SynchronizationHandle* wait(SynchronizationHandle*);

ParameterServerSynchronizationHandle*
parameterServerSynchronizationHandleFromFuture(size_t);
ParameterServerSynchronizationHandle* wait(ParameterServerSynchronizationHandle*);

///////////////////////////////////////////////////////////////////////////////
// Single producer, multiple consumer lockfree thread pool and vector of
// futures to wait on
///////////////////////////////////////////////////////////////////////////////
ThreadPool& collectiveOffloadThreadPool();
ThreadPool& parameterServerOffloadThreadPool();

std::vector<std::future<void>>& getCollectiveFutures();
std::vector<std::future<void>>& getParameterServerFutures();

void syncAll();

///////////////////////////////////////////////////////////////////////////////
// For local synchronization, avoid the cost of multithreaded MPI
///////////////////////////////////////////////////////////////////////////////
struct Barrier {
  bool ready;
  int rank;
  int size;
  std::vector<sem_t*> semaphores;

  explicit Barrier(const MPI::Intracomm& comm);
  ~Barrier();
  static Barrier* acquire();
  static void release(Barrier* s);

  inline void barrier() {
    _halfBarrier();
    _halfBarrier();
  }

 private:
  inline void _halfBarrier() {
    if (rank == 0) {
      sem_post(semaphores[rank]);
      sem_wait(semaphores[size - 1]);
    } else {
      sem_wait(semaphores[rank - 1]);
      sem_post(semaphores[rank]);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Vector of MPI::Request objects so we can offload MPI Ixyz calls directly
// from the main thread
///////////////////////////////////////////////////////////////////////////////
std::vector<MPI::Request>& getMPIRequests();
int enqueueMPIRequest(MPI::Request req);
// void syncMPIRequest(int requestIndex);

///////////////////////////////////////////////////////////////////////////////
// Communication plans used in custom collectives
///////////////////////////////////////////////////////////////////////////////
// Step -> startChunk -> (sendingChunk, receivingChunk)
struct MpiPlan {};
struct CudaPlan {};

typedef std::vector<std::vector<std::pair<long, long>>> AllReducePlan;

template<typename PlanType> std::pair<AllReducePlan, AllReducePlan> getPlan(
size_t totalChunks, size_t currentIndex, size_t prevIndex, size_t commRank, size_t commSize);

#ifdef TORCH_MPI_CUDA

namespace cuda {

struct SmallPinnedBufferProvider {
  bool ready;
  void* data;
  SmallPinnedBufferProvider() : ready(true), data(nullptr) {
    THCudaCheck(cudaMallocHost(
      &data,
      std::max(constants::kSmallBcastSizeGPU,
               constants::kSmallAllreduceSizeGPU) * sizeof(double)));
  }
  ~SmallPinnedBufferProvider() {
    THCudaCheck(cudaFreeHost(data));
  }
  void copyFrom(void* from, size_t size, cudaStream_t stream = 0) {
    THCudaCheck(cudaMemcpyAsync(data, from, size, cudaMemcpyDefault, stream));
  }
  void copyTo(void* to, size_t size, cudaStream_t stream = 0) {
    THCudaCheck(cudaMemcpyAsync(to, data, size, cudaMemcpyDefault, stream));
  }
  static SmallPinnedBufferProvider* acquire();
  static void release(SmallPinnedBufferProvider* buf);
};

// Given a cuda buffer, get a reusable matching pinned buffer of the same size
// Wasteful but works in a first approximation
void* getPinnedBuffer(void* devPtr, size_t elemSize);

///////////////////////////////////////////////////////////////////////////////
// IPC descriptors are necessary for cross-process P2P transfers
///////////////////////////////////////////////////////////////////////////////
struct IPCDesc {
  int rank_;
  int size_;
  void* data_;

  // Local information related to local device pointer
  int device;
  cudaIpcMemHandle_t memHandle;

  // Need open handle
  static constexpr int kHostnameLength = 1024;
  std::vector<int> allDevices;
  std::vector<void*> allDevicePointers;
  std::vector<std::string> allHostnames;

  // Need Allgather
  std::vector<cudaIpcMemHandle_t> allMemHandles;

  // for cleanup: pointers returned from cudaIpcOpenMemHandle.  If using a
  // caching allocator, these differ from the device pointers by an offset
  // into the cache.
  std::vector<void *> allMemHandlePtrs;

  // Sanity check that everyone is on the same machine or IPC communications
  // will fail.
  void sanityCheck(const MPI::Intracomm& comm);
  IPCDesc(THCState* state, void* data, const MPI::Intracomm& comm);
  ~IPCDesc();
};

std::unordered_map<void*, std::unique_ptr<IPCDesc>>& getIPCDescs();
IPCDesc* getIPCDesc(THCState* state, void* data, const MPI::Intracomm& comm);


///////////////////////////////////////////////////////////////////////////////
// High priority non-blocking stream/event pair for offloading collectives
///////////////////////////////////////////////////////////////////////////////
std::pair<cudaEvent_t, std::vector<cudaStream_t>> getCollectiveStreams();
std::vector<std::vector<cudaEvent_t>> getCollectiveIPCEvents(
  size_t nEvents, const MPI::Intracomm& comm);

} // ns cuda

#endif // TORCH_MPI_CUDA

}}}
