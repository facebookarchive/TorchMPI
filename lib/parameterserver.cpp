/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include <iostream>

#include "torch_mpi.h"

#include "TH.h"

#ifdef TORCH_MPI_CUDA
#include "torch_mpi_cuda.h"

#include "THC.h"
#endif

#include <cassert>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

//#define DEBUG_MPI 1

#ifdef DEBUG_MPI
std::mutex iomut;

#define DEBUG_LOG(msg)                          \
  {                                             \
    lock_guard<mutex> lg(iomut);                \
    cout << msg;                                \
  }

#define DEBUG_LOG_PS(src, dst, msgtype, tag)                            \
  {                                                                     \
    lock_guard<mutex> lg(iomut);                                        \
    cout << "thread: " << std::this_thread::get_id() << " rk: "         \
         << src << " " << msgtype << " " << dst << " tag: " << tag      \
         << " parameterserver @" << (size_t)(this) << "\n"  ;           \
  }
#else
#define DEBUG_LOG(msg)                                \
  ;
#define DEBUG_LOG_PS(src, dst, msgtype, tag)          \
  ;
#endif

#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)

/**********************************************************************
 ******************************** MPI *********************************
 **********************************************************************/
using namespace std;
using namespace torch::mpi::resources;

namespace torch { namespace mpi {

///////////////////////////////////////////////////////////////////////////
// Types Helpers
///////////////////////////////////////////////////////////////////////////
enum class THEnumType {
  Undefined = 0,
    FloatTensor = 1,
    DoubleTensor = 2,
#ifdef TORCH_MPI_CUDA
    CudaTensor = 10,
#endif
    Sentinel = 20
};

MPI::Datatype mpiTypeFromScalarType(float) { return MPI::Datatype(MPI_FLOAT); }
MPI::Datatype mpiTypeFromScalarType(double) { return MPI::Datatype(MPI_DOUBLE); }

MPI::Datatype mpiTypeFromEnumType(enum THEnumType et) {
  if (et == THEnumType::FloatTensor) { return MPI::Datatype(MPI_FLOAT); }
  if (et == THEnumType::DoubleTensor) { return MPI::Datatype(MPI_DOUBLE); }
#ifdef TORCH_MPI_CUDA
  if (et == THEnumType::CudaTensor) { return MPI::Datatype(MPI_FLOAT); }
#endif
  THError("Unsupported enum THEnumType");
  return MPI::Datatype(MPI_CHAR);
}

float scalarTypeFromTHTensorType(THFloatTensor* t) { return (float)0;}
double scalarTypeFromTHTensorType(THDoubleTensor* t) { return (double)0;}

template<typename THTensorType>
THEnumType enumTHTypeFromTHTensorType() {
  return THEnumType::Undefined;
}
template<> THEnumType enumTHTypeFromTHTensorType<THFloatTensor>() {
  return THEnumType::FloatTensor;
}
template<> THEnumType enumTHTypeFromTHTensorType<THDoubleTensor>() {
  return THEnumType::DoubleTensor;
}
#ifdef TORCH_MPI_CUDA
float scalarTypeFromTHTensorType(THCudaTensor* t) { return (float)0;}

template<> THEnumType enumTHTypeFromTHTensorType<THCudaTensor>() {
  return THEnumType::CudaTensor;
}
#endif


///////////////////////////////////////////////////////////////////////////
// Update Rules
///////////////////////////////////////////////////////////////////////////
struct BaseUpdateRule;
static std::vector<BaseUpdateRule*>& supportedUpdateRules();

struct BaseUpdateRule {
  std::string requested;

  BaseUpdateRule() : requested("none") {}

  BaseUpdateRule(std::string n) : requested(n) {}

  virtual void apply(THFloatTensor* local, THFloatTensor* received) {
    std::string err("Must override BaseUpdateRule CPU: ");
    err = err + name();
    THError(err.c_str());
  }
  virtual void apply(THDoubleTensor* local, THDoubleTensor* received) {
    std::string err("Must override BaseUpdateRule CPU: ");
    err = err + name();
    THError(err.c_str());
  }

#ifdef TORCH_MPI_CUDA
  virtual void apply(THCState* state, THCudaTensor* local, THCudaTensor* received) {
    std::string err("Must override BaseUpdateRule GPU: ");
    err = err + name();
    THError(err.c_str());
  }
#endif
  virtual string name() { return requested; };
  virtual ~BaseUpdateRule() {}
};

// Zero
struct UpdateRuleZero : public BaseUpdateRule {
  UpdateRuleZero() {}
  string name() override { return string("zero"); }
  void apply(THFloatTensor* local, THFloatTensor* received) override {
    torch::th::fill(local, static_cast<float>(0));
  }
  void apply(THDoubleTensor* local, THDoubleTensor* received) override {
    torch::th::fill(local, static_cast<double>(0));
  }
#ifdef TORCH_MPI_CUDA
  void apply(THCState* state, THCudaTensor* local, THCudaTensor* received) override {
    torch::thc::fill(state, local, static_cast<float>(0));
  }
#endif
};

// Copy
struct UpdateRuleCopy : public BaseUpdateRule {
  UpdateRuleCopy() {}
  string name() override { return string("copy"); }
  void apply(THFloatTensor* local, THFloatTensor* received) override {
    torch::th::copy(local, received);
  }
  void apply(THDoubleTensor* local, THDoubleTensor* received) override {
    torch::th::copy(local, received);
  }
#ifdef TORCH_MPI_CUDA
  void apply(THCState* state, THCudaTensor* local, THCudaTensor* received) override {
    torch::thc::copy(state, local, received);
  }
#endif
};

// Add
struct UpdateRuleAdd : public BaseUpdateRule {
  UpdateRuleAdd() {}
  string name() override { return string("add"); }
  void apply(THFloatTensor* local, THFloatTensor* received) override {
    torch::th::cadd<float, THFloatTensor>(local, received);
  }
  void apply(THDoubleTensor* local, THDoubleTensor* received) override {
    torch::th::cadd<double, THDoubleTensor>(local, received);
  }
#ifdef TORCH_MPI_CUDA
  void apply(THCState* state, THCudaTensor* local, THCudaTensor* received) override {
    torch::thc::cadd<float, THCudaTensor>(state, local, received);
  }
#endif
};

static std::vector<BaseUpdateRule*>& supportedUpdateRules() {
  static bool inited = false;
  static std::vector<BaseUpdateRule*> rules;
  if (!inited) {
    rules.push_back(new UpdateRuleZero());
    rules.push_back(new UpdateRuleCopy());
    rules.push_back(new UpdateRuleAdd());
    // TODO: More rules
    inited = true;
  }
  return rules;
}

void freeUpdateRules(BaseUpdateRule* r) {
  auto& rules = supportedUpdateRules();
  for (auto r : rules) {
    delete r;
  }
  rules.clear();
}

///////////////////////////////////////////////////////////////////////////
// Helper functions
///////////////////////////////////////////////////////////////////////////
int nextParameterServerInstance() {
  static mutex mut;
  lock_guard<mutex> lg(mut);
  static int g_parameterServerInstance = 0;
  return ++g_parameterServerInstance;
}

bool& terminateParameterServerThread() {
  static bool finalize = false;
  return finalize;
}

///////////////////////////////////////////////////////////////////////////
// DistributedParameterServer
///////////////////////////////////////////////////////////////////////////
struct DistributedParameterServer {
  int parameterServerInstance_; // MPI_Isend / MPI_IRecv tag disambiguation
  size_t nElement_;
  const CollectiveResources* comm_;

  enum THEnumType enumType_;

  void* localShard_;
  void* localShardReceive_;
  long startOffset_;
  long shardSize_;

  template<typename ScalarType, typename THTensorType>
  DistributedParameterServer(ScalarType* data, THTensorType* t, size_t nElement) :
      parameterServerInstance_(nextParameterServerInstance()),
      nElement_(nElement),
      comm_(acquireCollectiveResources(data)) {
    enumType_ = enumTHTypeFromTHTensorType<THTensorType>();

    auto p = getRange(commRank(comm_->comm->intraComm));
    startOffset_ = p.first;
    shardSize_ = p.second;

    // Point directly in the storage that we retained
    localShard_ = (void*) malloc(shardSize_ * sizeof(ScalarType));
    memcpy(localShard_, data + startOffset_, shardSize_ * sizeof(ScalarType));

    // TODO: cudaMalloc when necessary
    // leave unset
    localShardReceive_ = (void*) malloc(shardSize_ * sizeof(ScalarType));
    memset(localShardReceive_,
           255,
           shardSize_ * sizeof(ScalarType));

  }

  ~DistributedParameterServer() {
    free(localShard_);
    free(localShardReceive_);
  }

  std::pair<size_t, size_t> getRange(size_t rank) {
    // Every distributed shard has at least this many elements (floor)
    size_t commonShardSize = nElement_ / commSize(comm_->comm->intraComm);
    // There may exist remainder elements, assign 1 to each distributed shard
    // starting from 0 until we exhaust them all.
    size_t remainder = nElement_ - commonShardSize * commSize(comm_->comm->intraComm);
    assert(0 <= remainder && remainder < commSize(comm_->comm->intraComm));
    size_t shardSize = (rank < remainder) ?
      commonShardSize + 1 : commonShardSize;
    size_t startOffset =
      commonShardSize * rank + std::min(remainder, (size_t)(rank));
    return std::pair<size_t, size_t>(startOffset, shardSize);
  }

  size_t thisParameterServerTag(size_t tag) {
    if (tag > resources::kSentinelTag) {
      THError("Invalid MPI base tag %td\n", tag);
    }
    return parameterServerInstance_ * resources::kSentinelTag + tag;
  }

  std::string toString() {
    return std::string("DDD ") +
      std::to_string(static_cast<int>(enumType_));
  }

  // Client-side send with an update rule
  template<typename ScalarType>
  ParameterServerSynchronizationHandle* clientSend(ScalarType* data,
                                    const char* updateRuleName) {
    auto& futures = getParameterServerFutures();
    futures.push_back(
      parameterServerOffloadThreadPool().enqueue([=](){

          std::vector<MPI::Request> requests;
          for (int rank = 0; rank < commSize(comm_->comm->intraComm); ++rank) {
            DEBUG_LOG_PS(commRank(comm_->comm->intraComm),
                         rank,
                         "client post Isend rulename to",
                         thisParameterServerTag(kRuleTag));
            requests.push_back(
              comm_->comm->intraComm.Isend(updateRuleName,
                           strlen(updateRuleName),
                           MPI_CHAR,
                           rank,
                           thisParameterServerTag(resources::kRuleTag))
            );
          }

          for (int rank = 0; rank < commSize(comm_->comm->intraComm); ++rank) {
            DEBUG_LOG_PS(commRank(comm_->comm->intraComm),
                         rank,
                         "client post Isend data to",
                         thisParameterServerTag(kClientChunkTag));

            auto range = getRange(rank);
            requests[rank].Wait();
            // Ssend is important here, it lets requests be received in order and
            // allows to synchronize a wait(ParameterServerSynchronizationHandle) with an
            // enclosing Barrier. Be sure to never modify this without thinking very carefully.
            comm_->comm->intraComm.Ssend(
              data + range.first,
              range.second,
              mpiTypeFromScalarType(*data),
              rank,
              thisParameterServerTag(resources::kClientChunkTag));
          }

        }));
    return resources::parameterServerSynchronizationHandleFromFuture(
      futures.size() - 1);
  }

  // Client-side receive, asynchronously for latency hiding
  template<typename ScalarType>
  ParameterServerSynchronizationHandle* clientReceive(ScalarType* data) {
    auto& futures = getParameterServerFutures();
    futures.push_back(
      parameterServerOffloadThreadPool().enqueue([=](){
        std::vector<MPI::Request> requests;
        for (int rank = 0; rank < commSize(comm_->comm->intraComm); ++rank) {
          DEBUG_LOG_PS(commRank(comm_->comm->intraComm),
                       rank,
                       "post client Irecv from",
                       thisParameterServerTag(kServerChunkTag));
          auto range = getRange(rank);
          requests.push_back(
            std::move(
              comm_->comm->intraComm.Irecv(data + range.first,
                           range.second,
                           mpiTypeFromScalarType(*data),
                           rank,
                           thisParameterServerTag(resources::kServerChunkTag))
            )
          );
        }

        // Non-blocking sends of 1 byte to trigger the remote sends from
        // all the servers
        for (int rank = 0; rank < commSize(comm_->comm->intraComm); ++rank) {
          char trigger = 0;
          DEBUG_LOG_PS(commRank(comm_->comm->intraComm),
                       rank,
                       "post client Send trigger to",
                       thisParameterServerTag(resources::kTriggerServerSendShardTag));
          // Ignore the status, we'll wait on the Irecv thanks to transitive
          // dependencies via the server thread.
          comm_->comm->intraComm.Send(&trigger,
                      1,
                      MPI_CHAR,
                      rank,
                      thisParameterServerTag(resources::kTriggerServerSendShardTag));
        }

        for (auto& r : requests) { r.Wait(); }
    }));
    return resources::parameterServerSynchronizationHandleFromFuture(
      futures.size() - 1);
  }

  // Post a server receive which will handle all the pending messages
  // asynchronously
  void serverReceive() {
    // Iterate until probe returns empty
    while (1) {
      MPI::Status s;

      bool msg = comm_->comm->intraComm.Iprobe(
        MPI_ANY_SOURCE, thisParameterServerTag(resources::kRuleTag), s);
      if (!msg) {
        msg = comm_->comm->intraComm.Iprobe(
          MPI_ANY_SOURCE,
          thisParameterServerTag(resources::kTriggerServerSendShardTag),
          s);
      }

      if (!msg) {
        return;
      }

      DEBUG_LOG("thread: " << std::this_thread::get_id() <<
        " Server got from " << s.Get_source() << " tag: " << s.Get_tag() <<
        "\n");

      if (s.Get_tag() == thisParameterServerTag(resources::kRuleTag)) {
        // Regular path, receive a message and updateRule
        constexpr int kRuleNameLength = 1024;
        char updateRuleName[kRuleNameLength];
        memset(updateRuleName, 0, kRuleNameLength);
        if (s.Get_source() >= kRuleNameLength) {
          THError("Error: rule name too long (%d max)", kRuleNameLength);
        }

        DEBUG_LOG_PS(commRank(comm_->comm->intraComm),
                     s.Get_source(),
                     "server receive updateRule from",
                     thisParameterServerTag(kRuleTag));

        // 1. Get the update rule we just probed for
        comm_->comm->intraComm.Recv(updateRuleName,
                    s.Get_count(MPI_CHAR),
                    MPI_CHAR,
                    s.Get_source(),
                    thisParameterServerTag(resources::kRuleTag));

        BaseUpdateRule* updateRule = nullptr;
        for (auto r : supportedUpdateRules()) {
          if (string(updateRuleName) == string(r->name())) {
            updateRule = r;
            break;
          }
        }
        THAssert(updateRule);

        // 2. Get the chunk and apply the update rule.
        DEBUG_LOG_PS(commRank(comm_->comm->intraComm),
                     s.Get_source(),
                     "server receive from",
                     thisParameterServerTag(kClientChunkTag));
        comm_->comm->intraComm.Recv(
          localShardReceive_,
          shardSize_,
          mpiTypeFromEnumType(enumType_),
          s.Get_source(), // same source as what the probe said
          thisParameterServerTag(resources::kClientChunkTag));

        // Ugly but that's as much as I am willing to do atm ...
        {
#define UPDATE_RULE(EnumT, ScalarT, TensorT)                    \
          if (enumType_ == EnumT ) {                            \
            auto l = PPCAT(TensorT, Tensor_newWithStorage1d)(   \
              PPCAT(TensorT, Storage_newWithData)(              \
                (ScalarT*)localShard_, shardSize_),             \
              0,                                                \
              shardSize_,                                       \
              1);                                               \
            auto r = PPCAT(TensorT, Tensor_newWithStorage1d)(   \
              PPCAT(TensorT, Storage_newWithData)(              \
                (ScalarT*)localShardReceive_, shardSize_),      \
              0,                                                \
              shardSize_,                                       \
              1);                                               \
            updateRule->apply(l, r);                            \
            PPCAT(TensorT, Tensor_free)(l);                     \
            PPCAT(TensorT, Tensor_free)(r);                     \
            matched = true;                                     \
          }

          // Perform local update
          bool matched = false;
          UPDATE_RULE(THEnumType::FloatTensor, float, THFloat);
          UPDATE_RULE(THEnumType::DoubleTensor, double, THDouble);
          if (!matched) {
            THError("NYI: Unsupported type for UpdateRule");
          }

#undef UPDATE_RULE
        }
      } else if (s.Get_tag() ==
                 thisParameterServerTag(resources::kTriggerServerSendShardTag)) {

        DEBUG_LOG_PS(commRank(comm_->comm->intraComm),
                     s.Get_source(),
                     "server receive from",
                     thisParameterServerTag(kTriggerServerSendShardTag));

        // Regular path, receive a message and update
        char c;
        // 1. Receive signal from a client who requests the data
        // We already waited asynchronously above, we can just receive
        // synchronously now.
        comm_->comm->intraComm.Recv(&c,
                    1,
                    MPI_CHAR,
                    s.Get_source(),
                    thisParameterServerTag(
                      resources::kTriggerServerSendShardTag));

        DEBUG_LOG_PS(commRank(comm_->comm->intraComm),
                     s.Get_source(),
                     "server send localShard_ to",
                     thisParameterServerTag(kServerChunkTag));

        // 2. Perform the send requested with the signal
        // Do a synchronous send because we need to be sure the send
        // completed before we can modify the underlying buffer
        // (WAW and WAR dependencies).
        comm_->comm->intraComm.Send(localShard_,
                    shardSize_,
                    mpiTypeFromEnumType(enumType_),
                    s.Get_source(),
                    thisParameterServerTag(resources::kServerChunkTag));

        DEBUG_LOG_PS(commRank(comm_->comm->intraComm),
                     s.Get_source(),
                     "server done send localShard with val:",
                     *((float*)localShard_));
      }
    }
  }

};

static mutex PSMutex;

std::unordered_set<DistributedParameterServer*>&
parameterServers() {
  static std::unordered_set<DistributedParameterServer*> servers;
  return servers;
}


void launchParameterServer();

///////////////////////////////////////////////////////////////////////////
// DistributedParameterServer constructors
///////////////////////////////////////////////////////////////////////////
template<typename ScalarType, typename THTensorType>
DistributedParameterServer* newDistributedParameterServer(THTensorType* t) {
  if (!torch::th::isContiguous(t)) {
    THError("NYI: DistributedParameterServer only for contig tensors");
  }

  DistributedParameterServer* res;
  {
    lock_guard<mutex> lg(PSMutex);
    auto& servers = parameterServers();

    res = new DistributedParameterServer(
      torch::th::data<ScalarType, THTensorType>(t),
      t,
      torch::th::nElement(t));
    DEBUG_LOG("New DistributedParameterServer (instance: " <<
              res->parameterServerInstance_ << ")@" << res <<"\n");
    servers.insert(res);
    launchParameterServer();
  }

  return res;
}

#ifdef TORCH_MPI_CUDA

template<typename ScalarType, typename THTensorType>
DistributedParameterServer* newDistributedParameterServer(
    THCState* state, THTensorType* t) {
  if (!torch::thc::isContiguous(state, t)) {
    THError("NYI: DistributedParameterServer only for contig tensors");
  }

  DistributedParameterServer* res;
  {
    lock_guard<mutex> lg(PSMutex);
    auto& servers = parameterServers();
    res = new DistributedParameterServer(
      torch::thc::data<ScalarType, THTensorType>(state, t),
      t,
      torch::thc::nElement(state, t));
    servers.insert(res);
    launchParameterServer();
  }

  return res;
}

#endif


///////////////////////////////////////////////////////////////////////////
// DistributedParameterServer ptr holders and support for single helper
// thread scanning all available DistributedParameterServer and calling
// serverReceive on them.
///////////////////////////////////////////////////////////////////////////
void freeParameterServer(DistributedParameterServer* dps) {
  lock_guard<mutex> lg(PSMutex);
  auto& servers = parameterServers();
  servers.erase(dps);
  delete dps;
}

void freeParameterServers() {
  lock_guard<mutex> lg(PSMutex);
  auto& servers = parameterServers();
  for (auto dps : servers) {
    delete dps;
  }
  servers.clear();
}

// Call before joining the threadpool, not
void setTerminateParameterServerThread() {
  terminateParameterServerThread() = true;
}

std::thread& parameterServerThread() {
  static std::thread driver;
  return driver;
}

void launchParameterServer() {
  static bool inited = false;
  if (!inited) {
    inited = true;
    auto& driver = parameterServerThread();
    THAssert(!driver.joinable());
    driver = std::thread([=](){
      while(1) {
        {
          lock_guard<mutex> lg(PSMutex);
          if (terminateParameterServerThread()) {
            return;
          }
          auto& servers = parameterServers();
          for (auto dps : servers) {
            dps->serverReceive();
          }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });
  }
}

}} // ns torch::mpi


/**********************************************************************
 *********************** C Wrapper definitions ************************
 **********************************************************************/

using namespace torch::mpi;

extern "C" {

// TODO: instantiate for other types later
#define PS_INIT(ScalarType, THTensorType)                                \
  void* PPCAT(torchmpi_parameterserver_init_, THTensorType)             \
    (THTensorType* t) {                                                 \
    barrier(getMainThreadCommunicator().intraComm);                            \
    auto res = newDistributedParameterServer<ScalarType, THTensorType>(t); \
    barrier(getMainThreadCommunicator().intraComm);                            \
    return res;                                                         \
  }

PS_INIT(float, THFloatTensor);
PS_INIT(double, THDoubleTensor);

#define PS_SEND(ScalarType, THTensorType)                                     \
  ParameterServerSynchronizationHandle* PPCAT(torchmpi_parameterserver_send_, THTensorType) \
    (void* ps, THTensorType* t, const char* updateRuleName) {                \
    return reinterpret_cast<DistributedParameterServer*>(ps)->clientSend(    \
      torch::th::data<ScalarType, THTensorType>(t), updateRuleName);         \
  }

PS_SEND(float, THFloatTensor);
PS_SEND(double, THDoubleTensor);

#define PS_RECEIVE(ScalarType, THTensorType)                                \
  ParameterServerSynchronizationHandle* PPCAT(torchmpi_parameterserver_receive_, THTensorType) \
    (void* ps, THTensorType* t) {                                       \
    return reinterpret_cast<DistributedParameterServer*>(ps)->          \
      clientReceive(torch::th::data<ScalarType, THTensorType>(t));      \
  }

PS_RECEIVE(float, THFloatTensor);
PS_RECEIVE(double, THDoubleTensor);

#ifdef TORCH_MPI_CUDA

void* torchmpi_parameterserver_init_THCudaTensor(
    THCState* state, THCudaTensor* t) {
  barrier(getMainThreadCommunicator().intraComm);
  auto res = newDistributedParameterServer<float, THCudaTensor>(state, t);
  barrier(getMainThreadCommunicator().intraComm);
  return res;
}

ParameterServerSynchronizationHandle*
torchmpi_parameterserver_send_THCudaTensor(
    THCState* state, void* ps, THCudaTensor* t, const char* updateRuleName) {
  return reinterpret_cast<DistributedParameterServer*>(ps)->clientSend(
    torch::thc::data<float, THCudaTensor>(state, t), updateRuleName);
}

ParameterServerSynchronizationHandle*
torchmpi_parameterserver_receive_THCudaTensor(
    THCState* state, void* ps, THCudaTensor* t) {
  return reinterpret_cast<DistributedParameterServer*>(ps)->
    clientReceive(torch::thc::data<float, THCudaTensor>(state, t));
}

#endif

void torchmpi_parameterserver_free(void* ps) {
  barrier(getMainThreadCommunicator().intraComm);
  freeParameterServer(reinterpret_cast<DistributedParameterServer*>(ps));
  barrier(getMainThreadCommunicator().intraComm);
}

void torchmpi_parameterserver_free_all() {
  barrier(getMainThreadCommunicator().intraComm);
  freeParameterServers();
  barrier(getMainThreadCommunicator().intraComm);
}

ParameterServerSynchronizationHandle*
torchmpi_parameterserver_synchronize_handle(
    ParameterServerSynchronizationHandle* h)
{
  torch::mpi::resources::wait(h);
  return nullptr;
}

}
