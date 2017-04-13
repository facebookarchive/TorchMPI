/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "collectives_cuda.h"
#include "collectives.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "resources.h"

#ifdef TORCH_MPI_GLOO_CUDA
#include <gloo/cuda_allreduce_ring.h>
#include <gloo/cuda_allreduce_ring_chunked.h>
#include <gloo/cuda_broadcast_one_to_all.h>
#endif

/**********************************************************************
 ******************************** MPI *********************************
 **********************************************************************/
using namespace std;
using namespace torch::mpi;
using namespace torch::mpi::constants;
using namespace torch::mpi::resources;
using namespace torch::mpi::resources::cuda;

namespace torch { namespace mpi { namespace thc {

#define PREPARE(state, tensor)                                          \
  THCudaCheck(cudaGetLastError());                                      \
  if (!torch::thc::isContiguous(state, tensor)) {                       \
    THError("NYI: Collective only supported for contig tensors");       \
  }                                                                     \
  int device;                                                           \
  THCudaCheck(cudaGetDevice(&device));                                  \
  torch::mpi::thc::retainStorage(state, tensor);                        \
  auto stream = THCState_getCurrentStream(state);                       \
  auto tensorData = torch::thc::data<ScalarType>(state, tensor);        \
  auto nElement = torch::thc::nElement<THTensorType>(state, tensor);    \
  auto collectiveLevel = torch::mpi::getCollectiveSpan().first;         \
  CommunicatorGuard csOuter(collectiveLevel);                           \
  const CollectiveResourcesCuda* rOuter = acquireCollectiveResourcesCuda( \
    tensorData, Spin(true));

#define PREPARE_NCCL(state, tensor)                                     \
  PREPARE(state, tensor);                                               \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResourcesCuda* rInner =                               \
    acquireCollectiveResourcesCuda(tensorData,                          \
                                   Spin(true),                          \
                                   WithNCCLComm(true));                 \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();

#define PREPARE_GLOO(state, tensor)                                     \
  PREPARE(state, tensor);                                               \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResourcesCuda* rInner =                               \
    acquireCollectiveResourcesCuda(tensorData,                          \
                                   Spin(true),                          \
                                   WithNCCLComm(false),                 \
                                   WithGlooContext(true));              \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();

#define PREPARE_IPC(state, tensor)                                      \
  PREPARE(state, tensor);                                               \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResourcesCuda* rInner = acquireCollectiveResourcesCuda( \
    tensorData, Spin(true), WithNCCLComm(false),                        \
    WithGlooContext(false), WithEvents(true));                          \
  auto tensorDataBasePtr =                                              \
    torch::thc::data<ScalarType, THTensorType>(state, tensor) -         \
    tensor->storageOffset;                                              \
  auto desc =                                                           \
    getIPCDesc(state, tensorDataBasePtr, rInner->comm->intraComm);      \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();

#define PREPARE2(state, input, output, retainOutput)                    \
  THCudaCheck(cudaGetLastError());                                      \
  if (!torch::thc::isContiguous(state, input)) {                        \
    THError("NYI: Collective only supported for contig tensors");       \
  }                                                                     \
  torch::mpi::thc::retainStorage(state, input);                         \
  if (input != output && retainOutput) {                                \
    torch::mpi::thc::retainStorage(state, output);                      \
  }                                                                     \
  int device;                                                           \
  THCudaCheck(cudaGetDevice(&device));                                  \
  auto stream = THCState_getCurrentStream(state);                       \
  auto inputData = torch::thc::data<ScalarType>(state, input);          \
  auto outputData = (output) ?                                          \
    torch::thc::data<ScalarType>(state, output) : inputData;            \
  auto nElement = torch::thc::nElement<THTensorType>(state, input);     \
  auto nElementOutput = torch::thc::nElement<THTensorType>(state, output); \
  auto collectiveLevel = getCollectiveSpan().first;                     \
  CommunicatorGuard cs(collectiveLevel);                                \
  const CollectiveResourcesCuda* rOuter = acquireCollectiveResourcesCuda( \
    inputData, Spin(true));

#define PREPARE2_NCCL(state, input, output, retainOutput)               \
  PREPARE2(state, input, output, retainOutput);                         \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResourcesCuda* rInner =                               \
    acquireCollectiveResourcesCuda(inputData,                           \
                                   Spin(true),                          \
                                   WithNCCLComm(true));                 \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();

#define PREPARE2_GLOO(state, input, output, retainOutput)               \
  if (input != output) {                                                \
    THError("GLOO only supports inplace collectives");                  \
  }                                                                     \
  PREPARE2(state, input, output, retainOutput);                         \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResourcesCuda* rInner =                               \
    acquireCollectiveResourcesCuda(inputData,                           \
                                   Spin(true),                          \
                                   WithNCCLComm(false),                 \
                                   WithGlooContext(true));              \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();

#define PREPARE2_IPC(state, input, output, retainOutput)                \
  PREPARE2(state, input, output, retainOutput);                         \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResourcesCuda* rInner = acquireCollectiveResourcesCuda( \
    inputData, Spin(true), WithNCCLComm(false),                         \
    WithGlooContext(true), WithEvents(true));                           \
  auto outputDataBasePtr =                                              \
    torch::thc::data<ScalarType, THTensorType>(state, output) -         \
    output->storageOffset;                                              \
  auto desc =                                                           \
    getIPCDesc(state, outputDataBasePtr, rInner->comm->intraComm);      \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();


std::vector<cudaStream_t> preSyncHiPriStreams(cudaStream_t s) {
  auto p = getCollectiveStreams();
  THCudaCheck(cudaEventRecord(p.first, s));
  for (auto stream : p.second) {
    THCudaCheck(cudaStreamWaitEvent(stream, p.first, 0));
  }
  return p.second;
}

void postSyncHiPriStreams(cudaStream_t s) {
  auto p = getCollectiveStreams();
  for (auto stream : p.second) {
    THCudaCheck(cudaEventRecord(p.first, stream));
  }
  THCudaCheck(cudaStreamWaitEvent(s, p.first, 0));
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//                  Blocking collectives.                                    //
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
template<typename ScalarType>
void broadcast(ScalarType* inputData,
               int root,
               size_t nElement,
               const CollectiveResourcesCuda* r)
{
  r->comm->intraComm.Bcast(inputData, nElement, mpiType<ScalarType>(), root);
}

template<typename ScalarType>
void allreduce(ScalarType* inputData,
               ScalarType* outputData,
               size_t nElement,
               MPI::Op mpiRedOp,
               const CollectiveResourcesCuda* r)
{
  r->comm->intraComm.Allreduce(
    (outputData != inputData) ? inputData : MPI_IN_PLACE,
    outputData,
    nElement,
    mpiType<ScalarType>(),
    mpiRedOp);
}


template<typename ScalarType, typename THTensorType>
void sendreceive(THCState* state, THTensorType* tensor, int src, int dst) {
  PREPARE(state, tensor);

  rOuter->comm->intraComm.Sendrecv_replace(tensorData,
                                 nElement,
                                 mpiType<ScalarType>(),
                                 dst,
                                 kDefaultTag,
                                 src,
                                 kDefaultTag);

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));

  THCudaCheck(cudaGetLastError());
}

template<typename ScalarType>
void allgather(ScalarType* input,
               std::vector<ScalarType>& output,
               size_t nElement,
               const CollectiveResourcesCuda* r) {
    r->comm->intraComm.Allgather(
      input,
      nElement,
      mpiType<ScalarType>(),
      output.data(),
      nElement,
      mpiType<ScalarType>());
  }

template<typename ScalarType>
void allgatherv(ScalarType* input,
                ScalarType* output,
                size_t nElement,
                const std::vector<int>& counts,
                const std::vector<int>& displacements,
                const CollectiveResourcesCuda* r) {
  r->comm->intraComm.Allgatherv(
    input,
    nElement,
    mpiType<ScalarType>(),
    output,
    counts.data(),
    displacements.data(),
    mpiType<ScalarType>());
}

template<typename ScalarType, typename THTensorType>
void broadcast(THCState* state,
               THTensorType* tensor,
               int root)
{
  PREPARE(state, tensor);

  broadcast<ScalarType>(tensorData, root, nElement, rOuter);

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));

  THCudaCheck(cudaGetLastError());
}

template<typename ScalarType, typename THTensorType>
void reduce(THCState* state,
            THTensorType* input,
            THTensorType* output,
            int root,
            MPI::Op mpiRedOp)
{
  PREPARE2(state, input, output, true);

  if (outputData == inputData) {
    rOuter->comm->intraComm.Reduce(
      (commRank(rOuter->comm->intraComm) == root) ? MPI_IN_PLACE : inputData,
      outputData,
      nElement,
      mpiType<ScalarType>(),
      mpiRedOp,
      root);
  } else {
    rOuter->comm->intraComm.Reduce(
      inputData,
      outputData,
      nElement,
      mpiType<ScalarType>(),
      mpiRedOp,
      root);
  }

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));

  THCudaCheck(cudaGetLastError());
}

template<typename ScalarType, typename THTensorType>
void allreduce(THCState* state,
               THTensorType* input,
               THTensorType* output,
               MPI::Op mpiRedOp)
{
  PREPARE2(state, input, output, true);

  allreduce(inputData, outputData, nElement, mpiRedOp, rOuter);

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));

  THCudaCheck(cudaGetLastError());
}

template<typename ScalarType, typename THTensorType>
void allgather(THCState* state,
               THTensorType* input,
               THTensorType* output)
{
  if (input == output) {
    THError("inplace not supported for allgather");
  }
  PREPARE2(state, input, output, false);

  auto size = commSize(rOuter->comm->intraComm);
  std::vector<int> counts(size);

  // allgatherv takes int-typed counts / displacements
  int nElementInt = (int)nElement;
  allgather<int>(&nElementInt, counts, 1, rOuter);

  std::vector<int> displacements(size);
  displacements[0] = 0;
  for (int i = 1; i < size; ++i) {
    displacements[i] = counts[i-1] + displacements[i-1];
  }
  int outputSizeNeeded = displacements[size - 1] + counts[size - 1];
  if (outputSizeNeeded > nElementOutput) {
    THLongStorage *storageCopy = torch::thc::newSizeOf<THTensorType>(state, output);

    long outerStride = torch::thc::stride<THTensorType>(state, output, 0);
    if ( (outputSizeNeeded % outerStride) != 0 ) {
      THError("Size mismatch: assuming tensor gathered along last dimension, "
              "but outer stride of %d doesn't divide total size of %d\n",
              outerStride, outputSizeNeeded);
    }
    storageCopy->data[output->nDimension - 1] = outputSizeNeeded / outerStride;
    // TODO: creating a new tensor would be more efficient if we can't fit
    // realloc, but changes API since we would need to return new tensor.
    torch::thc::resizeNd(state, output, storageCopy->size,
                         storageCopy->data, NULL);
    outputData = torch::thc::data<ScalarType>(state, output);

    THLongStorage_free(storageCopy);
  }
  if (input != output) {
    torch::mpi::thc::retainStorage(state, output);
  }
  allgatherv<ScalarType>(inputData, outputData, nElement,
                         counts, displacements, rOuter);

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));

  THCudaCheck(cudaGetLastError());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//      P2P collectives perform barriers internally                          //
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
template<typename ScalarType>
SynchronizationHandle* broadcastp2pIPCImpl(ScalarType* dataPtr,
                                           int root,
                                           size_t nElement,
                                           const MPI::Intracomm& comm,
                                           IPCDesc* desc,
                                           size_t offset,
                                           cudaStream_t stream,
                                           const CollectiveIpcEvents &events)
{
  auto hiPriStreams = preSyncHiPriStreams(stream);
  THAssert(hiPriStreams.size() > 0);
  auto hiPriStream = hiPriStreams[0];
  if (nElement >= constants::kSmallBcastSizeGPU) {
    detail::broadcastp2pIPC<ScalarType>(
      dataPtr,
      root,
      nElement,
      desc,
      offset,
      comm,
      hiPriStream,
      events
    );
  } else {
    // This will become dead code eventually after we have tested enough
    THAssert(false);
    // TODO: Would be better to just use stock MPI here but for some reason,
    // I see registration error messages in this particular case
    auto b = SmallPinnedBufferProvider::acquire();
    if (root == commRank(comm)) {
      b->copyFrom(dataPtr, nElement * sizeof(ScalarType), hiPriStream);
      // Must sync to avoid Bcast too early
      THCudaCheck(cudaStreamSynchronize(hiPriStream));
    }
    comm.Bcast(b->data, nElement, mpiType<ScalarType>(), root);
    if (root != commRank(comm)) {
      b->copyTo(dataPtr, nElement * sizeof(ScalarType), hiPriStream);
      // Must sync to avoid releasing buffer too early
      THCudaCheck(cudaStreamSynchronize(hiPriStream));
    }
    SmallPinnedBufferProvider::release(b);
   }
  postSyncHiPriStreams(stream);
  return synchronizationHandleFromStream(stream);
}

template<typename ScalarType, typename THTensorType>
void broadcastp2p(THCState* state,
                  THTensorType* tensor,
                  int root)
{
  {
    // Latency-bound, better go through stock MPI implementation
    auto nElement = torch::thc::nElement<THTensorType>(state, tensor);
    if (nElement <= constants::kSmallBcastSizeGPU) {
      thc::broadcast<ScalarType>(state, tensor, root);
      return;
    }
  }

  PREPARE_IPC(state, tensor);
  if (hasInter) {
    // Release before calling!
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
    // For hierarchical broadcasts, participants need to agree on the root
    // value for different communicators. We don't have that yet so just
    // default to the mpi broadcast.
    thc::broadcast<ScalarType>(state, tensor, root);
    return;
  }
  auto sh = broadcastp2pIPCImpl<ScalarType>(tensorData,
                                            root,
                                            nElement,
                                            rInner->comm->intraComm,
                                            desc,
                                            tensor->storageOffset,
                                            stream,
                                            rInner->events);
  THCudaCheck(cudaGetLastError());
  resources::wait(sh);

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
}

template<typename ScalarType>
SynchronizationHandle* allreducep2pIPCImpl(ScalarType* inputData,
                                           ScalarType* outputData,
                                           size_t nElement,
                                           MPI::Op mpiRedOp,
                                           const MPI::Intracomm& comm,
                                           IPCDesc* desc,
                                           size_t offset,
                                           cudaStream_t stream,
                                           const CollectiveIpcEvents &events)
{
  auto hiPriStreams = preSyncHiPriStreams(stream);
  THAssert(hiPriStreams.size() > 0);
  auto hiPriStream = hiPriStreams[0];
  if (nElement >= constants::kSmallAllreduceSizeGPU) {
    // This performs collective calls internally
    detail::allreducep2pIPC<ScalarType>(
      inputData,
      outputData,
      nElement,
      mpiRedOp,
      desc,
      offset,
      comm,
      hiPriStreams,
      events
    );
  } else {
    // This will become dead code eventually after we have tested enough
    THAssert(false);
    auto b = SmallPinnedBufferProvider::acquire();
    b->copyFrom(inputData, nElement * sizeof(ScalarType), hiPriStream);
    // Must sync to avoid Allreduce too early
    THCudaCheck(cudaStreamSynchronize(hiPriStream));
    comm.Allreduce(
      MPI_IN_PLACE, b->data, nElement, mpiType<ScalarType>(), mpiRedOp);
    b->copyTo(outputData, nElement * sizeof(ScalarType), hiPriStream);
    // Must sync to avoid releasing buffer too early
    THCudaCheck(cudaStreamSynchronize(hiPriStream));
    SmallPinnedBufferProvider::release(b);
  }
  postSyncHiPriStreams(stream);
  return synchronizationHandleFromStream(stream);
}

template<typename ScalarType>
SynchronizationHandle* allreducep2pHierarchicalImpl(
    ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    IPCDesc* ipcDesc,
    size_t offset,
    cudaStream_t stream,
    bool hasIntra,
    bool hasInter,
    const CollectiveResourcesCuda* r)
{
  // Short P2P path
  if (!hasInter) {
    allreducep2pIPCImpl<ScalarType>(inputData,
                                    outputData,
                                    nElement,
                                    mpiRedOp,
                                    r->comm->intraComm,
                                    ipcDesc,
                                    offset,
                                    stream,
                                    r->events);
    // don't sync on stream here
    return synchronizationHandleFromStream(stream);
  }

  // If we get here we must go hierarchical
  if (!hasIntra) {
    auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
    THAssert(hiPriStreams.size() > 0);
    torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
      hasIntra ? outputData : inputData,
      outputData,
      nElement,
      mpiRedOp,
      r->comm->interComm,
      hiPriStreams);
    torch::mpi::thc::postSyncHiPriStreams(stream);
    // don't sync on stream here
    return synchronizationHandleFromStream(stream);
  }


  // Both inter and intr
  allreducep2pIPCImpl<ScalarType>(inputData,
                                  outputData,
                                  nElement,
                                  mpiRedOp,
                                  r->comm->intraComm,
                                  ipcDesc,
                                  offset,
                                  stream,
                                  r->events);
  if (torch::mpi::commSize(r->comm->interComm) > 1) {
    auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
    THAssert(hiPriStreams.size() > 0);
    torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
      hasIntra ? outputData : inputData,
      outputData,
      nElement,
      mpiRedOp,
      r->comm->interComm,
      hiPriStreams);
    torch::mpi::thc::postSyncHiPriStreams(stream);
    // don't sync on stream here
  }
  if (!r->comm->cartesian) {
    broadcastp2pIPCImpl(outputData,
                        0,
                        nElement,
                        r->comm->intraComm,
                        ipcDesc,
                        offset,
                        stream,
                        r->events);
    // don't sync on stream here
  }
  return synchronizationHandleFromStream(stream);
}

template<typename ScalarType, typename THTensorType>
void allreducep2pHierarchical(THCState* state,
                              THTensorType* input,
                              THTensorType* output,
                              MPI::Op mpiRedOp)
{
  PREPARE2_IPC(state, input, output, true);

  auto sh = allreducep2pHierarchicalImpl(inputData,
                                         outputData,
                                         nElement,
                                         mpiRedOp,
                                         desc,
                                         output->storageOffset,
                                         stream,
                                         hasIntra,
                                         hasInter,
                                         rInner);
  // TODO: ScopeGuard??
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));

  // Must sync on stream here
  resources::wait(sh);
}

template<typename ScalarType, typename THTensorType>
void allreducep2pFlat(THCState* state,
                      THTensorType* input,
                      THTensorType* output,
                      MPI::Op mpiRedOp)
{
  PREPARE2(state, input, output, true);

  auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
  THAssert(hiPriStreams.size() > 0);
  torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
    inputData,
    outputData,
    nElement,
    mpiRedOp,
    rOuter->comm->intraComm,
    hiPriStreams);
  torch::mpi::thc::postSyncHiPriStreams(stream);

  // TODO: ScopeGuard??
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));

  // Must sync on stream here
  THCudaCheck(cudaStreamSynchronize(stream));
}

template<typename ScalarType, typename THTensorType>
void allreducep2p(THCState* state,
                  THTensorType* input,
                  THTensorType* output,
                  MPI::Op mpiRedOp)
{
  {
    // Latency-bound, better go through stock MPI implementation
    auto nElement = torch::thc::nElement<THTensorType>(state, input);
    if (nElement <= constants::kSmallAllreduceSizeGPU) {
      thc::allreduce<ScalarType>(state, input, output, mpiRedOp);
      return;
    }
  }

  if (constants::kUseHierarchicalCollectives) {
    // If we have to go through TCP we cannot rely on cuda support to properly
    // perform CPU-GPU copies asynchronously. Write our own hierarchical
    // allreduce that goes through explicit copies ot pinned CPU buffers.
    allreducep2pHierarchical<ScalarType, THTensorType>(
      state, input, output, mpiRedOp);
  } else{
    // 1-level flat and simple Allreduce using MPI_Isend/MPI_Irecv backed by
    // RDMA.
    allreducep2pFlat<ScalarType, THTensorType>(
      state, input, output, mpiRedOp);
  }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//   All async collectives offload to the collectiveOffloadThreadPool        //
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* broadcastAsync(
    THCState* state, THTensorType* tensor, int root) {
  PREPARE(state, tensor);
  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));
      broadcast<ScalarType>(tensorData, root, nElement, rOuter);
      THCudaCheck(cudaGetLastError());
      // TODO: ScopeGuard??
      releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
  }));
  return synchronizationHandleFromFuture(futures.size() - 1);
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreduceAsync(
  THCState* state, THTensorType* input, THTensorType* output, MPI::Op mpiRedOp)
{
  PREPARE2(state, input, output, true);
  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));
      allreduce<ScalarType>(
        inputData, outputData, nElement, mpiRedOp, rOuter);
      THCudaCheck(cudaGetLastError());
      // TODO: ScopeGuard??
      releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
    }));
  return synchronizationHandleFromFuture(futures.size() - 1);
}

template<typename ScalarType>
SynchronizationHandle* broadcastp2pIPCAsyncImpl(
    ScalarType* tensorData,
    int root,
    size_t nElement,
    const CollectiveResourcesCuda* r,
    IPCDesc* desc,
    size_t offset,
    cudaStream_t stream) {
  int device;
  THCudaCheck(cudaGetDevice(&device));
  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));
      auto sh = broadcastp2pIPCImpl<ScalarType>(tensorData,
                                                root,
                                                nElement,
                                                r->comm->intraComm,
                                                desc,
                                                offset,
                                                stream,
                                                r->events);
      // Must sync on stream here
      resources::wait(sh);
      // TODO: ScopeGuard
      releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(r));
  }));
  return synchronizationHandleFromFuture(futures.size() - 1);
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* broadcastp2pAsync(THCState* state,
                                         THTensorType* tensor,
                                         int root)
{
  {
    // Latency-bound, better go through stock MPI implementation
    auto nElement = torch::thc::nElement<THTensorType>(state, tensor);
    if (nElement <= constants::kSmallBcastSizeGPU) {
      return thc::broadcastAsync<ScalarType>(state, tensor, root);
    }
  }

  PREPARE_IPC(state, tensor);
  if (hasInter) {
    // Release before calling!
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
    // For hierarchical broadcasts, participants need to agree on the root
    // value for different communicators. We don't have that yet so just
    // default to the mpi broadcast.
    auto res = thc::broadcastAsync<ScalarType>(state, tensor, root);
    return res;
  }
  // broadcastp2pIPCAsyncImpl must release rInner!!
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
  return broadcastp2pIPCAsyncImpl<ScalarType>(tensorData,
                                              root,
                                              nElement,
                                              rInner,
                                              desc,
                                              tensor->storageOffset,
                                              stream);
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreducep2pAsyncHierarchical(
  THCState* state, THTensorType* input, THTensorType* output, MPI::Op mpiRedOp)
{
  PREPARE2_IPC(state, input, output, true);

  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));

      // 2-level implementation going through cudaIPC for intra-node
      // and explicit pinned CPU buffers for inter-node.
      auto sh = allreducep2pHierarchicalImpl(inputData,
                                             outputData,
                                             nElement,
                                             mpiRedOp,
                                             desc,
                                             output->storageOffset,
                                             stream,
                                             hasIntra,
                                             hasInter,
                                             rInner);
      // Must sync on stream here
      resources::wait(sh);

      // TODO: ScopeGuard??
      releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
      releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
  }));

  return synchronizationHandleFromFuture(futures.size() - 1);
}


template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreducep2pAsyncFlat(
  THCState* state, THTensorType* input, THTensorType* output, MPI::Op mpiRedOp)
{
  PREPARE2_IPC(state, input, output, true);

  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));

      auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
      THAssert(hiPriStreams.size() > 0);
      torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
        inputData,
        outputData,
        nElement,
        mpiRedOp,
        rOuter->comm->intraComm,
        hiPriStreams);
      torch::mpi::thc::postSyncHiPriStreams(stream);

      // Must sync on stream here
      THCudaCheck(cudaStreamSynchronize(stream));

      // TODO: ScopeGuard??
      releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
      releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
  }));

  return synchronizationHandleFromFuture(futures.size() - 1);
}


template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreducep2pAsync(
  THCState* state, THTensorType* input, THTensorType* output, MPI::Op mpiRedOp)
{
  {
    // Latency-bound, better go through stock MPI implementation
    auto nElement = torch::thc::nElement<THTensorType>(state, input);
    if (nElement <= constants::kSmallAllreduceSizeGPU) {
      return thc::allreduceAsync<ScalarType>(state, input, output, mpiRedOp);
    }
  }

  if (constants::kUseHierarchicalCollectives) {
    // If we have to go through TCP we cannot rely on cuda support to properly
    // perform CPU-GPU copies asynchronously. Write our own hierarchical
    // allreduce that goes through explicit copies ot pinned CPU buffers.
    return allreducep2pAsyncHierarchical<ScalarType, THTensorType>(
      state, input, output, mpiRedOp);
  } else {
    // 1-level flat and simple Allreduce using MPI_Isend/MPI_Irecv backed by
    // RDMA.
    return allreducep2pAsyncFlat<ScalarType, THTensorType>(
      state, input, output, mpiRedOp);
  }
}

}} // ns mpi::thc


#ifdef TORCH_MPI_NCCL

namespace nccl { namespace thc {

// Collectives operating on THCuda*Tensor
template<typename ScalarType>
cudaStream_t broadcast(ScalarType* tensorData,
                       int root,
                       size_t nElement,
                       cudaStream_t stream,
                       const ncclComm_t& comm)
{
  int count, device, rank;
  NCCLCHECK(ncclCommCount(comm, &count));
  NCCLCHECK(ncclCommCuDevice(comm, &device));
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  VLOG_1(" ncclBroadcast device " << device << " rank " << rank
         << "/" << count << std::endl);
  NCCLCHECK(ncclBcast(tensorData,
                      nElement,
                      ncclType<ScalarType>(),
                      root,
                      comm,
                      stream));
  THCudaCheck(cudaGetLastError());
  return stream;
}

// Collectives operating on THCuda*Tensor
template<typename ScalarType, typename THTensorType>
cudaStream_t broadcastImpl(THCState* state, THTensorType* tensor, int root) {
  {
    // Latency-bound, better go through stock MPI implementation
    auto nElement = torch::thc::nElement<THTensorType>(state, tensor);
    if (nElement <= constants::kSmallBcastSizeGPU) {
      torch::mpi::thc::broadcast<ScalarType>(state, tensor, root);
      return THCState_getCurrentStream(state);
    }
  }

  PREPARE_NCCL(state, tensor);
  if (hasInter) {
    // Release before calling!
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
    // For hierarchical broadcasts, participants need to agree on the root
    // value for different communicators. We don't have that yet so just
    // default to the mpi broadcast.
    torch::mpi::thc::broadcast<ScalarType>(state, tensor, root);
    return stream;
  }
  nccl::thc::broadcast(tensorData, root, nElement, stream, *rInner->ncclComm);
  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
  return stream;
}

template<typename ScalarType, typename THTensorType>
void broadcast(THCState* state, THTensorType* tensor, int root) {
  THCudaCheck(cudaStreamSynchronize(
    nccl::thc::broadcastImpl<ScalarType, THTensorType>(
      state, tensor, root)));
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle*
broadcastAsync(THCState* state, THTensorType* tensor, int root) {
  return synchronizationHandleFromStream(
    nccl::thc::broadcastImpl<ScalarType, THTensorType>(
      state, tensor, root));
}


// Collectives operating on THCuda*Tensor
template<typename ScalarType>
cudaStream_t reduce(ScalarType* inputData,
                    ScalarType* outputData,
                    int root,
                    size_t nElement,
                    ncclRedOp_t ncclRedOp,
                    cudaStream_t stream,
                    const ncclComm_t& comm)
{
  int count, device, rank;
  NCCLCHECK(ncclCommCount(comm, &count));
  NCCLCHECK(ncclCommCuDevice(comm, &device));
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  VLOG_1(" ncclReduce device " << device << " rank " << rank
         << "/" << count << std::endl);
  NCCLCHECK(ncclReduce(inputData,
                       outputData,
                       nElement,
                       ncclType<ScalarType>(),
                       ncclRedOp,
                       root,
                       comm,
                       stream));
  THCudaCheck(cudaGetLastError());
  return stream;
}

// Collectives operating on THCuda*Tensor
template<typename ScalarType, typename THTensorType>
cudaStream_t reduceImpl(THCState* state,
                        THTensorType* input,
                        THTensorType* output,
                        int root,
                        ncclRedOp_t ncclRedOp)
{
  {
    // Latency-bound, better go through stock MPI implementation
    auto nElement = torch::thc::nElement<THTensorType>(state, input);
    if (nElement <= constants::kSmallBcastSizeGPU) {
      torch::mpi::thc::reduce<ScalarType>(state, input, output, root, mpiOp(ncclRedOp));
      return THCState_getCurrentStream(state);
    }
  }

  PREPARE2_NCCL(state, input, output, true);
  if (hasInter) {
    // Release before calling!
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
    // For hierarchical broadcasts, participants need to agree on the root
    // value for different communicators. We don't have that yet so just
    // default to the mpi broadcast.
    torch::mpi::thc::reduce<ScalarType>(state, input, output, root, mpiOp(ncclRedOp));
    return stream;
  }

  // Just reduce within node level, the NCCL communicator is unique
  nccl::thc::reduce(
    inputData, outputData, root, nElement, ncclRedOp, stream, *rInner->ncclComm);
  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
  return stream;
}

template<typename ScalarType, typename THTensorType>
void reduce(THCState* state,
            THTensorType* input,
            THTensorType* output,
            int root,
            ncclRedOp_t ncclRedOp) {
  THCudaCheck(cudaStreamSynchronize(reduceImpl<ScalarType, THTensorType>(
    state, input, output, root, ncclRedOp)));
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* reduceAsync(THCState* state,
                                   THTensorType* input,
                                   THTensorType* output,
                                   int root,
                                   ncclRedOp_t ncclRedOp) {
  return synchronizationHandleFromStream(reduceImpl<ScalarType, THTensorType>(
    state, input, output, root, ncclRedOp));
}


template<typename ScalarType>
cudaStream_t allreduce(ScalarType* inputData,
                       ScalarType* outputData,
                       size_t nElement,
                       ncclRedOp_t ncclRedOp,
                       cudaStream_t stream,
                       const ncclComm_t& comm)
{
  int count, device, rank;
  NCCLCHECK(ncclCommCount(comm, &count));
  NCCLCHECK(ncclCommCuDevice(comm, &device));
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  VLOG_1(" ncclAllReduce device " << device << " rank " << rank
         << "/" << count << std::endl);
  NCCLCHECK(ncclAllReduce(inputData,
                          outputData,
                          nElement,
                          ncclType<ScalarType>(),
                          ncclRedOp,
                          comm,
                          stream));
  THCudaCheck(cudaGetLastError());
  return stream;
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreduceImpl(THCState* state,
                                     THTensorType* input,
                                     THTensorType* output,
                                     ncclRedOp_t ncclRedOp)
{
  PREPARE2_NCCL(state, input, output, true);

  // Case 1. Intra only
  if (!hasInter) {
    // If we don't go cross NCCL invocations, never offload to helper thread
    // because we have no guarantee when kernels will actually get posted.
    // In this case asynchronicity should be handled by streams only
    auto res = synchronizationHandleFromStream(
      nccl::thc::allreduce<ScalarType>(inputData,
                                       outputData,
                                       nElement,
                                       ncclRedOp,
                                       stream,
                                       *rInner->ncclComm));
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
    return res;
  }

  // Case 2. Intra only
  if (!hasIntra) {
    auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
    THAssert(hiPriStreams.size() > 0);
    torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
      hasIntra ? outputData : inputData,
      outputData,
      nElement,
      mpiOp(ncclRedOp),
      rInner->comm->interComm,
      hiPriStreams);
    torch::mpi::thc::postSyncHiPriStreams(stream);
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
    return nullptr;
  }

  // Case 3. Both inter and intra
  auto lambda = [=]() {
    nccl::thc::allreduce<ScalarType>(inputData,
                                     outputData,
                                     nElement,
                                     ncclRedOp,
                                     stream,
                                     *rInner->ncclComm);
    auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
    THAssert(hiPriStreams.size() > 0);
    torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
      hasIntra ? outputData : inputData,
      outputData,
      nElement,
      mpiOp(ncclRedOp),
      rInner->comm->interComm,
      hiPriStreams);
    torch::mpi::thc::postSyncHiPriStreams(stream);
    if (!rInner->comm->cartesian) {
      nccl::thc::broadcast(outputData,
                           0,
                           nElement,
                           stream,
                           *rInner->ncclComm);
    }
    THCudaCheck(cudaStreamSynchronize(stream));
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
  };

  // Unfortunately trying to mix threads and multiple NCCL communicators
  // seems to create deadlocks.
  // So it seems no MPI overlapping of NCCL transfers with copies?
  // auto& futures = getCollectiveFutures();
  // futures.push_back(collectiveOffloadThreadPool().enqueue(lambda));
  // return synchronizationHandleFromFuture(futures.size() - 1);

  lambda();
  return synchronizationHandleFromStream(stream);
}

template<typename ScalarType, typename THTensorType>
void allreduce(THCState* state,
               THTensorType* input,
               THTensorType* output,
               ncclRedOp_t ncclRedOp) {
  resources::wait(
    nccl::thc::allreduceImpl<ScalarType, THTensorType>(
      state, input, output, ncclRedOp));

}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreduceAsync(THCState* state,
                                      THTensorType* input,
                                      THTensorType* output,
                                      ncclRedOp_t ncclRedOp) {
  return nccl::thc::allreduceImpl<ScalarType, THTensorType>(
    state, input, output, ncclRedOp);
}


}} // ns nccl::thc

#endif

// ns mpi::th

#ifdef TORCH_MPI_GLOO_CUDA

namespace gloo { namespace thc {

// Gloo does not instantiate the same types as TorchMPI -- need to map
template<typename T>
struct TypeToGlooType {
  typedef T type;
};

template<>
struct TypeToGlooType<char> {
  typedef int8_t type;
};

template<>
struct TypeToGlooType<int> {
  typedef int32_t type;
};

template<>
struct TypeToGlooType<long> {
  typedef int64_t type;
};

// Collectives operating on THCuda*Tensor
template<typename ScalarType>
cudaStream_t broadcast(ScalarType* tensorData,
                       int root,
                       size_t nElement,
                       cudaStream_t stream,
                       const shared_ptr<::gloo::mpi::Context>& context)
{
  typedef typename TypeToGlooType<ScalarType>::type GlooType;
  ::gloo::CudaBroadcastOneToAll<GlooType> broadcast(
      context, {(GlooType *)tensorData}, nElement, root, 0, {stream});
  broadcast.run();

  THCudaCheck(cudaGetLastError());
  return stream;
}

template<typename ScalarType, typename THTensorType>
cudaStream_t broadcastImpl(THCState* state, THTensorType* tensor, int root) {
  PREPARE_GLOO(state, tensor);
  gloo::thc::broadcast(tensorData, root, nElement, stream, rInner->glooContext);
  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
  return stream;
}

template<typename ScalarType, typename THTensorType>
void broadcast(THCState* state, THTensorType* tensor, int root) {
  THCudaCheck(cudaStreamSynchronize(
    gloo::thc::broadcastImpl<ScalarType, THTensorType>(
      state, tensor, root)));
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle*
broadcastAsync(THCState* state, THTensorType* tensor, int root) {
  return synchronizationHandleFromStream(
    gloo::thc::broadcastImpl<ScalarType, THTensorType>(
      state, tensor, root));
}

void checkReduceFunction(MPI::Op redOpt) {
  // Gloo doesn't support non-MPI_SUM for gpus (oversight?)
  if (redOpt != MPI::Op(MPI_SUM)) {
    THError("Only MPI_SUM supported by Gloo and MPI");
  }
}

template<typename ScalarType>
cudaStream_t allreduce(ScalarType* inputData,
                       ScalarType* outputData,
                       size_t nElement,
                       MPI::Op mpiRedOp,
                       cudaStream_t stream,
                       const shared_ptr<::gloo::mpi::Context>& context) {
  checkReduceFunction(mpiRedOp);
  typedef typename TypeToGlooType<ScalarType>::type GlooType;
  std::vector<GlooType *> v { (GlooType *)inputData };

#ifdef TORCH_MPI_CUDA_ALLREDUCE_CHUNKED_ENABLED
  if (nElement <= mpi::constants::kSmallAllreduceSizeGPU) {
#else
  if (true) {
#endif
    ::gloo::CudaAllreduceRing<GlooType> allreduce(
      context, v, nElement);
    allreduce.run();
  } else {
    ::gloo::CudaAllreduceRingChunked<GlooType> allreduce(
      context, v, nElement);
    allreduce.run();
  }
  THCudaCheck(cudaGetLastError());
  return stream;
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreduceImpl(THCState* state,
                                     THTensorType* input,
                                     THTensorType* output,
                                     MPI::Op mpiRedOp)
{
  PREPARE2_GLOO(state, input, output, true);

  // TODO: use high priority streams?
  auto res = synchronizationHandleFromStream(
    gloo::thc::allreduce<ScalarType>(inputData,
                                     outputData,
                                     nElement,
                                     mpiRedOp,
                                     stream,
                                     rInner->glooContext));
  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResourcesCuda*>(rOuter));
  return res;
}

template<typename ScalarType, typename THTensorType>
void allreduce(THCState* state,
               THTensorType* input,
               THTensorType* output,
               MPI::Op mpiRedOp) {
  resources::wait(
    gloo::thc::allreduceImpl<ScalarType, THTensorType>(
      state, input, output, mpiRedOp));
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreduceAsync(THCState* state,
                       THTensorType* input,
                       THTensorType* output,
                       MPI::Op mpiRedOp) {
  return gloo::thc::allreduceImpl<ScalarType, THTensorType>(
    state, input, output, mpiRedOp);
}

}} // ns gloo::thc

#endif

} // ns torch



/**********************************************************************
 *********************** C Wrapper definitions ************************
 **********************************************************************/
#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)

extern "C" {

/*********************** Broadcast ************************************/
#define DEFINE_BROADCAST(ScalarType, THCTensorType)                     \
  void PPCAT(torchmpi_broadcast_, THCTensorType)                        \
    (THCState* state, THCTensorType *input, int root) {                 \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::broadcast<ScalarType, THCTensorType>(              \
      state, input, root);                                              \
  }

#define DEFINE_BROADCAST_ASYNC(ScalarType, THCTensorType)               \
  SynchronizationHandle*                                                \
  PPCAT(torchmpi_async_broadcast_, THCTensorType)                       \
    (THCState* state, THCTensorType *input, int root) {                 \
  struct cudaPointerAttributes attributes;                              \
  THCudaCheck(cudaPointerGetAttributes(&attributes,                     \
    torch::thc::data<ScalarType, THCTensorType>(state, input)));        \
  return torch::mpi::thc::broadcastAsync<ScalarType, THCTensorType>(    \
  state, input, root);                                                  \
}

#define DEFINE_BROADCASTP2P(ScalarType, THCTensorType)                  \
  void PPCAT(torchmpi_p2p_broadcast_, THCTensorType)                    \
    (THCState* state, THCTensorType *input, int root) {                 \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::broadcastp2p<ScalarType, THCTensorType>(           \
      state, input, root);                                              \
  }

#define DEFINE_BROADCASTP2P_ASYNC(ScalarType, THCTensorType)            \
  SynchronizationHandle* PPCAT                                          \
  (torchmpi_async_p2p_broadcast_, THCTensorType)                        \
    (THCState* state, THCTensorType *input, int root) {                 \
  struct cudaPointerAttributes attributes;                              \
  THCudaCheck(cudaPointerGetAttributes(&attributes,                     \
    torch::thc::data<ScalarType, THCTensorType>(state, input)));        \
  return torch::mpi::thc::broadcastp2pAsync<ScalarType, THCTensorType>( \
  state, input, root);                                                  \
}

#define DEFINE_NCCL_BROADCAST(ScalarType, THCTensorType)        \
  void PPCAT(torchmpi_nccl_broadcast_, THCTensorType)           \
    (THCState* state, THCTensorType *input, int root) {         \
  torch::nccl::thc::broadcast<ScalarType, THCTensorType>(       \
    state, input, root);                                        \
  }

#define DEFINE_NCCL_BROADCAST_ASYNC(ScalarType, THCTensorType)          \
  SynchronizationHandle* PPCAT(torchmpi_async_nccl_broadcast_, THCTensorType) \
    (THCState* state, THCTensorType *input, int root) {                 \
    return torch::nccl::thc::broadcastAsync<ScalarType, THCTensorType>( \
      state, input, root);                                              \
  }

#define DEFINE_GLOO_BROADCAST(ScalarType, THCTensorType)        \
  void PPCAT(torchmpi_gloo_broadcast_, THCTensorType)           \
    (THCState* state, THCTensorType *input, int root) {         \
  torch::gloo::thc::broadcast<ScalarType, THCTensorType>(       \
    state, input, root);                                        \
  }

#define DEFINE_GLOO_BROADCAST_ASYNC(ScalarType, THCTensorType)          \
  SynchronizationHandle* PPCAT(torchmpi_async_gloo_broadcast_, THCTensorType) \
    (THCState* state, THCTensorType *input, int root) {                 \
    return torch::gloo::thc::broadcastAsync<ScalarType, THCTensorType>( \
      state, input, root);                                              \
  }

/*********************** Reduce ************************************/
#define DEFINE_REDUCE(ScalarType, THCTensorType)                        \
  void PPCAT(torchmpi_reduce_, THCTensorType)                           \
    (THCState* state, THCTensorType *input, THCTensorType *output, int root) { \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::reduce<ScalarType, THCTensorType>(                 \
      state, input, output, root, MPI_SUM);                             \
  }

#define DEFINE_NCCL_REDUCE(ScalarType, THCTensorType)                   \
  void PPCAT(torchmpi_nccl_reduce_, THCTensorType)                      \
    (THCState* state, THCTensorType *input, THCTensorType *output, int root) { \
  torch::nccl::thc::reduce<ScalarType, THCTensorType>(                  \
  state, input, output, root, ncclSum);                                 \
}

#define DEFINE_NCCL_REDUCE_ASYNC(ScalarType, THCTensorType)             \
  SynchronizationHandle* PPCAT(torchmpi_async_nccl_reduce_, THCTensorType) \
    (THCState* state, THCTensorType *input, THCTensorType *output, int root) { \
    return torch::nccl::thc::reduceAsync<ScalarType, THCTensorType>(    \
      state, input, output, root, ncclSum);                             \
  }

/*********************** Allreduce ************************************/
#define DEFINE_ALLREDUCE(ScalarType, THCTensorType)                     \
  void PPCAT(torchmpi_allreduce_, THCTensorType)(                       \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::allreduce<ScalarType, THCTensorType>(              \
      state, input, output, MPI_SUM);                                   \
  }

#define DEFINE_ALLREDUCE_ASYNC(ScalarType, THCTensorType)               \
  SynchronizationHandle*                                                \
  PPCAT(torchmpi_async_allreduce_, THCTensorType)(                      \
  THCState* state, THCTensorType *input, THCTensorType *output) {       \
  struct cudaPointerAttributes attributes;                              \
  THCudaCheck(cudaPointerGetAttributes(&attributes,                     \
    torch::thc::data<ScalarType, THCTensorType>(state, input)));        \
  return torch::mpi::thc::allreduceAsync<ScalarType, THCTensorType>(    \
    state, input, output, MPI_SUM);                                     \
}

#define DEFINE_ALLREDUCEP2P(ScalarType, THCTensorType)                  \
  void PPCAT(torchmpi_p2p_allreduce_, THCTensorType)(                   \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::allreducep2p<ScalarType, THCTensorType>(           \
      state, input, output, MPI_SUM);                                   \
  }

#define DEFINE_ALLREDUCEP2P_ASYNC(ScalarType, THCTensorType)            \
  SynchronizationHandle*                                                \
  PPCAT(torchmpi_async_p2p_allreduce_, THCTensorType)(                  \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    return torch::mpi::thc::allreducep2pAsync<ScalarType, THCTensorType>( \
      state, input, output, MPI_SUM);                                   \
  }

#define DEFINE_NCCL_ALLREDUCE(ScalarType, THCTensorType)                \
  void PPCAT(torchmpi_nccl_allreduce_, THCTensorType)(                  \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    torch::nccl::thc::allreduce<ScalarType, THCTensorType>(             \
      state, input, output, ncclSum);                                   \
  }

#define DEFINE_NCCL_ALLREDUCE_ASYNC(ScalarType, THCTensorType)          \
  SynchronizationHandle* PPCAT(torchmpi_async_nccl_allreduce_, THCTensorType)( \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    return torch::nccl::thc::allreduceAsync<ScalarType, THCTensorType>( \
      state, input, output, ncclSum);                                   \
  }

#define DEFINE_GLOO_ALLREDUCE(ScalarType, THCTensorType)                \
  void PPCAT(torchmpi_gloo_allreduce_, THCTensorType)(                  \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    torch::gloo::thc::allreduce<ScalarType, THCTensorType>(             \
      state, input, output, MPI_SUM);                                   \
  }

#define DEFINE_GLOO_ALLREDUCE_ASYNC(ScalarType, THCTensorType)          \
  SynchronizationHandle* PPCAT(torchmpi_async_gloo_allreduce_, THCTensorType)( \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    return torch::gloo::thc::allreduceAsync<ScalarType, THCTensorType>( \
      state, input, output, MPI_SUM);                                   \
  }

/*********************** Sendreceive **********************************/
#define DEFINE_SENDRECEIVE(ScalarType, THCTensorType)                   \
  void PPCAT(torchmpi_sendreceive_, THCTensorType)                      \
    (THCState* state, THCTensorType *input, int src, int dst) {         \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::sendreceive<ScalarType, THCTensorType>(            \
      state, input, src, dst);                                          \
  }

/*********************** Allgather **********************************/
#define DEFINE_ALLGATHER(ScalarType, THCTensorType)                     \
  void PPCAT(torchmpi_allgather_, THCTensorType)(                       \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::allgather<ScalarType, THCTensorType>(              \
      state, input, output);                                            \
  }

/**********************************************************************
 ********************** C Wrapper instantiations **********************
 **********************************************************************/
#define FUNCTIONS_TO_INSTANTIATE_ALWAYS(                \
  CPP_TYPE, TH_TENSOR_TYPE, THC_TENSOR_TYPE)            \
  DEFINE_BROADCAST(CPP_TYPE, THC_TENSOR_TYPE);          \
  DEFINE_BROADCAST_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);    \
  DEFINE_BROADCASTP2P(CPP_TYPE, THC_TENSOR_TYPE);       \
  DEFINE_BROADCASTP2P_ASYNC(CPP_TYPE, THC_TENSOR_TYPE); \
  DEFINE_REDUCE(CPP_TYPE, THC_TENSOR_TYPE);             \
  DEFINE_ALLREDUCE(CPP_TYPE, THC_TENSOR_TYPE);          \
  DEFINE_ALLREDUCE_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);    \
  DEFINE_ALLREDUCEP2P(CPP_TYPE, THC_TENSOR_TYPE);       \
  DEFINE_ALLREDUCEP2P_ASYNC(CPP_TYPE, THC_TENSOR_TYPE); \
  DEFINE_SENDRECEIVE(CPP_TYPE, THC_TENSOR_TYPE);        \
  DEFINE_ALLGATHER(CPP_TYPE, THC_TENSOR_TYPE);

#ifdef TORCH_MPI_NCCL
#define DEFINE_NCCL_FUNCTIONS(CPP_TYPE, THC_TENSOR_TYPE)        \
  DEFINE_NCCL_BROADCAST(CPP_TYPE, THC_TENSOR_TYPE);             \
  DEFINE_NCCL_BROADCAST_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);       \
  DEFINE_NCCL_REDUCE(CPP_TYPE, THC_TENSOR_TYPE);                \
  DEFINE_NCCL_REDUCE_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);          \
  DEFINE_NCCL_ALLREDUCE(CPP_TYPE, THC_TENSOR_TYPE);             \
  DEFINE_NCCL_ALLREDUCE_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);
#else
#define DEFINE_NCCL_FUNCTIONS(CPP_TYPE, THC_TENSOR_TYPE)
#endif

#ifdef TORCH_MPI_GLOO_CUDA
#define DEFINE_GLOO_FUNCTIONS(CPP_TYPE, THC_TENSOR_TYPE)        \
  DEFINE_GLOO_BROADCAST(CPP_TYPE, THC_TENSOR_TYPE);             \
  DEFINE_GLOO_BROADCAST_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);       \
  DEFINE_GLOO_ALLREDUCE(CPP_TYPE, THC_TENSOR_TYPE);             \
  DEFINE_GLOO_ALLREDUCE_ASYNC(CPP_TYPE, THC_TENSOR_TYPE)
#else
#define DEFINE_GLOO_FUNCTIONS(CPP_TYPE, THC_TENSOR_TYPE)
#endif

#define FUNCTIONS_TO_INSTANTIATE(                               \
  CPP_TYPE, TH_TENSOR_TYPE, THC_TENSOR_TYPE)                    \
  FUNCTIONS_TO_INSTANTIATE_ALWAYS(                              \
    CPP_TYPE, TH_TENSOR_TYPE, THC_TENSOR_TYPE);                 \
  DEFINE_NCCL_FUNCTIONS(CPP_TYPE, THC_TENSOR_TYPE);             \
  DEFINE_GLOO_FUNCTIONS(CPP_TYPE, THC_TENSOR_TYPE);

#include "generic/torch_collectives_wrappers.cpp.in"

void torchmpi_free_ipc_descriptors() {
  VLOG_1("torchmpi_free_ipc_descriptors" << endl);
  auto& descs = getIPCDescs();
  descs = unordered_map<void*, unique_ptr<IPCDesc>> ();
}

} // extern "C"
