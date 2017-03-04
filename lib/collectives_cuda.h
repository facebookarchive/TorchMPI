/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#ifndef TORCH_MPI_COLLECTIVES_CUDA_INC
#define TORCH_MPI_COLLECTIVES_CUDA_INC

#include "torch_mpi_cuda.h"

#include <THC.h>

#include "resources.h"

namespace torch {

namespace mpi { namespace thc {

using IPCDesc = torch::mpi::resources::cuda::IPCDesc;
using SynchronizationHandle = torch::mpi::resources::SynchronizationHandle;

namespace detail {

  template<typename ScalarType>
  void broadcastp2pIPC(ScalarType* data,
                       size_t root,
                       size_t nElement,
                       IPCDesc* desc,
                       size_t offset,
                       const MPI::Intracomm& comm,
                       cudaStream_t stream,
                       std::vector<std::vector<cudaEvent_t>> ipcEvents = std::vector<std::vector<cudaEvent_t>>());

  template<typename ScalarType>
  void allreducep2pIPC(
    const ScalarType*,
    ScalarType*,
    size_t,
    MPI::Op mpiRedOp,
    IPCDesc* desc,
    size_t offset,
    const MPI::Intracomm&,
    std::vector<cudaStream_t> copyStreams,
    std::vector<std::vector<cudaEvent_t>> ipcEvents = std::vector<std::vector<cudaEvent_t>>());

  template<typename ScalarType> void allreducep2pCrossNodes(
    const ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    const MPI::Intracomm& comm,
    std::vector<cudaStream_t> copyStreams);

}

// Collectives operating on THC*Tensor
template<typename ScalarType, typename THTensorType>
void broadcast(THCState* state,
               THTensorType* t,
               int src);

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* broadcastAsync(THCState* state,
                                      THTensorType* t,
                                      int src);

template<typename ScalarType, typename THTensorType>
void broadcastp2p(THCState* state,
                  THTensorType* t,
                  int src,
                  const MPI::Intracomm& comm,
                  IPCDesc* desc = nullptr);

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* broadcastp2pAsync(THCState* state,
                                         THTensorType* t,
                                         int src,
                                         const MPI::Intracomm& comm);

template<typename ScalarType, typename THTensorType>
void allreduce(THCState* state,
               THTensorType* t,
               THTensorType* output,
               MPI::Op mpiRedOp);

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreduceAsync(THCState* state,
                                      THTensorType* t,
                                      THTensorType* output,
                                      MPI::Op mpiRedOp);

template<typename ScalarType, typename THTensorType>
void allreducep2p(THCState* state,
                  THTensorType* t,
                  THTensorType* output,
                  MPI::Op mpiRedOp,
                  const MPI::Intracomm& comm,
                  IPCDesc* desc = nullptr);

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreducep2pAsync(THCState* state,
                                         THTensorType* t,
                                         THTensorType* output,
                                         MPI::Op mpiRedOp,
                                         const MPI::Intracomm& comm);

template<typename ScalarType, typename THTensorType>
void sendreceive(THCState* state,
                 THTensorType* t,
                 int src,
                 int dst);

}}}

#endif
