/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "../collectives.h"
#include "../resources.h"

#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <thread>
#include <vector>

/**********************************************************************
 ******************************** MPI *********************************
 **********************************************************************/
using namespace std;
using namespace torch::mpi::constants;
using namespace torch::mpi::resources;

namespace torch { namespace mpi { namespace th { namespace detail {

template<typename ScalarType> void broadcastp2p(
    ScalarType* tensorData,
    size_t root,
    size_t nElement,
    const MPI::Intracomm& comm) {
  auto rank = commRank(comm);
  auto size = commSize(comm);

  // TODO: Topologies
  std::vector<int> order(size);
  for (int i = 0; i < size; ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](int i1, int i2) {
    if (i1 == root) { return true; }
    return i1 < i2;
  });

  if (nElement * sizeof(ScalarType) <= kBcastSizeTreeBasedCPU) {
    // Tree-based
    std::vector<MPI::Request> reqRecv;
    for (int dist = 1; dist < size; dist = dist * 2) {
      THAssert(dist < kServerChunkTag);
      // At each dist, 'dist' ranks are sending from order[0 .. dist] to
      // order[dist+1 .. 2*dist]. Use a pull model
      for (int senderIndex = 0; senderIndex < dist; ++senderIndex) {
        int receiverIndex = senderIndex + dist;
        if (receiverIndex < size) {
          int sender = order[senderIndex];
          int receiver = order[receiverIndex];
          THAssert(sender != receiver);
          if (rank == sender) {
            comm.Send(
              tensorData, nElement, mpiType<ScalarType>(), receiver, dist);
          } else if (rank == receiver) {
            comm.Recv(
              tensorData, nElement, mpiType<ScalarType>(), sender, dist);
          }
        }
      }
    }
  } else {
    // Pipeline based
    int ind = 0, myPlace = -1;
    for (auto i : order) {
      if (i == rank) {
        myPlace = ind;
        break;
      }
      ind++;
    }
    THAssert(myPlace >= 0);
    THAssert(rank == order[myPlace]);

    auto chunkSize = std::max(
      static_cast<size_t>(kMinBufferSizeCPU),
      std::min(static_cast<size_t>(kMaxBufferSizeCPU),
               (nElement + kNumBuffersPerCollectiveCPU * size - 1) /
               (kNumBuffersPerCollectiveCPU * size)));
    auto rem = (nElement % chunkSize) ? 1 : 0;
    long totalChunks = nElement / chunkSize + rem;

    // Pipeline
    auto prev = (myPlace > 0) ? order[myPlace - 1] : -1;
    auto next = (myPlace < order.size() - 1) ? order[myPlace + 1] : -1;
    std::vector<MPI::Request> reqs(totalChunks);
    for (int chunk = 0; chunk < totalChunks; ++chunk) {
      size_t chunkStart = chunk * chunkSize;
      size_t thisSize = std::min(chunkSize, nElement - chunkStart);
      if (prev >= 0) {
        reqs[chunk] = comm.Irecv(
          tensorData + chunkStart, thisSize, mpiType<ScalarType>(), prev, chunk);
      }
    }
    for (int chunk = 0; chunk < totalChunks; ++chunk) {
      size_t chunkStart = chunk * chunkSize;
      size_t thisSize = std::min(chunkSize, nElement - chunkStart);
      if (next >= 0) {
        reqs[chunk].Wait();
        reqs[chunk] = comm.Issend(
          tensorData + chunkStart, thisSize, mpiType<ScalarType>(), next, chunk);
        reqs[chunk].Wait();
      }
    }
    for (auto& r : reqs) { r.Wait(); }
  }
}


template<typename ScalarType> void reduce(
  ScalarType* out,
  const ScalarType* in,
  size_t size,
  decltype(MPI::Op(MPI_SUM)) mpiRedOp)
{
  for (size_t i = 0; i < size; ++i) {
    out[i] += in[i];
  }
}

// CPU memory hog, fine for now
void* getBuffer(void* dataPtr, int bufferSize, int bufferIndex = 0) {
  struct BufferWrapper {
    std::unordered_map<void*, std::vector<void*>> buffers_;
    ~BufferWrapper() {
      for (auto kvp : buffers_) {
        for (auto buf : kvp.second) {
          free(buf);
        }
      }
    }
  };
  static BufferWrapper wrap; // Wrap buffers into RAII layer
  static mutex mut;
  lock_guard<mutex> lg(mut);

  THAssert(bufferIndex < constants::kNumBuffersPerCollectiveCPU);
  auto& buffers = wrap.buffers_;
  if (buffers.find(dataPtr) == buffers.end()) {
    auto v = std::vector<void*>(constants::kNumBuffersPerCollectiveCPU);
    buffers.emplace(dataPtr, v);
  }
  THAssert(bufferIndex < constants::kNumBuffersPerCollectiveCPU);
  if (!buffers[dataPtr][bufferIndex]) {
    buffers[dataPtr][bufferIndex] = malloc(bufferSize);
  }
  return buffers[dataPtr][bufferIndex];
}

template<typename ScalarType>
void allreducep2p(
    const ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    const MPI::Intracomm& comm) {
  if (mpiRedOp != MPI::Op(MPI_SUM)) {
    THError("NYI: MPI_allreducep2p only supported for MPI_SUM atm");
  }

  auto rank = commRank(comm);
  auto size = commSize(comm);
  auto next = (rank + 1) % size;
  auto prev = (rank + size - 1) % size;

  auto bufferSize =
    std::max(
      static_cast<unsigned long>(1 << 13),
      (nElement + size * constants::kNumBuffersPerCollectiveCPU - 1) /
      (size * constants::kNumBuffersPerCollectiveCPU));
  auto rem = (nElement % bufferSize) ? 1 : 0;
  long totalChunks = nElement / bufferSize + rem;

  auto pp = getPlan<MpiPlan>(totalChunks, rank, rank, rank, size);
  auto& planReduce = pp.first;
  auto& planBroadcast = pp.second;
  // TODO: remove this extra copy
  if (outputData != inputData) {
    memcpy(outputData, inputData, sizeof(ScalarType) * nElement);
  }

  for (long step = 0; step < size - 1; ++step) {
    auto& plan = planReduce[step];
    std::vector<MPI::Request> reqSend(totalChunks), reqRecv(totalChunks);

    // 1. Post all Irecv
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (receivingChunk > -1) {
        auto buf = getBuffer(
          outputData, bufferSize * sizeof(ScalarType), startChunkIndex);
        auto receiveStart = receivingChunk * bufferSize;
        auto receiveEnd =
          std::min((receivingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(receiveEnd - receiveStart + 1);

        if (receiveStart <= receiveEnd) {
          reqRecv[receivingChunk] = comm.Irecv(
            static_cast<ScalarType*>(buf),
            len,
            mpiType<ScalarType>(),
            prev,
            0);
        }
      }
    }

    // 2. Post all asynchronous ISend
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (sendingChunk > -1) {
        auto sendStart = sendingChunk * bufferSize;
        auto sendEnd =
          std::min((sendingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(sendEnd - sendStart + 1);

        if (sendStart <= sendEnd) {
          reqSend[sendingChunk] = comm.Issend(
            outputData + sendStart,
            len,
            mpiType<ScalarType>(),
            next,
            0);
        }
      }
    }

    // 3. Overlap compute and copies
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;

      if (receivingChunk > -1) {
        auto buf = getBuffer(
          outputData, bufferSize * sizeof(ScalarType), startChunkIndex);
        reqRecv[receivingChunk].Wait();

        auto receiveStart = receivingChunk * bufferSize;
        auto receiveEnd =
          std::min((receivingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(receiveEnd - receiveStart + 1);
        if (receiveStart <= receiveEnd) {
          reduce<ScalarType>(
            outputData + receiveStart,
            static_cast<ScalarType*>(buf),
            len,
            mpiRedOp
          );
        }
      }
    }

    // 4. Ensure all chunks are finished
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (sendingChunk > -1) {
        reqSend[sendingChunk].Wait();
      }
    }
  }

  for (long step = 0; step < size - 1; ++step) {
    auto& plan = planBroadcast[step];
    std::vector<MPI::Request> reqSend(totalChunks), reqRecv(totalChunks);

    // 1. Post all Irecv
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (receivingChunk > -1) {
        auto receiveStart = receivingChunk * bufferSize;
        auto receiveEnd =
          std::min((receivingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(receiveEnd - receiveStart + 1);

        if (receiveStart <= receiveEnd) {
          reqRecv[receivingChunk] = comm.Irecv(
            outputData + receiveStart,
            len,
            mpiType<ScalarType>(),
            prev,
            0);
        }
      }
    }

    // 2. Post all asynchronous ISend
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (sendingChunk > -1) {
        auto sendStart = sendingChunk * bufferSize;
        auto sendEnd =
          std::min((sendingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(sendEnd - sendStart + 1);
        if (sendStart <= sendEnd) {
          reqSend[sendingChunk] = comm.Issend(
            outputData + sendStart,
            len,
            mpiType<ScalarType>(),
            next,
            0);
        }
      }
    }

    // 3. Ensure all chunks are finished
    for (auto& r : reqSend) { r.Wait(); }
    for (auto& r : reqRecv) { r.Wait(); }
  }
}

}}}} // ns torch::mpi::th::detail


#define INSTANTIATE_broadcastp2p(TYPE)                          \
template void torch::mpi::th::detail::broadcastp2p<TYPE>(       \
    TYPE* tensorData,                                           \
    size_t root,                                                \
    size_t nElement,                                            \
    const MPI::Intracomm& comm);

INSTANTIATE_broadcastp2p(uint8_t);
INSTANTIATE_broadcastp2p(char);
INSTANTIATE_broadcastp2p(short);
INSTANTIATE_broadcastp2p(int);
INSTANTIATE_broadcastp2p(long);
INSTANTIATE_broadcastp2p(float);
INSTANTIATE_broadcastp2p(double);

#define INSTANTIATE_allreducep2p(TYPE)                          \
template void torch::mpi::th::detail::allreducep2p<TYPE>(       \
  const TYPE* inputData,                                        \
  TYPE* outputData,                                             \
  size_t nElement,                                              \
  decltype(MPI::Op(MPI_SUM)) mpiRedOp,                          \
  const MPI::Intracomm& comm);

INSTANTIATE_allreducep2p(uint8_t);
INSTANTIATE_allreducep2p(char);
INSTANTIATE_allreducep2p(short);
INSTANTIATE_allreducep2p(int);
INSTANTIATE_allreducep2p(long);
INSTANTIATE_allreducep2p(float);
INSTANTIATE_allreducep2p(double);
