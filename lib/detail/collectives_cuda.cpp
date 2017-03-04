/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "../collectives.h"
#include "../collectives_cuda.h"
#include "../resources.h"
#include "reduce_kernel.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <thread>
#include <vector>

using namespace std;
using namespace torch::mpi::constants;
using namespace torch::mpi::resources;

namespace torch { namespace mpi { namespace thc { namespace detail {

namespace {

long transferLen(long start, long end) {
  return end - start + 1;
}

std::pair<long, long>
transferBounds(long chunkIndex, long bufferSize, long vectorLength) {
  return std::make_pair(
    chunkIndex * bufferSize,
    std::min((chunkIndex + 1) * bufferSize - 1, vectorLength - 1));
}

} // ns anon

template<typename ScalarType> void broadcastp2pIPC(
    ScalarType* outputData,
    size_t root,
    size_t nElement,
    IPCDesc* desc,
    size_t offset,
    const MPI::Intracomm& comm,
    cudaStream_t stream,
    std::vector<std::vector<cudaEvent_t>> ipcEvents) {
  auto rank = commRank(comm);
  auto size = commSize(comm);

  // TODO: Non-bus topologies
  std::vector<int> order(size);
  for (int i = 0; i < size; ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](int i1, int i2) {
    if (i1 == root) { return true; }
    return desc->allDevices[i1] < desc->allDevices[i2];
  });

  barrier(comm);
  THCudaCheck(cudaStreamSynchronize(stream));

  if (nElement * sizeof(ScalarType) <= kBcastSizeTreeBasedGPU) {
    // Tree-based
    for (int dist = 1; dist < size; dist = dist * 2) {
      // At each dist, 'dist' ranks are sending from order[0 .. dist] to
      // order[dist+1 .. 2*dist]. Use a pull model
      for (int senderIndex = 0; senderIndex < dist; ++senderIndex) {
        int receiverIndex = senderIndex + dist;
        if (receiverIndex < size) {
          int sender = order[senderIndex];
          int receiver = order[receiverIndex];
          THAssert(sender != receiver);

          // Pull model
          if (receiver == rank) {
            THCudaCheck(cudaStreamWaitEvent(stream, ipcEvents[0][sender], 0));
            THCudaCheck(cudaStreamWaitEvent(stream, ipcEvents[0][receiver], 0));
            THCudaCheck(
              cudaMemcpyAsync(
                outputData,
                static_cast<ScalarType*>(desc->allDevicePointers[sender]) +
                  offset,
                nElement * sizeof(ScalarType),
                cudaMemcpyDefault,
                stream)
            );
            THCudaCheck(cudaEventRecord(ipcEvents[0][receiver], stream));
          }
        }
      }

      // At this point, sender spawned a non-blocking stream to perform the
      // send
      barrier(comm);
      THCudaCheck(cudaStreamSynchronize(stream));
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
      static_cast<size_t>(kMinBufferSizeGPU),
      std::min(static_cast<size_t>(kMaxBufferSizeGPU),
               (nElement + kNumBuffersPerCollectiveGPU * size - 1) /
               (kNumBuffersPerCollectiveGPU * size)));
    auto rem = (nElement % chunkSize) ? 1 : 0;
    long totalChunks = nElement / chunkSize + rem;
    // Pipeline + pull model
    for (int step = 0; step < totalChunks + size; ++step) {
      long chunk;
      if (step + 1 < myPlace || step - totalChunks >= myPlace) {
        chunk = -1;
      } else {
        chunk = step + 1 - myPlace;
      }
      // Pull model, the root never pulls anything
      if (myPlace > 0 && chunk >= 0) {
        int sender = order[myPlace - 1];
        int receiver = order[myPlace];
        size_t chunkStart = chunk * chunkSize;
        size_t thisSize = std::min(chunkSize, nElement - chunkStart);
        if (nElement > chunkStart) {
          THCudaCheck(cudaStreamWaitEvent(stream, ipcEvents[0][sender], 0));
          THCudaCheck(cudaStreamWaitEvent(stream, ipcEvents[0][receiver], 0));
          THCudaCheck(cudaMemcpyAsync(
            outputData + chunkStart,
            static_cast<ScalarType*>(desc->allDevicePointers[sender]) +
            offset + chunkStart,
            thisSize * sizeof(ScalarType),
            cudaMemcpyDefault,
            stream));
          THCudaCheck(cudaEventRecord(ipcEvents[0][receiver], stream));
        }
      }

      // At this point, sender spawned a non-blocking stream to perform the
      // send
      barrier(comm);
      THCudaCheck(cudaStreamSynchronize(stream));
    }
  }

  THCudaCheck(cudaGetLastError());
}


// Buffers with
constexpr uintptr_t kAlign = 256;
void* getGPUBuffer(size_t sizeNeeded, size_t bufferIndex) {
  struct Buffers {
    size_t size;
    std::vector<void*> buffers;
    Buffers(size_t kMaxBufferSizeGPU) : size(kMaxBufferSizeGPU + kAlign) {
      for (auto i = 0; i < constants::kMaxNumBuffersPerCollectiveGPU; ++i) {
        void* ptr;
        THCudaCheck(cudaMalloc(&ptr, size * sizeof(double)));
        buffers.push_back(ptr);
      }
    }
    ~Buffers() {
      for (auto buf : buffers) {
        THCudaCheck(cudaFree(buf));
      }
    }
  };
  // Wrap buffers into RAII layer
  static std::unordered_map<std::thread::id, Buffers> buffers;
  static mutex mut;
  lock_guard<mutex> lg(mut);
  auto buf = buffers.find(std::this_thread::get_id());
  if ((buf = buffers.find(std::this_thread::get_id())) == buffers.end()) {
    buffers.emplace(std::this_thread::get_id(), kMaxBufferSizeGPU);
  }
  buf = buffers.find(std::this_thread::get_id());
  THAssert(buf != buffers.end());
  THAssert(buf->second.size >= sizeNeeded + kAlign);
  size_t actualIndex = bufferIndex % buf->second.buffers.size();
  THAssert(actualIndex < constants::kMaxNumBuffersPerCollectiveGPU);
  return buf->second.buffers[actualIndex];
}

cudaStream_t getStream(std::vector<cudaStream_t>& streams, int index) {
  if (streams.size() == 0) { return 0; }
  return streams[index % streams.size()];
}

template<typename ScalarType> void allreducep2pIPC(
    const ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    IPCDesc* desc,
    size_t offset,
    const MPI::Intracomm& comm,
    std::vector<cudaStream_t> copyStreams,
    std::vector<std::vector<cudaEvent_t>> ipcEvents) {
  if (mpiRedOp != MPI::Op(MPI_SUM)) {
    THError("NYI: MPI_allreducep2pIPC only supported for MPI_SUM atm");
  }

  // Between 4K and 4M elements per transfer, such that we have as many
  // chunks as number of participants.
  auto rank = commRank(comm);
  auto size = commSize(comm);
  VLOG_1("@comm: " << &comm << " " <<
         rank << "/" << size << " start AllreduceP2PIPC" << endl);

  auto nBuffers = copyStreams.size();
  THAssert(nBuffers == ipcEvents.size());
  auto bufferSize = std::min(
    kMaxBufferSizeGPU / sizeof(ScalarType),
    std::max(kMinBufferSizeGPU / sizeof(ScalarType),
             (nElement + size * nBuffers - 1) / (size * nBuffers)));
  auto rem = (nElement % bufferSize) ? 1 : 0;
  long totalChunks = nElement / bufferSize + rem;

  // TODO: Non-bus topologies
  std::vector<int> order(size);
  for (int i = 0; i < size; ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](int i1, int i2) {
    return desc->allDevices[i1] < desc->allDevices[i2];
  });
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
  auto prevIndex = (myPlace + size - 1) % size;
  auto currentIndex = myPlace;
  auto prev = order[prevIndex];
  auto current = order[currentIndex];
  int dev;
  THCudaCheck(cudaGetDevice(&dev));
  THAssert(desc->allDevices[current] == dev);

  THAssert(copyStreams.size() == ipcEvents.size());
  THAssert(ipcEvents[0].size() == size);

  auto pp = getPlan<CudaPlan>(totalChunks, currentIndex, prevIndex, rank, size);
  auto& planReduce = pp.first;
  auto& planBroadcast = pp.second;
  if (inputData != outputData) {
    THCudaCheck(cudaMemcpyAsync(
      outputData,
      inputData,
      nElement * sizeof(ScalarType),
      cudaMemcpyDefault,
      getStream(copyStreams, 0)));
    THCudaCheck(cudaStreamSynchronize(getStream(copyStreams, 0)));
  }

  for (int step = 0; step < size - 1; ++step) {
    barrier(comm);

    auto& plan = planReduce[step];
    {
      // Wait that event on the previous device is done.
      int index = 0;
      for (auto& copyStream : copyStreams) {
        THCudaCheck(
          cudaStreamWaitEvent(copyStream, ipcEvents[index][prev], 0));
        index++;
      }
    }

    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      THAssert(sendingChunk == receivingChunk);

      if (receivingChunk < 0 || sendingChunk < 0) {
        continue;
      }

      // Pull model, receive-centric
      auto p = transferBounds(sendingChunk, bufferSize, nElement);
      auto start = p.first, end = p.second;
      auto buf = getGPUBuffer(bufferSize * sizeof(ScalarType), bufferIndex);
      auto dataPtr = outputData + start;
      auto alignData = reinterpret_cast<uintptr_t>(dataPtr) % kAlign;
      auto alignBuf = reinterpret_cast<uintptr_t>(buf) % kAlign;
      auto alignGpuBuffer = (kAlign + alignData - alignBuf) % kAlign;

      THCudaCheck(cudaMemcpyAsync(
         reinterpret_cast<void*>(
           reinterpret_cast<uintptr_t>(buf) + alignGpuBuffer),
         static_cast<ScalarType*>(desc->allDevicePointers[prev]) +
           offset + start,
         transferLen(start, end) * sizeof(ScalarType),
         cudaMemcpyDefault,
         getStream(copyStreams, bufferIndex)));

      reduce(dataPtr,
             reinterpret_cast<ScalarType*>(
               reinterpret_cast<uintptr_t>(buf) + alignGpuBuffer),
             transferLen(start, end),
             getStream(copyStreams, bufferIndex));
      THCudaCheck(cudaGetLastError());
    }

    {
      // At this point, prev spawned a non-blocking stream to perform the
      // send. Record an event on which we can wait.
      int index = 0;
      for (auto& copyStream : copyStreams) {
        THCudaCheck(cudaEventRecord(ipcEvents[index][current], copyStream));
        index++;
      }
    }

  }

  // Make sure every CPU thread reached this point so we can synchronize with
  // ipcEvents (barrier just below)

  for (long step = 0; step < size - 1; ++step) {
    barrier(comm);
    auto& plan = planBroadcast[step];

    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      THAssert(sendingChunk == receivingChunk);
      if (receivingChunk < 0 || sendingChunk < 0) {
        continue;
      }

      THCudaCheck(
        cudaStreamWaitEvent(
          getStream(copyStreams, bufferIndex),
          ipcEvents[bufferIndex % nBuffers][prev],
          0)
      );

      auto p = transferBounds(sendingChunk, bufferSize, nElement);
      auto start = p.first, end = p.second;
      THCudaCheck(cudaMemcpyAsync(
         outputData + start,
         static_cast<ScalarType*>(desc->allDevicePointers[prev]) +
         offset + start,
         transferLen(start, end) * sizeof(ScalarType),
         cudaMemcpyDefault,
         getStream(copyStreams, bufferIndex)));
    }

    // At this point, prev spawned a non-blocking stream to perform the
    // send.
    int index = 0;
    for (auto& copyStream : copyStreams) {
      THCudaCheck(cudaEventRecord(ipcEvents[index][current], copyStream));
      index++;
    }
  }

  for (auto& copyStream : copyStreams) {
    THCudaCheck(cudaStreamSynchronize(copyStream));
  }

  barrier(comm);

  THCudaCheck(cudaGetLastError());
}

template<typename ScalarType> void allreducep2pCrossNodesViaCPU(
    const ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    const MPI::Intracomm& comm,
    std::vector<cudaStream_t> copyStreams) {
  if (mpiRedOp != MPI::Op(MPI_SUM)) {
    THError("NYI: MPI_allreducep2p only supported for MPI_SUM atm");
  }

  barrier(comm);

  auto rank = commRank(comm);
  auto size = commSize(comm);
  auto next = (rank + 1) % size;
  auto prev = (rank + size - 1) % size;

  auto nBuffers = copyStreams.size();
  auto bufferSize = std::min(
    kMaxBufferSizeGPU / sizeof(ScalarType),
    std::max(kMinBufferSizeGPU / sizeof(ScalarType),
             (nElement + size * nBuffers - 1) / (size * nBuffers)));
  auto rem = (nElement % bufferSize) ? 1 : 0;
  long totalChunks = nElement / bufferSize + rem;

  VLOG_1(rank << "/" << size << " allreducep2pCrossNodesViaCPU: " <<
         totalChunks << " totalChunks\n");

  auto pp = getPlan<MpiPlan>(totalChunks, rank, rank, rank, size);
  auto& planReduce = pp.first;
  auto& planBroadcast = pp.second;
  if (inputData != outputData) {
    THCudaCheck(cudaMemcpyAsync(
      outputData,
      inputData,
      nElement * sizeof(ScalarType),
      cudaMemcpyDefault,
      getStream(copyStreams, 0)));
    THCudaCheck(cudaStreamSynchronize(getStream(copyStreams, 0)));
  }

  //////////////////////////////////////////////////////////////////////////////
  // Start Reduce phase
  //////////////////////////////////////////////////////////////////////////////
  for (long step = 0; step < size - 1; ++step) {
    auto& plan = planReduce[step];
    std::vector<MPI::Request> reqSend(totalChunks), reqRecv(totalChunks);

    // 1. Post all Irecv
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (receivingChunk > -1) {
        VLOG_1(rank << "/" << size << " 1. bidx: " << bufferIndex << "\n");
        // Pull model, receive-centric
        auto p = transferBounds(receivingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          auto cpubuf = torch::mpi::resources::cuda::getPinnedBuffer(
              outputData + start, bufferSize * sizeof(ScalarType));
          VLOG_1(rank << "/" << size <<
                 " 1. Post: " << bufferIndex << "@" << receivingChunk << "\n");
          reqRecv[receivingChunk] = comm.Irecv(
            cpubuf,
            transferLen(start, end),
            mpiType<ScalarType>(),
            prev,
            bufferIndex);
        }
      }
    }

    // 2. Post all DtoH copies
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (sendingChunk > -1) {
        VLOG_1(rank << "/" << size << " 2. bidx: " << bufferIndex << "\n");
        // Pull model, receive-centric
        auto p = transferBounds(sendingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          auto cpubuf = torch::mpi::resources::cuda::getPinnedBuffer(
              outputData + start, bufferSize * sizeof(ScalarType));
          THCudaCheck(cudaMemcpyAsync(
            cpubuf,
            outputData + start,
            transferLen(start, end) * sizeof(ScalarType),
            cudaMemcpyDefault,
            getStream(copyStreams, bufferIndex)));

        }
      }
    }

    // 3. Pipeline stream synchronize waiting for GPU-CPU
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (sendingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(sendingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          VLOG_1(rank << "/" << size << " 4. bidx: " << bufferIndex << "\n");
          auto cpubuf = torch::mpi::resources::cuda::getPinnedBuffer(
              outputData + start, bufferSize * sizeof(ScalarType));
          THCudaCheck(
            cudaStreamSynchronize(getStream(copyStreams, bufferIndex)));
          reqSend[sendingChunk] = comm.Issend(
            cpubuf,
            transferLen(start, end),
            mpiType<ScalarType>(),
            next,
            bufferIndex);
        }
      }
    }

    // 4. Receive and reduce
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;

      if (receivingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(receivingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          VLOG_1(rank << "/" << size << " 5. bidx: " << bufferIndex << "\n");
          auto cpubuf = torch::mpi::resources::cuda::getPinnedBuffer(
            outputData + start, bufferSize * sizeof(ScalarType));
          auto gpubuf =
            getGPUBuffer(bufferSize * sizeof(ScalarType), bufferIndex);
          // Wait for cpubuf to be filled
          VLOG_1(rank << "/" << size <<
                 " 5. wait: " << receivingChunk << "/" << totalChunks << "\n");
          reqRecv[receivingChunk].Wait();
          auto dataPtr = outputData + start;
          auto alignData = reinterpret_cast<uintptr_t>(dataPtr) % kAlign;
          auto alignBuf = reinterpret_cast<uintptr_t>(gpubuf) % kAlign;
          auto alignGpuBuffer = (kAlign + alignData - alignBuf) % kAlign;

          THCudaCheck(cudaMemcpyAsync(
            reinterpret_cast<void*>(
              reinterpret_cast<uintptr_t>(gpubuf) + alignGpuBuffer),
            cpubuf,
            transferLen(start, end) * sizeof(ScalarType),
            cudaMemcpyHostToDevice,
            getStream(copyStreams, bufferIndex)));

          reduce<ScalarType>(
            dataPtr,
            reinterpret_cast<ScalarType*>(
              reinterpret_cast<uintptr_t>(gpubuf) + alignGpuBuffer),
            transferLen(start, end),
            getStream(copyStreams, bufferIndex));

        }
      }
    }

    // 5. Ensure all chunks are finished, don't synchronize reductions we'll
    // just reuse streams
    for (auto& r : reqSend) { r.Wait(); }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Start Broadcast phase
  //////////////////////////////////////////////////////////////////////////////
  for (long step = 0; step < size - 1; ++step) {
    auto& plan = planBroadcast[step];
    std::vector<MPI::Request> reqSend(totalChunks), reqRecv(totalChunks);

    // 1. Post all Irecv
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (receivingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(receivingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          auto cpubuf = torch::mpi::resources::cuda::getPinnedBuffer(
              outputData + start, bufferSize * sizeof(ScalarType));
          reqRecv[receivingChunk] = comm.Irecv(
            cpubuf,
            transferLen(start, end),
            mpiType<ScalarType>(),
            prev,
            bufferIndex);
        }
      }
    }

    // 2. Post all DtoH copies
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (sendingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(sendingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          auto cpubuf = torch::mpi::resources::cuda::getPinnedBuffer(
              outputData + start, bufferSize * sizeof(ScalarType));
          THCudaCheck(cudaMemcpyAsync(
            cpubuf,
            outputData + start,
            transferLen(start, end) * sizeof(ScalarType),
            cudaMemcpyDefault,
            getStream(copyStreams, bufferIndex)));
        }
      }
    }

    // 3. Pipeline stream synchronize waiting for GPU-CPU
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (sendingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(sendingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          auto cpubuf = torch::mpi::resources::cuda::getPinnedBuffer(
            outputData + start, bufferSize * sizeof(ScalarType));
          THCudaCheck(
            cudaStreamSynchronize(getStream(copyStreams, bufferIndex)));
          reqSend[sendingChunk] = comm.Issend(
            cpubuf,
            transferLen(start, end),
            mpiType<ScalarType>(),
            next,
            bufferIndex);

        }
      }
    }

    // 4. Receive
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;

      if (receivingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(receivingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          auto cpubuf = torch::mpi::resources::cuda::getPinnedBuffer(
            outputData + start, bufferSize * sizeof(ScalarType));
          // Wait for cpubuf to be filled
          reqRecv[receivingChunk].Wait();
          THCudaCheck(cudaMemcpyAsync(
             outputData + start,
             cpubuf,
             transferLen(start, end) * sizeof(ScalarType),
             cudaMemcpyDefault,
             getStream(copyStreams, bufferIndex)));
        }
      }

    }

    // 4. Ensure all chunks are finished
    for (auto& r : reqSend) { r.Wait(); }
  }

  // Synchronize all copyStreams on exit
  barrier(comm);
  for (auto& copyStream : copyStreams) {
    THCudaCheck(cudaStreamSynchronize(copyStream));
  }

  THCudaCheck(cudaGetLastError());
}


template<typename ScalarType> void allreducep2pCrossNodesDirect(
    const ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    const MPI::Intracomm& comm,
    std::vector<cudaStream_t> copyStreams) {
  if (mpiRedOp != MPI::Op(MPI_SUM)) {
    THError("NYI: MPI_allreducep2p only supported for MPI_SUM atm");
  }

  auto rank = commRank(comm);
  auto size = commSize(comm);
  auto next = (rank + 1) % size;
  auto prev = (rank + size - 1) % size;

  auto nBuffers = copyStreams.size();
  auto bufferSize = std::min(
    kMaxBufferSizeGPU / sizeof(ScalarType),
    std::max(kMinBufferSizeGPU / sizeof(ScalarType),
             (nElement + size * nBuffers - 1) / (size * nBuffers)));
  auto rem = (nElement % bufferSize) ? 1 : 0;
  long totalChunks = nElement / bufferSize + rem;

  auto pp = getPlan<MpiPlan>(totalChunks, rank, rank, rank, size);
  auto& planReduce = pp.first;
  auto& planBroadcast = pp.second;
  if (inputData != outputData) {
    THCudaCheck(cudaMemcpyAsync(
      outputData,
      inputData,
      nElement * sizeof(ScalarType),
      cudaMemcpyDefault,
      getStream(copyStreams, 0)));
    THCudaCheck(cudaStreamSynchronize(getStream(copyStreams, 0)));
  }

  //////////////////////////////////////////////////////////////////////////////
  // Start Reduce phase
  //////////////////////////////////////////////////////////////////////////////
  for (long step = 0; step < size - 1; ++step) {
    auto& plan = planReduce[step];
    std::vector<MPI::Request> reqSend(totalChunks), reqRecv(totalChunks);

    // 1. Post all Irecv
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (receivingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(receivingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          auto gpubuf =
            getGPUBuffer(bufferSize * sizeof(ScalarType), bufferIndex);
          auto dataPtr = outputData + start;
          auto alignData = reinterpret_cast<uintptr_t>(dataPtr) % kAlign;
          auto alignBuf = reinterpret_cast<uintptr_t>(gpubuf) % kAlign;
          auto alignGpuBuffer = (kAlign + alignData - alignBuf) % kAlign;
          reqRecv[receivingChunk] = comm.Irecv(
            reinterpret_cast<ScalarType*>(
              reinterpret_cast<uintptr_t>(gpubuf) + alignGpuBuffer),
            transferLen(start, end),
            mpiType<ScalarType>(),
            prev,
            bufferIndex);
        }
      }
    }

    // 2. Pipeline stream synchronize waiting for GPU-GPU copies
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (sendingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(sendingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          THCudaCheck(
            cudaStreamSynchronize(getStream(copyStreams, bufferIndex)));
          reqSend[sendingChunk] = comm.Issend(
            (void*)(outputData + start),
            transferLen(start, end),
            mpiType<ScalarType>(),
            next,
            bufferIndex);
        }
      }
    }

    // 3. Receive data and start local reduction
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;

      if (receivingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(receivingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          auto gpubuf =
            getGPUBuffer(bufferSize * sizeof(ScalarType), bufferIndex);
          reqRecv[receivingChunk].Wait();
          auto dataPtr = outputData + start;
          auto alignData = reinterpret_cast<uintptr_t>(dataPtr) % kAlign;
          auto alignBuf = reinterpret_cast<uintptr_t>(gpubuf) % kAlign;
          auto alignGpuBuffer = (kAlign + alignData - alignBuf) % kAlign;
          reduce<ScalarType>(
            outputData + start,
            reinterpret_cast<ScalarType*>(
              reinterpret_cast<uintptr_t>(gpubuf) + alignGpuBuffer),
            transferLen(start, end),
            getStream(copyStreams, bufferIndex));

        }
      }
    }

    // 4. Ensure all chunks are finished, don't synchronize reductions now
    for (auto& r : reqSend) { r.Wait(); }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Start Broadcast phase
  //////////////////////////////////////////////////////////////////////////////
  for (long step = 0; step < size - 1; ++step) {
    auto& plan = planBroadcast[step];
    std::vector<MPI::Request> reqSend(totalChunks), reqRecv(totalChunks);

    // 1. Post all Irecv
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (receivingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(receivingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          reqRecv[receivingChunk] = comm.Irecv(
            outputData + start,
            transferLen(start, end),
            mpiType<ScalarType>(),
            prev,
            bufferIndex);
        }
      }
    }

    // 2. Make sure reduction finished before sending
    for (long startChunk = 0, bufferIndex = 0;
         startChunk < totalChunks;
         startChunk += (long)size, ++bufferIndex) {
      long sendingChunk = plan[bufferIndex].first;
      long receivingChunk = plan[bufferIndex].second;
      if (sendingChunk > -1) {
        // Pull model, receive-centric
        auto p = transferBounds(sendingChunk, bufferSize, nElement);
        auto start = p.first, end = p.second;
        if (start <= end) {
          THCudaCheck(
            cudaStreamSynchronize(getStream(copyStreams, bufferIndex)));
          reqSend[sendingChunk] = comm.Issend(
            outputData + start,
            transferLen(start, end),
            mpiType<ScalarType>(),
            next,
            bufferIndex);
        }
      }
    }

    // 3. Ensure all chunks are finished
    for (auto& r : reqSend) { r.Wait(); }
    for (auto& r : reqRecv) { r.Wait(); }
  }

  // Synchronize on exit
  barrier(comm);

  THCudaCheck(cudaGetLastError());
}

template<typename ScalarType> void allreducep2pCrossNodes(
    const ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    const MPI::Intracomm& comm,
    std::vector<cudaStream_t> copyStreams) {
  if (!constants::kUseStagedCollectives) {
    allreducep2pCrossNodesDirect(inputData,
                                 outputData,
                                 nElement,
                                 mpiRedOp,
                                 comm,
                                 copyStreams);
  } else {
    allreducep2pCrossNodesViaCPU(inputData,
                                 outputData,
                                 nElement,
                                 mpiRedOp,
                                 comm,
                                 copyStreams);
  }
}


}}}} // ns torch::mpi::thc::detail


// Explicit template instantiations
#define INSTANTIATE_broadcastp2pIPC(TYPE)                       \
  template void torch::mpi::thc::detail::broadcastp2pIPC<TYPE>( \
    TYPE*,                                                      \
    size_t,                                                     \
    size_t,                                                     \
    IPCDesc* desc,                                              \
    size_t offset,                                              \
    const MPI::Intracomm&,                                      \
    cudaStream_t stream,                                        \
    std::vector<std::vector<cudaEvent_t>> ipcEvents);

INSTANTIATE_broadcastp2pIPC(uint8_t);
INSTANTIATE_broadcastp2pIPC(char);
INSTANTIATE_broadcastp2pIPC(short);
INSTANTIATE_broadcastp2pIPC(int);
INSTANTIATE_broadcastp2pIPC(long);
INSTANTIATE_broadcastp2pIPC(float);
INSTANTIATE_broadcastp2pIPC(double);

#define INSTANTIATE_allreducep2pIPC(TYPE)                       \
  template void torch::mpi::thc::detail::allreducep2pIPC<TYPE>( \
    const TYPE*,                                                \
    TYPE*,                                                      \
    size_t,                                                     \
    decltype(MPI::Op(MPI_SUM)),                                 \
    IPCDesc* desc,                                              \
    size_t offset,                                              \
    const MPI::Intracomm&,                                      \
    std::vector<cudaStream_t> copyStreams,                      \
    std::vector<std::vector<cudaEvent_t>> ipcEvents);

INSTANTIATE_allreducep2pIPC(uint8_t);
INSTANTIATE_allreducep2pIPC(char);
INSTANTIATE_allreducep2pIPC(short);
INSTANTIATE_allreducep2pIPC(int);
INSTANTIATE_allreducep2pIPC(long);
INSTANTIATE_allreducep2pIPC(float);
INSTANTIATE_allreducep2pIPC(double);

#define INSTANTIATE_allreducep2pCrossNodes(TYPE)                        \
  template void torch::mpi::thc::detail::allreducep2pCrossNodes<TYPE>(  \
    const TYPE* inputData,                                              \
    TYPE* outputData,                                                   \
    size_t nElement,                                                    \
    MPI::Op mpiRedOp,                                                   \
    const MPI::Intracomm& comm,                                         \
    std::vector<cudaStream_t> copyStreams);

INSTANTIATE_allreducep2pCrossNodes(uint8_t);
INSTANTIATE_allreducep2pCrossNodes(char);
INSTANTIATE_allreducep2pCrossNodes(short);
INSTANTIATE_allreducep2pCrossNodes(int);
INSTANTIATE_allreducep2pCrossNodes(long);
INSTANTIATE_allreducep2pCrossNodes(float);
INSTANTIATE_allreducep2pCrossNodes(double);
