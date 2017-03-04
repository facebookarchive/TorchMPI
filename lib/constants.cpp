/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "constants.h"
#include <TH.h>

namespace torch { namespace mpi { namespace constants {

#ifdef TORCH_MPI_NCCL

template<> ncclDataType_t ncclType<char>() {
  return ncclChar;
}
template<> ncclDataType_t ncclType<int>() {
  return ncclInt;
}
template<> ncclDataType_t ncclType<float>() {
  return ncclFloat;
}
template<> ncclDataType_t ncclType<double>() {
  return ncclDouble;
}
template<> ncclDataType_t ncclType<long>() {
  return ncclInt64;
}
template<> ncclDataType_t ncclType<unsigned long>() {
  return ncclUint64;
}

MPI::Op mpiOp(ncclRedOp_t ncclRedOp) {
  if (ncclRedOp == ncclSum) { return MPI_SUM; }
  if (ncclRedOp == ncclProd) { return MPI_PROD; }
  if (ncclRedOp == ncclMax) { return MPI_MAX; }
  if (ncclRedOp == ncclMin) { return MPI_MIN; }
  THError("RedOpnot supported by both NCCL and MPI");
  return MPI_SUM;
}
#endif

template<> MPI::Datatype mpiType<char>() {
  return MPI_CHAR;
}
template<> MPI::Datatype mpiType<int>() {
  return MPI_INT;
}
template<> MPI::Datatype mpiType<float>() {
  return MPI_FLOAT;
}
template<> MPI::Datatype mpiType<double>() {
  return MPI_DOUBLE;
}
template<> MPI::Datatype mpiType<long>() {
  return MPI_LONG;
}
template<> MPI::Datatype mpiType<unsigned long>() {
  return MPI_UNSIGNED_LONG;
}

// After some point, constants become immutable
bool immutableConstants = false;

bool kUseStagedCollectives = false;
bool kUseHierarchicalCollectives = true;
bool kUseCartesianCommunicator = false;

int kSmallBcastSizeGPU = 1 << 13;
int kSmallAllreduceSizeGPU = 1 << 16;

int kSmallBcastSizeCPU = 1 << 13;
int kSmallAllreduceSizeCPU = 1 << 16;

int kMinBufferSizeGPU = 1 << 17;
int kMaxBufferSizeGPU = 1 << 20;
int kMinBufferSizeCPU = 1 << 17;
int kMaxBufferSizeCPU = 1 << 20;
int kBcastSizeTreeBasedCPU = 1 << 22;
int kBcastSizeTreeBasedGPU = 1 << 22;

int kNumBuffersPerCollectiveCPU = 3;
int kNumBuffersPerCollectiveGPU = 3;

int kCollectiveOffloadThreadPoolSize = 4;
int kNumAsyncCollectivesInFlight = 1 << 20;
int kParameterServerOffloadThreadPoolSize = 4;
int kNumAsyncParameterServersInFlight = 1 << 20;

}}} // torch::mpi::constants

using namespace torch::mpi::constants;

extern "C" {

  void torchmpi_set_flat_collectives() {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kUseHierarchicalCollectives = false;
  }
  void torchmpi_set_hierarchical_collectives() {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kUseHierarchicalCollectives = true;
  }
  void torchmpi_set_staged_collectives() {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kUseStagedCollectives = true;
  }
  void torchmpi_set_direct_collectives() {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kUseStagedCollectives = false;
  }
  void torchmpi_set_cartesian_communicator() {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kUseCartesianCommunicator = true;
  }
  void torchmpi_set_tree_communicator() {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kUseCartesianCommunicator = false;
  }
  void torchmpi_set_small_cpu_broadcast_size(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kSmallBcastSizeCPU = n;
  }
  void torchmpi_set_small_cpu_allreduce_size(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kSmallAllreduceSizeCPU = n;
  }
  void torchmpi_set_small_gpu_broadcast_size(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kSmallBcastSizeGPU = n;
  }
  void torchmpi_set_small_gpu_allreduce_size(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kSmallAllreduceSizeGPU = n;
  }
  void torchmpi_set_min_buffer_size_per_cpu_collective(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kMinBufferSizeCPU = n;
  }
  void torchmpi_set_min_buffer_size_per_gpu_collective(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kMinBufferSizeGPU = n;
  }
  void torchmpi_set_max_buffer_size_per_cpu_collective(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kMaxBufferSizeCPU = n;
  }
  void torchmpi_set_broadcast_size_cpu_tree_based(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kBcastSizeTreeBasedCPU = n;
  }
  void torchmpi_set_broadcast_size_gpu_tree_based(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kBcastSizeTreeBasedGPU = n;
  }
  void torchmpi_set_max_buffer_size_per_gpu_collective(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kMaxBufferSizeGPU = n;
  }
  void torchmpi_set_num_buffers_per_cpu_collective(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    if (n > kMaxNumBuffersPerCollectiveCPU) {
      THError("Unsupported n > kMaxNumBuffersPerCollectiveCPU");
    }
    kNumBuffersPerCollectiveCPU = n;
  }
  void torchmpi_set_num_buffers_per_gpu_collective(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    if (n > kMaxNumBuffersPerCollectiveGPU) {
      THError("Unsupported n > kMaxNumBuffersPerCollectiveGPU");
    }
    kNumBuffersPerCollectiveGPU = n;
  }
  void torchmpi_set_collective_num_threads(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kCollectiveOffloadThreadPoolSize = n;
  }
  void torchmpi_set_collective_thread_pool_size(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kNumAsyncParameterServersInFlight = n;
  }
  void torchmpi_set_parameterserver_num_threads(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kParameterServerOffloadThreadPoolSize = n;
  }
  void torchmpi_set_parameterserver_thread_pool_size(int n) {
    if (immutableConstants) {
      THError("Communications related constants are immutable after initialization");
    }
    kNumAsyncParameterServersInFlight = n;
  }

  int torchmpi_get_small_cpu_broadcast_size() {
    return kSmallBcastSizeCPU;
  }
  int torchmpi_get_small_cpu_allreduce_size() {
    return kSmallAllreduceSizeCPU;
  }
  int torchmpi_get_small_gpu_broadcast_size() {
    return kSmallBcastSizeGPU;
  }
  int torchmpi_get_small_gpu_allreduce_size() {
    return kSmallAllreduceSizeGPU;
  }
  int torchmpi_get_min_buffer_size_per_cpu_collective() {
    return kMinBufferSizeCPU;
  }
  int torchmpi_get_min_buffer_size_per_gpu_collective() {
    return kMinBufferSizeGPU;
  }
  int torchmpi_get_max_buffer_size_per_cpu_collective() {
    return kMaxBufferSizeCPU;
  }
  int torchmpi_get_broadcast_size_cpu_tree_based() {
    return kBcastSizeTreeBasedCPU;
  }
  int torchmpi_get_broadcast_size_gpu_tree_based() {
    return kBcastSizeTreeBasedGPU;
  }
  int torchmpi_get_max_buffer_size_per_gpu_collective() {
    return kMaxBufferSizeGPU;
  }
  int torchmpi_get_num_buffers_per_cpu_collective() {
    return kNumBuffersPerCollectiveCPU;
  }
  int torchmpi_get_num_buffers_per_gpu_collective() {
    return kNumBuffersPerCollectiveGPU;
  }
  int torchmpi_get_collective_num_threads() {
    return kCollectiveOffloadThreadPoolSize;
  }
  int torchmpi_get_collective_thread_pool_size() {
    return kNumAsyncParameterServersInFlight;
  }
  int torchmpi_get_parameterserver_num_threads() {
    return kParameterServerOffloadThreadPoolSize;
  }
  int torchmpi_get_parameterserver_thread_pool_size() {
    return kNumAsyncParameterServersInFlight;
  }


}
