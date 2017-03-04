/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

extern "C" {

  //////////////////////////////////////////////////////////////////////////////
  ///////////////////////// Basic functions ////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  // Start / stop must always be called on entry / exit of the program
  void torchmpi_start();
  void torchmpi_stop();

  // Rank / size / barrier are communicator dependent
  int torchmpi_rank();
  int torchmpi_size();

  // Is NCCL present
  int torchmpi_has_nccl();

  // Is Gloo present
  int torchmpi_has_gloo();

  // Is Gloo-cuda present
  int torchmpi_has_gloo_cuda();

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////// Communicators Management //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  // String representation of the available communicators at each node
  const char* torchmpi_communicator_names();

  // Push a new level of communicator, that discriminates by string key
  // Atm only a linear structure of communicators is supported (no tree
  // decomposition).
  // ***Collective***
  // This is a collective operation that must be called by all nodes in the
  // current communicator.
  int torchmpi_push_communicator(const char* key);

  // Sets the current communicator to the proper level
  // ***Collective***
  // This is a local operation but deadlocks will occur down
  // the road if mismatches occur. Better view this as a collective operation.
  void torchmpi_set_communicator(int level);

  // Queries whether the current communicator is cartesian or not
  bool torchmpi_is_cartesian_communicator();

  // Sets the levels of communicators use for collective operations
  // By default the 2 innermost levels are used for collectives.
  // This is useful for setting up communicators for parameterservers on top
  // of communicators for collectives.
  // ***Collective***
  // This is a local operation but deadlocks will occur down
  // the road if mismatches occur. Better view this as a collective operation.
  void torchmpi_set_collective_span(int begin, int end);

  // Returns the number of nodes in the current communicator
  int torchmpi_num_nodes_in_communicator();

  void torchmpi_free_ipc_descriptors();

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////// Important constants /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  // ***Collective***
  // The operations below are local but deadlocks will occur if different nodes
  // in a communicator use different implementations. Better treat them all as
  // collective operation.

  // By default collectives are implemented using 2 levels of hierarchy:
  //   1. intra-node/IPC level where GPUs can communicate using cudaIPC
  //   2. inter-node/IPC level where exchange are staged via CPU
  // In the particular context of RDMA when GDR is properly configured, GPUs
  // may communicate at very low latency without staging through the CPU.
  // For such cases a flat collective is better, use this function to activate
  // it.
  // Irrespective of whether flat or hierarchical collectives are used, one
  //   can also use explicitly staged CPU collectives (default) or direct
  //   MPI_Send/Recv transfers relying on MPI to do the proper thing.
  //
  // Caution: If GDR is improperly configured, staging via CPU will occur
  // automatically. In this case a flat collective implementation may be
  // limited by PCI-e bandwidth and the hierarchical implementation will still
  // perform better.
  //
  // Best way to be sure is to measure the performance of collectives using
  // the benchmarks, or implement an autotuner; YMMV.
  void torchmpi_set_flat_collectives();
  void torchmpi_set_hierarchical_collectives();
  void torchmpi_set_staged_collectives();
  void torchmpi_set_direct_collectives();
  void torchmpi_set_cartesian_communicator();
  void torchmpi_set_tree_communicator();

  // For collectives on small vectors, stock MPI implementations are usually
  // quite competitive. These functions change the trigger value.
  void torchmpi_set_small_cpu_broadcast_size(int n); // default 1 << 13
  void torchmpi_set_small_cpu_allreduce_size(int n); // default 1 << 16
  void torchmpi_set_small_gpu_broadcast_size(int n); // default 1 << 13
  void torchmpi_set_small_gpu_allreduce_size(int n); // default 1 << 16
  void torchmpi_set_min_buffer_size_per_cpu_collective(int n); // default (1 << 17)
  void torchmpi_set_min_buffer_size_per_gpu_collective(int n); // default (1 << 17)
  void torchmpi_set_max_buffer_size_per_cpu_collective(int n); // default (1 << 22)
  void torchmpi_set_max_buffer_size_per_gpu_collective(int n); // default (1 << 22)

  int torchmpi_get_small_cpu_broadcast_size();       // default 1 << 13
  int torchmpi_get_small_cpu_allreduce_size();       // default 1 << 16
  int torchmpi_get_small_gpu_broadcast_size();       // default 1 << 13
  int torchmpi_get_small_gpu_allreduce_size();       // default 1 << 16
  int torchmpi_get_min_buffer_size_per_cpu_collective();     // default (1 << 17)
  int torchmpi_get_min_buffer_size_per_gpu_collective();     // default (1 << 17)
  int torchmpi_get_max_buffer_size_per_cpu_collective();     // default (1 << 22)
  int torchmpi_get_max_buffer_size_per_gpu_collective();     // default (1 << 22)

  // How many buffers per collective should be used.
  // Note this number is per helper thread.
  void torchmpi_set_num_buffers_per_cpu_collective(int n); // default 1
  void torchmpi_set_num_buffers_per_gpu_collective(int n); // default 1
  int torchmpi_get_num_buffers_per_cpu_collective();
  int torchmpi_get_num_buffers_per_gpu_collective();

  //////////////////////////////////////////////////////////////////////////////
  /////////////////// Helper thread pools //////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // A single scripting language thread calls into async / wait operations on
  // collectives. All multithreaded operations occur in offloaded tasks.
  //////////////////////////////////////////////////////////////////////////////

  // ***Collective***
  // Operations below should be used like collectives of subtle deadlocks will
  // occur. Moreover, these specific operations should be used before any
  // collective or parameterserver operation is called since the threadpools
  // are initialized lazily on first use.

  void torchmpi_set_collective_num_threads(int n);           // default 4
  void torchmpi_set_collective_thread_pool_size(int n);      // default 1 << 20
  void torchmpi_set_parameterserver_num_threads(int n);      // default 4
  void torchmpi_set_parameterserver_thread_pool_size(int n); // default 1 << 20
  int torchmpi_get_collective_num_threads();
  int torchmpi_get_collective_thread_pool_size();
  int torchmpi_get_parameterserver_num_threads();
  int torchmpi_get_parameterserver_thread_pool_size();

  // Given a SynchronizationHandle corresponding to an asynchronous collective
  // in flight, this will wait for the collective to be compleed.
  // This is a local operation.
  SynchronizationHandle* torchmpi_synchronize_handle(SynchronizationHandle* h);

  //////////////////////////////////////////////////////////////////////////////
  /////////////////////////// Collectives //////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  void torchmpi_barrier();
  void customBarrier();

  // Instantiations for various collectives/scalar-tensor/CPU-GPU/sync-async
  // combinations: see collectives.cpp / collectives_cuda.cpp
  // Nomenclatura is:
  //   void torchmpi[_async][_impl]_collective_THXYZTensor([THCState*], args);
  // where impl is p2p, nccl or nothing for mpi
  // where collective is broadcast, reduce, allreduce or sendreceive
  // where THXYZTensor are the Torch CPU and GPU tensor types
  // Note not all combinations are supported, see torchmpi/init.lua for the
  // list of supported combinations.
}
