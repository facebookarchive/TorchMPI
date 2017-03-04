# TorchMPI

TorchMPI provides a simple abstraction for distributing training of Torch neural network models on multi-node multi-GPU clusters.
We support (a)synchronous data-parallel SGD, model-parallel SGD, CPU-side parameter server mode, as well as hierarchical multi-machine combinations of these modes of
distribution.
TorchMPI also makes oversubscription of mixed CPU-GPU models practical and allows mutual hiding of CPU computations, GPU computations and communications.

At the moment, TorchMPI provides the following functions and classes:
- [Basic Functionalities](https://github.com/facebookresearch/TorchMPI/docs/collectives.md) allow starting, synchronizing and stopping processes,
- [Collectives](https://github.com/facebookresearch/TorchMPI/docs/collectives.md) wrap a subset of MPI, NCCL, GLOO and a custom implementation of collectives useful for deeplearning. These collectives operate on dense Torch tensors and scalar values and come in synchronous or asynchronous flavors,
- [NN](https://github.com/facebookresearch/TorchMPI/docs/nn.md) extends ```torch.nn``` with support for synchronous and asynchronous collectives to automatically turn a
```torch.nn``` model to run on a distributed cluster of CPUs and GPUs,
- [Engine](https://github.com/facebookresearch/TorchMPI/docs/engine.md) is a [torchnet](https://github.com/torchnet/torchnet) style engine which allows simple networks to
train with (a)synchronous SGD while asynchronously loading data (when a proper datasetiterator is used),
- [Parameter Server](https://github.com/facebookresearch/TorchMPI/docs/parameterserver.md) exposes helper functions to shard a tensor across multiple processes and handle asynchronous client requests,
- [Communicators](https://github.com/facebookresearch/TorchMPI/docs/communicators.md) allows custom manipulation of groups of processes on which collectives and parameter server mode operates,
- [Launch scripts](https://github.com/facebookresearch/TorchMPI/docs/launchscripts.md) comprise a few options to simplify launching MPI jobs.

We also provide a set of [examples](https://github.com/facebookresearch/TorchMPI/examples/mnist/) which demonstrate the various modes of distribution on a trivial network.

## Getting started
The fastest way to use data-parallel SGD is to start from an existing CPU or **single** GPU Torch model.
Follow these 4 steps:
1. Add the following lines at the beginning of your script to start MPI with or without CUDA support.
   ```
      local mpi = require('torchmpi')
      mpi.start(config.usegpu)
   ```
2. Partition you local dataset using ```mpi.rank()``` and ```mpi.size()```, where ```mpi.rank()``` is in ```[0 .. mpi.size()-1]```
3. After your network is initialized, synchronize its weights across all participants
   ```
      local mpinn = require('torchmpi.nn')
      mpinn.synchronizeParameters(net)
   ```
4. After each backward step, synchronize your gradients
   ```
      mpinn.synchronizeGradients(state.network)
   ```
5. At the end, stop MPI to exit cleanly
   ```
      mpi.stop()
   ```

You can alternatively replace steps 3. and 4. by using a torchnet-style engine.
   ```
      require("torchmpi.engine.sgdengine")
      engine = tnt.AllReduceSGDEngine { ... }
      engine:train { ... }
      engine:test { ... }
   ```

You can ensure your GPU model is compliant by checking there is no ```cutorch.setDevice(...)``` calls anywhere in your source tree and that you are not using any of the parallel ```torch.nn``` containers.
For other modes of distribution involving asynchronous collectives, model parallel and parameter servers, see the mnist examples in the [examples](https://github.com/facebookresearch/TorchMPI/tree/master/examples/mnist) directory.

## Local install

Tested on Ubuntu Docker image @ [DockerHub](https://hub.docker.com/r/nicolasvasilache/torchmpi-rdma-devel/).

Please first check which dependencies you need:
  - MPI (e.g. [OpenMPI](https://www.open-mpi.org/software/ompi/v2.0/downloads/openmpi-2.0.1.tar.bz2)) (mandatory)
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-zone) (optional)
  - [cutorch](https://github.com/torch/cutorch) also mandatory (optional)
  - [NCCL](https://github.com/NVIDIA/nccl) for fast single node collectives (optional)
  - [GLOO](https://github.com/facebookincubator/gloo/tree/master/gloo) for non-mpi based collective algorithms (optional)

For instructions on how to install dependencies and OpenMPI, see [dependencies](https://github.com/facebookresearch/TorchMPI/dependencies/README.md).

If CUDA and cutorch are not found, the installation will install the CPU-only version.

**Note:** MPI must be built to enable the C++ API as well as support MPI_THREADS_MULTIPLE

**Note:** When using NCCL, you need to build both the static and dynamic libraries.

Once you are ready, just run the following `luarocks` command:
```sh
MPI_C_COMPILER=<path to>/mpicc MPI_CXX_COMPILER=<path to>/mpicxx MPI_CXX_COMPILE_FLAGS="-O3" <path to>/luarocks make rocks/torch_mpi-scm-1.rockspec
```
Note that on certain system your MPI compilers might have different
names. `MPI_C_COMPILER` and `MPI_CXX_COMPILER` are optional (the install
will try to find MPI), but if MPI is not found, _both_ must be specified.

# Additional details

## Computational Model
TorchMPI adopts the bulk-synchronous programming model that MPI popularized and uses the simplest possible model of compute.
Each process running a scripting language interpreter owns (and can schedule work on) 1 single GPU as well as 1 threadpool for collective communications and 1 threadpool for parameter server mode communications.
Locally, processes are pinned to GPUs in a round-robin fashion. When oversubscribing, multiple processes share the same GPU.

Advantages of this resource model include:
- no need for multi-stream and event synchronization across multiple GPUs in the application code. This is all handled at the collective level and does not leak in the user application.
- no problems of CPUs "being late to the party" in multi-GPU systems and no need to construct and schedule multi-threaded computational graphs to feed GPUs in time.
- automatically pin processes to CPU resources with numactl for improved overall CPU and I/O performance.
- asynchronous data loading operations are simplified by handling only 1 asynchronous stream for CPU-GPU and GPU-CPU transfers. We provide a torchnet-style engine with prefetch data calls to completely hide such transfers.

At the moment, determinism is required: all processes involved in a data-parallel or model-parallel operation need to train the same model and
issue their backward layers in the same order so that matching collectives are entered in the same order by all processes.
This restriction can be lifted at the cost of minor extra synchronizations.
If you need this feature, please reach out!

## Collectives
We provide minimal wrappers around MPI, NCCL, and GLOO collectives to perform synchronizations on Torch CPU and GPU tensors.
For asynchronous collectives, we provide an opaque handler and a wait primitive to abstract MPI_Request objects, CUDA streams or CPU-side futures.
Where necessary we developed a minimal set of collectives to alleviate certain issues:
- traditional MPI collectives have been tuned for latency of small messages but may perform poorly for large messages originating on GPUs
- NCCL collectives easily deadlock and extra synchronizations are necessary to avoid certain pathological cases
- within a group of cudaIPC capable devices, we found that the performance of NCCL Allreduce can be lower than a custom implementation based on cudaIPC

As a consequence we implemented a CPU and a GPU version of Broadcast and Allreduce which are usable alongside MPI, NCCL, or GLOO collectives.
These implementation come in 3 flavors: direct MPI_Isend / MPI_Irecv, staged via CPU and cudaIPC
Depending on your use case (synchronous vs asynchronous, overprovisioned vs not, CPU or GPU bound, small vs large messages),
you can switch implementations to get the best performance available.

## Terra
As GPUs get faster, CPUs are increasingly under pressure to deliver preprocessed data and offload kernel calls at higher rates.
The Terra language is a low level counterpart to Lua which emits compiled code via LLVM.
With Torch + Terra + MPI we get a familiar machine learning environment with scripting language capabilities and compiled performance.
A nice side-effect is that Terra automatically generates typed stubs for any C library and alleviates the need for any FFI code.
So far we have successfully used Terra to create a high-throughput dataset iterator which offloads preprocessing to a CPU threadpool
and asynchronously copies the data to GPU. Stay tuned for more details!

## Current Limitations
At the moment, TorchMPI processes are started with mpirun and require passwordless rsh/ssh connections between machines.
MPI implementations are also notorious for lack of standard IPv6 support.
Experiences with dynamic communicator creation with OpenMPI 1.10 over IPv6 were unsuccessful so we put a pin in it at the time being.

We currently have 2 open issues related to NCCL deadlocks when mixed with threading or overprovisioning.

We currently have an open issue related to GLOO Allreduce chunked implementation.