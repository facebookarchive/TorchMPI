# Basic functions

TorchMPI provide the following basic functions.
As a rule of thumb, avoid calling collective operations within conditionals.
In advanced use cases, make sure all processes you intend to synchronize are on the proper [communicator](https://github.com/facebookresearch/TorchMPI/docs/communicators.md) or deadlocks will ensue.

### torchmpi.start([boolean withCuda(false)][, boolean withIPCGroups(true)][, function customCommunicatorInit][, boolean withCartesianCommunicator(false)][, function collectiveCommunicator])
Collective operation that must be called by all processes to start MPI.
To enable cuda set ```withCuda``` to true.

By default, 2 communicators are created which group the MPI ranks by hostname.

If cuda is enabled, the default is to additionally group by hostnames and hostnames + set of cudaIPC capable devices.

A ```customCommunicatorInit``` function may additionally be specified to group ranks befire the default 2-level communicator is created
(see the hierarchical_communicators [test](https://github.com/facebookresearch/TorchMPI/test/hierarchical_communicators.lua) or
the mnist easgd + dataparallel [example](https://github.com/facebookresearch/TorchMPI/examples/mnist/mnist_parameterserver_easgd_dataparallel.lua).

The extra ```withCartesianCommunicator``` and ```collectiveCommunicator``` options can be used to
modify the default communicator constructor to experiment with new topologies. We envision this to be
useful for DGX-1 type systems.

### torchmpi.stop()
Collective operation which terminates the current MPI context. Once ```torchmpi.stop``` is called, you cannot issue any MPI calls anymore.

### torchmpi_barrier()
Collective operation which performs barrier synchronization across all processes in the current communicator.

### torchmpi.rank()
Returns an integer within 0 and ```torchmpi.size()-1``` which uniquely identifies
the process in the context of the current communicator.

### torchmpi.size()
Returns an integer which is the number of processes in the current communicator.

## Constants

### (boolean) torchmpi.withCuda
Set to true if ```torchmpi.start``` has been called with ```useCuda=true```.

### (boolean) torchmpi.hasNCCL
Set to ```true``` if NCCL is installed and available.

### (boolean) torchmpi.ipcGroups
Set to ```true``` if the default communicator discriminates by cudaIPC groups.

### (string) torchmpi.hostName
The name of the host on which the current rank resides
