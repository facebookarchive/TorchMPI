# Collectives
We provide minimal wrappers around MPI, GLOO and NCCL collectives to perform synchronizations on Torch CPU and GPU tensors.
In the case of asynchronous collectives, an opaque handler is returned and a wait primitive is exposed to abstract MPI_Request objects, CUDA streams or CPU-side futures.
Where necessary we developed a minimal set of collectives to alleviate certain issues:
- traditional MPI collectives may perform poorly for large messages originating on GPUs
- NCCL collectives may deadlock and extra synchronizations are necessary to avoid certain pathological cases
- within a group of cudaIPC capable devices, we found that the performance of NCCL Allreduce can be lower than that of a custom implementation

As a consequence we implemented a CPU and a GPU version of Broadcast and Allreduce which are usable alongside MPI or NCCL collectives.
These implementations come in direct MPI_Isend / MPI_Irecv, staged via CPU and cudaIPC.
Depending on your use case (synchronous or asynchronous, overprovisioned or not, CPU or GPU bound, small or large messages),
you can switch implementations to get the best performance available.

# Manually calling collectives
The collectives we expose follow the generic patterns.
For synchronous collectives: ```torchmpi.[impl.][name.][type]()```, where:
- impl is either omitted (for the stock mpi implementation), ```nccl```, ```gloo```, or ```p2p```
- name is either of ```broadcast```, ```reduce```, ```sendreceive``` or ```allreduce```
- type is a CPU of GPU Torch tensor type

For asynchronous collectives: ```userdata<SynchronizationHandle*> torchmpi.async.[impl.][name.][type]()```, where:
- impl is either omitted (for the stock mpi implementation), ```nccl```, ```gloo```, or ```p2p```
- name is either of ```broadcast```, ```reduce```, ```sendreceive``` or ```allreduce```
- type is a CPU of GPU Torch tensor type
Asynchronous collectives return a ```SynchronizationHandle``` object on which you should call ```torchmpi.syncHandle()```
to ensure the communication has completed (see e.g. the
[mnist_allreduce_async](https://github.com/facebookresearch/TorchMPI/examples/mnist/mnist_allreduce_async.lua) example).

## Scalar collectives
We provide synchronous scalar collectives offloaded to MPI.
The nomenclature for such functions is ```torchmpi.[name.]_[type]``` where:
- name is either of ```broadcast```, ```reduce```, ```sendreceive``` or ```allreduce```
- type is either of ```char```, ```short```, ```int```, ```long```, ```float``` or ```double```

## Collective API

The collectives described above all follow the same API outlined here. For examples see the relevant
[test](https://github.com/facebookresearch/TorchMPI/test/collectives_all.lua).

### broadcastTensor(number root, tensor tensor)
### reduceTensor(number root, tensor input[, tensor output = input])
### allreduceTensor(tensor input[, tensor output = input])
### sendreceiveTensor(tensor tensor, number src, number dst)

## Collective availability
Not every collective specified by the [Manually Calling Collectives](#manually-calling-collectives)
pattern is currently implemented (e.g. MPI.p2p.reduceTensor) or available on every installation
(e.g you may have compiled without nccl support).  After calling ```torchmpi.start(...)``` you
may view the status of every collective by calling ```torchmpi.torchmpi.collectiveAvailability()```.

The function returns a string representing two tables (one for cpu, one for gpu) which maps
every collective to its status in your installation.  The status can be one of:
- available (usable)
- unimplemented (not implemented in your version of torchmpi)
- unavailable (implemented by your version of torchmpi but not available in your installation)

Here is a sample output of that function call for an installation with nccl but without gloo support:
```
cpu = {
	MPI.broadcastTensor                 	->	 available
	MPI.reduceTensor                    	->	 available
	MPI.allreduceTensor                 	->	 available
	MPI.sendreceiveTensor               	->	 available
	MPI.p2p.broadcastTensor             	->	 available
	MPI.p2p.reduceTensor                	->	 unimplemented
	MPI.p2p.allreduceTensor             	->	 available
	MPI.p2p.sendreceiveTensor           	->	 unimplemented
	MPI.gloo.broadcastTensor            	->	 unavailable
	MPI.gloo.reduceTensor               	->	 unimplemented
	MPI.gloo.allreduceTensor            	->	 unavailable
	MPI.gloo.sendreceiveTensor          	->	 unimplemented
	MPI.async.broadcastTensor           	->	 available
	MPI.async.reduceTensor              	->	 available
	MPI.async.allreduceTensor           	->	 available
	MPI.async.sendreceiveTensor         	->	 unimplemented
	MPI.async.p2p.broadcastTensor       	->	 available
	MPI.async.p2p.reduceTensor          	->	 available
	MPI.async.p2p.allreduceTensor       	->	 available
	MPI.async.p2p.sendreceiveTensor     	->	 unimplemented
	MPI.async.gloo.broadcastTensor      	->	 unavailable
	MPI.async.gloo.reduceTensor         	->	 unimplemented
	MPI.async.gloo.allreduceTensor      	->	 unavailable
	MPI.async.gloo.sendreceiveTensor    	->	 unimplemented
}
gpu = {
	MPI.broadcastTensor                 	->	 available
	MPI.reduceTensor                    	->	 available
	MPI.allreduceTensor                 	->	 available
	MPI.sendreceiveTensor               	->	 available
	MPI.p2p.broadcastTensor             	->	 available
	MPI.p2p.reduceTensor                	->	 unimplemented
	MPI.p2p.allreduceTensor             	->	 available
	MPI.p2p.sendreceiveTensor           	->	 unimplemented
	MPI.gloo.broadcastTensor            	->	 unavailable
	MPI.gloo.reduceTensor               	->	 unimplemented
	MPI.gloo.allreduceTensor            	->	 unavailable
	MPI.gloo.sendreceiveTensor          	->	 unimplemented
	MPI.nccl.broadcastTensor            	->	 available
	MPI.nccl.reduceTensor               	->	 available
	MPI.nccl.allreduceTensor            	->	 available
	MPI.nccl.sendreceiveTensor          	->	 unimplemented
	MPI.nccl.p2p.broadcastTensor        	->	 available
	MPI.nccl.p2p.reduceTensor           	->	 available
	MPI.nccl.p2p.allreduceTensor        	->	 available
	MPI.nccl.p2p.sendreceiveTensor      	->	 unimplemented
	MPI.async.broadcastTensor           	->	 available
	MPI.async.reduceTensor              	->	 unimplemented
	MPI.async.allreduceTensor           	->	 available
	MPI.async.sendreceiveTensor         	->	 unimplemented
	MPI.async.p2p.broadcastTensor       	->	 available
	MPI.async.p2p.reduceTensor          	->	 unimplemented
	MPI.async.p2p.allreduceTensor       	->	 available
	MPI.async.p2p.sendreceiveTensor     	->	 unimplemented
	MPI.async.gloo.broadcastTensor      	->	 unavailable
	MPI.async.gloo.reduceTensor         	->	 unimplemented
	MPI.async.gloo.allreduceTensor      	->	 unavailable
	MPI.async.gloo.sendreceiveTensor    	->	 unimplemented
	MPI.async.nccl.broadcastTensor      	->	 available
	MPI.async.nccl.reduceTensor         	->	 available
	MPI.async.nccl.allreduceTensor      	->	 available
	MPI.async.nccl.sendreceiveTensor    	->	 unimplemented
	MPI.async.nccl.p2p.broadcastTensor  	->	 available
	MPI.async.nccl.p2p.reduceTensor     	->	 available
	MPI.async.nccl.p2p.allreduceTensor  	->	 available
	MPI.async.nccl.p2p.sendreceiveTensor 	->	 unimplemented
}
```

# Collective selector
### torchmpi.collectiveSelectorToString()
Returns a string representation of the ```torchmpi.collectiveSelector``` which shows which version
is called in [```mpi.nn```](https://github.com/facebookresearch/TorchMPI/torchmpi/nn.lua)

### (table) torchmpi.collectiveSelector
The collectives selected by default can be modified by manipulating this table.

# Extra functions

### torchmpi.async.initStreams(boolean async [false])
TorchMPI uses and exposes the following asynchronous streams for hiding communications.
We require the user to explicitly call this function to register streams so that there are no synchronization surprises with models that don't synchronize with the default stream.
If called with ```async=true``` the exposed streams are initialized as ```cudaStreamNonBlocking```.

### userdata<SynchronizationHandle*> torchmpi.syncHandle(userdata<SynchronizationHandle*> h)
Given a ```userdata<SynchronizationHandle*>``` object returned by an asynchronous collective, this makes the CPU wait on the collective to finish.

### torchmpi.C.torchmpi_free_ipc_descriptors()
Releases all tensors that have been retained for the purpose of collective operations.
Frees all the CUDA IPC resources allocated for interprocess communications.
In TorchMPI, resources are allocated on-demand.
Under multiple communicator contexts, coarse-grained resource management of collectives simplifies bookeeping issues.

### torchmpi.collectiveAvailability()
Returns a string mapping each collective to its status ({available, unimplemented, unavailable})
on your system.  See [Collective availability](#collective-availability) for more information.
