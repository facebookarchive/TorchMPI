# Parameter server mode
The following functions are used to create, synchronize and operate on
At the moment these functions are supported for FloatTensor and DoubleTensor only.

## High-level API

### torchmpi.Update{nn.Module network, number shardingCommunicator [0], number dataparallelCommunicator [0], number updateFrequency [25], number initDelay[100], number prefetch[0]}
This is the base class for parameter server mode updates.
An instance of ```torchmpi.Update``` initializes a parameterserver handle for each parameter in the network.
The shardingCommunicator is the communicator on whose ranks each parameter tensor is sharded.
The dataparallelCommunicator is the unit of worker granularity with dataparallel synchronous SGD.
After an initial warmup phase of ```updateFrequency``` calls to the ```self.update(number step)``` function,
a local parameter update occurs every ```updateFrequency``` steps.
Asynchronous ```self._send``` operations are issued based on the overloaded ```self._send(number step)``` function.
Asynchronous ```self._fetch``` operations are issued ```prefetch``` steps before the next ```self.update(number step)```.
Outstanding ```self._send``` operations are synchronized before a ```self._fetch``` operation is issued.
The outstanding ```self._fetch``` operation is sycnhronized befor the ```self.update``` is applied.

### torchmpi.DownpourUpdate{..., number sendFrequency[1], function localUpdate}
Derives from torchmpi.Update and implements the [Downpour](https://research.google.com/archive/large_deep_networks_nips2012.html) algorithm.
Adds a ```self._send``` operation that is issued every ```sendFrequency``` steps.
A local update is applied to each accumulated gradient before ```self._send``` occurs.
The ```self._integrate``` function simply copies the received server parameters after synchronization.

### torchmpi.EASGDUpdate{..., number beta [0.9]}
Derives from torchmpi.Update and implements the mplements the [Elastic Averaging SGD](https://arxiv.org/abs/1412.6651) algorithm.
Adds the ```beta``` parameter as described in the litterature.

See the [examples](https://github.com/facebookresearch/TorchMPI/examples/mnist/) directory for more details and samples.

## Low-level API

### torchmpi.parameterserver.init(tensor t)
Collective operation which shards a tensor ```t``` on all ranks in the current communicator.
On the first such call, a threadpool is initialized to handle client requests in the
background.
The default initialization for each shard is
```
               shard[0]   shard[1]  ...  shard[mpi.size()]
from rank:        0          1              mpi.size()
```


### torchmpi.parameterserver.initTensors(ListOf(tensor) ts)

### userdata<void*> torchmpi.parameterserver.send(userdata<void*> psh, tensor t, rawstring updateRule)
Issues an asynchronous send operation on a parameterserver handle and returns a synchronization handle.
The ```updateRule``` is applied on the server side.
Atm the update rules supported are ```zero```, ```copy``` and ```add```.

### userdata<void*> torchmpi.parameterserver.receive(userdata<void*> psh, tensor t)
Issues an asynchronous receive operation on a parameterserver handle and returns a synchronization handle.

### torchmpi.parameterserver.syncHandle(userdata<void*> sh)
Ensures that a send or a receive operation is complete.

### torchmpi.parameterserver.free(userdata<void*> psh)
Frees the sharded tensor associated with the parameterserver handle.

For code samples, see the [test case](https://github.com/facebookresearch/TorchMPI/test/parameterserver.lua).