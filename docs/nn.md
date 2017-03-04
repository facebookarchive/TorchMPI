# NN

### torchmpi.nn.synchronizeParameters(nn.Module network, boolean withAllreduce [false])
Traverses the network and synchronizes its parameters.
By default calls ```brodcast``` from rank 0.
When called with ```withAllreduce=true```, calls ```allreduce``` which averages the parameters of all networks.
The particular implementation of ```broadcast``` or ```allreduce``` used is determined by ```torchmpi.collectiveSelector```.

### torchmpi.nn.synchronizeGradients(nn.Module network)
Traverses the network and synchronizes its gradients with ```allreduce```.
The particular implementation of ```allreduce``` used is determined by ```torchmpi.collectiveSelector```.

### torchmpi.nn.checkWithAllreduce(nn.Module network)
Sanity check, traverses all weights in the network and checks that ```weight:abs():mean() == torchmpi.allreduce_double(weight:abs():mean()) / torchmpi.size()```.

### torchmpi.nn.async.registerAsyncMPIBackward(nn.Module network, number syncGradientFrequency)
Overrides the ```network.backward``` function and interleave the proper asynchronous ```allreduce``` operations.

**Note:** to use this you first need to call ```torchmpi.async.initStream(...)``` to register TorchMPI's cuda stream objects.