# Engine

We provide the following [torchnet](https://github.com/torchnet/torchnet) engines to simplify the process of writing distributed training while asynchronously loading data
on the GPU (if the datasetiterator implements the prefetch operator)

### Class tnt.AllreduceSGDEngine{boolean usegpu [false], boolean async [false], boolean devicesync [true], boolean dynamicnetwork [true]}
Engine class which supports synchronous SGD with allreduce collective calls occuring either:
- synchronously after gradient accumulations (```async=false```)
- asynchronously interleaved with gradient accumulations with a final synchronization  (```async=true```)

If the DatasetIterator object passed to the constructor supports the ```prefetch``` operation, CPU-GPU transfers can be hidden by GPU compute.

When ```devicesync=true``` is specified, we perform ```cutorch.synchronize()``` calls:
- after the next data sample has been retrieved and,
- after the criterion has been computed

When ```dynamicnetwork=true```, the asynchronous backward registration is performed at each step to account for potential parameter reallocations in the network.
