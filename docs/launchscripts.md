# Scripts
We provide test scripts for [CPU](https://github.com/facebookresearch/TorchMPI/scripts/test_cpu.sh) and [GPU](https://github.com/facebookresearch/TorchMPI/scripts/test_gpu.sh).
Additionally, we provide the following convenience scripts which we find useful.

## [scripts/wrap.sh](https://github.com/facebookresearch/TorchMPI/scripts/test_cpu.sh) and [GPU](https://github.com/facebookresearch/TorchMPI/scripts/wrap.sh)
By default this decomposes the available CPU cores into independent ```numactl``` domains.
Additionally setting the following environment variables may be useful:
- ```NVPROF=1``` launches mpi processes in nvprof mode and dumps the output on each node in ```/tmp/human_[local_mpi_rank]``` and ```/tmp/nvprof_[local_mpi_rank]```. You can open the ```nvprof_*``` files in **nvvp** and examine the timelines.
- ```LOG_TO_FILE=1``` redirects stderr and stdout to ```/tmp/mpi_[local_mpi_rank]```

A typical launch with OpenMPI may resemble:
```mpirun -n 4 --npernode 2 --bind-to none --hostfile ${HOSTFILE} -x CUDA_VISIBLE_DEVICES=2,3 ./scripts/wrap.sh luajit ./test/collectives_all.lua -storage inplace -execution sync -processor gpu -tests p2p -benchmark -hierarchical false -staged false -numBuffers 2```, where -x environment variables are forwarded to all processes.

## [scripts/ompirun.sh](https://github.com/facebookresearch/TorchMPI/scripts/test_cpu.sh) and [GPU](https://github.com/facebookresearch/TorchMPI/scripts/ompirun.sh)
Sets up different options and larger transfer sizes than activated by default in OpenMPI.
These options usually work better for deep learning workloads but feel free to experiment and let us know your findings.
You could call as follow:
```
./scripts/ompirun.sh -n 4 -npernode 4 -hostfile ${HOSTFILE} ./scripts/wrap.sh ${LUAJIT} ./test/collectives_all.lua -storage inplace -execution sync -processor gpu -tests p2p -benchmark -hierarchical false -staged false -numBuffers 2
```
To query OpenMPI available runtime options run for instance:
```ompi_info --param btl openib --level 9```