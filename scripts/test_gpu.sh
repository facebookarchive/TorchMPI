#! /bin/bash

set -e

if ! command cat '1+1' | bc ; then
    echo "Need program bc to run this test"
    exit 1
fi

if ! test -e ./scripts/wrap.sh; then
    echo "Please run test from torchmpi base directory"
    exit 1
fi

LUAJIT=${LUAJIT:=luajit}

#########################################################################################################
# GPU tests
#########################################################################################################

# The following test suite is made for 2 nodes, 4 GPU per node which is the basic Nimbix setup.
# Single node tests
mpirun -n 4 --bind-to none ./scripts/wrap.sh ${LUAJIT} ./test/collectives_all.lua -processor gpu
mpirun -n 2 --bind-to none ./scripts/wrap.sh ${LUAJIT} ./test/collectives_all.lua -tests p2p -processor gpu
mpirun -n 4 --bind-to none ./scripts/wrap.sh ${LUAJIT} ./test/parameterserver.lua

# Longer tests, single node (examples)
mpirun -n 4 --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_allreduce.lua -usegpu
mpirun -n 4 --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_allreduce_async.lua -usegpu

# Parameterserver basic GPU support via CPU
mpirun -n 4  --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_downpour.lua -usegpu
mpirun -n 4  --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_easgd.lua -usegpu
mpirun -n 4  --bind-to none ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_easgd_dataparallel.lua -usegpu

if test ${HOSTFILE}; then
    # No hostfile, no multi-node for you!
    stat ${HOSTFILE}

    # Multi-node tests
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./test/collectives_all.lua -processor gpu

    # Custom hierarchical collectives have both cartesian and non-cartesian communicators, run a loop to test all
    export NUM_GPUS=$(nvidia-smi -L | wc -l)
    export NUM_NODES=$(cat ${HOSTFILE} | wc -l)
    export ub=$(echo ${NUM_GPUS}*${NUM_NODES} | bc)
    for n in $(seq 2 $ub); do
        mpirun -n ${n} -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./test/collectives_all.lua -processor gpu -tests p2p -cartesian
        mpirun -n ${n} -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./test/collectives_all.lua -processor gpu -tests nccl -cartesian
    done

    # Longer tests, multi-node (examples)
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_allreduce.lua -usegpu
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_allreduce_async.lua -usegpu

    # Parameterserver basic GPU support via CPU
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_downpour.lua -usegpu
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_easgd.lua -usegpu
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_easgd_dataparallel.lua -usegpu
fi

# TODO: make this work properly in general
#cleanup() {
#    err=$?
#    echo "cleanup script"
#    pkill -9 terra && pkill -9 luajit
#    HOSTFILE=${HOSTFILE} . ./scripts/kill.sh
#    exit $err
#}
#trap cleanup SIGHUP SIGINT SIGTERM SIGEXIT
#
