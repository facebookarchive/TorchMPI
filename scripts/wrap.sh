#!/bin/sh

set -e

test $(which bc) || (echo "wrap.sh need bc, please install (e.g. sudo apt-get install bc)" && exit 1)
test $(which numactl) || (echo "wrap.sh need numactl, please install (e.g. sudo apt-get install numactl)" && exit 1)
test $(which hostname) || (echo "wrap.sh need hostname, please install" && exit 1)

set +e

# Set the LOG_TO_FILE variable so each process logs to /tmp/mpi_${RANK}
LOG_TO_FILE=${LOG_TO_FILE:=}

ARGS=$@
LOCAL_RANK=${MV2_COMM_WORLD_LOCAL_RANK:=${OMPI_COMM_WORLD_LOCAL_RANK:=${PMI_LOCAL_RANK:=0}}}
LOCAL_SIZE=${MV2_COMM_WORLD_LOCAL_SIZE:=${OMPI_COMM_WORLD_LOCAL_SIZE:=${PMI_LOCAL_SIZE:=1}}}

if test $(which nvidia-smi); then
    NGPUS=$(nvidia-smi -L | wc -l)
    MY_GPU=$(expr ${LOCAL_RANK} % ${NGPUS})
fi

if test $1 = 'mpirun'; then
    echo "Usage: 'mpirun -n 2 wrap.sh cmd'"
    echo "Detected incorrect form: 'wrap.sh mpirun -n 2 cmd'"
    echo "Exiting"
    exit
fi

sockets=$(numactl --hardware | grep -c cpus)
cores=$(grep -c processor /proc/cpuinfo)
core_per_socket=$(echo "${cores} / ${sockets}" | bc)
core_per_process=$(echo "${cores} / ${LOCAL_SIZE}" | bc) # IF hyperthreading IS ENABLED AND we don't want hyperthreading, divide by 2.


echo "sockets core_per_socket core_per_process = ${sockets} ${core_per_socket} ${core_per_process}"
if test -z ${NO_NUMACTL}; then
    if test ${core_per_process} -eq 0; then
        # Clear overprovisioning, just free for all!
        NUMACTL=""
    else
        core_begin=$(echo "${LOCAL_RANK} * ${core_per_process}" | bc)
        core_end=$(echo "${core_begin} + ${core_per_process} - 1" | bc)
        NUMACTL="numactl --physcpubind=${core_begin}-${core_end}"
    fi
fi

# NUMACTL=""

MPI_MY_NODE=$(hostname)
MPI_NODES=$(hostname)
if test ${HOSTFILE}; then
    MPI_NODES=$(cat ${HOSTFILE})
fi

set -ex

if test ${NVPROF}; then
    # NVPROF='/usr/local/cuda/bin/nvprof --print-gpu-trace --print-api-trace --profile-from-start off -f --log-file /tmp/human_'${LOCAL_RANK}' -o /tmp/nvprof_'${LOCAL_RANK}
    NVPROF='/usr/local/cuda/bin/nvprof --print-gpu-trace --print-api-trace --unified-memory-profiling off  -f --log-file /tmp/human_'${LOCAL_RANK}' -o /tmp/nvprof_'${LOCAL_RANK}
    # NVPROF='/usr/local/cuda-7.5/bin/nvprof --print-gpu-trace --print-api-trace --profile-from-start off -f --log-file /tmp/human_'${LOCAL_RANK}' -o /tmp/nvprof_'${LOCAL_RANK}
    # NVPROF='/usr/local/cuda-7.5/bin/nvprof --print-gpu-trace --print-api-trace -f --log-file /tmp/human_'${LOCAL_RANK}' -o /tmp/nvprof_'${LOCAL_RANK}
fi

if test ${LOG_TO_FILE}; then
    MPI_MY_NODE=${MPI_MY_NODE} MPI_NODES=${MPI_NODES} ${NUMACTL} ${NVPROF} ${ARGS} >/tmp/mpi_${LOCAL_RANK} 2>&1
else
    if test ${LOCAL_RANK}; then
        MPI_MY_NODE=${MPI_MY_NODE} MPI_NODES=${MPI_NODES} ${NUMACTL} ${NVPROF} ${ARGS}
    else
        MPI_MY_NODE=${MPI_MY_NODE} MPI_NODES=${MPI_NODES} ${NUMACTL} ${NVPROF} ${ARGS} 1>/dev/null 2>/dev/null
    fi
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
