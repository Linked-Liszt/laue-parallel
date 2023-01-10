#!/bin/bash
PYTHON_PATH=/eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/python 
CONFIG_PATH=../configs/AL30/config-64_gpu.yml
START_IM=0

num_gpus=$(nvidia-smi -L | wc -l)
# need to assign GPUs in reverse order due to topology
# See Polaris Device Affinity Information https://www.alcf.anl.gov/support/user-guides/polaris/hardware-overview/machine-overview/index.html
gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))


if ((${PMI_LOCAL_RANK} < ${num_gpus}))
then
    export CUDA_VISIBLE_DEVICES=$gpu
    export NSYS_ENV=CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
    echo “RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}”
else
    export CUDA_VISIBLE_DEVICES=""
    export NSYS_ENV=CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\"
    echo “RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} NO GPU”
fi

nsys profile -o laue_${PMI_RANK} --mpi-impl=mpich \
-e  ${NSYS_ENV} \
${PYTHON_PATH} \
../laue_parallel.py \
${CONFIG_PATH} \
--start_im ${START_IM} \
--no_rank_check