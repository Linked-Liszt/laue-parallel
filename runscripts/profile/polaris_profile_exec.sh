#!/bin/bash
if [ -z ${PYTHONPATH+x} ]; 
then 
    echo "PYTHONPATH is not set. No job was queued."; 
    exit 1
else 
    echo "Using Python path '${PYTHONPATH}'"; 
fi

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
    nsys profile -o laue_${PMI_RANK} \
    -e  ${NSYS_ENV} \
    ${PYTHONPATH} \
    ../laue_parallel.py \
    ${CONFIG_PATH} \
    --start_im ${START_IM} 
else
    export CUDA_VISIBLE_DEVICES=N
    export NSYS_ENV=CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
    echo “RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} NO GPU”
    ${PYTHONPATH} \
    ../laue_parallel.py \
    ${CONFIG_PATH} \
    --start_im ${START_IM} 
fi
