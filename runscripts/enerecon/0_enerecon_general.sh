#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle 
#PBS -l place=scatter 
#PBS -q debug
#PBS -A APSDataAnalysis 

NUM_NODES=1
RANKS_PER_NODE=32
START_IM=0


CONDA_PATH=/eagle/APSDataAnalysis/mprince/lau_env_polaris

PROJ_NAME=laue_enerecon_test

AFFINITY_PATH=../runscripts/set_soft_affinity.sh
CONFIG_PATH=/eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/configs/Si_mount_2/recon25_tp_15mn_0.yml


cd ${PBS_O_WORKDIR}

#module load gsl
#module load cray-hdf5
module load conda
conda activate ${CONDA_PATH}

export MPICH_GPU_SUPPORT_ENABLED=0

# MPI and OpenMP settings
NRANKS_PER_NODE=${RANKS_PER_NODE}
NDEPTH=2
NTHREADS=2

NTOTRANKS=$(( NUM_NODES * NRANKS_PER_NODE ))
echo \"NUM_OF_NODES= ${NUM_NODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}\"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth --env NNODES=${NUM_NODES}  --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads \
    ${AFFINITY_PATH} \
    python \
    ../laue_parallel.py \
    ${CONFIG_PATH} \
    --b \
    --no_load_balance \
    --prod_output