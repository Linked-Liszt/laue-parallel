#!/bin/sh
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A APSDataAnalysis

# Could change backend to fix parallel error
#module unload cray-mpich
#module unload craype-network-ofi
#module load craype-network-ucx
#module load cray-mpich-ucx

# Change to working directory
cd ${PBS_O_WORKDIR}

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=32
NDEPTH=2
NTHREADS=2

CONFIGPATH=/eagle/projects/APSDataAnalysis/mprince/lau/laue-parallel/configs/config-full.yml

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads \
    /eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/python \
    /eagle/projects/APSDataAnalysis/mprince/lau/laue-parallel/laue_parallel.py \
    ${CONFIGPATH} \

mpiexec -n ${NNODES} --ppn 1 --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads \
    /eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/python \
    /eagle/projects/APSDataAnalysis/mprince/lau/laue-parallel/recon_parallel.py \
    ${CONFIGPATH} \
