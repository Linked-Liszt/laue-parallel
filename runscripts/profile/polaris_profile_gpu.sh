#!/bin/bash
NUM_NODES=1
RANKS_PER_NODE=32
START_IM=0
PROJ_NAME=laue_gpu_profile

AFFINITY_PATH=../runscripts/set_gpu_affinity.sh
CONFIG_PATH=../configs/AL30/config-64_gpu.yml
PROFILE_PATH=../runscripts/profile/polaris_profile_exec.sh

if [ -z ${PYTHONPATH+x} ]; 
then 
    echo "PYTHONPATH is not set. No job was queued."; 
    exit 1
else 
    echo "Using Python path '${PYTHONPATH}'"; 
fi

echo "
cd \${PBS_O_WORKDIR}

# MPI and OpenMP settings
NNODES=\`wc -l < \$PBS_NODEFILE\`
NRANKS_PER_NODE=${RANKS_PER_NODE}
NDEPTH=2
NTHREADS=2

NTOTRANKS=\$(( NNODES * NRANKS_PER_NODE ))
echo \"NUM_OF_NODES= \${NNODES} TOTAL_NUM_RANKS= \${NTOTRANKS} RANKS_PER_NODE= \${NRANKS_PER_NODE} THREADS_PER_RANK= \${NTHREADS}\"

mpiexec -n \${NTOTRANKS} --ppn \${NRANKS_PER_NODE} --depth=\${NDEPTH} --cpu-bind depth --env NNODES=\${NNODES}  --env OMP_NUM_THREADS=\${NTHREADS} -env OMP_PLACES=threads \\
    ${PROFILE_PATH}
" | \
qsub -A APSDataAnalysis \
-q debug \
-l select=${NUM_NODES}:system=polaris \
-l walltime=0:30:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} 