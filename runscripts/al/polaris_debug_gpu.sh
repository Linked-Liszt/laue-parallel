NUM_NODES=1
RANKS_PER_NODE=16
START_IM=0
PROJ_NAME=laue_debug_gpu

AFFINITY_PATH=../runscripts/set_gpu_affinity.sh
CONFIG_PATH=../configs/AL30/config-64_gpu.yml

if [ -z ${PYTHONPATH+x} ]; 
then 
    echo "PYTHONPATH is not set. No job was queued."; 
    exit 1
else 
    echo "Using Python path '${PYTHONPATH}'"; 
fi

echo "
cd \${PBS_O_WORKDIR}
printenv CUDA_VISIBLE_DEVICES
echo \${CUDA_VISIBLE_DEVICES}

# MPI and OpenMP settings
NNODES=\`wc -l < \$PBS_NODEFILE\`
NRANKS_PER_NODE=${RANKS_PER_NODE}
NDEPTH=1
NTHREADS=1

NTOTRANKS=\$(( NNODES * NRANKS_PER_NODE ))
echo \"NUM_OF_NODES= \${NNODES} TOTAL_NUM_RANKS= \${NTOTRANKS} RANKS_PER_NODE= \${NRANKS_PER_NODE} THREADS_PER_RANK= \${NTHREADS}\"

mpiexec -n \${NTOTRANKS} --ppn \${NRANKS_PER_NODE} --depth=\${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=\${NTHREADS} -env OMP_PLACES=threads \\
    ${AFFINITY_PATH} \\
    ${PYTHONPATH} \\
    ../laue_parallel.py \\
    ${CONFIG_PATH} \\
    --start_im ${START_IM} \\
    --no_rank_check

" | \
qsub -A APSDataAnalysis \
-q debug \
-l select=${NUM_NODES}:system=polaris \
-l walltime=1:00:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} 