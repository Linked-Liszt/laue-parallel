NUM_NODES=10
RANKS_PER_NODE=32
START_IM=2
PROJ_NAME=laue_al
QUEUE=preemptable

AFFINITY_PATH=../runscripts/set_gpu_affinity.sh
CONFIG_PATH=../configs/AL30/config_gladier_debug.yml

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
    ${AFFINITY_PATH} \\
    ${PYTHONPATH} \\
    ../laue_parallel.py \\
    ${CONFIG_PATH} \\
    --start_im ${START_IM} \\

mpiexec -n \${NNODES} --ppn 3 --depth=\${NDEPTH} --cpu-bind depth --env NNODES=\${NNODES} --env OMP_NUM_THREADS=\${NTHREADS} -env OMP_PLACES=threads \\
    ${AFFINITY_PATH} \\
    ${PYTHONPATH} \\
    ../recon_parallel.py \\
    ${CONFIG_PATH} \\
    --start_im ${START_IM} \\
" | \
qsub -A APSDataAnalysis \
-q ${QUEUE} \
-l select=${NUM_NODES}:system=polaris \
-l walltime=24:00:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} 