NUM_NODES=8
RANKS_PER_NODE=8
START_IM=0
PROJ_NAME=laue_scaling_test
QUEUE=preemptable

AFFINITY_PATH=../runscripts/set_gpu_affinity.sh
PYTHONPATH=/eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/python 
CONFIG_PATH=../configs/AL30/config-full.yml

echo "
cd \${PBS_O_WORKDIR}

# MPI and OpenMP settings
NNODES=\`wc -l < \$PBS_NODEFILE\`
NRANKS_PER_NODE=${RANKS_PER_NODE}
NDEPTH=2
NTHREADS=2

NTOTRANKS=\$(( NNODES * NRANKS_PER_NODE ))
echo \"NUM_OF_NODES= \${NNODES} TOTAL_NUM_RANKS= \${NTOTRANKS} RANKS_PER_NODE= \${NRANKS_PER_NODE} THREADS_PER_RANK= \${NTHREADS}\"

mpiexec -n \${NTOTRANKS} --ppn \${NRANKS_PER_NODE} --depth=\${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=\${NTHREADS} -env OMP_PLACES=threads \\
    ${AFFINITY_PATH} \\
    ${PYTHONPATH} \\
    ../laue_parallel.py \\
    ${CONFIG_PATH} \\
    --start_im ${START_IM} \\

mpiexec -n \${NNODES} --ppn 1 --depth=\${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=\${NTHREADS} -env OMP_PLACES=threads \\
    ${AFFINITY_PATH} \\
    ${PYTHONPATH} \\
    ../recon_parallel.py \\
    ${CONFIG_PATH} \\
    --start_im ${START_IM} \\
" | \
qsub -A APSDataAnalysis \
-q ${QUEUE} \
-l select=${NUM_NODES}:system=polaris \
-l walltime=0:50:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} 