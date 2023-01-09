NUM_NODES=1
RANKS_PER_NODE=1
START_IM=0
PROJ_NAME=laue_gpu_profile

AFFINITY_PATH=../runscripts/set_gpu_affinity.sh
PYTHON_PATH=/eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/python 
CONFIG_PATH=../configs/AL30/config-64_gpu.yml

echo "
cd \${PBS_O_WORKDIR}

# MPI and OpenMP settings
NNODES=\`wc -l < \$PBS_NODEFILE\`
NRANKS_PER_NODE=${RANKS_PER_NODE}
NDEPTH=2
NTHREADS=2

export nsysOutput=output_A100_nsys\${PMI_RANK}
echo \"nsysOutput=\$nsysOutput\"

NTOTRANKS=\$(( NNODES * NRANKS_PER_NODE ))
echo \"NUM_OF_NODES= \${NNODES} TOTAL_NUM_RANKS= \${NTOTRANKS} RANKS_PER_NODE= \${NRANKS_PER_NODE} THREADS_PER_RANK= \${NTHREADS}\"

mpiexec -n \${NTOTRANKS} --ppn \${NRANKS_PER_NODE} --depth=\${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=\${NTHREADS} -env OMP_PLACES=threads \\
    ${AFFINITY_PATH} \\
    nsys profile \\
    ${PYTHON_PATH} \\
    ../laue_parallel_gpu_only.py \\
    ${CONFIG_PATH} \\
    --start_im ${START_IM} \\

" | \
qsub -A APSDataAnalysis \
-q debug \
-l select=${NUM_NODES}:system=polaris \
-l walltime=0:30:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} 