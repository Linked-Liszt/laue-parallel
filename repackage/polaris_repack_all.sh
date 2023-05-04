NUM_NODES=1
RANKS_PER_NODE=11
PROJ_NAME=laue-repack
QUEUE=debug
WALLTIME=01:00:00

PYTHONPATH=/eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/python
SCRIPT_PATH=/eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/repackage/repack_polaris.py


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
    ${PYTHONPATH} ${SCRIPT_PATH} $@  --s

" | \
qsub -A APSDataAnalysis \
-q ${QUEUE} \
-l select=${NUM_NODES}:system=polaris \
-l walltime=${WALLTIME} \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} \
