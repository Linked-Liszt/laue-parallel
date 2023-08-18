NUM_NODES=1
RANKS_PER_NODE=32
START_IM=0

PROJ_NAME=laue_perf_test

CONFIG_PATH=/eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/configs/AL30/config_perf.yml
PYTHONPATH=/eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/python
CWD=/eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/logs_perf


cd ${CWD}

echo "
cd \${PBS_O_WORKDIR}

# MPI and OpenMP settings
NNODES=\`wc -l < \$PBS_NODEFILE\`
NRANKS_PER_NODE=${RANKS_PER_NODE}
NDEPTH=2
NTHREADS=2

NTOTRANKS=\$(( NNODES * NRANKS_PER_NODE ))
echo \"NUM_OF_NODES= \${NNODES} TOTAL_NUM_RANKS= \${NTOTRANKS} RANKS_PER_NODE= \${NRANKS_PER_NODE} THREADS_PER_RANK= \${NTHREADS}\"

export CUDA_VISIBLE_DEVICES=1

${PYTHONPATH} \\
../scripts/perf_testbed.py \\
${CONFIG_PATH} \\
" | \
qsub -A APSDataAnalysis \
-q debug \
-l select=${NUM_NODES}:system=polaris \
-l walltime=1:00:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} \