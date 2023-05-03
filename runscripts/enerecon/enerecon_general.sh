NUM_NODES=1
RANKS_PER_NODE=32
START_IM=0


CONDA_PATH=/eagle/APSDataAnalysis/mprince/lau_env_polaris

PROJ_NAME=laue_enerecon

AFFINITY_PATH=../runscripts/set_soft_affinity.sh
CONFIG_PATH=$1
MASK_PATH=$2


echo "
cd \${PBS_O_WORKDIR}

module load gsl
module load cray-hdf5
module load conda
conda activate ${CONDA_PATH}

export MPICH_GPU_SUPPORT_ENABLED=0

# MPI and OpenMP settings
NNODES=\`wc -l < \$PBS_NODEFILE\`
NRANKS_PER_NODE=${RANKS_PER_NODE}
NDEPTH=2
NTHREADS=2

NTOTRANKS=\$(( NNODES * NRANKS_PER_NODE ))
echo \"NUM_OF_NODES= \${NNODES} TOTAL_NUM_RANKS= \${NTOTRANKS} RANKS_PER_NODE= \${NRANKS_PER_NODE} THREADS_PER_RANK= \${NTHREADS}\"

mpiexec -n \${NTOTRANKS} --ppn \${NRANKS_PER_NODE} --depth=\${NDEPTH} --cpu-bind depth --env NNODES=\${NNODES}  --env OMP_NUM_THREADS=\${NTHREADS} -env OMP_PLACES=threads \\
    ${AFFINITY_PATH} \\
    python \\
    ../laue_parallel.py \\
    ${CONFIG_PATH} \\
    --start_im ${START_IM} \\
    --no_load_balance \\
    --mask ${MASK_PATH} \\
    --prod_output

" | \
qsub -A APSDataAnalysis \
-q preemptable \
-l select=${NUM_NODES}:system=polaris \
-l walltime=05:00:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} \
