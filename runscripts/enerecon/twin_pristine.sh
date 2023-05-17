NUM_NODES=10
RANKS_PER_NODE=32
START_IM=0

echo $1

CWD=/eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/logs_enerecon
CONDA_PATH=/eagle/APSDataAnalysis/mprince/lau_env_polaris

PROJ_NAME=laue_twin_pristine_$1

AFFINITY_PATH=../runscripts/set_soft_affinity.sh
CONFIG_PATH=../configs/twin_pristine/recon_10/recon_10_tp_$1.yml

mkdir -p ${CWD}
cd ${CWD}

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
    --mask /eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/im_masks/0024_mask2_forpolaris_TwinPristine_d20_2023_05_16.npy \\
    --b \\
    --prod_output

" | \
qsub -A APSDataAnalysis \
-q preemptable \
-l select=${NUM_NODES}:system=polaris \
-l walltime=24:00:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} \
