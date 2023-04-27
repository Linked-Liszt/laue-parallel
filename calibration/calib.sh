NUM_NODES=1
RANKS_PER_NODE=5
START_IM=0


CWD=/eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/calibration/logs
CONDA_PATH=/eagle/APSDataAnalysis/mprince/lau_env_polaris


PROJ_NAME=laue_calib


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

python ../calibrate-autofocus.py $1 $2

" | \
qsub -A APSDataAnalysis \
-q preemptable \
-l select=${NUM_NODES}:system=polaris \
-l walltime=12:00:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} \
