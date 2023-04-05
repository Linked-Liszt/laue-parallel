NUM_NODES=1
RANKS_PER_NODE=32
INPUT_DIR=/eagle/APSDataAnalysis/LAUE/scans/consgeo_Ni2_10mN_125.h5
OUTPUT_DIR=../outputs_combined/recon/consgeo_Ni2_10mN_125
REPACK_DIR=../outputs_combined/repack
START_IM=0

BASENAME=$(/usr/bin/basename ${INPUT_DIR})
POINT_NAME=$(/usr/bin/basename ${INPUT_DIR} .h5)
PROJ_NAME=laue_test_${POINT_NAME}

AFFINITY_PATH=../runscripts/set_soft_affinity.sh
CONFIG_PATH=/eagle/APSDataAnalysis/mprince/lau/dev/laue-gladier/funcx_launch/launch_scripts/config_gladier_stack_temp.yml
CONDA_PATH=/eagle/APSDataAnalysis/mprince/lau_env_polaris
CWD=/eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/logs_scaling

PYTHONPATH=/eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/python

REPACK_SCRIPT=/eagle/APSDataAnalysis/mprince/lau/dev/laue-gladier/repackage/repack_polaris.py
REPACK_INPUT="$(/usr/bin/dirname "${OUTPUT_DIR}")"

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

mpiexec -n \${NTOTRANKS} --ppn \${NRANKS_PER_NODE} --depth=\${NDEPTH} --cpu-bind depth --env NNODES=\${NNODES}  --env OMP_NUM_THREADS=\${NTHREADS} -env OMP_PLACES=threads \\
    ${AFFINITY_PATH} \\
    ${PYTHONPATH} \\
    ../laue_parallel.py \\
    ${CONFIG_PATH} \\
    --override_input ${INPUT_DIR} \\
    --override_output ${OUTPUT_DIR} \\
    --start_im ${START_IM} \\
    --no_load_balance \\
    --prod_output


mpiexec -n 1 --ppn 1 --depth=\${NDEPTH} --cpu-bind depth --env NNODES=\${NNODES}  --env OMP_NUM_THREADS=\${NTHREADS} -env OMP_PLACES=threads \\
    ${PYTHONPATH} ${REPACK_SCRIPT} ${REPACK_INPUT} ${REPACK_DIR} --p ${POINT_NAME}

" | \
qsub -A 9169 \
-q debug \
-l select=${NUM_NODES}:system=polaris \
-l walltime=01:00:00 \
-l filesystems=home:eagle \
-l place=scatter \
-N ${PROJ_NAME} \
-W block=true 
