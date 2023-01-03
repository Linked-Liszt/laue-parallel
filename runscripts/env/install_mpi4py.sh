#!/bin/bash
PIP_PATH=/eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/pip

# DOES NOT WORK WITH NVHPC
module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

# Info to recompile mpi4py also appears to work with GNU over NVHPC
${PIP_PATH} uninstall --yes mpi4py 
env MPICC=/opt/cray/pe/craype/2.7.15/bin/cc ${PIP_PATH} install --no-cache-dir mpi4py