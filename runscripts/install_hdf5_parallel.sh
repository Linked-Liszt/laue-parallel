#!/bin/bash
PIP_PATH=/eagle/projects/APSDataAnalysis/mprince/lau_env_polaris/bin/pip

# DOES NOT WORK WITH NVHPC
module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

module load cray-hdf5-parallel
export HDF5_MPI="ON"
# HDF5 uses CC, but wants C compiler (NOT C++)
export CC=/opt/cray/pe/craype/2.7.15/bin/cc
${PIP_PATH} uninstall --yes h5py
${PIP_PATH} install --no-binary=h5py h5py

# Info to recompile mpi4py also appears to work with GNU over NVHPC
# env MPICC=/opt/cray/pe/craype/2.7.15/bin/cc ${PIP_PATH} install --no-cache-dir mpi4py