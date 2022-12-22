import h5py
import argparse
import os
import numpy as np
import math
import cold
from typing import Dict, List, Tuple, BinaryIO
from mpi4py import MPI
import datetime

DATASETS = ['lau', 'pos', 'sig', 'ind']
IND_PATH = 'ind'
PROC_OUT_DIR = 'proc_results'
RECON_OUT_DIR = 'recon'
ALL_OUTS = 'all_recons_test'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to reconstruct output from the proc dumps in a run.'
    )
    parser.add_argument(
        'config_fp',
        help='path to the config used to create the run'
    )
    parser.add_argument(
        '--start_im',
        type=int,
        help='Specify a start image through command line.'
    )
    return parser.parse_args()


def reconstruct_backup(base_path, frame, grid_size, im_num, all_dir, comm):
    """
    Build file infrastructure, gather metadata, and use hdf5 parallel 
    over mpi to write proc outputs to a single reconstructed file.  
    """
    
    
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    dim_y = frame[1] - frame[0]
    dim_x = frame[3] - frame[2]
    im_dim = dim_y

    if dim_y != dim_x:
        raise NotImplementedError("Can only reconstruct square images!")

    if not os.path.exists(os.path.join(base_path, str(im_num))):
        return 

    out_dir = os.path.join(base_path, str(im_num), RECON_OUT_DIR)
    backup_dir = os.path.join(base_path, str(im_num), PROC_OUT_DIR)

    avail_dsets = None
    dims = None
    # Determine global data properties
    if rank == 0:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(all_dir):
            os.makedirs(all_dir)
        
        dims, avail_dsets = get_dataset_props(os.path.join(backup_dir, '0.hd5'))

    max_proc = grid_size ** 2
    avail_dsets = comm.bcast(avail_dsets, root=0)
    dims = comm.bcast(dims, root=0)


    # Place data into recon hdf5
    with h5py.File(os.path.join(all_dir, f'recon_{im_num}.hd5'), 'w', driver='mpio', comm=MPI.COMM_WORLD) as recon_f:
        start_time = datetime.datetime.now() 
        recon_dsets = build_datasets(recon_f, avail_dsets, dims, im_dim)

        write_time = datetime.datetime.now()
        proc_subset = np.array_split(list(range(max_proc)), world_size)[rank]
        for i in proc_subset:
            with h5py.File(os.path.join(backup_dir, f'{i}.hd5'), 'r') as proc_f:
                proc_ind = np.array(proc_f[IND_PATH])
                proc_ind[:,0] -= frame[0]
                proc_ind[:,1] -= frame[2]

                for ds_path in avail_dsets:
                    ds = np.array(proc_f[ds_path])
                    for j, ind in enumerate(proc_ind):
                        if len(dims[ds_path]) == 1:
                            recon_dsets[ds_path][ind[0], ind[1]] = ds[j]

                        elif len(dims[ds_path]) == 2:
                            recon_dsets[ds_path][:, ind[0], ind[1]] = ds[j]

    with open(f'time.log', 'a+') as time_f:
        end_time = datetime.datetime.now()
        time_f.write(f'{rank} wt: {(end_time - write_time).total_seconds()}\n{(end_time-start_time).total_seconds()}\n{len(proc_subset)}\n\n')


def get_dataset_props(ds_path: str) -> Tuple[dict, List[str]]:
    """
    Determine the datasets available in a proc dump. Also 
    gather the dims of each dataset for placement.
    """
    dims = {}
    avail_datasets = []
    with h5py.File(ds_path, 'r') as h5_f:
        for ds_path in DATASETS:
            if ds_path in h5_f:
                dims[ds_path] = h5_f[ds_path].shape
                avail_datasets.append(ds_path)
    return dims, avail_datasets


def build_datasets(h5_file: BinaryIO, avail_dsets: dict, dims: dict, im_dim: int) -> dict:
    """
    Set up the h5 metadata before beginning parallel write.

    NOTE: Metadata calls are BLOCKING and SYNCHRONOUS! 
    """
    h5_dsets = {}
    for ds_path in avail_dsets:
        if len(dims[ds_path]) == 1:
            h5_dsets[ds_path] = h5_file.create_dataset(ds_path, (im_dim, im_dim), dtype='f4')

        elif len(dims[ds_path]) == 2:
            h5_dsets[ds_path] = h5_file.create_dataset(ds_path, (dims[ds_path][1], im_dim, im_dim), dtype='f4')

    return h5_dsets


def write_proc_output(proc_f: BinaryIO, recon_dsets: dict, avail_dsets: dict, frame: list, dims: dict) -> None:
    """
    Write data from a single proc file to the output datasets. Use the ind keys
    to determine the position of the writes. 
    """

    # Calculate write positions from frame offset for image crops


def recon_from_config(comm, config_fp, override_start=None):
    """
    Read scan metadata and override options to begin reconstruction
    """
    mpi_rank = comm.Get_rank()
    conf_file, conf_comp, conf_geo, conf_algo = cold.config(config_fp)


    base_path = conf_file['output']
    all_outs = os.path.join(conf_file['output'], ALL_OUTS)

    im_num = conf_comp['scanstart']
    if override_start is not None:
        im_num = override_start

    reconstruct_backup(base_path, 
                        conf_file['frame'],
                        conf_comp['gridsize'],
                        im_num,
                        all_outs,
                        comm)


if __name__ == '__main__':
    args = parse_args()
    comm = MPI.COMM_WORLD
    try:
        recon_from_config(comm, args.config_fp, args.start_im)
    except Exception as e:
        import traceback
        with open('err_recon.log', 'a+') as err_f:
            err_f.write(str(e) + '\n') # MPI term output can break.
            err_f.write('Traceback: \n')
            err_f.write(traceback.format_exc())
        comm.Abort(1) # Term run early to prevent hang.
