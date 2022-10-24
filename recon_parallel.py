from ast import arg, parse
from multiprocessing.sharedctypes import Value
from turtle import back
import h5py
import argparse
import os
import numpy as np
import math
import cold
from mpi4py import MPI

DATASETS = ['lau', 'pos', 'sig', 'ind']
PROC_OUT_DIR = 'h5_backup'
RECON_OUT_DIR = 'recon'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to reconstruct output from the proc dumps in a run.'
    )
    parser.add_argument(
        'config_fp',
        help='path to the config used to create the run'
    )
    return parser.parse_args()


def reconstruct_backup(backup_dir, out_dir, scan_no, im_dim, im_num):
    max_proc = -1
    for fp in os.listdir(backup_dir):
        rank = int(fp.split('_')[1].split('.')[0])
        if int(fp.split('_')[0]) == scan_no and rank > max_proc:
            max_proc = rank
    max_proc += 1
    print(f'Found max rank: {max_proc}')

    dims = {}
    avail_datasets = []
    with h5py.File(os.path.join(backup_dir, f'{scan_no}_0.hd5'), 'r') as h5_f:
        for ds_path in DATASETS:
            if ds_path in h5_f:
                dims[ds_path] = h5_f[ds_path].shape
                avail_datasets.append(ds_path)
    
    

    raw_ds = {}
    for ds_path in avail_datasets:
        if len(dims[ds_path]) == 1:
            raw_ds[ds_path ]= np.zeros((dims[ds_path][0] * max_proc,))
        
        elif len(dims[ds_path]) == 2:
            raw_ds[ds_path]= np.zeros((dims[ds_path][0] * max_proc, dims[ds_path][1]))

        else:
            raise NotImplementedError(f'New dim! {len(dims[ds_path])}')


    print('Constructing ind')
    num_splits = int(math.sqrt(max_proc))
    print(f'Calculated {num_splits} splits')
    grid_size = im_dim // num_splits

    assert (im_dim % num_splits) == 0

    ind = []
    for i in range(max_proc):
        start_x, start_y = np.divmod(i, num_splits)
        start_x *= grid_size
        start_y *= grid_size

        ind_rows = []
        for i in range(start_x, start_x + grid_size):
            ind_rows.append(np.column_stack(
                (np.full(grid_size, i),
                np.arange(start_y, start_y + grid_size))
            ))
        ind_grid = np.concatenate(ind_rows)
        ind.append(ind_grid)
    ind = np.concatenate(ind)


    print('Gathering proc data')
    for i in range(max_proc):
        with h5py.File(os.path.join(backup_dir, f'0_{i}.hd5'), 'r') as h5_f_in:
            for ds_path in avail_datasets:
                if len(dims[ds_path]) == 1:
                    raw_ds[ds_path][i * dims[ds_path][0] : (i + 1) * dims[ds_path][0]] = np.array(h5_f_in[ds_path])

                elif len(dims[ds_path]) == 2:
                    raw_ds[ds_path][i * dims[ds_path][0] : (i + 1) * dims[ds_path][0], :] = np.array(h5_f_in[ds_path])
    
    reshapes = {}
    for ds_path in avail_datasets:
        if len(dims[ds_path]) == 1:
            reshapes[ds_path]= np.zeros((im_dim, im_dim))

        elif len(dims[ds_path]) == 2:
            reshapes[ds_path] = np.zeros((im_dim, im_dim, dims[ds_path][1]))


    print('Starting reshape placement')
    #TODO: Could be done faster with broadcast
    for i in range(raw_ds[avail_datasets[0]].shape[0]):
        for ds_path in avail_datasets:
            if len(dims[ds_path]) == 1:
                reshapes[ds_path][ind[i][0], ind[i][1]] = raw_ds[ds_path][i]
            
            elif len(dims[ds_path]) == 2:
                reshapes[ds_path][ind[i][0], ind[i][1], :] = raw_ds[ds_path][i]

    for ds_path in avail_datasets:
        if len(dims[ds_path]) == 2:
            reshapes[ds_path] = np.swapaxes(reshapes[ds_path], 0, 2)
            reshapes[ds_path] = np.swapaxes(reshapes[ds_path], 1, 2)

    print('Writing out')
    with h5py.File(os.path.join(out_dir, f'im_{im_num}_r{scan_no}.hd5'), 'w') as h5_f_out:
        for ds_path in avail_datasets:
            h5_f_out.create_dataset(ds_path, data=reshapes[ds_path])


def recon_from_config(comm, config_fp):
    mpi_rank = comm.Get_rank()
    conf_file, conf_comp, conf_geo, conf_algo = cold.config(config_fp)
    dim_y = conf_file['frame'][1] - conf_file['frame'][0]
    dim_x = conf_file['frame'][3] - conf_file['frame'][2]

    if dim_y != dim_x:
        raise NotImplementedError("Can only reconstruct square images!")

    proc_dump_dir = os.path.join(conf_file['output'], PROC_OUT_DIR)
    recon_out_dir = os.path.join(conf_file['output'], RECON_OUT_DIR)

    if mpi_rank == 0:
        if not os.path.exists(recon_out_dir):
            os.mkdir(recon_out_dir)
    comm.Barrier()

    im_num = mpi_rank + conf_comp['scanstart']

    if mpi_rank < conf_comp['scannumber']:
        reconstruct_backup(proc_dump_dir, 
                           recon_out_dir,
                           mpi_rank,
                           dim_y,
                           im_num)


if __name__ == '__main__':
    args = parse_args()
    comm = MPI.COMM_WORLD
    try:
        recon_from_config(comm, args.config_fp)
    except Exception as e:
        print(e)
        with open('err_recon.log', 'a+') as err_f:
            err_f.write(str(e)) # MPI term output can break.
        comm.Abort(1) # Term run early to prevent hang.
