import h5py
import argparse
import os
import numpy as np
import math
import cold
import shutil
import datetime
import glob

#DATASETS = ['lau', 'pos', 'sig', 'ind']
DATASETS = ['lau', 'pos']
IND_PATH = 'ind'
PROC_OUT_DIR = 'proc_results'
RECON_OUT_DIR = 'recon'
ALL_OUTS = 'all_recons'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to reconstruct output from the proc dumps in a run.'
    )
    parser.add_argument(
        '--p',
        help='path override for manual running'
    )
    return parser.parse_args()


def reconstruct_backup(base_path, scan_no, frame, im_id, all_dir):
    start_time = datetime.datetime.now()
    dim_y = frame[1] - frame[0]
    dim_x = frame[3] - frame[2]
    im_dim = dim_y

    if dim_y != dim_x:
        raise NotImplementedError("Can only reconstruct square images!")

    max_proc = -1
    print(f'Rank {scan_no} processing im {im_id}')
    if not os.path.exists(os.path.join(base_path, str(im_id))):
        return 

    backup_dir = os.path.join(base_path, str(im_id), PROC_OUT_DIR)
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    out_dir = os.path.join(base_path, str(im_id), RECON_OUT_DIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fp in os.listdir(backup_dir):
        rank = int(fp.split('.')[0])
        if rank > max_proc:
            max_proc = rank
    max_proc += 1
    print(f'Found max rank: {max_proc}')


    dims = {}
    avail_datasets = []
    with h5py.File(os.path.join(backup_dir, f'0.hd5'), 'r') as h5_f:
        for ds_path in DATASETS:
            if ds_path in h5_f:
                dims[ds_path] = h5_f[ds_path].shape
                avail_datasets.append(ds_path)

    reshapes = {}
    for ds_path in avail_datasets:
        if len(dims[ds_path]) == 1:
            reshapes[ds_path]= np.zeros((im_dim, im_dim))

        elif len(dims[ds_path]) == 2:
            reshapes[ds_path] = np.zeros((im_dim, im_dim, dims[ds_path][1]))

    read_time = datetime.datetime.now()

    print('Placing proc data')
    for i in range(max_proc):
        with h5py.File(os.path.join(backup_dir, f'{i}.hd5'), 'r') as h5_f_in:
            proc_ind = np.array(h5_f_in[IND_PATH])
            proc_ind[:,0] -= frame[0]
            proc_ind[:,1] -= frame[2]
            for ds_path in avail_datasets:
                ds = np.array(h5_f_in[ds_path])
                for j, ind in enumerate(proc_ind):
                    if len(dims[ds_path]) == 1:
                        reshapes[ds_path][ind[0], ind[1]] = ds[j]

                    elif len(dims[ds_path]) == 2:
                        reshapes[ds_path][ind[0], ind[1], :] = ds[j]
                        
    for ds_path in avail_datasets:
        if len(dims[ds_path]) == 2:
            reshapes[ds_path] = np.swapaxes(reshapes[ds_path], 0, 2)
            reshapes[ds_path] = np.swapaxes(reshapes[ds_path], 1, 2)

    write_time = datetime.datetime.now()
    print('Writing out')
    with h5py.File(os.path.join(all_dir, f'{im_id}.h5'), 'w') as h5_f_out:
        for ds_path in avail_datasets:
            h5_f_out.create_dataset(ds_path, data=reshapes[ds_path])


def recon_manual_from_config(path_override):
    base_path, scan_id = os.path.split(path_override)
    all_outs = os.path.join(base_path, ALL_OUTS)

    config_fp = glob.glob(os.path.join(base_path, '*.yml'))[0]

    conf_file, conf_comp, conf_geo, conf_algo = cold.config(config_fp)

    if not os.path.exists(all_outs):
        os.makedirs(all_outs)

    reconstruct_backup(base_path, 
                        0,
                        conf_file['frame'],
                        scan_id,
                        all_outs)

def recon_from_config(comm, base_path, override_start=None):
    raise NotImplementedError #TODO: MPI Queue
    mpi_rank = comm.Get_rank()

    all_outs = os.path.join(base_path, ALL_OUTS)

    if mpi_rank == 0:
        if not os.path.exists(all_outs):
            os.makedirs(all_outs)
    comm.Barrier()

    if override_start is not None:
        im_num = mpi_rank + override_start

    if mpi_rank < conf_comp['scannumber']:
        reconstruct_backup(base_path, 
                           mpi_rank,
                           conf_file['frame'],
                           im_num,
                           all_outs)

def force_write_log(rank: int, msg: str) -> None:
    """
    Use this to force writes for debugging. PBS sometimes doesn't flush
    std* outputs. MPI faults clobber greedy flushing of default python
    logs.
    """
    with open(f'{rank}.log', 'a') as log_f:
        log_f.write(f'{datetime.datetime.now()} | {msg}\n')

if __name__ == '__main__':
    args = parse_args()

    if args.p is not None:
        recon_manual_from_config(args.p)
    
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        if comm.Get_rank() == 0:
            force_write_log(0, 'Starting Recon')

        try:
            recon_from_config(comm, args.override_dir)
        except Exception as e:
            import traceback
            with open('err_recon.log', 'a+') as err_f:
                err_f.write(str(e) + '\n') # MPI term output can break.
                err_f.write('Traceback: \n')
                err_f.write(traceback.format_exc())
            comm.Abort(1) # Term run early to prevent hang.

        if comm.Get_rank() == 0:
            force_write_log(0, 'Ending Recon')
