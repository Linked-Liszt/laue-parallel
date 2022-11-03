#!/usr/bin/env python3
from faulthandler import disable
import cold
import h5py
import numpy as np
import os
import shutil
import json
from mpi4py import MPI
import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to perform a parallel coded aperture post-processing.'
    )
    parser.add_argument(
        'config_path',
        help='Path to .yaml cold configuration'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enables cold debug'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Prints cold grid allocations and terminates run before processing'
    )
    parser.add_argument(
        '--log_time',
        action='store_true',
        help='Enable time logging into time_logs output dir'
    )
    parser.add_argument(
        '--h5_backup',
        action='store_true',
        help='Enable backup of individual proc data into a new h5_backup output dir'
    )
    parser.add_argument(
        '--disable_recon',
        action='store_true',
        help='Disable in-script reconstruction. Also forces h5 backup'
    )
    parser.add_argument(
        '--start_im',
        type=int,
        help='Specify a start image through command line.'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Activate cprofile for the 0th rank',
    )
    return parser.parse_args()


def time_wrap(func, time_data, time_key):
    """
    Timing decorator to log the timing of various operations. 
    """
    def wrap(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        time_data[time_key] = (datetime.datetime.now() - start_time).total_seconds()
        return result
    
    return wrap


def compute_absloute_ind(im_dim, num_splits, rank):
    """
    Compute a grid of absolute indexes of the frame of a given 
    rank. Returns a (grid_size ** 2, 2) array of indices 
    """
    grid_size = im_dim // num_splits

    start_x, start_y = np.divmod(rank % (num_splits ** 2), num_splits)
    start_x *= grid_size
    start_y *= grid_size

    ind_rows = []
    for i in range(start_x, start_x + grid_size):
        ind_rows.append(np.column_stack(
            (np.full(grid_size, i),
            np.arange(start_y, start_y + grid_size))
        ))
    return np.concatenate(ind_rows)


def parallel_laue(comm, path, dry_run=False, debug=False, log_time=False, h5_backup=False, disable_recon=False, start_im=None):
    """
    Run cold processing in parallel.
    """

    # Log host name and rank
    proc = MPI.Get_processor_name()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Read parameters from the config file
    file, comp, geo, algo = cold.config(path)

    if start_im is None:
        scan_start = comp['scanstart']
    else:
        scan_start = start_im
    print(f'Start IM: {start_im}')
    out_path = os.path.join(file['output'], str(start_im))

    if log_time:
        setup_start_time = datetime.datetime.now()
        time_data = {}
        time_dir = os.path.join(out_path, 'time_logs')
        if rank == 0:
            if not os.path.exists(time_dir):
                os.makedirs(time_dir)
    
    if h5_backup:
        h5_backup_dir = os.path.join(out_path, 'h5_backup')
        if rank == 0:
            if not os.path.exists(h5_backup_dir):
                os.makedirs(h5_backup_dir)
    

    no = comp['scannumber']
    gr = comp['gridsize']

    scanpoint, pointer = np.divmod(rank, size // no)
    m, n = np.divmod(pointer % (gr ** 2), gr)

    scan_comm = MPI.COMM_WORLD.Split(color=scanpoint, key=rank)
    
    chunkm = np.divide(file['frame'][1] - file['frame'][0], gr)
    chunkn = np.divide(file['frame'][3] - file['frame'][2], gr)
    frame0 = file['frame']
    file['range'] = [int((scan_start + scanpoint) * file['range'][1]), int((scan_start + scanpoint + 1) * file['range'][1]), 1]
    file['frame'] = [int(file['frame'][0] + m * chunkm), int(file['frame'][0] + (m + 1) * chunkm), int(file['frame'][2] + n * chunkn), int(file['frame'][2] + (n + 1) * chunkm)]

    print(proc, size, rank, file['range'], file['frame'], scanpoint, pointer, m, n)

    valid_num_procs = (gr ** 2) * no 
    if valid_num_procs != size:
        raise ValueError(f"Incorrect number of procs assigned! Procs must equal gridsize^2 * scannumber. Num proces expected: {valid_num_procs}")

    if log_time:
        time_data['setup_time'] = (datetime.datetime.now() - setup_start_time).total_seconds()

    if dry_run:
        exit()

    # Load data

    if log_time:
        cold.load = time_wrap(cold.load, time_data, 'cold_load')
    data, ind = cold.load(file)

    
    # Reconstruct
    if log_time:
        cold.decode = time_wrap(cold.decode, time_data, 'cold_decode')
    pos, sig, scl = cold.decode(data, ind, comp, geo, algo, debug=debug)

    if log_time:
        cold.resolve = time_wrap(cold.resolve, time_data, 'cold_resolve')
    dep, lau = cold.resolve(data, ind, pos, sig, geo, comp)
    
    print(rank, lau.shape, ind.shape, pos.shape, sig.shape, dep.shape, frame0, file['frame'])

    if log_time:
        write_start_time = datetime.datetime.now()

    if h5_backup or disable_recon:
        with h5py.File(os.path.join(h5_backup_dir, f'{scanpoint}_{pointer}.hd5'), 'w') as hf:
            hf.create_dataset('pos', data=pos)
            hf.create_dataset('sig', data=sig)
            hf.create_dataset('ind', data=ind)
            hf.create_dataset('lau', data=lau)
            hf.create_dataset('frame', data=file['frame'])
        print(f'Proc {rank} finished backup')

    if comp['h5parallel'] and not disable_recon:
        #TODO: Reshape on IND. 
        print(f'H5 parallel write: {rank}: {pointer * lau.shape[0]}, {(pointer + 1) * lau.shape[0]}, {lau.shape}')
        scan_comm.Barrier()
        with h5py.File(os.path.join(out_path, f'out{scanpoint}.hdf5'), 'w', driver='mpio', comm=scan_comm) as h5_f:
            dset_lau = h5_f.create_dataset('lau', (lau.shape[0] * int(size/no), lau.shape[1]), dtype=float)
            dset_lau[pointer * lau.shape[0] : (pointer + 1) * lau.shape[0], :] = lau
            scan_comm.Barrier()
            dset_pos = h5_f.create_dataset('pos', (pos.shape[0] * int(size/no),),  dtype=float)
            dset_pos[pointer * pos.shape[0] : (pointer + 1) * pos.shape[0]] = pos


    elif not disable_recon:
        print(f'Proc {rank} beginning interative recon')
        reduced = comm.gather([scanpoint, ind, lau, pos, sig, dep], root=0)

        if pointer == 0:
            sig = []
            lau = []
            pos = []
            for i in range(int(size/no)):
                if reduced[i][0] == scanpoint:
                    sig.append(reduced[i][4])
                    lau.append(reduced[i][2])
                    pos.append(reduced[i][3])

            sig = np.concatenate(sig)
            lau = np.concatenate(lau)
            pos = np.concatenate(pos)
            
            ind_full = []
            im_dim = frame0[1] - frame0[0] # MUST BE SQUARE
            for i in range(gr ** 2):
                ind_full.append(compute_absloute_ind(im_dim, gr, i))
            ind_full = np.concatenate(ind_full)

            lau_reshape = np.zeros((im_dim, im_dim, lau.shape[1]))
            pos_reshape = np.zeros((im_dim, im_dim))
            sig_reshape = np.zeros((im_dim, im_dim), sig.shape([1]))

            for i in range(lau.shape[0]):
                lau_reshape[ind_full[i][0], ind_full[i][1], :] = lau[i]
                pos_reshape[ind_full[i][0], ind_full[i][1]] = pos[i]
                sig_reshape[ind_full[i][0], ind_full[i][1], :] = sig[i]

            lau_reshape = np.swapaxes(lau_reshape, 0, 2)
            lau_reshape = np.swapaxes(lau_reshape, 1, 2)
            sig_reshape = np.swapaxes(sig_reshape, 0, 2)
            sig_reshape = np.swapaxes(sig_reshape, 1, 2)

            with h5py.File(out_path +'/'+str(scanpoint) + '.hd5', 'w') as hf:
                hf.create_dataset('pos', data=pos_reshape)
                hf.create_dataset('lau', data=lau_reshape)
                hf.create_dataset('sig', data=sig_reshape)

    if log_time:
        time_data['write_time'] = (datetime.datetime.now() - write_start_time).total_seconds()
        time_data['walltime'] = (datetime.datetime.now() - setup_start_time).total_seconds()
        with open(os.path.join(time_dir, f'proc_{rank}.json'), 'w') as time_f:
            json.dump(time_data, time_f)

    if rank == 0:
        shutil.copy2(path, out_path)


if __name__ == '__main__':
    args = parse_args()
    comm = MPI.COMM_WORLD

    try:
        if comm.Get_rank() == 0 and args.profile:
            import cProfile
            cProfile.run('parallel_laue('
                            + 'comm=comm, '
                            + 'path=args.config_path, '
                            + 'dry_run=args.dry_run, '
                            + 'debug=args.debug, '
                            + 'log_time=args.log_time, '
                            + 'h5_backup=args.h5_backup,'
                            + 'disable_recon=args.disable_recon, '
                            + 'start_im=args.start_im)',
                        'laue.profile')
            comm.Abort(0)
        else:
            parallel_laue(
                comm=comm,
                path=args.config_path, 
                dry_run=args.dry_run, 
                debug=args.debug, 
                log_time=args.log_time, 
                h5_backup=args.h5_backup,
                disable_recon=args.disable_recon,
                start_im=args.start_im)

    except Exception as e:
        import traceback
        with open('err.log', 'a+') as err_f:
            err_f.write(str(e) + '\n') # MPI term output can break.
            err_f.write('Traceback: \n')
            err_f.write(traceback.format_exc())
        comm.Abort(1) # Term run early to prevent hang.
