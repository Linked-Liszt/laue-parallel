#!/usr/bin/env python3

import cold
import fire
import h5py
import numpy as np
import os
import shutil
import json
from mpi4py import MPI
import datetime

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


def main(path, dry_run=False, debug=False, log_time=False, h5_backup=False):
    """Runs the reconstruction workflow given parameters 
    in a configuration file.

    Parameters
    ----------
    path: string
        Path of the YAML file with configuration parameters.

    Returns
    -------
        None
    """

    # Log host name and rank
    proc = MPI.Get_processor_name()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Read parameters from the config file
    file, comp, geo, algo = cold.config(path)

    if log_time:
        setup_start_time = datetime.datetime.now()
        time_data = {}
        time_dir = os.path.join(file['output'], 'time_logs')
        if rank == 0:
            if not os.path.exists(time_dir):
                os.mkdir(time_dir)
    
    if h5_backup:
        h5_backup_dir = os.path.join(file['output'], 'h5_backup')
        if rank == 0:
            if not os.path.exists(h5_backup_dir):
                os.mkdir(h5_backup_dir)
    
    no = comp['scannumber']
    gr = comp['gridsize']

    scanpoint, pointer = np.divmod(rank, size // no)
    m, n = np.divmod(pointer % (gr ** 2), gr)

    scan_comm = MPI.COMM_WORLD.Split(color=scanpoint, key=rank)
    
    chunkm = np.divide(file['frame'][1] - file['frame'][0], gr)
    chunkn = np.divide(file['frame'][3] - file['frame'][2], gr)
    frame0 = file['frame']
    file['range'] = [int(scanpoint * file['range'][1]), int((scanpoint + 1) * file['range'][1]), 1]
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

    if comp['h5parallel']:
        print(f'h5 write: {rank}, {pointer * lau.shape[0]}, {(pointer + 1) * lau.shape[0]}, {lau.shape}')
        if h5_backup:
            with h5py.File(os.path.join(h5_backup_dir, f'{scanpoint}_{pointer}.hd5'), 'w') as hf:
                hf.create_dataset('pos', data=pos)
                hf.create_dataset('lau', data=lau)
        scan_comm.Barrier()
        with h5py.File(os.path.join(file['output'], f'out{scanpoint}.hdf5'), 'w', driver='mpio', comm=scan_comm) as h5_f:
            dset_lau = h5_f.create_dataset('lau', (lau.shape[0] * int(size/no), lau.shape[1]), dtype=float)
            dset_lau[pointer * lau.shape[0] : (pointer + 1) * lau.shape[0], :] = lau
            scan_comm.Barrier()
            dset_pos = h5_f.create_dataset('pos', (pos.shape[0] * int(size/no),),  dtype=float)
            dset_pos[pointer * pos.shape[0] : (pointer + 1) * pos.shape[0]] = pos


    else:
        reduced = scan_comm.gather([scanpoint, ind, lau, pos, sig, dep], root=0)

        if pointer == 0:

            ind = []
            lau = []
            pos = []
            for i in range(int(size/no)):
                if reduced[i][0] == scanpoint:
                    ind.append(reduced[i][1])
                    lau.append(reduced[i][2])
                    pos.append(reduced[i][3])

            ind = np.concatenate(ind)
            lau = np.concatenate(lau)
            pos = np.concatenate(pos)

            with h5py.File(file['output'] +'/'+str(scanpoint) + '.hd5', 'w') as hf:
                hf.create_dataset('pos', data=pos)
                hf.create_dataset('lau',data=lau)

    if log_time:
        time_data['write_time'] = (datetime.datetime.now() - write_start_time).total_seconds()
        time_data['walltime'] = (datetime.datetime.now() - setup_start_time).total_seconds()
        with open(os.path.join(time_dir, f'proc_{rank}.json'), 'w') as time_f:
            json.dump(time_data, time_f)

    if rank == 0:
        shutil.copy2(path, file['output'])


if __name__ == '__main__':
    fire.Fire(main)
