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


def main(path, dry_run=False, debug=False, log_time=False):
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

    if log_time:
        setup_start_time = datetime.datetime.now()
        time_data = {}
        time_dir = os.path.join(file['output'], 'time_logs')
        if rank == 0:
            if not os.path.exists(time_dir):
                os.path.mkdir(time_dir)


    # Read parameters from the config file
    file, comp, geo, algo = cold.config(path)
    
    no = comp['scannumber']
    gr = comp['gridsize']

    scanpoint, pointer = np.divmod(rank, size // no)
    m, n = np.divmod(pointer, gr)

    scan_comm = MPI.COMM_WORLD.Split(color=scanpoint, key=rank)
    
    chunkm = np.divide(file['frame'][1] - file['frame'][0], gr)
    chunkn = np.divide(file['frame'][3] - file['frame'][2], gr)
    frame0 = file['frame']
    file['range'] = [int(scanpoint * file['range'][1]), int((scanpoint + 1) * file['range'][1]), 1]
    file['frame'] = [int(file['frame'][0] + m * chunkm), int(file['frame'][0] + (m + 1) * chunkm), int(file['frame'][2] + n * chunkn), int(file['frame'][2] + (n + 1) * chunkm)]

    print(proc, size, rank, file['range'], file['frame'], scanpoint, pointer, m, n)

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
    
    #cold.saveimg(file['output'] + '/pos-' + str(rank) + '/pos', pos, ind, (2048, 2048), file['frame'])
    #cold.saveimg(file['output'] + '/lau-' + str(m) + '/lau', lau, ind, (2048, 2048), frame0, swap=True)

    print (rank, lau.shape, ind.shape, pos.shape, sig.shape, dep.shape, frame0, file['frame'])

    # the three lines below are for parallel hdf5

    # this is not tested code, I had it once and tested, but deleted by mistake. Trying to retrieve it. It's saving
    # in npy, we would need to save it as hd5

    if log_time:
        write_start_time = datetime.datetime.now()

    if comp['h5_parallel']:
        f = h5py.File(file['output'] + 'out' + str(scanpoint) + '.hdf5', 'w', driver='mpio', comm=comm)
        dset = f.create_dataset('lau', (lau.shape[0] * int(size/no), lau.shape[1]), dtype=float)
        dset[pointer * lau.shape[0] : (pointer + 1) * lau.shape[0], :] = lau

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

            if not os.path.exists(file['output'] + '/pos-' + str(scanpoint)):
                os.mkdir(file['output'] + '/pos-' + str(scanpoint))

            if not os.path.exists(file['output'] + '/lau-' + str(scanpoint)):
                os.mkdir(file['output'] + '/lau-' + str(scanpoint))

            with h5py.File(file['output'] +'/'+str(scanpoint) + '.hd5', 'w') as hf:
                hf.create_dataset('pos', data=pos)
                hf.create_dataset('lau',data=lau)

    if log_time:
        time_data['write_time'] = (datetime.datetime.now() - write_start_time).total_seconds()
        with open(os.path.join(time_dir, f'proc_{rank}.json'), 'w') as time_f:
            json.dump(time_f, time_data)

    if rank == 0:
        shutil.copy2(path, file['output'])


if __name__ == '__main__':
    fire.Fire(main)
