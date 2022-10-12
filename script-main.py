#!/usr/bin/env python3

import cold
import fire
import h5py
import numpy as np
import os
import shutil
import logging
from mpi4py import MPI
import time 
import dxchange

def main(path, h5_out=False, dry_run=False, debug=False):
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
    
    no = comp['scannumber']
    gr = comp['gridsize']

    scanpoint, pointer = np.divmod(rank, size // no)
    m, n = np.divmod(pointer, gr)
    
    chunkm = np.divide(file['frame'][1] - file['frame'][0], gr)
    chunkn = np.divide(file['frame'][3] - file['frame'][2], gr)
    frame0 = file['frame']
    file['range'] = [int(scanpoint * file['range'][1]), int((scanpoint + 1) * file['range'][1]), 1]
    file['frame'] = [int(file['frame'][0] + m * chunkm), int(file['frame'][0] + (m + 1) * chunkm), int(file['frame'][2] + n * chunkn), int(file['frame'][2] + (n + 1) * chunkm)]

    print(proc, size, rank, file['range'], file['frame'], scanpoint, pointer, m, n)

    if dry_run:
        exit()

    # Load data
    data, ind = cold.load(file)

    t = time.time()
    
    # Reconstruct
    pos, sig, scl = cold.decode(data, ind, comp, geo, algo, debug=debug)
    dep, lau = cold.resolve(data, ind, pos, sig, geo, comp)
    
    #cold.saveimg(file['output'] + '/pos-' + str(rank) + '/pos', pos, ind, (2048, 2048), file['frame'])
    #cold.saveimg(file['output'] + '/lau-' + str(m) + '/lau', lau, ind, (2048, 2048), frame0, swap=True)

    print (rank, time.time() - t, lau.shape, ind.shape, pos.shape, sig.shape, dep.shape, frame0, file['frame'])

    # the three lines below are for parallel hdf5
#    f = h5py.File(file['output'] + 'out' + str(scanpoint) + '.hdf5', 'w', driver='mpio', comm=comm)
#    dset = f.create_dataset('lau', (lau.shape[0] * int(size/no), lau.shape[1]), dtype=float)
#    dset[pointer * lau.shape[0] : (pointer + 1) * lau.shape[0], :] = lau

    # this is not tested code, I had it once and tested, but deleted by mistake. Trying to retrieve it. It's saving
    # in npy, we would need to save it as hd5
    reduced = MPI.COMM_WORLD.gather([scanpoint, ind, lau, pos, sig, dep], root=0)

    # Inefficent, should only bcast to pointer=0 procs
    reduced = MPI.COMM_WORLD.bcast(reduced, root=0)
    if pointer == 0:

        ind = []
        lau = []
        pos = []
        for i in range(int(size/no) * scanpoint, int(size/no) * (scanpoint + 1)):
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


        if h5_out:
            with h5py.File(file['output'] +'/'+str(scanpoint) + '.hd5', 'w') as hf:
                hf.create_dataset('pos', data=pos)
                hf.create_dataset('lau',data=lau)
        else:
            np.save(file['output'] + '/pos-' + str(scanpoint) + '/pos', pos)
            np.save(file['output'] + '/lau-' + str(scanpoint) + '/lau', lau)

    # DEBUG
    if rank == 0:
        shutil.copy2(path, file['output'])

        # can test code in cold if it is saving ok
       # cold.saveimg(file['output'] + '/pos-' + str(scanpoint) + '/pos', pos, ind, (2048, 2048), frame0)
       # cold.saveimg(file['output'] + '/lau-' + str(scanpoint) + '/lau', lau, ind, (2048, 2048), frame0, swap=True)


    #reduced = None
    #if rank == 0:
    #    reduced = np.empty(size)
    #reduced = MPI.COMM_WORLD.gather([scanpoint, ind, lau, pos, sig, dep], root=0)
    
    #if rank == 0:
    #    for m in range(no):
    #        ind = []
    #        lau = []
    #        pos = []
    #        for n in range(size):
    #            if reduced[n][0] == m:
    #                ind.append(reduced[n][1])
    #                lau.append(reduced[n][2])
    #                pos.append(reduced[n][3])

    #        ind = np.concatenate(ind)
    #        lau = np.concatenate(lau)
    #        pos = np.concatenate(pos)
    #        print (lau.shape, ind.shape, pos.shape)
    #        cold.saveimg(file['output'] + '/pos-' + str(m) + '/pos', pos, ind, (2048, 2048), frame0) 
    #        cold.saveimg(file['output'] + '/lau-' + str(m) + '/lau', lau, ind, (2048, 2048), frame0, swap=True)

    print (time.time() - t)

if __name__ == '__main__':
    fire.Fire(main)
