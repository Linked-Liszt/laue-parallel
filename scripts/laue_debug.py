#!/usr/bin/env python3
from faulthandler import disable
import cold
import h5py
import numpy as np
import os
import json
from mpi4py import MPI
import datetime
import argparse
import dataclasses
import copy
import shutil

def parse_args():
    """
    Script arguments. Use this as a reference for script usage. 
    """
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


@dataclasses.dataclass
class OutDirs():
    pfx: str = None
    time: str = None
    config: str = None
    proc_results: str = None

@dataclasses.dataclass
class ColdConfig():
    file: dict
    comp: dict
    geo: dict
    algo: dict
    scanpoint: int = None
    pointer: int = None

@dataclasses.dataclass
class TimeData():
    setup_start: datetime.time = None
    write_start: datetime.time = None
    times: dict = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class ColdResult():
    data = None
    ind = None
    pos = None
    sig = None
    scl = None
    dep = None
    lau = None
    ene = None


def time_wrap(func, time_data: dict, time_key: str):
    """
    Timing decorator to log the timing of various operations. 
    """
    def wrap(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        time_data[time_key] = (datetime.datetime.now() - start_time).total_seconds()
        return result
    
    return wrap


def make_paths(cold_config: ColdConfig, rank: int) -> OutDirs:
    """
    Make the necessary output directories for processes to dump
    results into.  
    """
    out_dirs = OutDirs

    im_num = cold_config.comp['scanstart'] 
    print(f"Rank {rank} processing IM: {im_num}")

    out_dirs.pfx = os.path.join(cold_config.file['output'], str(im_num))

    out_dirs.time = os.path.join(out_dirs.pfx, 'time_logs')
    if rank == 0:
        if not os.path.exists(out_dirs.time):
            os.makedirs(out_dirs.time)
    
    out_dirs.proc_results = os.path.join(out_dirs.pfx, 'proc_results')
    if rank == 0:
        if not os.path.exists(out_dirs.proc_results):
            os.makedirs(out_dirs.proc_results)

    out_dirs.config = os.path.join(out_dirs.pfx, 'configs')
    if rank == 0:
        if not os.path.exists(out_dirs.config):
            os.makedirs(out_dirs.config)
    
    return out_dirs


def spatial_decompose(comm, cold_config: ColdConfig, rank: int) -> ColdConfig:
    """
    Perform the calculations to determine what area of the image a single the
    process should perform calculations on. 

    Returns: 
        cc: a copy of the cold config with the parameters set for the individual
            process 
    """
    cc = copy.deepcopy(cold_config)

    no = cc.comp['scannumber']
    gr = cc.comp['gridsize']

    size = comm.Get_size()

    cc.scanpoint, cc.pointer = np.divmod(rank, size // no)
    m, n = np.divmod(cc.pointer % (gr ** 2), gr)

    chunkm = np.divide(cc.file['frame'][1] - cc.file['frame'][0], gr)
    chunkn = np.divide(cc.file['frame'][3] - cc.file['frame'][2], gr)
    cc.file['range'] = [int((cc.comp['scanstart'] + cc.scanpoint) * cc.file['range'][1]), 
                                 int((cc.comp['scanstart'] + cc.scanpoint + 1) * cc.file['range'][1]), 
                                 1]

    cc.file['frame'] = [int(cc.file['frame'][0] + m * chunkm), 
                                 int(cc.file['frame'][0] + (m + 1) * chunkm), 
                                 int(cc.file['frame'][2] + n * chunkn), 
                                 int(cc.file['frame'][2] + (n + 1) * chunkm)]

    print(size, rank, cc.file['range'], cc.file['frame'], cc.scanpoint, cc.pointer, m, n)

    valid_num_procs = (gr ** 2) * no 
    if valid_num_procs != size:
        raise ValueError(f"Incorrect number of procs assigned! Procs must equal gridsize^2 * scannumber. Num proces expected: {valid_num_procs}")
    
    return cc


def process_cold(args, cold_config: ColdConfig, time_data: TimeData, start_range: list, rank: int) -> ColdResult:
    """
    Performs the image stack calculations via the cold library. 
    """
    cold.reset_mask_cache(True, True)

    cr = ColdResult()
    shift = 25 * cold_config.geo['scanner']['step'] * 1e-3
    print(shift)

    if not cold_config.file['stacked']:
        cold_config.file['range'] = start_range

    cold.load = time_wrap(cold.load, time_data.times, 'cold_load')
    cr.data, cr.ind = cold.load(cold_config.file)
    print(f'Data Shape: {cr.data.shape}')
    print(f'Data Shape: {cr.ind.shape}')
    num_px = 50
    cr.data = cr.data[10000:10000 + num_px]
    cr.ind = cr.ind[10000:10000 + num_px]
    print(f'Data Shape: {cr.data.shape}')
    print(f'Data Shape: {cr.ind.shape}')

    
    # Reconstruct
    cold.decode = time_wrap(cold.decode, time_data.times, 'cold_decode')
    cr.pos, cr.sig, cr.scl, cr.ene, _ = cold.decode(cr.data, cr.ind, cold_config.comp, cold_config.geo, cold_config.algo, debug=args.debug)
    #cr.pos, cr.sig, cr.scl, cr.ene  = cold.decode(cr.data, cr.ind, cold_config.comp, cold_config.geo, cold_config.algo, debug=args.debug)

    cold.resolve = time_wrap(cold.resolve, time_data.times, 'cold_resolve')
    cr.dep, cr.lau = cold.resolve(cr.data, cr.ind, cr.pos, cr.sig, cold_config.geo, cold_config.comp)
    
    print(rank, cr.lau.shape, cr.ind.shape, cr.pos.shape, cr.sig.shape, cr.dep.shape, cold_config.file['frame'], cold_config.file['frame'])

    return cr

def write_output(cold_config: ColdConfig, out_dirs: OutDirs, cold_result: ColdResult, rank: int) -> None:
    """
    Takes the output from cold processing and writes to a file. Currently, each process dumps its individual 
    output into a h5 file which is then reconstructed by a script. 

    TODO: Could be performed over MPI or h5+MPI but data seems too large for infrastructure, so would likely 
          require some sort of batching system to do in the same script.
    """
    with h5py.File(os.path.join(out_dirs.proc_results, f'{cold_config.pointer}.hd5'), 'w') as hf:
        hf.create_dataset('pos', data=cold_result.pos)
        hf.create_dataset('sig', data=cold_result.sig)
        hf.create_dataset('scl', data=cold_result.scl)
        hf.create_dataset('ind', data=cold_result.ind)
        hf.create_dataset('lau', data=cold_result.lau)
        hf.create_dataset('ene', data=cold_result.ene)
        hf.create_dataset('frame', data=cold_config.file['frame'])
    print(f'Proc {rank} finished backup')


def write_time(out_dirs: OutDirs, time_data: TimeData, rank: int) -> None:
    """
    Writes time logs to output directory.  
    """
    time_data.times['write_time'] = (datetime.datetime.now() - time_data.write_start).total_seconds()
    time_data.times['walltime'] = (datetime.datetime.now() - time_data.setup_start).total_seconds()
    with open(os.path.join(out_dirs.time, f'proc_{rank}.json'), 'w') as time_f:
            json.dump(time_data.times, time_f)

def parallel_laue(comm, args):
    """
    Main script function to set up output, spatially decompose, compute cold
    results, and write all relevant data to disk. 
    """
    #rank = comm.Get_rank()
    rank = 0

    cold_config = ColdConfig(*cold.config(args.config_path))
    if args.start_im is not None:
        cold_config.comp['scanstart'] = args.start_im
    
    num_pix = (cold_config.file['frame'][1] - cold_config.file['frame'][0]) * (cold_config.file['frame'][3] - cold_config.file['frame'][2])
    print(f'Number of Pixels {num_pix}')
    
    time_data = TimeData()
    time_data.setup_start = datetime.datetime.now()

    """
    cold_config = spatial_decompose(comm, cold_config, rank)

    comm.Barrier()

    with open(os.path.join(out_dirs.config, f'{rank}.pkl'), 'wb') as conf_f:
        pickle.dump(cold_config, conf_f)

    time_data.setup = (datetime.datetime.now() - time_data.setup_start).total_seconds()

    if args.dry_run:
        exit()
    """
    out_dirs = make_paths(cold_config, rank)

    start_range = cold_config.file['range']

    cold_result = process_cold(args, cold_config, time_data, start_range, rank)

    runtime = (datetime.datetime.now() - time_data.setup_start).total_seconds()
    print(f'{runtime} (s) runtime')

    write_output(cold_config, out_dirs, cold_result, rank)
    """
    time_data.write_start = datetime.datetime.now()
    write_time(out_dirs, time_data, rank)

    # Copy config to output
    if rank == 0:
    """
    shutil.copy2(args.config_path, out_dirs.pfx)


if __name__ == '__main__':
    args = parse_args()
    #comm = MPI.COMM_WORLD
    comm = None

    if args.profile:
        import cProfile
        cProfile.run('parallel_laue(comm, args)', 'laue-local.profile')
    else:
        parallel_laue(comm, args)
