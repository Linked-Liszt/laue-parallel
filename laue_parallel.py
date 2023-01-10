#!/usr/bin/env python3
import cold
import h5py
import numpy as np
import os
import shutil
import json
from mpi4py import MPI
import datetime
import argparse
import dataclasses
import copy
import pickle

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
        '--mpi_recon',
        action='store_true',
        help='Enable reconstruction of individual proc data over MPI'
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
    parser.add_argument(
        '--no_rank_check',
        action='store_true',
        help='Debug feature to ignore the check for number of ranks',
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
    frame = None

    # Allow for easy iteration 
    def __getitem__(self, item):
        return getattr(self, item)

OUT_DSETS = ['pos', 'sig', 'ind', 'lau']

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

    num_grid = cold_config.comp['gridsize'] ** 2
    im_num = cold_config.comp['scanstart'] + (rank // num_grid)
    print(f"Rank {rank} processing IM: {im_num}")

    out_dirs.pfx = os.path.join(cold_config.file['output'], str(im_num))

    out_dirs.time = os.path.join(out_dirs.pfx, 'time_logs')
    if rank % num_grid == 0:
        if not os.path.exists(out_dirs.time):
            os.makedirs(out_dirs.time)
    
    out_dirs.proc_results = os.path.join(out_dirs.pfx, 'proc_results')
    if rank % num_grid == 0:
        if not os.path.exists(out_dirs.proc_results):
            os.makedirs(out_dirs.proc_results)

    out_dirs.config = os.path.join(out_dirs.pfx, 'configs')
    if rank % num_grid == 0:
        if not os.path.exists(out_dirs.config):
            os.makedirs(out_dirs.config)
    
    return out_dirs


def spatial_decompose(comm, cold_config: ColdConfig, rank: int, no_rank_check: bool) -> ColdConfig:
    """
    Perform the calculations to determine what area of the image a single the
    process should perform calculations on. 

    Returns: 
        cc: a copy of the cold config with the parameters set for the individual
            process 
    """
    cc = copy.deepcopy(cold_config)

    no = 1 # TODO: Develop parallel runs in the same job
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
    if not no_rank_check and valid_num_procs != size:
        raise ValueError(f"Incorrect number of procs assigned! Procs must equal gridsize^2 * scannumber. Num proces expected: {valid_num_procs}")
    
    return cc


def process_cold(args, cold_config: ColdConfig, time_data: TimeData, rank: int) -> ColdResult:
    """
    Performs the image stack calculations via the cold library. 
    """

    cr = ColdResult()

    cold.load = time_wrap(cold.load, time_data.times, 'cold_load')
    cr.data, cr.ind = cold.load(cold_config.file)
    
    # Reconstruct
    cold.decode = time_wrap(cold.decode, time_data.times, 'cold_decode')
    cr.pos, cr.sig, cr.scl = cold.decode(cr.data, cr.ind, cold_config.comp, cold_config.geo, cold_config.algo, debug=args.debug)

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

    NOTE: Via collective testing (https://github.com/h5py/h5py/blob/master/examples/collective_io.py)
          this implementation of parallel HDF5 does NOT yield perf gains writing to the same dset, unless
          across multiple nodes.
    """
    with h5py.File(os.path.join(out_dirs.proc_results, f'{cold_config.pointer}.hd5'), 'w') as hf:
        for dset in OUT_DSETS:
            hf.create_dataset(dset, data=cold_result[dset])
        hf.create_dataset('frame', data=cold_config.file['frame'])
    print(f'Proc {rank} finished backup')


def write_recon_p2p(cold_config: ColdConfig, start_frame, cold_result: ColdResult, comm) -> None:
    """
    Reconstruct single hdf5 file output, transferring data via MPI.
    MPI performs significantly faster than disk-based transfer. 

    NOTE: Collective operations at this scale break, so use p2p transfer.
          Also could use parallel HDF5 for more speedup, likely a reduced set
          of write ranks.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Hold sends till rank 0 is ready
    comm.Barrier()

    cold_result.frame = cold_config.file['frame']
    if rank != 0:
        comm.send(cold_result, dest=0)

    else:
        dims, reshapes = build_recon_metadata(cold_config, cold_result)
        fill_reshapes(cold_result, start_frame, reshapes, dims)
        for recv_rank in range(1, size):
            recv_result = comm.recv(source=recv_rank)
            fill_reshapes(recv_result, start_frame, reshapes, dims)

        out_fp = os.path.join(cold_config.file['output'], 'all_recons_mpi') 
        if not os.path.exists(out_fp):
            os.makedirs(out_fp)

        with h5py.File(os.path.join(out_fp, f"im_{cold_config.comp['scanstart']}.hd5"), 'w') as h5_f_out:
            for ds_path in OUT_DSETS:
                h5_f_out.create_dataset(ds_path, data=reshapes[ds_path])

    comm.Barrier()


def fill_reshapes(cold_result, start_frame, reshapes, dims):
    proc_ind = copy.deepcopy(cold_result.ind)
    proc_ind[:,0] -= start_frame[0]
    proc_ind[:,1] -= start_frame[2]
    for ds_path in OUT_DSETS:
        ds = cold_result[ds_path]
        for j, ind in enumerate(proc_ind):
            if len(dims[ds_path]) == 1:
                reshapes[ds_path][ind[0], ind[1]] = ds[j]

            elif len(dims[ds_path]) == 2:
                reshapes[ds_path][:, ind[0], ind[1]] = ds[j]


def build_recon_metadata(cold_config, cold_result):
    dims = {}
    for ds_path in OUT_DSETS:
        dims[ds_path] = cold_result[ds_path].shape

    im_dim = (cold_config.file['frame'][1] - cold_config.file['frame'][0]) * cold_config.comp['gridsize']

    reshapes = {}
    for ds_path in OUT_DSETS:
        if len(dims[ds_path]) == 1:
            reshapes[ds_path]= np.zeros((im_dim, im_dim))

        elif len(dims[ds_path]) == 2:
            reshapes[ds_path] = np.zeros((dims[ds_path][1], im_dim, im_dim))

    return dims, reshapes


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
    rank = comm.Get_rank()

    cold_config = ColdConfig(*cold.config(args.config_path))

    if ('CUDA_VISIBLE_DEVICES' not in os.environ 
         or os.environ['CUDA_VISIBLE_DEVICES'] == ""
         or os.environ['CUDA_VISIBLE_DEVICES'] == "N"):
        print(f'R: {rank} overriding batch size to 8')
        cold_config.comp['batch_size'] = 8
 
    im_start = cold_config.comp['scanstart']
    if args.start_im is not None:
        im_start = args.start_im

    start_range = cold_config.file['range']
    start_frame = cold_config.file['frame']
    for im_num in range(im_start, im_start + cold_config.comp['scannumber']):
        cold_config.comp['scanstart'] = im_num
        cold_config.file['range'] = start_range
        cold_config.file['frame'] = start_frame

        time_data = TimeData()
        time_data.setup_start = datetime.datetime.now()

        cold_config = spatial_decompose(comm, cold_config, rank, args.no_rank_check)

        out_dirs = make_paths(cold_config, rank)
        comm.Barrier()

        with open(os.path.join(out_dirs.config, f'{rank}.pkl'), 'wb') as conf_f:
            pickle.dump(cold_config, conf_f)

        time_data.setup = (datetime.datetime.now() - time_data.setup_start).total_seconds()

        if args.dry_run:
            exit()

        cold_result = process_cold(args, cold_config, time_data, rank)

        time_data.write_start = datetime.datetime.now()
        write_output(cold_config, out_dirs, cold_result, rank)

        if args.mpi_recon:
            write_recon_p2p(cold_config, start_frame, cold_result, comm)

        write_time(out_dirs, time_data, rank)

        # Copy config to output
        if rank == 0:
            shutil.copy2(args.config_path, out_dirs.pfx)
        

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
    comm = MPI.COMM_WORLD

    try:
        if comm.Get_rank() == 0 and args.profile:
            import cProfile
            cProfile.run('parallel_laue(comm, args)')
            comm.Abort(0)
        else:
            parallel_laue(comm, args)

    except Exception as e:
        import traceback
        with open('err.log', 'a+') as err_f:
            err_f.write(f'{str(e)} {comm.Get_rank()} \n') # MPI term output can break.
            err_f.write('Traceback: \n')
            err_f.write(traceback.format_exc())
        comm.Abort(1) # Term run early to prevent hang.
