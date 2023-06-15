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
import math
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
        '--profile',
        action='store_true',
        help='Activate cprofile for the 0th rank',
    )
    parser.add_argument(
        '--no_load_balance',
        action='store_true',
        help='Disable GPU/CPU load balancing.',
    )
    parser.add_argument(
        '--override_input',
        type=str,
        help='Override the input directory',
    )
    parser.add_argument(
        '--override_output',
        type=str,
        help='Override the output directory',
    )
    parser.add_argument(
        '--mask',
        type=str,
        help='Path to the mask np array',
    )
    parser.add_argument(
        '--prod_output',
        action='store_true',
        help='Enable separated debug and prod outputs.',
    )
    parser.add_argument(
        '--b',
        action='store_true',
        help='Enable batch processing of stacked files',
    )
    return parser.parse_args()


@dataclasses.dataclass
class OutDirs():
    pfx: str = None
    pfx_prod: str = None
    prod_proc_results: str = None
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
    ele = None
    pathlen = None

    # Allow for easy iteration 
    def __getitem__(self, item):
        return getattr(self, item)

OUT_DEBUG_DSETS = ['pos', 'sig', 'ind', 'lau', 'ene', 'pathlen']
OUT_DSETS = ['lau', 'ind']
OUT_DTYPES= {
    'pos': 'int32',
    'ind': 'int32',
    'sig': 'float32',
    'lau': 'float32',
    'ene': 'float32',
    'pathlen': 'float32',
}

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


def make_paths(cold_config: ColdConfig, rank: int, prod_output: bool) -> OutDirs:
    """
    Make the necessary output directories for processes to dump
    results into.  
    """
    out_dirs = OutDirs

    im_num = cold_config.comp['scanstart'] 
    print(f"Rank {rank} processing IM: {im_num}")

    if prod_output:
        out_dirs.pfx_prod =cold_config.file['output']
        out_dirs.prod_proc_results = os.path.join(out_dirs.pfx_prod, 'proc_results')
        if rank == 0:
            if not os.path.exists(out_dirs.pfx_prod):
                os.makedirs(out_dirs.pfx_prod)

        if rank == 0:
            if not os.path.exists(out_dirs.prod_proc_results):
                os.makedirs(out_dirs.prod_proc_results)

        out_dirs.pfx = f"{cold_config.file['output']}_debug"
    else:
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


def spatial_decompose(comm, cold_config: ColdConfig, rank: int, no_load_balance: bool) -> ColdConfig:
    """
    Perform the calculations to determine what area of the image a single the
    process should perform calculations on. 

    Returns: 
        cc: a copy of the cold config with the parameters set for the individual
            process 
    """     

    cc = copy.deepcopy(cold_config)

    no = 1 # TODO: Develop parallel runs in the same job

    size = comm.Get_size()

    cc.scanpoint, cc.pointer = np.divmod(rank, size // no)

    cc.file['range'] = [int((cc.comp['scanstart'] + cc.scanpoint) * cc.file['range'][1]), 
                                 int((cc.comp['scanstart'] + cc.scanpoint + 1) * cc.file['range'][1]), 
                                 1]

    n_lines = cc.file['frame'][1] - cc.file['frame'][0]
    if no_load_balance:
        proc_lines, rem = divmod(n_lines, size)
        frame_start = rank * proc_lines + min(rank, rem)
        frame_end = (rank + 1) * proc_lines + min(rank + 1, rem)
        cc.comp['batch_size'] = cc.comp['batch_size_gpu']
        cc.comp['use_gpu'] = True
    else:
        frame_start, frame_end, is_gpu = load_balance(rank, size, n_lines)
        cc.comp['use_gpu'] = is_gpu
        if is_gpu:
            cc.comp['batch_size'] = cc.comp['batch_size_gpu']
        else:
            cc.comp['batch_size'] = cc.comp['batch_size_cpu']


    cc.file['frame'] = [frame_start, 
                        frame_end, 
                        cc.file['frame'][2], 
                        cc.file['frame'][3]]
    
    print(size, rank, cc.file['range'], cc.file['frame'], cc.scanpoint, cc.pointer)
    
    return cc

def load_balance(rank, size, n_lines):
    GPU_PER_NODE = 4
    GPU_CPU_RATIO = 2 # TODO: Tune this and proc count to not clobber GPU.

    # Divide vertically
    n_nodes = int(os.environ['NNODES']) 
    n_gpu = n_nodes * GPU_PER_NODE
    n_cpu = size - n_gpu
    rank_per_node = int(size / n_nodes)

    total_ratio = n_gpu * GPU_CPU_RATIO + n_cpu
    lines_per_cpu = max(1, math.floor(n_lines / total_ratio))

    # If GPU rank
    is_gpu = rank % rank_per_node < GPU_PER_NODE
    if is_gpu:
        node_idx, gpu_idx = divmod(rank, rank_per_node)
        gpu_rank = (node_idx * GPU_PER_NODE) + gpu_idx

        all_gpu_lines = n_lines - lines_per_cpu * n_cpu
        gpu_lines, rem = divmod(all_gpu_lines, n_gpu)

        frame_start = gpu_rank * gpu_lines + min(gpu_rank, rem)
        frame_end = (gpu_rank + 1) * gpu_lines + min(gpu_rank + 1, rem)

    else:
        node_idx, cpu_idx = divmod(rank, rank_per_node)
        cpu_per_node = int(size / n_nodes) - GPU_PER_NODE
        cpu_rank = (node_idx * cpu_per_node) + (cpu_idx - GPU_PER_NODE)

        cpu_start = n_lines - (lines_per_cpu * n_cpu)

        frame_start = cpu_start + (cpu_rank * lines_per_cpu)
        frame_end = cpu_start + ((cpu_rank + 1) * lines_per_cpu)
    
    return frame_start, frame_end, is_gpu


def load_distribute_thresh(comm, cc: ColdConfig, time_data: TimeData, start_range, rank: int) -> ColdResult:
    """
    Performs load balancing based on the threshold of the config. 
    Data is loaded and thresholded by each process, and each one independently 
    
    """
    cr = ColdResult()

    cc.pointer = rank

    size = comm.Get_size()

    if not cc.file['stacked']:
        cc.file['range'] = start_range

    cold.load = time_wrap(cold.load, time_data.times, 'cold_load')
    cr.data, cr.ind = cold.load(cc.file)

    cr.data = np.array_split(cr.data, size)[rank]
    cr.ind = np.array_split(cr.ind, size)[rank]

    print(rank, f'Rank: {rank} processing {np.shape(cr.data)} pixels')

    cc.comp['use_gpu'] = True

    return cc, cr


def load_distribute_mask(comm, cc: ColdConfig, time_data: TimeData, start_range, mask_fp, rank: int) -> ColdResult:
    cr = ColdResult()
    mask = np.load(mask_fp).astype(int)

    cc.pointer = rank
    size = comm.Get_size()

    if not cc.file['stacked']:
        cc.file['range'] = start_range

    cold.load = time_wrap(cold.load, time_data.times, 'cold_load')
    cr.data, cr.ind = cold.load(cc.file, collapsed=False)

    num_px = np.count_nonzero(mask)
    mask_data = np.zeros((num_px, cr.data.shape[2]))
    mask_ind = np.zeros((num_px, cr.ind.shape[1]))

    xs, ys = np.nonzero(mask)
    for i, (x, y) in enumerate(zip(xs, ys)):
        mask_data[i] = cr.data[x, y]
        mask_ind[i] = np.asarray([x, y])


    cr.data = np.array_split(mask_data, size)[rank]
    cr.ind = np.array_split(mask_ind, size)[rank]

    print(rank, f'Rank: {rank} processing {np.shape(cr.data)} pixels')

    cc.comp['use_gpu'] = True

    return cc, cr


def process_cold(args, cr: ColdResult, cold_config: ColdConfig, time_data: TimeData, start_range: list, rank: int) -> ColdResult:
    """
    Performs the image stack calculations via the cold library. 
    """

    
    # Reconstruct
    cold.decode = time_wrap(cold.decode, time_data.times, 'cold_decode')
    cr.pos, cr.sig, cr.scl, cr.ene, cr.pathlen = cold.decode(cr.data, cr.ind, cold_config.comp, cold_config.geo, cold_config.algo, debug=args.debug)

    cold.resolve = time_wrap(cold.resolve, time_data.times, 'cold_resolve')
    cr.dep, cr.lau = cold.resolve(cr.data, cr.ind, cr.pos, cr.sig, cold_config.geo, cold_config.comp)
    
    print(rank, cr.lau.shape, cr.ind.shape, cr.pos.shape, cr.sig.shape, cr.dep.shape, cold_config.file['frame'], cold_config.file['frame'])

    return cr


def write_output(cold_config: ColdConfig, out_dirs: OutDirs, cold_result: ColdResult, rank: int, prod_output: bool) -> None:
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
        for dset in OUT_DEBUG_DSETS:
            hf.create_dataset(dset, data=cold_result[dset], dtype=OUT_DTYPES[dset])
        hf.create_dataset('frame', data=cold_config.file['frame'])

    if prod_output:
        with h5py.File(os.path.join(out_dirs.prod_proc_results, f'{cold_config.pointer}.hd5'), 'w') as hf:
            for dset in OUT_DSETS:
                hf.create_dataset(dset, data=cold_result[dset], dtype=OUT_DTYPES[dset])
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
        dims, reshapes = build_recon_metadata(cold_config, cold_result, start_frame)
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


def build_recon_metadata(cold_config, cold_result, start_frame):
    dims = {}
    for ds_path in OUT_DSETS:
        dims[ds_path] = cold_result[ds_path].shape

    im_dim = (start_frame[1] - start_frame[0]) 

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
    print(time_data.times)


def parallel_laue(comm, args):
    """
    Main script function to set up output, spatially decompose, compute cold
    results, and write all relevant data to disk. 
    """
    rank = comm.Get_rank()

    cold_config = ColdConfig(*cold.config(args.config_path))

    if args.override_input is not None:
        cold_config.file['path'] = args.override_input
    
    if args.override_input is not None:
        cold_config.file['output'] = args.override_output

    start_range = cold_config.file['range']
    start_frame = cold_config.file['frame']
    start_in_path = cold_config.file['path']
    start_out_path = cold_config.file['output']

    if args.b:
        files = list(os.listdir(cold_config.file['path']))
        # Extracts scan number seperated by '_' and sorts: myscan_[scan_no].h5
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    else:
        files = [cold_config.file['path']]

    for input_file in files:
        cold_config.file['range'] = start_range
        cold_config.file['frame'] = start_frame

        if args.b:
            file_basename = os.path.splitext(input_file)[0]
            input_path = start_in_path
            output_path = start_out_path
            if args.override_input is not None:
                input_path = args.override_input
            if args.override_output is not None:
                output_path = args.override_output

            cold_config.file['path'] = os.path.join(input_path, input_file)
            cold_config.file['output'] = os.path.join(output_path, file_basename)


        time_data = TimeData()
        time_data.setup_start = datetime.datetime.now()

        # TODO: Integrate Thresh into SD
        #cold_config = spatial_decompose(comm, cold_config, rank, args.no_load_balance)

        out_dirs = make_paths(cold_config, rank, args.prod_output)
        if rank == 0:
            if os.path.isdir(cold_config.file['path']):
                base_file = os.path.join(cold_config.file['path'], os.listdir(cold_config.file['path'])[0])
            else:
                base_file = cold_config.file['path']
            shutil.copy(base_file, out_dirs.pfx_prod)

        comm.Barrier()

        with open(os.path.join(out_dirs.config, f'{rank}.pkl'), 'wb') as conf_f:
            pickle.dump(cold_config, conf_f)

        time_data.setup = (datetime.datetime.now() - time_data.setup_start).total_seconds()

        #cold_config, cold_result = load_distribute_thresh(comm, cold_config, time_data, start_range, rank)
        if 'pixmask' in cold_config.file:
            pix_mask = cold_config.file['pixmask']
        else:
            pix_mask = args.mask
        cold_config, cold_result = load_distribute_mask(comm, cold_config, time_data, start_range, pix_mask, rank)

        if args.dry_run:
            exit()

        cold_result = process_cold(args, cold_result, cold_config, time_data, start_range, rank)

        time_data.write_start = datetime.datetime.now()
        write_output(cold_config, out_dirs, cold_result, rank, args.prod_output)

        if args.mpi_recon:
            write_recon_p2p(cold_config, start_frame, cold_result, comm)

        write_time(out_dirs, time_data, rank)

        # Copy config to output
        if rank == 0:
            shutil.copy2(args.config_path, out_dirs.pfx)
            if args.prod_output:
                shutil.copy2(args.config_path, out_dirs.pfx_prod)
        

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
