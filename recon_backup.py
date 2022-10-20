from ast import arg, parse
from turtle import back
import h5py
import argparse
import os
import numpy as np
from tqdm import tqdm
import math

DATASETS = ['lau', 'pos', 'sig', 'ind']
IM_DIM = 2048

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to reconstruct output from a failed run.'
    )
    parser.add_argument(
        'backup_dir',
        help='directory with proc output file dumps'
    )
    parser.add_argument(
        'scan_number',
        type=int,
        help='Specify scan number to reconstruct'
    )

    return parser.parse_args()

def reconstruct_backup(backup_dir, scan_no):
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
    
    out_dir = os.path.join(os.sep.join(backup_dir.split(os.sep)[:-2]), 'recon')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

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
    grid_size = IM_DIM // num_splits

    assert (IM_DIM % num_splits) == 0

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
    for i in tqdm(range(max_proc)):
        with h5py.File(os.path.join(backup_dir, f'0_{i}.hd5'), 'r') as h5_f_in:
            for ds_path in avail_datasets:
                if len(dims[ds_path]) == 1:
                    raw_ds[ds_path][i * dims[ds_path][0] : (i + 1) * dims[ds_path][0]] = np.array(h5_f_in[ds_path])

                elif len(dims[ds_path]) == 2:
                    raw_ds[ds_path][i * dims[ds_path][0] : (i + 1) * dims[ds_path][0], :] = np.array(h5_f_in[ds_path])
    
    reshapes = {}
    for ds_path in avail_datasets:
        if len(dims[ds_path]) == 1:
            reshapes[ds_path]= np.zeros((IM_DIM, IM_DIM))

        elif len(dims[ds_path]) == 2:
            reshapes[ds_path] = np.zeros((IM_DIM, IM_DIM, dims[ds_path][1]))


    print('Starting reshape placement')
    #TODO: Could be done faster with broadcast
    for i in tqdm(range(raw_ds[avail_datasets[0]].shape[0])):
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
    with h5py.File(os.path.join(out_dir, f'{scan_no}_recon.hd5'), 'w') as h5_f_out:
        for ds_path in avail_datasets:
            h5_f_out.create_dataset(ds_path, data=reshapes[ds_path])

if __name__ == '__main__':
    args = parse_args()
    reconstruct_backup(args.backup_dir, args.scan_number)
