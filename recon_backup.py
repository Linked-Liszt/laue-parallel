from ast import arg, parse
from turtle import back
import h5py
import argparse
import os
import numpy as np
from tqdm import tqdm

LAU_PATH = 'lau'
POS_PATH = 'pos'
IM_DIM = 2048
NUM_SPLITS = 16

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

    with h5py.File(os.path.join(backup_dir, f'{scan_no}_0.hd5'), 'r') as h5_f:
        lau_dim = h5_f[LAU_PATH].shape
        pos_dim = h5_f[POS_PATH].shape
    
    out_dir = os.path.join(os.sep.join(backup_dir.split(os.sep)[:-2]), 'recon')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    lau = np.zeros((lau_dim[0] * max_proc, lau_dim[1]))
    pos = np.zeros((pos_dim[0] * max_proc,))

    print('Constructing ind')
    num_splits = NUM_SPLITS
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
            lau[i * lau_dim[0] : (i + 1) * lau_dim[0], :] = np.array(h5_f_in[LAU_PATH])
            pos[i * pos_dim[0] : (i + 1) * pos_dim[0]] = np.array(h5_f_in[POS_PATH])
    
    lau_reshape = np.zeros((IM_DIM, IM_DIM, lau_dim[1]))
    pos_reshape = np.zeros((IM_DIM, IM_DIM))

    print('Starting reshape placement')
    #TODO: Could be done faster with broadcast
    for i in tqdm(range(lau.shape[0])):
        lau_reshape[ind[i][0], ind[i][1], :] = lau[i]
        pos_reshape[ind[i][0], ind[i][1]] = pos[i]

    lau_reshape = np.swapaxes(lau_reshape, 0, 2)
    lau_reshape = np.swapaxes(lau_reshape, 1, 2)

    print('Writing out')
    with h5py.File(os.path.join(out_dir, f'{scan_no}_recon.hd5'), 'w') as h5_f_out:
        h5_f_out.create_dataset('lau', data=lau_reshape)
        h5_f_out.create_dataset('pos', data=pos_reshape)

if __name__ == '__main__':
    args = parse_args()
    reconstruct_backup(args.backup_dir, args.scan_number)
