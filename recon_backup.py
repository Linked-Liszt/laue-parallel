from ast import arg, parse
from turtle import back
import h5py
import argparse
import os
import numpy as np

LAU_PATH = 'lau'
POS_PATH = 'pos'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to reconstruct output from a failed run.'
    )
    parser.add_argument(
        'backup_dir',
        help='directory with proc output file dumps'
    )
    parser.add_argument(
        'scan_number'
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

    with h5py.File(os.path.join(out_dir, f'{scan_no}_recon.hd5'), 'w') as h5_f_out:
        dset_lau = h5_f_out.create_dataset('lau', (lau_dim[0] * max_proc, lau_dim[1]), dtype=float)
        dset_pos = h5_f_out.create_dataset('pos', (pos_dim[0] * max_proc,), dtype=float)
        for i in range(max_proc):
            with h5py.File(os.path.join(backup_dir, f'0_{i}.hd5'), 'r') as h5_f_in:
                dset_lau[i * lau_dim[0] : (i + 1) * lau_dim[0], :] = np.array(h5_f_in[LAU_PATH])
                dset_pos[i * pos_dim[0] : (i + 1) * pos_dim[0]] = np.array(h5_f_in[POS_PATH])
                


if __name__ == '__main__':
    args = parse_args()
    reconstruct_backup(args.backup_dir, args.scan_number)
