import argparse
import argparse
import numpy as np
import h5py

LAU_PATH = 'lau'
POS_PATH = 'pos'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_fp_1')
    parser.add_argument('h5_fp_2')
    return parser.parse_args()


def comp_h5_files(h5_fp_1, h5_fp_2):
    args = parse_args()
    with h5py.File(h5_fp_1, 'r') as h5_f:
        print(f'Lau 1 Shape: {h5_f[LAU_PATH].shape}')
        lau_1 = np.array(h5_f[LAU_PATH])
        print(f'Pos 1 Shape: {h5_f[POS_PATH].shape}')
        pos_1 = np.array(h5_f[POS_PATH])

    with h5py.File(h5_fp_2, 'r') as h5_f:
        print(f'Lau 2 Shape: {h5_f[LAU_PATH].shape}')
        lau_2 = np.array(h5_f[LAU_PATH])
        print(f'Pos 2 Shape: {h5_f[POS_PATH].shape}')
        pos_2 = np.array(h5_f[POS_PATH])
    
    print(f'AllClose Lau {np.allclose(lau_1, lau_2)}')
    print(f'AllClose Pos {np.allclose(pos_2, pos_2)}')
    print(f'Dif Lau {np.sum(lau_1-lau_2)}')
    print(f'Dif Pos {np.sum(pos_2-pos_2)}')



if __name__ == '__main__':
    args = parse_args()
    comp_h5_files(args.h5_fp_1, args.h5_fp_2)