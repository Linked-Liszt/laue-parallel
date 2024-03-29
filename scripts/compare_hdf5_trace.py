import argparse
import argparse
import numpy as np
import h5py

paths = ['lau']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_fp_1')
    parser.add_argument('h5_fp_2')
    return parser.parse_args()


def comp_h5_files(h5_fp_1, h5_fp_2):
    args = parse_args()
    for ds_path in paths:
        print(f'\n----{ds_path}----')
        with h5py.File(h5_fp_1, 'r') as h5_f:
            print(f'{ds_path} 1 Shape: {h5_f[ds_path].shape}')
            ds_1 = np.array(h5_f[ds_path])

        with h5py.File(h5_fp_2, 'r') as h5_f:
            print(f'{ds_path} 2 Shape: {h5_f[ds_path].shape}')
            ds_2 = np.array(h5_f[ds_path])

        if np.any(np.isnan(ds_1)):
            print(f'WARN: nans detected in {ds_path} 1 ({np.count_nonzero(np.isnan(ds_1))} nans)')
            ds_1 = np.nan_to_num(ds_1)

        if np.any(np.isnan(ds_2)):
            print(f'WARN: nans detected in {ds_path} 2 ({np.count_nonzero(np.isnan(ds_2))} nans)')
            ds_2 = np.nan_to_num(ds_2)

        
        print(f'Sum {ds_path} 1 {np.sum(ds_1)}')
        print(f'Sum {ds_path} 2 {np.sum(ds_2)}')
        print(f'AllClose {ds_path} {np.allclose(ds_1, ds_2)}')
        print(f'Dif {ds_path} {np.sum(ds_1-ds_2)}')
        diffs = np.where((np.isclose(ds_1, ds_2)) == False)
        if len(ds_1.shape) == 2:
            print(f'Total Diffs {len(diffs[0])})')
            for i in range(len(diffs[0])):
                print(f'{diffs[0][i]}, {diffs[1][i]}')
                print(f'{ds_1[diffs[0][i], diffs[1][i]]} | {ds_2[diffs[0][i], diffs[1][i]]}')
            
                input()

        if len(ds_1.shape) == 1:
            print(f'Total Diffs {len(diffs)}))')
            for i in range(len(diffs)):
                print(f'{diffs[i]}')
                print(f'{ds_1[diffs[i]]} | {ds_2[diffs[i]]}')



if __name__ == '__main__':
    args = parse_args()
    comp_h5_files(args.h5_fp_1, args.h5_fp_2)