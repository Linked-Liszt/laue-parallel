"""
Script to package files into single hdf5 files with all layers. 
"""
import os
import argparse
import cold
import dataclasses
import re
import h5py
import pathlib
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fp')
    parser.add_argument('output_fp')
    parser.add_argument('--dim', default=2048)
    parser.add_argument('--n_im', default=20)
    return parser.parse_args()

@dataclasses.dataclass
class ColdConfig():
    file: dict
    comp: dict
    geo: dict
    algo: dict
    scanpoint: int = None
    pointer: int = None

def main():
    args = parse_args()

    if not os.path.exists(args.output_fp):
        os.makedirs(args.output_fp)
    

    cc = ColdConfig(*cold.config(args.config_fp))

    files = pathlib.Path(cc.file['path']).glob('*.' + cc.file['ext'])
    files = sorted(files, key=lambda path: int(re.sub('\D', '', str(path))))

    for scanpoint in range(args.n_im):
        print(f'Processing scan {scanpoint}')
        
        cc_range = [int((scanpoint) * cc.file['range'][1]), 
                            int((scanpoint + 1) * cc.file['range'][1]), 
                            1]

        im_stack = np.zeros((cc.file['range'][1], args.dim, args.dim), np.int32)
        print(np.shape(im_stack))
        dset_type = None
        for im_idx in range(cc_range[0], cc_range[1], cc_range[2]):
            print(files[im_idx])
            with h5py.File(files[im_idx], 'r') as in_f:
                im_stack[im_idx - cc_range[0], :, :] = in_f[cc.file['h5']['key']][:]
                if dset_type is None:
                    dset_type = in_f[cc.file['h5']['key']].dtype


        with h5py.File(os.path.join(args.output_fp, f'im_{scanpoint:04}.h5'), 'w') as out_f:
            out_f.create_dataset('data', data=im_stack, dtype=dset_type)
        


    
    

    

    
        


if __name__ == '__main__':
    main()