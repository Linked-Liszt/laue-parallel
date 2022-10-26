import argparse
import h5py
from PIL import Image
from skimage import io
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5')
    parser.add_argument('tiff')
    return parser.parse_args()

def compare_h5_tiff(h5_fp, tiff_fp):
    with h5py.File(h5_fp, 'r') as h5_f:
        print(f"Pos 1 Shape: {h5_f['lau'].shape}")
        ds = h5_f['pos'][1, 0:100]
        print(ds)
        ds = np.array(h5_f['pos'])
        print(np.sum(ds))
    
    im = Image.open(tiff_fp)
    im_arr = np.array(im)
    print(im_arr.shape)
    print(np.sum(im_arr))
    print(im_arr[2, 100:300])


if __name__ == '__main__':
    args = parse_args()
    compare_h5_tiff(args.h5, args.tiff)