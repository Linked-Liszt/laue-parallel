import h5py
import sys
import argparse
import os
import numpy as np


def readhdf(fname):
  def scan_node(g, tabs=0):
      elems = []
      for k, v in g.items():
          if isinstance(v, h5py.Dataset):
              print('dataset ' + v.name)
              print(f'dataset shape: {v.shape}')
              print(f'dtype {v.dtype}')
              #print(f'dataset: {np.array(v)}')

  with h5py.File(fname, 'r') as f:
      scan_node(f)


def main(arg):
  parser = argparse.ArgumentParser()
  parser.add_argument("hd5file", help="hd5 file name")
  args = parser.parse_args()
  readhdf(args.hd5file)


if __name__ == "__main__":
  exit(main(sys.argv[1:]))
