import h5py

KEY='/entry1/data/data'

with h5py.File('/eagle/APSDataAnalysis/mprince/lau/data/Al30_thick_maskS1_Z/Al30_thick_maskS1_Z_1212.h5', 'r') as h5_f:
    print(f'Lau 2 Shape: {h5_f[KEY].shape}')