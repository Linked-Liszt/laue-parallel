import os
import h5py
import numpy as np

DATASETS = ['ene', 'pathlen']
IND_PATH ='ind'

def repack_ene(file_path, data_path, out_path):
    point_file = os.path.basename(file_path)
    point_name = point_file.split('.')[0]

    data_dir = os.path.join(data_path, point_name, 'proc_results')
    data_files = os.listdir(data_dir)
    new_dir = os.path.join(out_path)

    ff = h5py.File(os.path.join(data_dir, data_files[0]), 'r')
    print(f"{file_path}_{ff['lau'].shape}")
    stack = ff['lau'].shape[1]
    frame = ff['frame']
    y_dim = frame[3] - frame[2]
    x_dim = y_dim


    reshapes = {}
    for ds_path in DATASETS:
        reshapes[ds_path]= np.zeros((y_dim, x_dim))
    
    print(data_files)
    for data_fp in data_files:
        with h5py.File(os.path.join(data_dir, data_fp), 'r') as h5_f_in:
            proc_ind = np.array(h5_f_in[IND_PATH])
            proc_ind[:,0] -= frame[0]
            proc_ind[:,1] -= frame[2]

            for ds_path in DATASETS:
                ds = np.array(h5_f_in[ds_path])
                for j, ind in enumerate(proc_ind):
                    reshapes[ds_path][ind[0], ind[1]] = ds[j]

    with h5py.File(os.path.join(out_path, f'{point_name}_ene.h5'), 'w') as h5_f_out:
        for ds_path in DATASETS:
            h5_f_out.create_dataset(ds_path, data=reshapes[ds_path])