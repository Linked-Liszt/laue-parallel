import copy
import os
import h5py
import numpy as np
import time
import yaml
from multiprocessing import cpu_count, Pool, shared_memory
from functools import partial


start = time.time()
data_dir = os.environ['DATA_DIR']
data_files = os.listdir(data_dir)
data_files = [data_dir + '/' + fn for fn in data_files]
# get shapes from the last file
ff = h5py.File(data_files[0], 'r')
stack = ff['lau'].shape[1]
frame = ff['frame']
y_dim = frame[3] - frame[2]
x_dim = y_dim

lau_set_sm = shared_memory.SharedMemory(create=True, size=x_dim * y_dim * stack * 8)
lau_set = np.ndarray((x_dim, y_dim, stack), dtype=float, buffer=lau_set_sm.buf)

def process_chunk(filename, def_param=lau_set):
    f = h5py.File(filename, 'r')
    frame = f['frame']
    lau = f['lau']
    ind = f['ind']
    # adjust the index to the current frame and flatten it
    new_ind = (ind[:, 0] - frame[0]) * y_dim + ind[:, 1]
    # create a new ndarray that the laue will be expanded to according to index
    lau_arr = np.zeros(((frame[1] - frame[0]) * (frame[3] - frame[2]), lau.shape[1]))
    # fill out the laue values by index
    lau_arr[new_ind] = lau
    # reshape the array
    lau_arr = np.reshape(lau_arr, ((frame[1] - frame[0]), (frame[3] - frame[2]), lau.shape[1]))
    # write to the corresponding chunk of shared memory
    lau_set[frame[0] : frame[1], :, :] = lau_arr
    f.close()


pool_size = min(len(data_files), cpu_count() * 2)
pool = Pool(processes=pool_size)
pool.map(process_chunk, data_files)
pool.close()
pool.join()
pool.terminate()

mid = time.time()
print('mid time', mid - start)

def write_to_hd5(dir, data_set, data_ind_d):
    def set_group(grp, key, val):
        if type(val) is dict:
            sg = grp.create_group(key)
            for skey, sval in val.items():
                set_group(sg, skey, sval)
        else:
            grp.attrs[key] = val

    (data, ind, attrs) = data_ind_d
    fn = dir + '/out_' + str(ind) + '.hd5'
    f = h5py.File(fn, "w")
    f.create_dataset(data_set, data=data)
    # attach attributes
    for key, value in attrs.items():
        grp = f.create_group(key)
        set_group(grp, key, value)
    f.close()

# create directory where results will go
new_dir = os.environ['OUT_DIR']
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)

# get attributes from a file
attr_file = os.environ['ATTR_FILE']
stream = open(attr_file, 'r')
dc = yaml.load(stream, Loader=yaml.FullLoader)
stream.close()

func = partial(write_to_hd5, new_dir, 'lau')
iterable = [(lau_set[:, :, i], i, copy.deepcopy(dc)) for i in range(stack)]
with Pool(processes=pool_size) as pool:
    pool.map_async(func, iterable)
    pool.close()
    pool.join()
    pool.terminate()

lau_set_sm.close()
lau_set_sm.unlink()

stop = time.time()
print('took ', stop-start, 'secs')

