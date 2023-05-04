import numpy as np
import h5py
import sys

h5_fp = sys.argv[1]
thresh = float(sys.argv[2])

with h5py.File(h5_fp, 'r') as h5_f:
    arr = np.asarray(h5_f['entry1/data/data'])

sumed = np.sum(arr, axis=0)

thresholded = np.all(arr < 20, axis=0).astype('float')

nonzero = np.count_nonzero(np.all(thresholded, axis=0).astype('float'))
print(f'Thresh Pixels {nonzero}')
print(f'Analyzed Pixels {(2048 * 2048) - nonzero}')

CALIB_3_X1800 = np.asarray([[1617, 6],
        [396, 11],
        [1766, 33],
        [156, 61],
        [1237, 165],
        [507, 196],
        [808, 198],
        [1232, 248],
        [1667, 256],
        [1981, 296],
        [215, 337],
        [590, 342],
        [1879, 434],
        [1223, 450],
        [899, 510],
        [1538, 543],
        [1217, 603],
        [488, 640],
        [785, 723],
        [1211, 723],
        [573, 747],
        [1959, 756],
        [1643, 791],
        [841, 844],
        [1860, 852],
        [1205, 900],
        [1576, 905],
        [883, 939],
        [120, 985],
        [1526, 995],
        [190, 1017],
        [1200, 1026],
        [251, 1046],
        [353, 1099],
        [1197, 1119],
        [470, 1166],
        [558, 1221],
        [1940, 1306],
        [682, 1307],
        [1844, 1346],
        [764, 1371],
        [1710, 1410],
        [867, 1458],
        [1621, 1458],
        [1509, 1525],
        [1180, 1781],
        [557, 1826],
        [454, 1840],
        [167, 1943],
        [1828, 1974]])

for pixel_idx in CALIB_3_X1800:
    print(thresholded[pixel_idx[0], pixel_idx[1]])