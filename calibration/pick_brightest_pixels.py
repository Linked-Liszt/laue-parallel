import h5py
import matplotlib.pyplot as plt
import numpy as np
import os


h5_fp = "/data34b/Run2023-1/Sheyfer423_Indenter/Si_Norcada_90_calib2/SiNorcada90_calib_pos1_1.h5"

with h5py.File(h5_fp, 'r') as h5_f:
    im = np.asarray(h5_f['entry1/data/data'])


indicies = np.array([
        [1618, 6],
        [396, 11], 
        [1766, 33], 
        [156, 60], 
        [1237, 165], 
        [507, 195], 
        [808, 198], 
        [1232, 248],
        [1667, 256], 
        [1980, 295],
        [215, 337], 
        [589, 342], 
        [1878, 434], 
        [1222, 450],
        [899, 510],
        [1543, 557],
        [1216, 603], 
        [487, 641],
        [785, 723], 
        [1211, 723], 
        [573, 747],
        [1959, 756], 
        [1642, 791], 
        [841, 844],
        [1860, 852],
        [1204, 900], 
        [1576, 905],
        [883, 939], 
        [120, 985], 
        [1526, 995], 
        [189, 1017],
        [1200, 1025],
        [250, 1046],
        [353, 1099],
        [1197, 1119], 
        [470, 1166], 
        [557, 1221], 
        [1939, 1305],
        [681, 1307], 
        [1843, 1346],
        [764, 1371], 
        [1709, 1410], 
        [867, 1458],
        [1620, 1458],
        [1509, 1525], 
        [1179, 1781],
        [542, 1818],
        [453, 1840], 
        [167, 1943],
        [1828, 1974]
        ])

def find_nearby_highest_point(im, index, dist=25):
    xl = max(index[0] - dist, 0)
    xh = min(index[0] + dist, im.shape[0])
    yl = max(index[1] - dist, 0)
    yh = min(index[1] + dist, im.shape[1])

    search_area = im[xl:xh, yl:yh]
    max_pix_idx = np.where(search_area == np.max(search_area))

    ret_idx = [xl + max_pix_idx[0][0], yl + max_pix_idx[1][0]]
    return ret_idx

out_indicies = []
for i in indicies:
    out_indicies.append(find_nearby_highest_point(im, i, dist=25))
    print(f'og: {i}\t new: {out_indicies[-1]}')

print('\n\n')

print('[' + ',\n'.join(map(str, out_indicies)) + ']')




