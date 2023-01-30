in_dir = '/eagle/APSDataAnalysis/mprince/lau/data/Al30_thick_maskCam2v1S3right_xline_dZ_macro'

out_dir = '/eagle/APSDataAnalysis/mprince/lau/data/Al30_masks_sorted'

import os
import shutil

for file in os.listdir(in_dir):
    mask_id = file[4]
    out_path = os.path.join(out_dir, f'mask_{mask_id}')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    in_path = os.path.join(in_dir, file)
    out_path = os.path.join(out_path, file)
    shutil.copy(in_path, out_path)
    print(f's: {in_path} d: {out_path}')


