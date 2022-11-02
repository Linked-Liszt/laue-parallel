import cold
import os
import argparse
import copy

INSERT_LINES = [35, 41]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to perform a parallel coded aperture post-processing.'
    )
    parser.add_argument(
        'script_fp',
        help='runscript path'
    )
    parser.add_argument(
        'config_fp',
        help='Path to .yaml cold configuration'
    )
    return parser.parse_args()

def build_scripts(script_fp, config_fp):
    conf_file, conf_comp, conf_geo, conf_algo = cold.config(config_fp)

    with open(script_fp, 'r') as script_f:
        script_lines_base = script_f.readlines()

    range_end = conf_file['range'][1]
    
    max_im = -1
    for im in os.listdir(conf_file['path']):
        max_im = max(int(im.split('_')[-1].split('.')[-2]), max_im)
    
    if max_im % range_end != 0:
        raise ValueError(f"Invalid Number of Images Detected {max_im}/{range_end}")
    
    num_ims = int(max_im / range_end)
    for im in range(num_ims):
        script_lines = copy.deepcopy(script_lines_base)
        for insert_idx in INSERT_LINES:
            script_lines.insert(insert_idx, f'    --start_im {im}\n')
        
        with open(f'runscripts_prod/im_{im}_laue.sh', 'w') as script_f:
            script_f.writelines(script_lines)


if __name__ == '__main__':
    args = parse_args()
    build_scripts(args.script_fp, args.config_fp)
