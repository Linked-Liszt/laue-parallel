import argparse
import cold
from tabulate import tabulate

paths = ['lau', 'pos', 'sig', 'ind']

EST_PX_SEC = 150

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fp')
    parser.add_argument('--n',
                        default=16,
                        type=int,
                        help='number of processes per node')
    parser.add_argument('--max',
                        default=32,
                        type=int,
                        help='max nodes to check')
    parser.add_argument('--thresh',
                        default=10,
                        type=int,
                        help='threshold for unfilled procs')
    return parser.parse_args()


def print_possible_nodes(config_fp, n_procs, max_nodes, thresh):
    conf_file, conf_comp, conf_geo, conf_algo = cold.config(config_fp)

    frame = conf_file['frame']

    dim_y = frame[1] - frame[0]
    dim_x = frame[3] - frame[2]
    im_dim = dim_y

    if dim_y != dim_x:
        raise NotImplementedError("Can only reconstruct square images!")

    gr_size = 1
    req_procs = gr_size ** 2
    table_rows = []
    while req_procs <= (max_nodes * n_procs):
        if im_dim % gr_size == 0:
            for n_nodes in range(1, max_nodes + 1):
                avail_procs = n_procs * n_nodes
                if avail_procs - req_procs <= thresh and avail_procs >= req_procs:
                    row = []
                    row.append(f'{gr_size}/{req_procs}')
                    row.append(f'{n_nodes}/{avail_procs}')
                    row.append(f'{avail_procs - req_procs}')
                    num_px = (im_dim / gr_size) ** 2
                    row.append(f'{num_px}')
                    row.append(f'{num_px / EST_PX_SEC / 60:.2f}')
                    table_rows.append(row)
            
                if avail_procs > req_procs:
                    break

        gr_size += 1
        req_procs = gr_size ** 2
    
    headers = ['Grid/Procs', 'Nodes/Procs', 'Unused', 'Num Pixels', 'Est. Time']
    print(tabulate(table_rows, headers, tablefmt='grid'))

        


if __name__ == '__main__':
    args = parse_args()
    print_possible_nodes(args.config_fp, args.n, args.max, args.thresh)