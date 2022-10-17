import argparse
import argparse
import json
import os
from time import time
import numpy as np

TIME_INTRVALS = ['setup_time', 'cold_load', 'cold_decode', 'cold_resolve', 'write_time']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('time_dir')
    return parser.parse_args()


def main():
    args = parse_args()

    all_time_data = {}
    for time_interval in TIME_INTRVALS:
        all_time_data[time_interval] = []

    for time_fp in os.listdir(args.time_dir):
        with open(os.path.join(args.time_dir, time_fp), 'r') as time_f:
            time_data = json.load(time_f)
            for time_interval in TIME_INTRVALS:
                all_time_data[time_interval].append(time_data[time_interval])

    for time_interval in TIME_INTRVALS:
        print(f'{time_interval} {np.mean(all_time_data[time_interval])} {np.std(all_time_data[time_interval])}')



if __name__ == '__main__':
    main()