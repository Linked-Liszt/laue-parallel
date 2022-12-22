import subprocess
import os
import copy
import time
import argparse


NUM_NODES = 32

PROD_CMD = [
    'qsub',
    '-A',
    'APSDataAnalysis',
    '-q',
    'prod',
    '-l'
    f'select={NUM_NODES}:system=polaris',
    '-l'
    'walltime=02:00:00',
    '-l',
    'filesystems=home:eagle',
    '-l',
    'place=scatter',
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to queue a series of scripts on polaris'
    )
    parser.add_argument(
        'runscripts_dir',
        help='Path to folder containing bash scripts'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Prints cold grid allocations and terminates run before processing'
    )
    return parser.parse_args()


def queue_scripts(runscripts_dir: str, dry_run: bool) -> None:
    runscripts_dir = os.path.abspath(runscripts_dir)
    for script in os.listdir(runscripts_dir):
        prod_cmd = copy.deepcopy(PROD_CMD)
        prod_cmd.append(os.path.join(runscripts_dir, script))
        print(prod_cmd)
        if not dry_run:
            subprocess.run(prod_cmd)
            time.sleep(5)
    

if __name__ == '__main__':
    args = parse_args()
    queue_scripts(args.runscripts_dir, args.dry_run)
