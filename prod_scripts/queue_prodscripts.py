import subprocess
import os
import copy
import time

RUNSCRIPTS_DIR = '/eagle/APSDataAnalysis/mprince/lau/laue-parallel/runscripts_prod'

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
    'walltime=1:00:00',
    '-l',
    'filesystems=home:eagle',
    '-l',
    'place=scatter',
]

for script in os.listdir(RUNSCRIPTS_DIR):
    prod_cmd = copy.deepcopy(PROD_CMD)
    prod_cmd.append(os.path.join(RUNSCRIPTS_DIR, script))
    print(prod_cmd)
    subprocess.run(prod_cmd)
    time.sleep(5)