#!/usr/bin/env python3

import cold
import fire
import numpy as np
import logging
import time 

def main(path, debug=False):
    """Runs the reconstruction workflow given parameters 
    in a configuration file.

    Parameters
    ----------
    path: string
        Path of the YAML file with configuration parameters.

    Returns
    -------
        None
    """
    t = time.time()

    # Read parameters from the config file
    file, comp, geo, algo = cold.config(path)

    # Load data
    data, ind = cold.load(file)

    # Reconstruct
    pos, sig, scl = cold.decode(data, ind, comp, geo, algo, debug=debug)
    dep, lau = cold.resolve(data, ind, pos, sig, geo, comp)

    #cold.saveimg(file['output'] + '/pos-' + str(scanpoint) + '/pos-' + str(m) + '-' + str(n), pos, ind, (2048, 2048), file['frame']) 
    #cold.saveimg(file['output'] + '/lau-' + str(scanpoint) + '/lau-' + str(m) + '-' + str(n), lau, ind, (2048, 2048), file['frame'], swap=True)

    print (time.time() - t)

if __name__ == '__main__':
    fire.Fire(main)
