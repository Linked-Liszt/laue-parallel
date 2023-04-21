#!/usr/bin/env python3

from re import M
from tkinter import N
import cold
import fire
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage, stats, optimize, interpolate
from matplotlib.gridspec import GridSpec
import os
import calib_indicies
import matplotlib
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'font.size': 9, 'font.family' : 'times'})
from matplotlib.pyplot import cm
import warnings
warnings.filterwarnings('ignore')


def main(path1, path2, debug=False):
    """Runs the reconstruction workflow given parameters 
    in a configuration file.

    Parameters
    ----------
    path: string
        Path of the YAML file with configuration parameters.

    debug: bool
        If True, plots the fitted signals. 

    Returns
    -------
        None
    """

    def costs(vals, ww, dat, ind, comp, geo, algo, a=0, debug=True):

        # Update optimization parameters
        geo['mask']['focus']['cenx'] = vals[0]
        geo['mask']['focus']['dist'] = vals[1]
        # geo['mask']['focus']['dist'] = vals[1] + 0.05 * ww
        geo['mask']['focus']['anglez'] = vals[2]
        geo['mask']['focus']['angley'] = vals[3]
        geo['mask']['focus']['anglex'] = vals[4]
        geo['mask']['focus']['cenz'] = vals[5]

        pos, sig, scl, ene = cold.decode(dat, ind, comp, geo, algo)
        dep, lau = cold.resolve(dat, ind, pos, sig, geo, comp)

        # Weights based on squared intensity
        weights = np.sqrt(np.max(dat, axis=1))
        ii = np.argsort(weights)

        # cost function evaluations for lau (TODO: increase the resolution)
        grid = geo['source']['grid']
        x = np.arange(*grid)
        indp0 = np.zeros((lau.shape[0]), dtype='int32')
        for m in range(lau.shape[0]):
            ha0 = np.cumsum(lau[m])
            if np.max(ha0) > 0:
                ha0 = ha0 / np.max(ha0)
                indp0[m] = np.min(np.where(ha0 >= 0.5))
        focus = x[indp0.flatten()]
        costs = np.sum(np.power(np.multiply(focus, weights), 2))

        if debug == True:
            fig = plt.figure(figsize=(6, 8))
            gs = GridSpec(10, 1, figure=fig)
            color = cm.RdBu(np.linspace(0, 1, lau.shape[0]))
            fig.add_subplot(gs[0:9, 0])
            for m in range(lau.shape[0]):
                plt.step(x, lau[ii[m]] / (np.max(lau[ii[m]])) - ii[m] - 1, color=color[m], where='mid', linewidth=0.8)
            plt.grid(True, linewidth=0.8)
            
            fig.add_subplot(gs[9, 0])
            plt.step(x, dep / dep.max(), 'darkorange', where='mid', linewidth=1)
            plt.ylim((-0.15, 1.15))
            plt.grid(True, linewidth=0.8)
            plt.title(str(vals), fontsize=7)
                
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.04, top=0.97, hspace=2)
            if not os.path.exists('tmp/' + geo['mask']['cal']['path'] + '/autofocus'):
                os.makedirs('tmp/' + geo['mask']['cal']['path'] + '/autofocus')
            plt.savefig('tmp/' + geo['mask']['cal']['path'] + '/autofocus/autofocus-' + str(a) + '-' + str(geo['mask']['cal']['id']) + '.png', dpi=300)
            plt.close()

        return costs
    
    indices = calib_indicies.CALIB_3_X1800


    # Load calibration datasets
    number = 2
    file1, comp1, geo1, algo1 = cold.config(path1)
    file2, comp2, geo2, algo2 = cold.config(path2)
    # file3, comp3, geo3, algo3 = cold.config(path3)
    # file4, comp4, geo4, algo4 = cold.config(path4)
    # file5, comp5, geo5, algo5 = cold.config(path5)
    dat1, ind = cold.load(file1, index=indices) 
    dat2, ind = cold.load(file2, index=indices) 

    #dat1 = np.load('tmp/' + geo1['mask']['cal']['path'] + '/data/dat1.npy')
    #dat2 = np.load('tmp/' + geo2['mask']['cal']['path'] + '/data/dat2.npy')
    # dat3 = np.load('tmp/' + geo3['mask']['cal']['path'] + '/data/dat3.npy')
    # dat4 = np.load('tmp/' + geo4['mask']['cal']['path'] + '/data/dat4.npy')
    # dat5 = np.load('tmp/' + geo5['mask']['cal']['path'] + '/data/dat5.npy')
    #ind = np.load('tmp/' + geo2['mask']['cal']['path'] + '/data/ind.npy')

    # dat = [dat1, dat2, dat3, dat4, dat5]
    # comp = [comp1, comp2, comp3, comp4, comp5]
    # geo = [geo1, geo2, geo3, geo4, geo5]
    # algo = [algo1, algo2, algo3, algo4, algo5]
    # dat = [dat2, dat3, dat4, dat5]
    # comp = [comp2, comp3, comp4, comp5]
    # geo = [geo2, geo3, geo4, geo5]
    # algo = [algo2, algo3, algo4, algo5]
    dat = [dat1, dat2]
    comp = [comp1, comp2]
    geo = [geo1, geo2]
    algo = [algo1, algo2]

    # Sequence of coordinates
    seq = np.array([0, 1, 2, 3, 4, 5], dtype='int')

    # Search regions
    lbound = np.array([0.4, 1.4, 42.35, -2, -2, -2]) 
    ubound = np.array([0.5, 1.7, 44.35,  2,  2,  2])
    mpoint = 0.5 * (lbound + ubound)

    # Parameter init
    vl = mpoint.copy()
    vu = mpoint.copy()
    vm = mpoint.copy()

    a = 0
    for nn in range(10): # For each CG iteration
        for qq in range(6): # For each coordinate

            # Lower bound cost
            costl = 0
            for ww in range(number):
                vl[seq[qq]] = lbound[seq[qq]]
                costl += costs(vl, ww, dat[ww], ind, comp[ww], geo[ww], algo[ww], a, debug=True)
                a += 1

            # Upper bound cost
            costu = 0
            for ww in range(number):
                vu[seq[qq]] = ubound[seq[qq]]
                costu += costs(vu, ww, dat[ww], ind, comp[ww], geo[ww], algo[ww], a, debug=True)
                a += 1
                

            for it in range(10): # For each binary search iteration
                
                # Middle point cost
                costm = 0
                for ww in range(number):
                    vm[seq[qq]] = (vu[seq[qq]] + vl[seq[qq]]) * 0.5
                    costm += costs(vm, ww, dat[ww], ind, comp[ww], geo[ww], algo[ww], a, debug=True)
                    a += 1

                # Update points and bounds
                costlm = costl - costm
                costum = costu - costm
                if costlm > costum:
                    vl[seq[qq]] = vm[seq[qq]]
                    costl = costm
                else:
                    vu[seq[qq]] = vm[seq[qq]]
                    costu = costm
                vm[seq[qq]] = (vu[seq[qq]] + vl[seq[qq]]) * 0.5

                # Save/print results
                print (str(nn) + '-' + str(qq) + '-' + str(it) + ': ' + str(vm[0:7]))

if __name__ == '__main__':
    fire.Fire(main)

