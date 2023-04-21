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
import matplotlib
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'font.size': 9, 'font.family' : 'times'})
from matplotlib.pyplot import cm
import warnings
warnings.filterwarnings('ignore')


def main(path1, path2, path3, path4, path5, debug=False):
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
        geo['mask']['focus']['dist'] = vals[1] + 0.05 * ww
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
        costs = np.sum(np.power(np.multiply(x[indp0.flatten()], weights), 2))

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
    

    # for calib1
    indices = np.array([
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

    # Load calibration datasets
    number = 5
    file1, comp1, geo1, algo1 = cold.config(path1)
    file2, comp2, geo2, algo2 = cold.config(path2)
    file3, comp3, geo3, algo3 = cold.config(path3)
    file4, comp4, geo4, algo4 = cold.config(path4)
    file5, comp5, geo5, algo5 = cold.config(path5)

    # Load config parameters
    dat1, ind = cold.load(file1, index=indices) 
    dat2, ind = cold.load(file2, index=indices) 
    dat3, ind = cold.load(file3, index=indices) 
    dat4, ind = cold.load(file4, index=indices) 
    dat5, ind = cold.load(file5, index=indices) 

    dat = [dat1, dat2, dat3, dat4, dat5]
    comp = [comp1, comp2, comp3, comp4, comp5]
    geo = [geo1, geo2, geo3, geo4, geo5]
    algo = [algo1, algo2, algo3, algo4, algo5]

    # Sequence of coordinates
    seq = np.array([0, 1, 2, 3, 4, 5], dtype='int')

    # Search regions
    lbound = np.array([0.44, 1.3, 42.3, -2, -2, -2]) 
    ubound = np.array([0.52, 1.8, 44.3,  2,  2,  2])
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
                print (str(nn) + '-' + str(qq) + '-' + str(it) + ': ' + str(vm[0:6]))

if __name__ == '__main__':
    fire.Fire(main)


