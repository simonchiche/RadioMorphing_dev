#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:33:05 2022

@author: chiche
"""


import numpy as np
import matplotlib.pyplot as plt


font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

def PlotMeanRMS(ZenithCut, MeanResidual, RMSResidual):

    plt.figure()
    plt.subplot(211)
    plt.scatter(ZenithCut,  MeanResidual)
    plt.ylim(-0.4, 0.4)
    #plt.ylim(-0.25, 0.25)
    plt.ylabel("Mean")
    plt.subplot(212)
    plt.scatter(ZenithCut, RMSResidual, color = 'orange', marker ="*")
    plt.xlabel("Target zenith [Deg.]")
    plt.ylabel("RMS")
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig("/Users/chiche/Desktop/Mean_RMSvsThetaRelErrorPeak.pdf")
    plt.show()
    
def PlotMeanRMSPlaneDistance(Dplane, MeanResidual, RMSResidual):
    
    plt.figure()
    plt.subplot(211)
    plt.scatter(Dplane,  MeanResidual)
    plt.ylim(-0.4, 0.1)
    plt.ylabel("Mean")
    plt.subplot(212)
    plt.scatter(Dplane, RMSResidual, color = 'orange', marker ="*")
    plt.xlabel("Plane Distance [km]")
    plt.ylabel("RMS")
    plt.ylim(0, 1)
    plt.tight_layout()
    #plt.savefig("scaling_int3d_error_peak_mean_rms_sub_voltage.pdf")
    plt.show()

def PlotRMSDistrib(data, zenith, savepath):
    plt.hist(data, bins=50, color="skyblue", edgecolor='black', linewidth=0.5, alpha=0.8)
    plt.xlabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
    plt.ylabel("Number of antennas")
    plt.title("Zenith = %.1f°" %zenith)
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(savepath + "/RMSDistribZenith%.1f.pdf" %zenith, bbox_inches = "tight")
    plt.show()
    return

def GetMeanRMSerr(EnergyFiltered, ZenithFiltered, error_all, energy_threshold, filteredpath, PLOT=False):

    ZenithFilteredcut = np.unique(ZenithFiltered)
    Meanerr_filtered = np.zeros(len(ZenithFilteredcut))
    RMSerr_filtered = np.zeros(len(ZenithFilteredcut))

    for i in range(len(ZenithFilteredcut)):
        indices = np.where((ZenithFiltered == ZenithFilteredcut[i]) & (EnergyFiltered>energy_threshold))[0]
        err_zen = {k: error_all.get(k) for k in indices if k in error_all}
        err_1d = [val for arr in err_zen.values() for val in arr]
        err_1d = np.array(err_1d)

        if(PLOT):
            PlotRMSDistrib(err_1d, ZenithFilteredcut[i], filteredpath)
        Meanerr_filtered[i] = np.mean(err_1d)
        RMSerr_filtered[i] = np.std(err_1d)

    return ZenithFilteredcut, Meanerr_filtered, RMSerr_filtered