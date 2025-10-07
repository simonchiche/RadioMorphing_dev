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