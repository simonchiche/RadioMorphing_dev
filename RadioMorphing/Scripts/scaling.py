#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:28:02 2021

@author: chiche
"""

from ModuleScale import EnergyScale, GeomagneticScale, DensityScale, DensityScaleBiasCorrected, CerenkovStretch
import matplotlib.pyplot as plt
import numpy as np
import sys
   
#def myscale(sim_file, primary, energy, zenith, azimuth):
def myscale(RefShower, TargetShower, simxmax):
    
    # Number of reference antennas used for the scaling
    Nant = RefShower.nant
    
    # Translation in the shower plane
    TargetShower.pos, TargetShower.traces = RefShower.GetinShowerPlane()
    #sys.exit()
    
    # Energy scaling
    TargetShower.traces[:,Nant:], kE = EnergyScale(RefShower, TargetShower)

    # Geomagnetic scaling
    TargetShower.traces[:,2*Nant:3*Nant], kgeo = \
    GeomagneticScale(RefShower, TargetShower)
    
    # Bias correction: not to be used, only for testing
    #TargetShower.traces[:,2*Nant:3*Nant], TargetShower.traces[:,3*Nant:4*Nant]\
    #,TargetShower.xmaxpos, krho_geo, krho_ce = \
    #DensityScaleBiasCorrected(RefShower, TargetShower) 

    # Density scaling
    TargetShower.traces[:,2*Nant:3*Nant], TargetShower.traces[:,3*Nant:4*Nant]\
    ,TargetShower.xmaxpos, krho_geo, krho_ce = \
    DensityScale(RefShower, TargetShower)
    #sys.exit(TargetShower.xmaxpos)
    TargetShower.xmaxpos = simxmax # for the tests only # TODO: remove for relase version
        
    # Layout and traces stretching
    TargetShower.pos, TargetShower.traces[:,Nant:], kstretch = \
    CerenkovStretch(RefShower, TargetShower)

    # Back in the geographic plane
    TargetShower.pos, TargetShower.traces = TargetShower.GetinGeographicFrame()

    Pos = TargetShower.pos
    #plt.scatter(Pos[:,0], Pos[:,1], s=1, c='k', label='Target Antennas')
    #plt.show()
    #sys.exit()
        
    # TODO: include magnetic field scaling   
    
    return TargetShower, krho_geo
        