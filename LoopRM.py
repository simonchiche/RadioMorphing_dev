#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 02:24:48 2021

@author: chiche
"""

import glob
import numpy as np
from RunRadioMorphing import run
from coreRadiomorphing_ground import extractData
import sys
from ScalingTestRadiomorphing import Scalingcheck
from interpolation_test_ground import test_interpolation

# Test library to compare the RM results with ZHAireS simulations
simulations = glob.glob("./TargetShowers/*.hdf5") 

# We initialize the RM test variables
ILDFvxbAll, ILDFvxvxbAll , ItotAll , krho_all, RefTheta, \
TargetTheta, DplaneRef, DplaneTarget = [], [], [], [], [], [], [], []
RefEnergy, TargetEnergy, RefPhi, TargetPhi = [], [], [], []
ResidualPeakAll, TargetPeakAll, RefPeakAll, OmegaAll = [], [], [], []

# We loop the RM over the ZHS test library
for i in range(len(simulations)):
 
# =============================================================================
#                           Initialization
# =============================================================================
    
    # We set the RM target parameters to values of 
    # the ZHS test library simulations 
    
    # Simulated ZHS shower
    Simulated_path = simulations[i]
        
    print("Are you using perp planes? (True or False)")
    PerpPlanes = ("True" == input())
    if(PerpPlanes): 
        print("test", PerpPlanes)
        simulations[i] =  simulations[i].split("Stshp_Stshp")[0] + simulations[i].split("Stshp_Stshp")[1]
    
    
    SimulatedShower = extractData(Simulated_path)
    # We load the antennas positions
    np.savetxt("./DesiredPositions/desired_pos.txt", SimulatedShower.pos)  
    # We load the different target shower parameters
    energy = SimulatedShower.energy
    xmaxsim = SimulatedShower.xmaxpos
    zenith = 180 - SimulatedShower.zenith
    azimuth = float(simulations[i].split("_")[4])
    
    # we check that the RM simulated shower 
    #is not already in the reference  library
    zenithfile =  float(simulations[i].split("_")[3])
    filename = simulations[i].split("/")[-1]
    path = "./Simulations/SelectedPlane/theta_%.1f/" %(zenithfile) + filename
    ref_sim =  glob.glob(path)
    if(len(ref_sim)>0): continue 
    
# =============================================================================
#                         Radio Morphing launching
# =============================================================================
        
    TargetShower, RefShower, efield_interpolated, w_interpolated = \
    run(energy, zenith, azimuth, xmaxsim)
    #sys.exit()
    
# =============================================================================
#                         Radio Morphing tests
# =============================================================================
    
    # Tests of the scaling 
    # TODO: warning the scaling test affect the final results
    ScalingTest = False
    if(ScalingTest):
        ILDFvxb, ILDFvxvxb, Itot = \
        Scalingcheck(TargetShower, SimulatedShower, RefShower)

    # Tests of the interpolation
    InterpolationTest = True
    if(InterpolationTest):
        ResidualPeak, TargetPeak, RefPeak = \
        test_interpolation(SimulatedShower,  TargetShower,\
                           efield_interpolated, True)
        
        ResidualPeakAll.append(ResidualPeak)
        TargetPeakAll.append(TargetPeak)
        RefPeakAll.append(RefPeak)
        OmegaAll.append(w_interpolated)
        #print(ResidualPeak)
        #print(np.mean(abs(ResidualPeak[~np.isnan(ResidualPeak)])))
        print(np.mean(ResidualPeak[~np.isnan(ResidualPeak)]), \
              np.std(ResidualPeak[~np.isnan(ResidualPeak)]))
        Trigger = 0
        if(Trigger!=0):
            CleanRef =  np.array(RefPeak)[~np.isnan(ResidualPeak)]
            CleanRes = ResidualPeak[~np.isnan(ResidualPeak)]
            print(np.mean(CleanRes[CleanRef>Trigger]), np.std(CleanRes[CleanRef>Trigger]))
        Trigger = False
        if(Trigger):
            ErrorPeakTrigger75 = ResidualPeak[RefPeak>75]
            print(np.mean(abs(ErrorPeakTrigger75)))
            
# =============================================================================
#                         Storing the results
# =============================================================================
        
    if(ScalingTest):  
        ILDFvxbAll.append(ILDFvxb), ILDFvxvxbAll.append(ILDFvxvxb), \
        ItotAll.append(Itot)
    RefTheta.append(RefShower.zenith),
    TargetTheta.append(TargetShower.zenith),\
    DplaneRef.append(RefShower.distplane), \
    DplaneTarget.append(TargetShower.distplane),\
    RefEnergy.append(RefShower.energy),\
    TargetEnergy.append(TargetShower.energy),\
    RefPhi.append(RefShower.azimuth),\
    TargetPhi.append(TargetShower.azimuth)

# Ref and Target parameters
np.savetxt("RefTargetParameters.txt", np.transpose([RefTheta,TargetTheta, \
RefEnergy,TargetEnergy, RefPhi,TargetPhi, DplaneRef, DplaneTarget]))

# LDF error
#np.savetxt("LDFScalingTest.txt", np.transpose(\
#[ILDFvxbAll, ILDFvxvxbAll, ItotAll]))

#peak error, TODO: inclure peaktime
np.savetxt("PeakResidual.txt", ResidualPeakAll)
np.savetxt("PeakTarget.txt", TargetPeakAll)
np.savetxt("PeakRef.txt", RefPeakAll)
np.savetxt("OmegaAngle.txt", OmegaAll)


    

