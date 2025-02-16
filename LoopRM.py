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
#from interpolation_test_ground import test_interpolation
from InterpolationTest.TestFilter import test_interpolation
import matplotlib.pyplot as plt
from ScalingTests.CompareLDF import CompareLDF, GetRMtraces
plt.ion()

Local = True
Lyon = not(Local)

# Test library to compare the RM results with ZHAireS simulations
if(Local):
    simulations = glob.glob("./TargetShowers/*.hdf5") 
else:
    simulations = np.loadtxt\
    ("/sps/trend/chiche/RadiomorphingUptoDate/TargetPath.txt", dtype = 'str')
    start =  int(sys.argv[1]) 
    stop =  int(sys.argv[2])
    if(stop>len(simulations)): stop = len(simulations) 
    print(len(simulations))

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
    
    if(Local):
        print("Are you using perp planes? (True or False)")
        PerpPlanes = False#("True" == input())
        if(PerpPlanes): 
            print("test", PerpPlanes)
            simulations[i] =  simulations[i].split("Stshp_Stshp")[0] + \
            simulations[i].split("Stshp_Stshp")[1]
        
    
    SimulatedShower = extractData(Simulated_path)
    # We load the antennas positions
    if(Local):
        np.savetxt("./DesiredPositions/desired_pos.txt", SimulatedShower.pos)
    elif(Lyon):
        np.savetxt\
        ("/sps/trend/chiche/RadiomorphingUptoDate/DesiredPositions/desired_pos%.d.txt" \
         %start, SimulatedShower.pos)  
        
    # We load the different target shower parameters
    energy = SimulatedShower.energy
    xmaxsim = SimulatedShower.xmaxpos
    zenith = 180 - SimulatedShower.zenith
    azimuth = float(simulations[i].split("_")[4])
    
    # we check that the RM simulated shower 
    #is not already in the reference  library
    zenithfile =  float(simulations[i].split("_")[3])
    filename = simulations[i].split("/")[-1]
    if(Local):
        path = "./Simulations/SelectedPlane/theta_%.1f/" %(zenithfile) + filename
    elif(Lyon):
        path = \
        "/sps/trend/chiche/RadiomorphingUptoDate/Simulations/SelectedPlane/theta_%.1f/" \
        %(zenithfile) + filename
    ref_sim =  glob.glob(path)
    if(len(ref_sim)>0): continue 
    
# =============================================================================
#                         Radio Morphing launching
# =============================================================================
    if(Local):
        TargetShower, RefShower, efield_interpolated, w_interpolated, IndexAll = \
        run(energy, zenith, azimuth, xmaxsim)
    elif(Lyon):
            TargetShower, RefShower, efield_interpolated, w_interpolated, IndexAll = \
    run(energy, zenith, azimuth, xmaxsim, start)
    #fsys.exit()
    
# =============================================================================
#                         Radio Morphing tests
# =============================================================================
    '''
    # Tests of the scaling 
    #GetRMtraces(TargetShower, SimulatedShower, RefShower)
    RMtime, RMx, RMy, RMz, index, refTime, refEx, refEy, refEz = GetRMtraces(SimulatedShower, TargetShower, efield_interpolated, IndexAll)
    print(np.shape(refTime), np.shape(RMtime))
    sys.exit()
    '''
    # TODO: warning the scaling test affect the final results
    ScalingTest = True
    if(ScalingTest):
        ILDFvxb, ILDFvxvxb, Itot = \
        Scalingcheck(TargetShower, SimulatedShower, RefShower)
    sys.exit()
    # Tests of the interpolation
    InterpolationTest = True
    if(InterpolationTest):
        ResidualPeak, TargetPeak, RefPeak = \
        test_interpolation(SimulatedShower,  TargetShower,\
                           efield_interpolated, IndexAll, True)
        sys.exit()

        
        plt.scatter(w_interpolated, ResidualPeak, label = "M = %.2f, R = %.2f" %(np.mean(ResidualPeak[~np.isnan(ResidualPeak)]), np.std(ResidualPeak[~np.isnan(ResidualPeak)])))
        plt.xlabel("$\omega$ [Deg.]")
        plt.ylabel("Relative peak error")
        plt.legend()
        plt.title("$\\theta=$%.f$\degree$" %(180-TargetShower.zenith))
        plt.tight_layout()
        #plt.savefig("/Users/chiche/Desktop/RelErrorVsOmega_theta%.f_BiasCorrected.pdf" %(180-TargetShower.zenith))
        plt.show()
        
        ResidualPeakAll.append(ResidualPeak)
        TargetPeakAll.append(TargetPeak)
        RefPeakAll.append(RefPeak)
        OmegaAll.append(w_interpolated)
        #print(ResidualPeak)
        #print(np.mean(abs(ResidualPeak[~np.isnan(ResidualPeak)])))
        print(np.mean(ResidualPeak[~np.isnan(ResidualPeak)]), \
              np.std(ResidualPeak[~np.isnan(ResidualPeak)]))
        #print(ResidualPeak[~np.isnan(ResidualPeak)])
        #print(np.std(ResidualPeak[(~np.isnan(ResidualPeak)) & (ResidualPeak<0.95)]), "HEEERE")
        #np.savetxt("error86_5.txt", ResidualPeak[~np.isnan(ResidualPeak)])
        plt.plot(ResidualPeak[~np.isnan(ResidualPeak)])
        plt.xlabel("AntennaID")
        plt.ylabel("Relative peak error")
        plt.tight_layout()
        #plt.savefig("Resdiual86_5_D150km")
        plt.show()
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
if(Local):
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

elif(Lyon):
    np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/RefTargetParameters%.2d.txt" %start, np.transpose([RefTheta,TargetTheta, \
    RefEnergy,TargetEnergy, RefPhi,TargetPhi, DplaneRef, DplaneTarget]))
    
    # LDF error
    #np.savetxt("LDFScalingTest.txt", np.transpose(\
    #[ILDFvxbAll, ILDFvxvxbAll, ItotAll]))
    
    #peak error, TODO: inclure peaktime
    np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/PeakResidual%.2d.txt" %start, ResidualPeakAll)
    np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/PeakTarget%.2d.txt" %start, np.array(TargetPeakAll))
    np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/PeakRef%.2d.txt" %start, np.array(RefPeakAll))
    np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/OmegaAngle%.2d.txt" %start, np.array(OmegaAll))
        


    

