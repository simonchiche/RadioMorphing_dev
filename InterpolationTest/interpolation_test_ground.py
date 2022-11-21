# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:39:15 2020

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from module_signal_process import filter_traces
import scipy
import sys


def test_interpolation(SimulatedShower, TargetShower, efield_interpolated,\
                       Display = True):
    
    
# =============================================================================
#                            Initialization
# =============================================================================

    Traces = SimulatedShower.traces
    NantRefStarshape = SimulatedShower.InitialShape
    NantRefLayout = SimulatedShower.NantTraces
    NantRefCrossCheck = NantRefLayout - NantRefStarshape
    NantTargetLayout = 176#TargetShower.nant - TargetShower.InitialShape        
    
    filtering = TargetShower.filter
    if filtering:
        time_sample = int(len(Traces[:,0]))
        Traces = filter_traces(Traces, NantRefLayout, time_sample)
    
# =============================================================================
#                  Extracting the reference Traces
# =============================================================================
    
    use_cross_check = False
    if(use_cross_check):
        refTime, refEx, refEy, refEz = [], [], [], []
        for i in range(4* NantRefLayout):
            if((i>=NantRefLayout - NantRefCrossCheck) & (i<NantRefLayout)): 
                refTime.append(Traces[:,i])
            if((i>=2*NantRefLayout -NantRefCrossCheck) & (i<2*NantRefLayout)): 
                refEx.append(Traces[:,i])
            if((i>=3*NantRefLayout -NantRefCrossCheck) & (i<3*NantRefLayout)): 
                refEy.append(Traces[:,i])
            if((i>=4*NantRefLayout -NantRefCrossCheck) & (i<4*NantRefLayout)): 
                refEz.append(Traces[:,i])
    
    UseStarshape = not(use_cross_check)    
    if(UseStarshape):
        refTime, refEx, refEy, refEz = [], [], [], []
        for i in range(4*NantRefLayout):
            if((i>=0) & (i<NantTargetLayout)): 
                refTime.append(Traces[:,i])
            if((i>=NantRefLayout) & (i<NantRefLayout + NantTargetLayout)): \
                refEx.append(Traces[:,i])
            if((i>=2*NantRefLayout) & (i<2*NantRefLayout + NantTargetLayout)): 
                refEy.append(Traces[:,i])
            if((i>=3*NantRefLayout) & (i<3*NantRefLayout + NantTargetLayout)): 
                refEz.append(Traces[:,i])
                
# =============================================================================
#                   Loading the Target Traces
# =============================================================================
  
    refEx, refEy, refEz = np.array(refEx),  np.array(refEy),  np.array(refEz)
    error_peak,rm_peak, zh_peak, diff_time_all  = [], [], [], []
    
    print(NantTargetLayout)
    #sys.exit()
    for i in range(NantTargetLayout):
        
        # We load the Target traces
        Load = True
        try:
           # targetTime, targetEx, targetEy, targetEz = np.loadtxt(\
           #"./OutputDirectory/DesiredTraces_%d.txt"%i, unpack = True)
            targetTime, targetEx, targetEy, targetEz = \
            efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
            efield_interpolated[i][:,2], efield_interpolated[i][:,3]
        except(IndexError):
            Load = False
            rm_peak.append(np.nan)
            zh_peak.append(np.nan)
            error_peak.append(np.nan)
        if(Load):
            targetTime, targetEx, targetEy, targetEz = \
            efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
            efield_interpolated[i][:,2], efield_interpolated[i][:,3]
            #t, targetEx, targetEy, targetEz = np.loadtxt(\
            #"./OutputDirectory/DesiredTraces_%d.txt"%i, unpack = True)

# =============================================================================
#                      Optional Post Filtering
# =============================================================================
           
            # In case we want to filter the reference and the target traces 
            #after the interpolation
            post_filtering = False
            if post_filtering:
                
                # Filtering the reference Traces
                time_sample = int(len(refTime[0]))
                RefTracesArray = np.transpose\
                (np.array([refTime[i], refEx[i], refEy[i], refEz[i]]))
                TracesRefFiltered = \
                filter_traces(RefTracesArray, 1, time_sample)
                refEx[i], refEy[i], refEz[i] = TracesRefFiltered[:,1], \
                TracesRefFiltered[:,2], TracesRefFiltered[:,3]

                # Filtering the target Traces
                time_sample = int(len(targetTime))
                TracesTargetFiltered = filter_traces(np.transpose(np.array\
                ([targetTime,targetEx,targetEy,targetEz])), 1, time_sample)
                
                targetTime, targetEx, targetEy, targetEz = \
                TracesTargetFiltered[:,0], TracesTargetFiltered[:,1], \
                TracesTargetFiltered[:,2], TracesTargetFiltered[:,3]
            

# =============================================================================
#                       Ref/Target comparison
# =============================================================================

            # We compute the total Efield
            refEtot =  np.sqrt(refEx**2 + refEy**2 + refEz**2)
            targetEtot = np.sqrt(targetEx**2 + targetEy**2 + targetEz**2)
            
            # We shift the Traces time window
            refTime[i] = refTime[i] - refTime[i][0]
            targetTime = targetTime -targetTime[0]
            refTimeArray = 0.5*np.arange(0, len(refEtot[i]), 1)
            targetTimeArray = 0.5*np.arange(0, len(targetEtot), 1)
            
            #if(max(targetEtot)>1e4):
                
            #print(max(targetEtot), max(refEtot[i]))
            Ndisplay = 60
            if((Display) & (i<Ndisplay)):
                
                plt.plot(refTimeArray, refEtot[i], label = "simulation") 
                plt.plot(targetTimeArray, targetEtot, label = "interpolation")
                plt.xlabel("Time [ns]")
                plt.ylabel("E [$\mu V/m$]")
                plt.legend()
                #plt.xlim(150, 300)
                plt.tight_layout()
                #plt.savefig\
                #("./InterpolationTest_Etot_antenna_scale_int3D%.d.pdf" %i)
                plt.show()
                print(i)
            
            PeakrefEtot = max(refEtot[i])
            PeaktargetEtot = max(targetEtot)
            
            error_peak.append((PeakrefEtot- PeaktargetEtot)/PeakrefEtot)
            #print((PeakrefEtot- PeaktargetEtot)/PeakrefEtot)
            rm_peak.append(PeaktargetEtot)
            zh_peak.append(PeakrefEtot)                  
    
    return np.array(error_peak), np.array(rm_peak), np.array(zh_peak)