#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys




def GetRMtraces(SimulatedShower, TargetShower, efield_interpolated, IndexAll):
        
    Traces = SimulatedShower.traces
    NantRefStarshape = SimulatedShower.InitialShape
    NantRefLayout = SimulatedShower.NantTraces
    NantRefCrossCheck = NantRefLayout - NantRefStarshape
    NantTargetLayout = TargetShower.nant - TargetShower.InitialShape   
    
# =============================================================================
#                  Extracting the reference Traces
# =============================================================================

    refTime, refEx, refEy, refEz = [], [], [], []
    for i in range(4*NantRefLayout):
        if((i>=0) & (i<NantTargetLayout)): 
            refTime.append(Traces[:,i])
            print(i, NantTargetLayout, "time")
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
    print(np.shape(refTime)), print(np.shape(refEx))
    print(NantTargetLayout)
    RMtime, RMx, RMy, RMz, index = [], [], [], [], []
    
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
            #print("OULAH")
        if(Load):
            targetTime, targetEx, targetEy, targetEz = \
            efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
            efield_interpolated[i][:,2], efield_interpolated[i][:,3]
            
            #print("target", np.shape(targetTime))
            RMtime.append(targetTime)
            RMx.append(targetEx)
            RMy.append(targetEy)
            RMz.append(targetEz)
            index.append(i)
            #t, targetEx, targetEy, targetEz = np.loadtxt(\
            #"./OutputDirectory/DesiredTraces_%d.txt"%i, unpack = True)

    return RMtime, RMx, RMy, RMz, index, refTime, refEx, refEy, refEz



def CompareLDF(SimulatedShower, TargetShower, efield_interpolated, IndexAllr):

    RMtime, RMx, RMy, RMz, index, refTime, refEx, refEy, refEz = \
        GetRMtraces(SimulatedShower, TargetShower, efield_interpolated, IndexAll)

    
    return