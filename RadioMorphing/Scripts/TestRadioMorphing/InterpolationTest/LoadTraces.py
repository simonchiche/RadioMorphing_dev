#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from module_signal_process import filter_traces

def LoadZHSsimTraces(SimulatedShower, TargetShower):

    Traces = SimulatedShower.traces
    # Number of reference antennas (176 by default)
    NantRefStarshape = SimulatedShower.InitialShape
    NantRefLayout = SimulatedShower.NantTraces
    NantRefCrossCheck = NantRefLayout - NantRefStarshape
    NantTargetLayout = TargetShower.nant - TargetShower.InitialShape   
    filtering = TargetShower.filter
   
    if filtering:
        time_sample = int(len(Traces[:,0]))
        Traces = filter_traces(Traces, NantRefLayout, time_sample)
    
    # Traces of the "true" reference shower
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
     
    refEx, refEy, refEz = np.array(refEx),  np.array(refEy),  np.array(refEz) 

    return refEx, refEy, refEz

def LoadRMscaledTrace(efield_interpolated, error_peak,rm_peak, zh_peak, RMtime, RMx, RMy, RMz, index, diff_time_all, Load, i):

    # We check if a solution is found for the RM traces at the antenna "i"
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
        # If there is q solution we load the RM traces 
    if(Load):
        targetTime, targetEx, targetEy, targetEz = \
        efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
        efield_interpolated[i][:,2], efield_interpolated[i][:,3]
        
        RMtime.append(targetTime)
        RMx.append(targetEx)
        RMy.append(targetEy)
        RMz.append(targetEz)
        index.append(i)
        #t, targetEx, targetEy, targetEz = np.loadtxt(\
        #"./OutputDirectory/DesiredTraces_%d.txt"%i, unpack = True)