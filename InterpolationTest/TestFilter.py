#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:56:15 2024

@author: chiche
"""


# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:39:15 2020

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from module_signal_process import filter_traces, filter_single_trace
import scipy
from scipy.signal import hilbert
import sys
from scipy.fftpack import rfft, fft
import subprocess

def test_interpolation(SimulatedShower, TargetShower, efield_interpolated,\
                       IndexAll, Display = True):
    
    
# =============================================================================
#                            Initialization
# =============================================================================

    Traces = SimulatedShower.traces
    NantRefStarshape = SimulatedShower.InitialShape
    NantRefLayout = SimulatedShower.NantTraces
    NantRefCrossCheck = NantRefLayout - NantRefStarshape
    NantTargetLayout = TargetShower.nant - TargetShower.InitialShape   
    filtering = TargetShower.filter

    if filtering:
        time_sample = int(len(Traces[:,0]))
        Traces = filter_traces(Traces, NantRefLayout, time_sample)
    
# =============================================================================
#                  Extracting the reference Traces
# =============================================================================

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
            
            RMtime.append(targetTime)
            RMx.append(targetEx)
            RMy.append(targetEy)
            RMz.append(targetEz)
            index.append(i)
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
            
            refEtot =  refEy#np.sqrt(refEx**2 + refEy**2 + refEz**2)
            targetEx = targetEx.astype(float) #TODO: change in Lyon
            targetEy = targetEy.astype(float) #TODO: change in Lyon
            targetEz = targetEz.astype(float) #TODO: change in Lyon

            targetEtot = targetEy#np.sqrt(targetEx**2 + targetEy**2 + targetEz**2)
            
            # We shift the Traces time window
            refTime[i] = refTime[i] - refTime[i][0]
            targetTime = targetTime -targetTime[0]
            refTimeArray = 0.5*np.arange(0, len(refEtot[i]), 1)
            targetTimeArray = 0.5*np.arange(0, len(targetEtot), 1)
            
            post_filtering = False
            if post_filtering: refEtot[i] = \
            filter_single_trace(refTimeArray, refEtot[i], 1, len(refEtot[i]))
            if post_filtering: targetEtot = \
            filter_single_trace(targetTimeArray, targetEtot, 1, len(targetEtot))
        

            #if(max(targetEtot)>1e4):
                
            #print(max(targetEtot), max(refEtot[i]))
            

            
            Ndisplay = 40
            start = 80
            # Third option only if one plane is tested
            if((Display) & (i>start) & (i<(start + Ndisplay))):# & (np.min(abs(IndexAll-i)) == 0) ):
                
                plt.plot(refTimeArray, abs(hilbert(refEy[i])), label = "simulation") 
                plt.plot(targetTimeArray, abs(hilbert(targetEy)), label = "interpolation")
                plt.xlabel("Time [ns]")
                plt.ylabel("E [$\mu V/m$]")
                plt.legend()
                kmax = np.argmax(targetEtot)
                tmax = targetTimeArray[kmax]
                plt.xlim(tmax-150, tmax+150)
                plt.tight_layout()
                #plt.savefig\
                #("./InterpolationTest_Etot_antenna_scale_int3D%.d.pdf" %i)
                plt.show()
                print(i)
                
                Fourier = True
                if(Fourier == True):
                    if(max(refEy[i])>0):
                        #refEtot[i] = filter_single_trace(refTimeArray, refEtot[i], 1, len(refEtot[i]))
                        TF = np.fft.fft(refEy[i])
                        T = 0.5/1e9
                        N = len(refEy[i])
                        xf = np.fft.fftfreq(N, T)
                        #xf  = np.linspace(0.0, 1.0/(2.0*T), int(N/2.0))/1e6
                        plt.plot(xf[:N // 2]/1e6, 1.0/N * np.abs(TF)[:N // 2])
                        plt.xlabel("frequency [MHz]")
                        plt.ylabel("TF")
                        plt.xlim(0,400)
                        
                        #print(np.shape(targetTimeArray))
                        #targetEtot = filter_single_trace(targetTimeArray, targetEtot, 1, len(targetEtot))
                        #sys.exit()
                        TF2 = fft(targetEy)
                        T = 0.5/1e9
                        N = len(targetEy)
                        xf = np.fft.fftfreq(N, T)
                        #xf  = np.linspace(0.0, 1.0/(2.0*T), int(N/2.0))/1e6
                        plt.plot(xf[:N // 2]/1e6, 1.0/N * np.abs(TF2)[:N // 2])
                        plt.xlabel("frequency [MHz]")
                        plt.ylabel("TF")
                        plt.xlim(0,400)
                        plt.show()
                
            PeakrefEtot = max(abs(hilbert(refEy[i])))#max(refEtot[i])
            PeaktargetEtot = max(abs(hilbert(targetEy)))#max(targetEtot)
            #print(PeaktargetEtot)
        
            # condition when we test only one plane so not all the antennas
            # this conditon checks wheter i is contained in IndexAll, i.e., 
            # if the efield for the i-antenna was computed.
            TestOnePlane = False
            if(TestOnePlane):
                if(np.min(abs(IndexAll-i)) == 0):
                    #print(i)
                    error_peak.append((PeakrefEtot- PeaktargetEtot)/PeakrefEtot)
                    #print((PeakrefEtot- PeaktargetEtot)/PeakrefEtot)
                    rm_peak.append(PeaktargetEtot)
                    zh_peak.append(PeakrefEtot)                  
            else:
                if(PeakrefEtot>0):
                        #print(i)
                        error_peak.append((PeakrefEtot- PeaktargetEtot)/PeakrefEtot)
                        #print((PeakrefEtot- PeaktargetEtot)/PeakrefEtot)
                        rm_peak.append(PeaktargetEtot)
                        zh_peak.append(PeakrefEtot)   
                else:               
                    rm_peak.append(np.nan)
                    zh_peak.append(np.nan)
                    error_peak.append(np.nan)
    
    path = "/Users/chiche/Desktop/RadioMorphingUptoDate/RMFilterTests/Traces/"
    SaveDir = "E4_th63_phi0_0"
    cmd = "mkdir -p " + path + SaveDir
    p =subprocess.Popen(cmd, cwd=os.getcwd(), shell=True)
    stdout, stderr = p.communicate()

    SaveTraces = True 
    if(SaveTraces):
        np.savetxt(path + SaveDir + "/ZHStime.txt", refTimeArray)
        np.savetxt(path + SaveDir + "/ZHSx.txt", refEx)
        np.savetxt(path + SaveDir + "/ZHSy.txt", refEy)
        np.savetxt(path + SaveDir + "/ZHSz.txt", refEz)
        
        np.savetxt(path + SaveDir + "/RMtime.txt", np.array(RMtime))
        np.savetxt(path + SaveDir + "/RMx.txt", np.array(RMx))
        np.savetxt(path + SaveDir + "/RMy.txt", np.array(RMy))
        np.savetxt(path + SaveDir + "/RMz.txt", np.array(RMz))
        np.savetxt(path + SaveDir + "/RMindex.txt", np.array(index))
                
    return np.array(error_peak), np.array(rm_peak), np.array(zh_peak)