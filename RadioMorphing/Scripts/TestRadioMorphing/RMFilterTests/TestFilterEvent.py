#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:35:15 2024

@author: chiche
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import module_signal_process
import importlib
importlib.reload(module_signal_process)
from module_signal_process import filter_traces, filter_single_trace
from ModuleFilter import LoadData, CorrectPadding, ApplyFilter, MaxHilbert, GetRelError, CompareTraces, CompareTF, PlotError, CompareGivenTrace, PlotErrorVsAmplitude, GetErrvsTrigg, PlotErrvsTrigg, Plot_rms_ErrvsTrig, PlotHistErr, CorrectPaddingantenna, calculate_rms_correlation
from scipy.signal import hilbert
from scipy.signal import correlate

### Loading the data
MainDir = "RMresults_09_08_24" 
SaveDir = "E1_th75_phi0_4" #"E1_th75_phi0_4"   #Stshp_Iron_3.98_85.8_90.0_4
Filter = False
TriggerThreshold =60
fmin = 50e6
fmax = 200e6
if(TriggerThreshold == 0): triggLabel = "noTrigg"
if(TriggerThreshold == 60): triggLabel = "3sigma"
if(TriggerThreshold == 110): triggLabel = "5sigma"
if(fmin == 30e6): flabel = "30_80"
if(fmin == 50e6): flabel = "50_200"
if(not(Filter)): flabel = "full_band"
method = "peak"

band = flabel + "_" + triggLabel + "_" + method

SavePath = "/Users/chiche/Desktop/" + MainDir + "/" + SaveDir + "_" + band
path = "./DataRMFilterTests/Traces/"
ZHStime, ZHSx, ZHSy, ZHSz, RMtime, RMx, RMy, RMz, index, Nant =  LoadData(SaveDir, path)

### Padding correction
Nmax = int(max(index))
Padding = False
if(Padding):
    RMx = CorrectPadding(ZHSx, RMx, Nant, index)
    RMy = CorrectPadding(ZHSy, RMy, Nant, index)
    RMz = CorrectPadding(ZHSz, RMz, Nant, index)
    RMtime = ZHStime


### INITIALISATION
dt = 0.5/1e9
k=0
Nplot = 0
Ndisplay = 176

RMpeak, ZHSpeak, ZHSfilteredpeak, error = \
np.zeros(len(RMx)), np.zeros(len(RMx)), np.zeros(len(RMx)), np.zeros(len(RMx))

error_correlation = []
# LOOP OVER THE ANTENNAS
### Si jamais dans index la dernier antenne (176) n'a pas pu etre  morphee alors une iteration en k en trop est ajoutee, dans l'ideal il faudrait modifier le code pour prendre en compte ce cas, je le corrige ici de maniere grossiere en imposant
### for i in range(Nmax) au lieu de for i in range(Nant), où Nmax = max(index) idem pour correct padding
Nmax = int(max(index))
for i in range(Nmax):
    # We skip the antennas for which the RM failed
    if(i == index[k]):
        ### FILTERING THE TRACES
        # Even if we dont filter we still set a trigger condition on the filtered ZHS signals
        ZHS_filtered_x, ZHS_filtered_y, ZHS_filtered_z = np.copy(ZHSx), np.copy(ZHSy), np.copy(ZHSz)
        ZHS_filtered_x[i,:], ZHS_filtered_y[i,:], ZHS_filtered_z[i,:] =  ApplyFilter(ZHStime, ZHS_filtered_x[i,:], ZHS_filtered_y[i,:], ZHS_filtered_z[i,:], i, 30e6, 80e6)
       
        # We take the channel with the highest amplitude to compare with RM
        abs_arrays = [np.abs(ZHS_filtered_x[i,:]), np.abs(ZHS_filtered_y[i,:]), np.abs(ZHS_filtered_z[i,:])]
        max_index = np.argmax([np.max(arr) for arr in abs_arrays])
        ZHSmain  = [ZHS_filtered_x, ZHS_filtered_y, ZHS_filtered_z][max_index]
        #print(np.diff(ZHSmain, ZHS_filtered_x)),
        if(Filter):
            #ZHS traces 
            ZHSx[i,:], ZHSy[i,:], ZHSz[i,:] = ApplyFilter(ZHStime, ZHSx[i,:], ZHSy[i,:], ZHSz[i,:], i, fmin, fmax)
            ZHSmain_true = [ZHSx, ZHSy, ZHSz][max_index]
            # Scaled RM traces
            if(Padding):
                RMx[k,:], RMy[k,:], RMz[k,:] = ApplyFilter(RMtime, RMx[k,:], RMy[k,:], RMz[k,:], k, fmin, fmax)
            else:
                #print(RMtime)
                #RMx[k,:], RMy[k,:], RMz[k,:] = ApplyFilter(np.array(RMtime[k,:]), RMx[k,:], RMy[k,:], RMz[k,:], k, fmin, fmax)
                RMx[k,:], RMy[k,:], RMz[k,:] = ApplyFilter(RMtime[k,:], RMx[k,:], RMy[k,:], RMz[k,:], k, fmin, fmax)
                RMmain = [RMx, RMy, RMz][max_index]
        else:
            ZHSmain_true = [ZHSx, ZHSy, ZHSz][max_index]
            RMmain = [RMx, RMy, RMz][max_index]
                                                                                                               
            
        ### COMPUTATION OF THE ERROR
        if(method == "peak"):  RMpeak[k] = MaxHilbert(RMy[k,:])
        if(method == "sum"):  RMpeak[k] =np.sum(abs(hilbert(RMx[k,:])))
        if(method == "peak"): ZHSpeak[k]= MaxHilbert(ZHSy[i,:]) 
        if(method == "sum"): ZHSpeak[k]=  np.sum(abs(hilbert(ZHSx[i,:])))
        ZHSfilteredpeak[k] = MaxHilbert(ZHSmain[i,:])
        error[k] =  (ZHSpeak[k] - RMpeak[k])/ZHSpeak[k]

        
        # EVALUATING AND PLOTTING THE RESULTS
        if((Nplot<Ndisplay) & (max(abs(ZHSmain[i,:]))>TriggerThreshold) & (k>0)):
            print(i)
            ### Trace comparison
            #print(k)
            PlotTraces = False
            if(PlotTraces): bindiff = CompareTraces(RMy[k,:],ZHSy[i,:])

            #### CORRELATION #####
            signal1 =  RMmain[k,:]
            signal2= ZHSmain_true[i,:]
            print(i)
            rms_diff = calculate_rms_correlation(signal1, signal2)
            error_correlation.append(rms_diff)
            print("peak error", error[k]*100)
            print("RMS correlation error:", rms_diff*100)
            if(rms_diff*100>30):
                print(i, "LARGE ERROR")
            
            ### Fourier transform
            TF = False
            if(TF):
                CompareTF(ZHSy[i,:], RMy[k,:], dt)
            
            Nplot = Nplot + 1
        #print(i,k)
        k = k +1
        

# PLOTTING ERROR RESULTS
# Plot given trace
ant_id = 6
CompareGivenTrace(ant_id, index, ZHSx, RMx)

# Error
PlotError(error, ZHSfilteredpeak, TriggerThreshold, SavePath)

# Error vs amplitude
PlotErrorVsAmplitude(1, error, ZHSpeak, SavePath)

# Error vs trigger threshold
#trigger, ErrVsTrigg, STDvsTrigg = GetErrvsTrigg(500, ZHSfilteredpeak,error)

# RMS Error vs trigger threshold
#PlotErrvsTrigg(trigger, ErrVsTrigg) 
#Plot_rms_ErrvsTrig(trigger, STDvsTrigg)


PlotHistErr(error,ZHSfilteredpeak, TriggerThreshold, SavePath)


error_correlation = np.array(error_correlation)
plt.hist(error_correlation, bins=50, density=True)
print(max(abs(error_correlation))*100, " % max error")
np.std(error_correlation)