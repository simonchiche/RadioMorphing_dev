#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:35:15 2024

@author: chiche
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from module_signal_process import filter_traces, filter_single_trace
from scipy.signal import hilbert

SaveDir = "E4_th63_phi0_0"
path = "/Users/chiche/Desktop/RadioMorphingUptoDate/RMFilterTests/Traces/"
ZHStime = np.loadtxt(path + SaveDir + "/ZHStime.txt")
ZHSx = np.loadtxt(path + SaveDir + "/ZHSx.txt")
ZHSy = np.loadtxt(path + SaveDir + "/ZHSy.txt")
ZHSz = np.loadtxt(path + SaveDir + "/ZHSz.txt")

RMtime = np.loadtxt(path + SaveDir + "/RMtime.txt")
RMx = np.loadtxt(path + SaveDir + "/RMx.txt")
RMy = np.loadtxt(path + SaveDir + "/RMy.txt")
RMz = np.loadtxt(path + SaveDir + "/RMz.txt")
index = np.loadtxt(path + SaveDir + "/RMindex.txt")


def CorrectPadding(E1, E2, Nant, index):
    
    k= 0
    RME = []
    for i in range(Nant):
         if(i == index[k]):
             RME.append(np.pad(E2[k,:], (0, len(E1[i]) - len(E2[k,:])), 'constant'))
             
             k = k +1
    return np.array(RME)

Nant = len(ZHSx)
dt = 0.5/1e9
k=0
Nplot = 0
Ndisplay = 120
Filter = True

Padding = False
if(Padding):
    RMx = CorrectPadding(ZHSx, RMx, Nant, index)
    RMy = CorrectPadding(ZHSy, RMy, Nant, index)
    RMz = CorrectPadding(ZHSz, RMz, Nant, index)
    
    RMtime = ZHStime

RMpeak, ZHSpeak, error = \
np.zeros(len(RMx)), np.zeros(len(RMx)), np.zeros(len(RMx))

for i in range(Nant):
    
    if(i == index[k]):
        
        if(Filter):
            # ZHS
            time_sample = int(len(ZHStime))
            ZHSTrace = np.transpose\
            (np.array([ZHStime, ZHSx[i,:], ZHSy[i,:], ZHSz[i,:]]))
            ZHSfiltered = \
            filter_traces(ZHSTrace, 1, time_sample)
                
            ZHSx[i,:], ZHSy[i,:], ZHSz[i,:] = ZHSfiltered [:,1], \
            ZHSfiltered [:,2], ZHSfiltered [:,3]
            
            # RM
            time_sample = len(RMx[0,:])
            if(Padding):
                RMTrace = np.transpose\
                (np.array([RMtime, RMx[k,:], RMy[k,:], RMz[k,:]]))
            else:
                RMTrace = np.transpose\
                (np.array([RMtime[k,:], RMx[k,:], RMy[k,:], RMz[k,:]]))
                
            RMfiltered = \
            filter_traces(RMTrace, 1, time_sample)
            
            RMx[k,:], RMy[k,:], RMz[k,:] = RMfiltered [:,1], \
            RMfiltered [:,2], RMfiltered [:,3]
            
            
        ### Computation of the error   
        RMpeak[k] = max(abs(hilbert(RMy[k,:])))
        ZHSpeak[k]=  max(abs(hilbert(ZHSy[i,:])))
        error[k] =  (ZHSpeak[k] - RMpeak[k])/ZHSpeak[k]
        
        #
        if((Nplot<Ndisplay) & (max(abs(ZHSy[i,:]))>0) & (k>100)):
            print(k)
            plt.plot(abs(hilbert(ZHSy[i,:])))
            plt.plot(abs(hilbert(RMy[k,:])))
            plt.show()
            
            TF = False
            if(TF):
                TFzhs = np.fft.fft(ZHSy[i,:])
                Nzhs = len(ZHSy[i,:])
                xf_zhs = np.fft.fftfreq(Nzhs, dt)
                 
                TFrm = np.fft.fft(RMy[k,:])
                Nrm = len(RMy[k,:])
                xf_rm = np.fft.fftfreq(Nrm, dt)
                
                plt.plot(xf_zhs[:Nzhs // 2]/1e6, 1.0/Nzhs * np.abs(TFzhs)[:Nzhs // 2])
                plt.plot(xf_rm[:Nrm // 2]/1e6, 1.0/Nrm * np.abs(TFrm)[:Nrm // 2])
                plt.ylabel("TF")
                plt.xlim(0,400)
                plt.show()
            
            Nplot = Nplot + 1
        k = k +1


# Plot error
plt.plot(error)
plt.show()
print(np.std(error))


l = 6
l2 = int(index[l])
plt.plot(abs(hilbert(ZHSy[l,:])))
plt.plot(abs(hilbert(RMy[l2,:])))
plt.xlim(300,800)
plt.show()


plt.plot(abs(error)*1e3, label = r"error $\times$ 10000")
plt.plot(ZHSpeak, label = "ZHS peak [50-200 MHz]")
#plt.yscale("log")
plt.xlabel("Antenna ID")
plt.legend()
plt.savefig("/Users/chiche/Desktop/Peak_vs_error_lin.pdf")
plt.show()


trigger = np.linspace(0,500,51)
max_error  = np.zeros(len(trigger))
std = np.zeros(len(trigger))

for i in range(len(trigger)):
    
    max_error[i] = max(abs(error[ZHSpeak>trigger[i]]))
    std[i] = np.std(error[ZHSpeak>trigger[i]])
    
    
plt.scatter(trigger, max_error, label ="50-200 MHz")
plt.xlabel("Threshold [$\mu V/m$]")
plt.ylabel("Maximum relative error")
plt.savefig("/Users/chiche/Desktop/MaxRelErr.pdf")
plt.show()


plt.scatter(trigger, std)
plt.xlabel("Threshold [$\mu V/m$]")
plt.ylabel("RMS relative error")
plt.savefig("/Users/chiche/Desktop/RMSrelErr.pdf")
plt.show()










