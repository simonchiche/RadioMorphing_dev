import numpy as np

from module_signal_process import filter_traces
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import correlate


def LoadData(SaveDir, path):

    ZHStime = np.loadtxt(path + SaveDir + "/ZHStime.txt")
    ZHSx = np.loadtxt(path + SaveDir + "/ZHSx.txt")
    ZHSy = np.loadtxt(path + SaveDir + "/ZHSy.txt")
    ZHSz = np.loadtxt(path + SaveDir + "/ZHSz.txt")

    RMtime = np.loadtxt(path + SaveDir + "/RMtime.txt")
    RMx = np.loadtxt(path + SaveDir + "/RMx.txt")
    RMy = np.loadtxt(path + SaveDir + "/RMy.txt")
    RMz = np.loadtxt(path + SaveDir + "/RMz.txt")
    index = np.loadtxt(path + SaveDir + "/RMindex.txt")

    Nant = len(ZHSx)

    return ZHStime, ZHSx, ZHSy, ZHSz, RMtime, RMx, RMy, RMz, index, Nant

def CorrectPadding(E1, E2, Nant, index):
    
    k= 0
    #plt.plot(E1)
    #plt.plot(E2)
    #plt.show()
    RME = []
    for i in range(Nant):
         if(i == index[k]):
             RME.append(np.pad(E2[k,:], (0, len(E1[i]) - len(E2[k,:])), 'constant'))
             
             k = k +1
    return np.array(RME)


def CorrectPaddingantenna(E1, E2):
    len1, len2 = len(E1), len(E2)
    max_len = max(len1, len2)

    if len1 < max_len:
        E1 = np.pad(E1, (0, max_len - len1), mode='constant', constant_values=0)
    if len2 < max_len:
        E2 = np.pad(E2, (0, max_len - len2), mode='constant', constant_values=0)

    return E1, E2



def ApplyFilter(time, Ex, Ey, Ez, i, fmin, fmax):

    #fmin, fmax to be given in Hz (usually 1e6 factor)

    time_sample = int(len(time))
    Trace = np.transpose\
            (np.array([time, Ex, Ey, Ez]))
    Tracefiltered = \
            filter_traces(Trace, 1, time_sample, fmin, fmax)
            
    Ex, Ey, Ez = Tracefiltered[:,1], \
            Tracefiltered[:,2], Tracefiltered[:,3]
    
    return Ex, Ey, Ez

def MaxHilbert(E):

    return max(abs(hilbert(E)))


def GetRelError(Erm, Ezhs):

    Erm_peak = MaxHilbert(Erm)
    Ezhs_peak = MaxHilbert(Ezhs)
    error = (Ezhs_peak - Erm_peak)/Ezhs_peak

    return Erm_peak, Ezhs_peak, error

def CompareTraces(Erm, Ezhs):


    #plt.plot(abs(hilbert(Ezhs)))
    #plt.plot(abs(hilbert(Erm)))
    plt.plot(Ezhs)
    plt.plot(Erm)
    bindiff = np.argmax(abs(hilbert(Erm))) - np.argmax(abs(hilbert(Ezhs)))
    #print("bindiff:", np.argmax(abs(hilbert(Erm)))- np.argmax(abs(hilbert(Ezhs))))
    plt.xlim(300, 800)
    plt.show()

    return int(bindiff)

def CompareTF(Ezhs, Erm,  dt):

    TFzhs = np.fft.fft(Ezhs)
    Nzhs = len(Ezhs)
    xf_zhs = np.fft.fftfreq(Nzhs, dt)
        
    TFrm = np.fft.fft(Erm)
    Nrm = len(Erm)
    xf_rm = np.fft.fftfreq(Nrm, dt)
    
    plt.plot(xf_zhs[:Nzhs // 2]/1e6, 1.0/Nzhs * np.abs(TFzhs)[:Nzhs // 2])
    plt.plot(xf_rm[:Nrm // 2]/1e6, 1.0/Nrm * np.abs(TFrm)[:Nrm // 2])
    plt.ylabel("TF")
    plt.xlim(0,400)
    plt.show()

    return

def PlotError(error, E, TriggerThreshold, savepath):

    os.makedirs(savepath, exist_ok=True)

    plt.plot(error[E>TriggerThreshold])
    plt.ylabel("Relative Error")
    plt.xlabel("Antenna ID")
    err_rms = np.std(error[E>TriggerThreshold])
    print(err_rms)
    plt.title("RMS = %.3f, Trigger = %.d $\mu V/m$" %(err_rms, TriggerThreshold))
    plt.savefig(savepath + "/RelErr.pdf", bbox_inches = "tight")
    plt.show()

    return

def CompareGivenTrace(index_zhs, index, Ex_zhs, Ex_rm):

    index_rm = int(index[index_zhs])
    plt.plot(abs(hilbert(Ex_zhs[index_zhs,:])))
    plt.plot(abs(hilbert(Ex_rm[index_rm,:])))
    plt.xlim(300,800)
    plt.show()

    return


def PlotErrorVsAmplitude(scale, error, Epeak, savepath):

    plt.plot(abs(error)*scale, label = r"error")# $\times$ 1000")
    plt.plot(Epeak, label = "ZHS peak full band")# [50-200 MHz]")
    #plt.yscale("log")
    plt.xlabel("Antenna ID")
    plt.legend()
    plt.ylabel("RelError / Peak-x")
    plt.yscale("log")
    plt.savefig(savepath + "/RelErrVsAmplitude.pdf", bbox_inches = "tight")
    #plt.savefig("/Users/chiche/Desktop/Peak_vs_error_lin.pdf")
    plt.show()

    return

def GetErrvsTrigg(HighThreshold, Epeak, error):

    LowThreshold  = 0 
    bins = 51
    trigger = np.linspace(LowThreshold, HighThreshold,bins)
    ErrVsTrigg  = np.zeros(len(trigger))
    STDvsTrigg = np.zeros(len(trigger))

    for i in range(len(trigger)):
        
        ErrVsTrigg[i] = max(abs(error[Epeak>trigger[i]]))
        STDvsTrigg[i] = np.std(error[Epeak>trigger[i]])

    return trigger, ErrVsTrigg, STDvsTrigg


def PlotErrvsTrigg(trigger, ErrVsTrigg):

    plt.scatter(trigger, ErrVsTrigg, label ="50-200 MHz")
    plt.xlabel("Threshold [$\mu V/m$]")
    plt.ylabel("Maximum relative error")
    #plt.savefig("/Users/chiche/Desktop/MaxRelErr.pdf")
    plt.show()

    return

def Plot_rms_ErrvsTrig(trigger, STDvsTrigg):
    plt.scatter(trigger, STDvsTrigg)
    plt.xlabel("Threshold [$\mu V/m$]")
    plt.ylabel("RMS relative error")
    #plt.savefig("/Users/chiche/Desktop/RMSrelErr.pdf")
    plt.show()

    return


def PlotHistErr(error, Epeak, threshold, savepath):
    
    error = error[Epeak>threshold]
    plt.hist(abs(error), bins =100, edgecolor = "black", color="skyblue", linewidth =1)
    plt.xlabel("realtive error")
    plt.ylabel("Nant")
    plt.savefig(savepath + "/HistErr.pdf", bbox_inches = "tight")
    plt.show()

    return

def calculate_rms_correlation(signal1, signal2):

    plt.plot(signal1, label='Signal 1', color='blue')
    plt.plot(signal2, label='Signal 2', color='red')
    plt.legend()
    plt.show()
    print("max args")
    print(np.argmax(abs(hilbert(signal1))), np.argmax(abs(hilbert(signal2))))
    arg1 = np.argmax(abs(hilbert(signal1)))
    arg2 = np.argmax(abs(hilbert(signal2)))

    arg1 = np.argmax(abs(signal1))
    arg2 = np.argmax(abs(signal2))
    if(np.sign(signal1[arg1])*np.sign(signal2[arg2])>0):
        diff = arg1 - arg2
        print(signal1[arg1], signal2[arg2])
        print("SAME SIGNAL")
    else:
        if(np.sign(signal2[arg2])>0):
            arg1 = np.argmax(signal1)
        else:
            arg1 = np.argmin(signal1)
        
        diff = arg1 - arg2
        #print(signal1[arg1], signal2[arg2])
        #print(min(signal2))
        #arg2 = np.argmax(abs(hilbert(signal2)))
        #print(signal2[arg2])
        #print("INVERTED SIGNAL")

    #plt.plot(signal1[diff:], label='Signal 1', color='blue')
    #plt.plot(signal2, label='Signal 2', color='red')
    #plt.legend()
    #plt.show()
    
    x1 = np.linspace(0, len(signal1[diff:])-1, len(signal1[diff:]))
    x2 = np.linspace(0, len(signal2)-1, len(signal2))
    #plt.scatter(x1, abs(hilbert(signal1[diff:])), label='Signal 1', color='blue')
    #plt.scatter(x2, abs(hilbert(signal2)), label='Signal 2', color='red')
    #plt.legend()
    #plt.xlim(250,500)
    #plt.show()
    

    Padding = False
    if(Padding):
        signal1 = CorrectPaddingantenna(signal1, signal2)

    corr = correlate(signal1, signal2, mode="full")
    lags = np.arange(-len(signal1) + 1, len(signal2))
    best_lag = lags[np.argmax(corr)]
    print("Décalage optimal:", best_lag)
    bindiff = np.argmax(abs(hilbert(signal1))) - np.argmax(abs(hilbert(signal2)))
    best_lag=  diff #bindiff
    
    if best_lag > 0:
        aligned1 = signal1[best_lag:]
        aligned2 = signal2[:len(aligned1)]
    else:
        aligned1 = signal1[:len(signal2)+best_lag]
        aligned2 = signal2[-best_lag:len(signal2)]
    
    #CompareTraces(aligned1,aligned2)
    min_len = min(len(aligned1), len(aligned2))
    a1 = aligned1[:min_len]
    a2 = aligned2[:min_len]
    x1 = np.linspace(0, len(a1)-1, len(a1))
    x2 = np.linspace(0, len(a2)-1, len(a2))
    plt.scatter(x1, a1, label='Signal 1', color='blue', s=10)  
    plt.scatter(x2, a2, label='Signal 2', color='red', s=10)
    maxid = np.argmax(abs(hilbert(a1)))
    plt.xlim(maxid-100, maxid+100)
    plt.show()
    print(np.argmax(abs(hilbert(a1))), np.argmax(abs(hilbert(a2))))

    mask = np.abs(a2) > 0.2 * np.max(np.abs(a2))  # garder seulement les points significatifs
    a1 = a1[mask]
    a2 = a2[mask]

    #rms_diff = np.sqrt(np.mean(((a2 - a1) / a2)**2))
    #rms_diff = np.sqrt(np.mean((a1 - a2)**2))/max(abs(a2))
    
    #sign = np.sign(np.max(np.abs(a1)) - np.max(np.abs(a2)))
    #signed_rms_diff = sign * rms_diff
    signed_rms_diff = np.mean(((a2 - a1) / a2))

    #if(signed_rms_diff*100>8):
        #print("LARGE ERROR")
        #print(len(a1), len(a2))
        #plt.scatter(x1[mask], a1, label='Signal 1', color='blue', s=10)  
        #plt.scatter(x2[mask], a2, label='Signal 2', color='red', s=10)
        #maxid = np.argmax(abs(hilbert(a1)))
        #plt.xlim(maxid-100, maxid+100)
        #plt.show()


    return signed_rms_diff
