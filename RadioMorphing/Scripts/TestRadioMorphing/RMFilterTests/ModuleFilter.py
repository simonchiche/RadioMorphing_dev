import numpy as np

from module_signal_process import filter_traces
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os

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
    RME = []
    for i in range(Nant):
         if(i == index[k]):
             RME.append(np.pad(E2[k,:], (0, len(E1[i]) - len(E2[k,:])), 'constant'))
             
             k = k +1
    return np.array(RME)


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
    print("bindiff:", np.argmax(abs(hilbert(Erm)))- np.argmax(abs(hilbert(Ezhs))))
    plt.show()

    return

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