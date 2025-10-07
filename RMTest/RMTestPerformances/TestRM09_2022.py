#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:48:51 2022

@author: chiche
"""


import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from ModuleTestRM import get_ZenithCut, AverageOnZenith, \
AverageOnZenithOneAzimuth,AverageOnOmega, AverageOnRefPeak, AverageOnDplane,\
AverageOnDplaneRel, AverageOnOmegaZenithCut
from PlotTestRM import PlotMeanRMS, PlotMeanRMSPlaneDistance


font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)


# =============================================================================
#                             Initialisation
# =============================================================================

#files = glob.glob("./RMTestResults/*.txt")
NFileGroup =  44

PeakResidualAllAntennas = []
TargetZenithAllAntennas = []
RefZenithAllAntennas = []
RefAzimuthAllAntennas = []
TargetAzimuthAllAntennas = []
TargetEnergyAllAntennas = []
OmegaAllAntennas = []
RefPeakAllAntennas = []
TargetPeakAllAntennas = []
MeanPeakResidualAll = []
RMSPeakResidualAll = []
TargetZenithAllSim = []
DplaneRefAllAntennas = []
DplaneTargetAllAntennas = []
RMSResidualAllSim = []
TallSim = []
# =============================================================================
#                                 Loading
# =============================================================================
#path ="./RMResultsAllPlanesAllThetaBiasCorrected06_03_23/"
#path ="./RMResultsAllPlanesAllTheta01_03_23/"
##path ="./RMresultsManuscript/"
#path ="./RMResults_03_2024/"
#path ="./RMResults_03_2024_with_bias_correction_2/"
path ="./Data/RMResults_03_2024_without_87.2_ref/"



#path = "./NewMatiasSim_theta68_25_10/"
# Loop over all jobs
for i in range(NFileGroup):

    start = i*100
    PeakRef =  np.loadtxt(path + "PeakRef%.2d.txt" %start)
    PeakTarget =  np.loadtxt(path + "PeakTarget%.2d.txt" %start)
    PeakResidual =  np.loadtxt(path + "PeakResidual%.2d.txt" %start)
    Parameters = np.loadtxt(path + "RefTargetParameters%.2d.txt" %start)
    Omega = np.loadtxt(path + "OmegaAngle%.2d.txt" %start)
    Tsim = np.loadtxt(path + "Tsim%.2d.txt" %start)
    
    # Loop over all sims per jobs
    for j in range(len(PeakRef[:,0])):

        TargetZenithAllSim.append(180 - Parameters[j,1])
        TallSim.append(Tsim[j])
        # Loop over all antennas per sim
        Nant = len(PeakResidual[0,:])
        for k in range(Nant):
            PeakResidualAllAntennas.append(PeakResidual[j,k])
            TargetZenithAllAntennas.append(180 - Parameters[j,1])
            RefZenithAllAntennas.append(180 - Parameters[j,0])
            TargetAzimuthAllAntennas.append(Parameters[j,5])
            TargetEnergyAllAntennas.append(Parameters[j,3])
            RefAzimuthAllAntennas.append(Parameters[j,4])
            OmegaAllAntennas.append(Omega[j,k])
            RefPeakAllAntennas.append(PeakRef[i,j])
            DplaneRefAllAntennas.append(Parameters[j,6])
            DplaneTargetAllAntennas.append(Parameters[j,7])
        RMSResidualAllSim.append(np.std(PeakResidual[j,:][~np.isnan(PeakResidual[j,:])]))
        if(np.std(PeakResidual[j,:][~np.isnan(PeakResidual[j,:])])>0.4):
            print(i, j, Parameters[j,3], 180 - Parameters[j,1], Parameters[j,5],)
#sys.exit()
    
TallSim = np.array(TallSim)    
TargetZenithAllSim = np.array(TargetZenithAllSim)
TargetZenithCut = get_ZenithCut(TargetZenithAllSim)       
TargetZenithAllAntennas = np.array(TargetZenithAllAntennas)
RefZenithAllAntennas = np.array(RefZenithAllAntennas)
RefAzimuthAllAntennas = np.array(RefAzimuthAllAntennas)
TargetAzimuthAllAntennas = np.array(TargetAzimuthAllAntennas)
TargetEnergyAllAntennas = np.array(TargetEnergyAllAntennas)
PeakResidualAllAntennas = np.array(PeakResidualAllAntennas)
OmegaAllAntennas = np.array(OmegaAllAntennas)
RefPeakAllAntennas = np.array(RefPeakAllAntennas)
DplaneRefAllAntennas = np.array(DplaneRefAllAntennas)
DplaneTargetAllAntennas = np.array(DplaneTargetAllAntennas)


Trigger = True#False
plt.scatter(TargetZenithAllAntennas, RefPeakAllAntennas)
plt.xscale("log")
plt.show()
'''
for i in range(len(zenith_sort_cut)): 
    plt.hist(RefPeakAllAntennas[TargetZenithAllAntennas == zenith_sort_cut[i]], density = True)
    plt.yscale("log")
    plt.show()
'''
#plt.scatter(RefPeakAllAntennas[TargetZenithAllAntennas == TargetZenithCut[0]], PeakResidualAllAntennas[TargetZenithAllAntennas == TargetZenithCut[0])

plt.scatter(OmegaAllAntennas, RefPeakAllAntennas)
plt.show()
#sys.exit()
if(Trigger):
    TriggerValue = 60
    OldPeak = PeakResidualAllAntennas
    TargetZenithAllAntennas = TargetZenithAllAntennas[RefPeakAllAntennas>TriggerValue]
    RefZenithAllAntennas = RefZenithAllAntennas[RefPeakAllAntennas>TriggerValue]
    TargetAzimuthAllAntennas = TargetAzimuthAllAntennas[RefPeakAllAntennas>TriggerValue]
    TargetEnergyAllAntennas = TargetEnergyAllAntennas[RefPeakAllAntennas>TriggerValue]
    PeakResidualAllAntennas = PeakResidualAllAntennas[RefPeakAllAntennas>TriggerValue]
    print(len(PeakResidualAllAntennas)/len(OldPeak))
    #sys.exit()
    OmegaAllAntennas = OmegaAllAntennas[RefPeakAllAntennas>TriggerValue]
    DplaneRefAllAntennas = DplaneRefAllAntennas[RefPeakAllAntennas>TriggerValue]
    DplaneTargetAllAntennas = DplaneTargetAllAntennas[RefPeakAllAntennas>TriggerValue]
    RefPeakAllAntennas = RefPeakAllAntennas[RefPeakAllAntennas>TriggerValue]
    for j in range(len(PeakRef[:,0])):
        RMSResidualAllSim.append(np.std(PeakResidual[j,:][~np.isnan(PeakResidual[j,:])]))



FilterAzimuth0 = False

if(FilterAzimuth0):
    TargetZenithAllAntennas = TargetZenithAllAntennas[TargetAzimuthAllAntennas!=0]
    RefZenithAllAntennas = RefZenithAllAntennas[TargetAzimuthAllAntennas!=0]
    PeakResidualAllAntennas = PeakResidualAllAntennas[TargetAzimuthAllAntennas!=0]
    OmegaAllAntennas = OmegaAllAntennas[TargetAzimuthAllAntennas!=0]
    DplaneRefAllAntennas = DplaneRefAllAntennas[TargetAzimuthAllAntennas!=0]
    DplaneTargetAllAntennas = DplaneTargetAllAntennas[TargetAzimuthAllAntennas!=0]
    RefPeakAllAntennas = RefPeakAllAntennas[TargetAzimuthAllAntennas!=0]
    TargetEnergyAllAntennas = TargetEnergyAllAntennas[TargetAzimuthAllAntennas!=0]
    TargetAzimuthAllAntennas = TargetAzimuthAllAntennas[TargetAzimuthAllAntennas!=0]
   
FilterEnergy= True
if(FilterEnergy):
    TargetZenithAllAntennas = TargetZenithAllAntennas[TargetEnergyAllAntennas != 0.12589]
    RefZenithAllAntennas = RefZenithAllAntennas[TargetEnergyAllAntennas != 0.12589]
    PeakResidualAllAntennas = PeakResidualAllAntennas[TargetEnergyAllAntennas != 0.12589]
    OmegaAllAntennas = OmegaAllAntennas[TargetEnergyAllAntennas != 0.12589]
    DplaneRefAllAntennas = DplaneRefAllAntennas[TargetEnergyAllAntennas != 0.12589]
    DplaneTargetAllAntennas = DplaneTargetAllAntennas[TargetEnergyAllAntennas != 0.12589]
    RefPeakAllAntennas = RefPeakAllAntennas[TargetEnergyAllAntennas != 0.12589]
    TargetAzimuthAllAntennas = TargetAzimuthAllAntennas[TargetEnergyAllAntennas != 0.12589]
    TargetEnergyAllAntennas = TargetEnergyAllAntennas[TargetEnergyAllAntennas != 0.12589]

FilterOmega= False
if(FilterOmega):
    Condition1 =  list((OmegaAllAntennas<0.25))
    Condition2 = list((OmegaAllAntennas>0.75))
    ConditonAll  = (Condition1 or Condition2)
    TargetZenithAllAntennas = TargetZenithAllAntennas[ConditonAll]
    RefZenithAllAntennas = RefZenithAllAntennas[ConditonAll]
    PeakResidualAllAntennas = PeakResidualAllAntennas[ConditonAll]
    DplaneRefAllAntennas = DplaneRefAllAntennas[ConditonAll]
    DplaneTargetAllAntennas = DplaneTargetAllAntennas[ConditonAll]
    RefPeakAllAntennas = RefPeakAllAntennas[ConditonAll]
    TargetAzimuthAllAntennas = TargetAzimuthAllAntennas[ConditonAll]
    TargetEnergyAllAntennas = TargetEnergyAllAntennas[ConditonAll]
    OmegaAllAntennas = OmegaAllAntennas[ConditonAll]

   
# =============================================================================
#                                Analysis
# =============================================================================

MeanPeakResidualAll , RMSPeakResidualAll = AverageOnZenith(\
            PeakResidualAllAntennas, TargetZenithCut, TargetZenithAllAntennas)

MeanPeakResidualAllPhi270 , RMSPeakResidualAllPhi270 = AverageOnZenithOneAzimuth(\
PeakResidualAllAntennas, TargetZenithCut, TargetZenithAllAntennas, 270.0,\
TargetAzimuthAllAntennas)

MeanPeakResidualAllPhi0 , RMSPeakResidualAllPhi0 = AverageOnZenithOneAzimuth(\
PeakResidualAllAntennas, TargetZenithCut, TargetZenithAllAntennas, 0.0,\
TargetAzimuthAllAntennas)

MeanPeakResidualAllPhi180 , RMSPeakResidualAllPhi180 = AverageOnZenithOneAzimuth(\
PeakResidualAllAntennas, TargetZenithCut, TargetZenithAllAntennas, 180.0,\
TargetAzimuthAllAntennas)

OmegaBins, BinnedOmegaResidual, BinnedOmegaRMS = \
AverageOnOmega(OmegaAllAntennas, PeakResidualAllAntennas)

OmegaBinsZenithCut, BinnedOmegaResidualZenithCut, BinnedOmegaRMSZenithCut = \
AverageOnOmegaZenithCut(OmegaAllAntennas, PeakResidualAllAntennas, \
                        TargetZenithCut, TargetZenithAllAntennas)

RefPeakBins, BinnedRefPeakResidual, BinnedRefPeakRMS = \
AverageOnRefPeak(RefPeakAllAntennas, PeakResidualAllAntennas)

error_0_1000 = PeakResidualAllAntennas[(abs(RefPeakAllAntennas)>0)  & (abs(RefPeakAllAntennas)<1e3)]
error_1000_10000 = PeakResidualAllAntennas[(abs(RefPeakAllAntennas)>1e3)  & (abs(RefPeakAllAntennas)<1e4)]
error_10000_100000 = PeakResidualAllAntennas[(abs(RefPeakAllAntennas)>1e4)  & (abs(RefPeakAllAntennas)<1e5)]

Distplane = (DplaneRefAllAntennas - DplaneTargetAllAntennas)/1e3
DistplaneRel  = (DplaneRefAllAntennas - DplaneTargetAllAntennas)/DplaneRefAllAntennas

DplaneBins, BinnedDplaneResidual, BinnedDplanePeakRMS = \
AverageOnDplane(abs(Distplane), PeakResidualAllAntennas)

DplaneRelBins, BinnedDplaneRelResidual, BinnedDplaneRelPeakRMS = \
AverageOnDplaneRel(abs(DistplaneRel), PeakResidualAllAntennas)


# =============================================================================
#                               Plots
# =============================================================================
# =============================================================================
#                        RadioMorphing Performances
# =============================================================================

# Total Mean and RMS 
print(" RADIOMORPHING PERFOMANCES MEAN AND RMS")
print("\n")
print("Phi = 0")
PlotMeanRMS(TargetZenithCut, MeanPeakResidualAllPhi0, RMSPeakResidualAllPhi0)
print("Phi = 180")
PlotMeanRMS(TargetZenithCut, MeanPeakResidualAllPhi180, RMSPeakResidualAllPhi180)
print("Phi = 270")
PlotMeanRMS(TargetZenithCut, MeanPeakResidualAllPhi270, RMSPeakResidualAllPhi270)
print("All Azimuth")
PlotMeanRMS(TargetZenithCut, MeanPeakResidualAll, RMSPeakResidualAll)


plt.scatter(TargetZenithCut, MeanPeakResidualAll)
#plt.axhline(y = 0.2, color = 'r', linestyle = '--')
#plt.axvline(x = 82.5, color = 'green', linestyle = '--')
plt.xlabel("target zenith [Deg.]")
plt.ylabel("Mean")
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/MeanvsThetaRelErrorPeakCorrectedBias.pdf")
plt.show()

plt.scatter(TargetZenithCut, RMSPeakResidualAll)
plt.axhline(y = 0.2, color = 'r', linestyle = '--')
plt.axvline(x = 82.5, color = 'green', linestyle = '--')
plt.xlabel("target zenith [Deg.]")
plt.ylabel("RMS")
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/RMSvsThetaRelErrorPeakCorrectedBias.pdf")
plt.show()

#plt.scatter(TargetZenithCut, MeanPeakResidualAll)


# =============================================================================
#                         Dependency with azimuth angle
# =============================================================================

print("\n")
print("DEPENDENCY OF THE ERROR WITH AZIMUTH")
print("Mean")
# Mean dependency with azimuth angle
plt.scatter(TargetZenithCut, MeanPeakResidualAllPhi0, label = "$\phi = 0^{\circ}$")
plt.scatter(TargetZenithCut, MeanPeakResidualAllPhi270, label = "$\phi = 90^{\circ}$")
plt.scatter(TargetZenithCut, MeanPeakResidualAllPhi180, label = "$\phi = 180^{\circ}$")
plt.legend()
plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
plt.xlabel("Target zenith [Deg.]")
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/MeanvsThetavsPhiRelErrorPeakCorrectedBias.pdf")
#plt.savefig("/Users/chiche/Desktop/MeanRelErrorPeakAzimuthvsTheta.pdf")
plt.show()

print("RMS")
# RMS dependency with azimuth angle
plt.scatter(TargetZenithCut, RMSPeakResidualAllPhi0, label = "$\phi = 0^{\circ}$")
plt.scatter(TargetZenithCut, RMSPeakResidualAllPhi270, label = "$\phi = 90^{\circ}$")
plt.scatter(TargetZenithCut, RMSPeakResidualAllPhi180, label = "$\phi = 180^{\circ}$")
#plt.ylabel("RMS relative peak deviation")
plt.ylabel("RMS($\delta$)")
plt.xlabel("Target zenith [Deg.]")
plt.legend()
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/RMSvsThetavsPhiRelErrorPeakCorrectedBias.pdf")
#plt.savefig("/Users/chiche/Desktop/RMSRelErrorPeakAzimuthvsTheta.pdf")
plt.show()

# =============================================================================
#                       Error with omega angle
# =============================================================================
print("\n")
print("DEPENDENCY OF THE ERROR WITH THE OMEGA ANGLE")
print("Splitting the results in terms of zenith")
# Residuals binned in omega values
for i in range(len(TargetZenithCut)):
    plt.errorbar(OmegaBinsZenithCut[i], BinnedOmegaResidualZenithCut[i], \
                 yerr = BinnedOmegaRMSZenithCut[i], fmt = 'o', label = '$\\theta = %.2f^{\circ}$' %TargetZenithCut[i])
   # plt.scatter(OmegaBinsZenithCut[i], BinnedOmegaRMSZenithCut[i], c= 'orange')
    #print(np.mean(BinnedOmegaRMSZenithCut[i]))
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("/Users/chiche/Desktop/RelPeakErrorvsOmegavsTheta%.2f_CorrectedBias.pdf" %TargetZenithCut[i])
    #plt.savefig("/Users/chiche/Desktop/OmegaRMtest/RelPeakErrorvsOmegavsTheta%.2f.pdf" %TargetZenithCut[i])
    plt.show()
    


print("All Zenith")
# Residuals binned in omega values
plt.errorbar(OmegaBins, BinnedOmegaResidual, yerr = BinnedOmegaRMS, fmt = 'o', label = "All Zenith")
plt.xlabel("$\omega$ [Deg.]")
plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
plt.legend()
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/OmegaRMtest/RelPeakErrorvsOmegavsAllTheta.pdf")
#plt.savefig("/Users/chiche/Desktop/RelPeakErrorvsOmegaAllZenithCorrectedBias.pdf")
plt.show()

#plt.scatter(OmegaBins, BinnedOmegaRMS)
# =============================================================================
#                     Error with reference peak amplitude
# =============================================================================

# Residuals binned in Ref Peak values
plt.errorbar(RefPeakBins, BinnedRefPeakResidual, yerr = BinnedRefPeakRMS, fmt = 'o')
plt.xlabel("$Ref peak$ [$\mu V/m$]")
plt.ylabel("relative peak deviation")
plt.xscale("log")
plt.tight_layout()
plt.show()

# =============================================================================
#                         Error distribution
# =============================================================================

plt.hist(error_0_1000, bins = 100, edgecolor = "red", color = 'white', \
         density = 'True', histtype=u'step', linewidth=1)

plt.hist(error_1000_10000, bins = 100, edgecolor = "blue", color = 'white', \
         density = 'True', histtype=u'step', linewidth=2)

plt.hist(error_10000_100000, bins = 100, edgecolor = "black", color = 'white', \
         density = 'True', histtype=u'step', linewidth=3)

#plt.xlabel("Mean relative difference peak amplitude")
plt.xlabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
plt.ylabel("#")
plt.legend(["$E<1e3\,\mu V/m$", "$1e3<E<1e4\,\mu V/m$", "$E>1e4\,\mu V/m$"], fontsize =12)
plt.xlim(-1,1)
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/HistMeanRelPeakError.pdf")
plt.show()


plt.hist(error_10000_100000, bins = 60, edgecolor = "black", color = 'white', \
         density = 'True', histtype=u'step', linewidth=3)

plt.xlabel("Mean relative difference peak amplitude")
plt.ylabel("#")
plt.legend(["$E<1e3\,\mu V/m$", "$1e3<E<1e4\,\mu V/m$", "$E>1e4\,\mu V/m$"], fontsize =12)
plt.xlim(-1,1)
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/HistMeanRelPeakErrorOneBinCorrectedBias.pdf")
plt.show()

plt.hist(RMSResidualAllSim, bins = 100, edgecolor = "black", color = 'white', \
         density = 'True', histtype=u'step', linewidth=3)

plt.xlabel("Mean relative difference peak amplitude")
plt.ylabel("#")
plt.legend(["$E<1e3\,\mu V/m$", "$1e3<E<1e4\,\mu V/m$", "$E>1e4\,\mu V/m$"], fontsize =12)
plt.xlim(0,0.5)
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/HistRMSRelPeakErrorOneBinCorrectedBias.pdf")
plt.show()

# =============================================================================
#                   Dependency with plane distance
# =============================================================================

#plotMeanRMSPlaneDistance(DplaneBins, BinnedDplaneResidual, BinnedDplanePeakRMS)
#plt.scatter(DplaneBins, BinnedDplaneResidual)
#plt.show()

#plt.scatter(DplaneBins, BinnedDplanePeakRMS)
#plt.show()
#PlotMeanRMSPlaneDistance(DplaneRelBins, BinnedDplaneRelResidual, BinnedDplaneRelPeakRMS)

#plt.scatter(DplaneRelBins, BinnedDplaneRelResidual)
#plt.show()
#plt.scatter(DplaneRelBins, BinnedDplaneRelPeakRMS)
#plt.show()

# =============================================================================
#                    Dependency with the Target Energy
# =============================================================================

plt.scatter(TargetEnergyAllAntennas, PeakResidualAllAntennas, s=0.1)
plt.xlabel("Target Energy [EeV]")
plt.ylabel("relative peak error")
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/RMSrelErrorPeakvsEnergyCorrectedBias.png")
plt.show()

TargetEnergyCut = get_ZenithCut(TargetEnergyAllAntennas)
N_Ebins = len(TargetEnergyCut)
MeanPeakResidualAllAntennasEbins = np.zeros(N_Ebins)
RMS_PeakResidualAllAntennasEbins = np.zeros(N_Ebins)

NoNanRes = PeakResidualAllAntennas[~np.isnan(PeakResidualAllAntennas)]
NoNanE = TargetEnergyAllAntennas[~np.isnan(PeakResidualAllAntennas)]

for i in range(N_Ebins):
    MeanPeakResidualAllAntennasEbins[i] = \
    np.mean(NoNanRes[NoNanE ==  TargetEnergyCut[i]])
    
    RMS_PeakResidualAllAntennasEbins[i] = \
    np.std(NoNanRes[NoNanE ==  TargetEnergyCut[i]])
    
# Residuals binned in E bins
plt.errorbar(TargetEnergyCut, MeanPeakResidualAllAntennasEbins, \
             yerr = RMS_PeakResidualAllAntennasEbins, fmt = 'o')
plt.xlabel("Target Energy [EeV]")
plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
#plt.legend()
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/MeanRelPeakErrorEbins.png")
plt.show()

plt.scatter(TargetEnergyCut, MeanPeakResidualAllAntennasEbins)
plt.scatter(TargetEnergyCut, RMS_PeakResidualAllAntennasEbins)
plt.xlabel("Target Energy [EeV]")
plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
#plt.legend()
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/MeanRelPeakErrorEbins.png")
plt.show()


plt.scatter(TargetEnergyCut, RMS_PeakResidualAllAntennasEbins)
plt.show()

# =============================================================================
#                  Distribution of amplitude versus omega
# =============================================================================

tzen = 67.75
plt.scatter(OmegaAllAntennas[RefZenithAllAntennas == tzen ], \
            RefPeakAllAntennas[RefZenithAllAntennas == tzen ])
plt.yscale("log")
plt.show()

plt.scatter(OmegaAllAntennas, RefPeakAllAntennas, s= 0.1)
plt.show()

plt.hist(OmegaAllAntennas[RefPeakAllAntennas>0], bins = 10, \
         edgecolor = "red", color = 'white', \
         density = 'False', histtype=u'step', linewidth=1)
plt.show()

# =============================================================================
#                    Distribution of amplitude
# =============================================================================

plt.hist(RefPeakAllAntennas, bins = 1000, edgecolor = "black", color = 'white', \
         density = 'True', histtype=u'step', linewidth=1)
#plt.xlim(0,10000)
plt.xscale("log")
plt.xlabel("Reference peak amplitude [$\mu V/m$]")
plt.ylabel("#")
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/HistRefPeak.pdf")
plt.show()


# =============================================================================
#                       Computation time
# =============================================================================

MeanTsim = np.zeros(len(TargetZenithCut))

for i in range(len(TargetZenithCut)):
    
    MeanTsim[i] = np.mean(TallSim[TargetZenithAllSim == TargetZenithCut[i]])



plt.scatter(TargetZenithCut, MeanTsim)
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("Mean CPU time [s]")
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/MeanCpuTime.png")
plt.show()





