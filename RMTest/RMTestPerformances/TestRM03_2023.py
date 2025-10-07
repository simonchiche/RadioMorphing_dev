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

# =============================================================================
#                                 Loading
# =============================================================================
##path ="./RMResultsAllPlanesAllTheta01_03_23/"
##path ="./RMResults_03_2024/"
#path = "RMResults_03_2024_with_bias_correction"
#path ="./Data/RMResults_03_2024/"
path ="./Data/RMResults_03_2024_without_87.2_ref/"


# Loop over all jobs

Trigger = True
for i in range(NFileGroup):

    start = i*100
    PeakRef =  np.loadtxt(path + "PeakRef%.2d.txt" %start)
    PeakTarget =  np.loadtxt(path + "PeakTarget%.2d.txt" %start)
    PeakResidual =  np.loadtxt(path + "PeakResidual%.2d.txt" %start)
    Parameters = np.loadtxt(path + "RefTargetParameters%.2d.txt" %start)
    Omega = np.loadtxt(path + "OmegaAngle%.2d.txt" %start)
    
    # Loop over all sims per jobs
    for j in range(len(PeakRef[:,0])):

        TargetZenithAllSim.append(180 - Parameters[j,1])
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

if(Trigger):
    TriggerValue = 100
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
    
    RMSResidualAllSim = []
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
CutEnergy = 0.12589
if(FilterEnergy):
    TargetZenithAllAntennas = TargetZenithAllAntennas[TargetEnergyAllAntennas != CutEnergy]
    RefZenithAllAntennas = RefZenithAllAntennas[TargetEnergyAllAntennas != CutEnergy]
    PeakResidualAllAntennas = PeakResidualAllAntennas[TargetEnergyAllAntennas != CutEnergy]
    OmegaAllAntennas = OmegaAllAntennas[TargetEnergyAllAntennas != CutEnergy]
    DplaneRefAllAntennas = DplaneRefAllAntennas[TargetEnergyAllAntennas != CutEnergy]
    DplaneTargetAllAntennas = DplaneTargetAllAntennas[TargetEnergyAllAntennas != CutEnergy]
    RefPeakAllAntennas = RefPeakAllAntennas[TargetEnergyAllAntennas != CutEnergy]
    TargetAzimuthAllAntennas = TargetAzimuthAllAntennas[TargetEnergyAllAntennas != CutEnergy]
    TargetEnergyAllAntennas = TargetEnergyAllAntennas[TargetEnergyAllAntennas != CutEnergy]

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
#plt.savefig("/Users/chiche/Desktop/scaling_int3d_error_peak_mean_sub_voltage_new.pdf")
plt.show()

plt.scatter(TargetZenithCut, RMSPeakResidualAll)
plt.axhline(y = 0.2, color = 'r', linestyle = '--')
plt.axvline(x = 82.5, color = 'green', linestyle = '--')
plt.xlabel("target zenith [Deg.]")
plt.ylabel("RMS")
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/RM_full_rms_03_2024_0")
plt.show()
#sys.exit()


# =============================================================================
#                       Test pour une énergie
# =============================================================================

TargetEnergyCut = get_ZenithCut(TargetEnergyAllAntennas)       

ConditionEnergy = TargetEnergyAllAntennas ==TargetEnergyCut[14]
MeanPeakResidualAll , RMSPeakResidualAll = AverageOnZenith(\
            PeakResidualAllAntennas[ConditionEnergy], TargetZenithCut, TargetZenithAllAntennas[ConditionEnergy])

plt.scatter(TargetZenithCut, RMSPeakResidualAll)
plt.show()

# =============================================================================
#                       Généralisation
# =============================================================================
RefZenithCut =  get_ZenithCut(RefZenithAllAntennas)  
MeanPeakResidualAllOneEnergy = dict()
RMSPeakResidualAllOneEnergy = dict()
RefZenithCut[4] =84.95

for i in range(len(TargetEnergyCut)):
    ConditionEnergy = TargetEnergyAllAntennas ==TargetEnergyCut[i]
    MeanPeakResidualAllOneEnergy[i] , RMSPeakResidualAllOneEnergy[i] = AverageOnZenith(\
    PeakResidualAllAntennas[ConditionEnergy], TargetZenithCut, TargetZenithAllAntennas[ConditionEnergy])
    
    ConditionZenith = np.in1d(TargetZenithCut, RefZenithCut).tolist()
    not_ConditionZenith = [not i for i in ConditionZenith]
    plt.scatter(np.array(TargetZenithCut)[ConditionZenith],  \
    RMSPeakResidualAllOneEnergy[i][ConditionZenith], label = "$\\theta^{t} = \\theta^{r}$")
    plt.scatter(np.array(TargetZenithCut)[not_ConditionZenith],  \
    RMSPeakResidualAllOneEnergy[i][not_ConditionZenith], label = "$\\theta^{t} \\neq \\theta^{r}$")
    plt.legend(loc = "upper left")
    plt.xlabel("target zenith [Deg.]")
    plt.ylabel("RMS($\delta$)")
    plt.title("E = %.2f EeV" %TargetEnergyCut[i], fontsize = 14)
    plt.tight_layout()
    #plt.savefig("/Users/chiche/Desktop/EnergyResults/RMSRelErrorPeakvsThetavsEnergy%.2f.pdf" %TargetEnergyCut[i])
    #plt.savefig("/Users/chiche/Desktop/Eplot/RMSRelErrorPeakvsThetavsEnergy%.2f.pdf" %TargetEnergyCut[i])
    plt.show()
    
    
# =============================================================================
#                         Dependency with azimuth angle
# =============================================================================

print("\n")
print("DEPENDENCY OF THE ERROR WITH AZIMUTH")
print("Mean")
# Mean dependency with azimuth angle
plt.scatter(TargetZenithCut, MeanPeakResidualAllPhi0, label = "$\phi = 0^{\circ}$")
plt.scatter(TargetZenithCut, MeanPeakResidualAllPhi180, label = "$\phi = 180^{\circ}$")
plt.scatter(TargetZenithCut, MeanPeakResidualAllPhi270, label = "$\phi = 270^{\circ}$")
plt.ylabel("Mean relative peak deviation")
plt.legend()
plt.show()

print("RMS")
# RMS dependency with azimuth angle
plt.scatter(TargetZenithCut, RMSPeakResidualAllPhi0, label = "$\phi = 0^{\circ}$")
plt.scatter(TargetZenithCut, RMSPeakResidualAllPhi180, label = "$\phi = 180^{\circ}$")
plt.scatter(TargetZenithCut, RMSPeakResidualAllPhi270, label = "$\phi = 270^{\circ}$")
plt.ylabel("RMS relative peak deviation")
plt.legend()
plt.show()



plt.scatter(RefZenithAllAntennas, TargetZenithAllAntennas-RefZenithAllAntennas)
plt.show()
