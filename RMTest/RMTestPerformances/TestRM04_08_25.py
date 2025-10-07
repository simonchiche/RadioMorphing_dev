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


# PLOT for article
RefZenithCut =  get_ZenithCut(RefZenithAllAntennas) 
ConditionZenith = np.in1d(TargetZenithCut, RefZenithCut).tolist()
not_ConditionZenith = [not i for i in ConditionZenith]

plt.scatter(np.array(TargetZenithCut)[ConditionZenith], MeanPeakResidualAll[ConditionZenith], marker ="x", color = '#0072B2', s = 65, label = "$\\theta^{t} = \\theta^{\\rm ref}$")
plt.scatter(np.array(TargetZenithCut)[not_ConditionZenith], MeanPeakResidualAll[not_ConditionZenith], marker ="x", color ="#E69F00", s = 60, label = "$\\theta^{t} \\neq \\theta^{\\rm ref}$")
#plt.axhline(y = 0.2, color = 'r', linestyle = '--')
#plt.axvline(x = 82.5, color = 'green', linestyle = '--')
plt.xlabel("target zenith [Deg.]")
plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.axhline(0, color='#B22222', linestyle='--', alpha=0.6)
plt.tight_layout()
#plt.savefig("MeanvsThetaRelErrorPeak.pdf", bbox_inches = "tight")
plt.show()

#plt.scatter(np.array(TargetZenithCut)[ConditionZenith], RMSPeakResidualAll[ConditionZenith], marker ='*', s= 60, color="#4C72B0")
#plt.scatter(np.array(TargetZenithCut)[not_ConditionZenith], RMSPeakResidualAll[not_ConditionZenith], marker ='*', s= 60, color="#E69F00")
plt.scatter(TargetZenithCut, RMSPeakResidualAll, marker ='*', s= 65, color="#4C72B0")
plt.axhline(y = 0.2, color = '#C44E52', linestyle = '--')
plt.axhline(y = 0.12, color = '#C44E52', linestyle = '--')
plt.axvline(x = 80, color = 'black', linestyle = '--')
plt.axvspan(80, 90, color='orange', alpha=0.15, label="Region > 80°")
plt.xlabel("target zenith [Deg.]")
plt.ylabel("$\sigma{(\\delta})$")
plt.text(
    67, 0.122, "12% limit", 
    color='#C44E52', 
    fontsize=12, 
    va='bottom',  # ancre verticale (texte au-dessus de la ligne)
    ha='right'    # ancre horizontale
)
plt.text(
    67, 0.202, "20% limit", 
    color='#C44E52', 
    fontsize=12, 
    va='bottom',  # ancre verticale (texte au-dessus de la ligne)
    ha='right'    # ancre horizontale
)
plt.text(
    89.7, 0.19, "highly inclined showers", 
    color='black', 
    fontsize=10, 
    va='bottom',  # ancre verticale (texte au-dessus de la ligne)
    ha='right'    # ancre horizontale
)
plt.tight_layout()
plt.ylim(0.08,0.22)
#plt.savefig("RMSvsThetaRelErrorPeak.pdf", bbox_inches = "tight")
plt.show()

#plt.scatter(TargetZenithCut, MeanPeakResidualAll)


# =============================================================================
#                       Error with omega angle
# =============================================================================

# PLOT for article
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

# PLOT for article
print("All Zenith")
# Residuals binned in omega values
plt.errorbar(OmegaBins, BinnedOmegaResidual, fmt = 'x', label = "All Zenith", color = "black", markersize = 7)
plt.plot(OmegaBins, BinnedOmegaResidual, color ="black", linewidth =0.2)
plt.fill_between(OmegaBins, BinnedOmegaResidual - BinnedOmegaRMS, BinnedOmegaResidual + BinnedOmegaRMS, alpha=0.2,color="#009E73")
plt.xlabel("$\omega$ [Deg.]")
plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
plt.legend()
plt.tight_layout()
#plt.savefig("/Users/chiche/Desktop/OmegaRMtest/RelPeakErrorvsOmegavsAllTheta.pdf")
#plt.savefig("/Users/chiche/Desktop/RelPeakErrorvsOmegaAllZenithCorrectedBias.pdf")
plt.grid(True, linestyle='--', alpha=0.5)
#plt.savefig("RelPeakErrorvsOmegavsAllTheta.pdf", bbox_inches = "tight")
plt.show()



# =============================================================================
#                    Dependency with the Target Energy
# =============================================================================

MeanPeakResidualsEbins_phi = dict()
RMSPeakResidualsEbins_phi = dict()
PhiBins= np.unique(TargetAzimuthAllAntennas)
bin_edges = np.linspace(-0.5, 0.5, 40) 
### PLOT for article
for k in range(len(PhiBins)):
    label = "$\\varphi = %.d^{\circ}$" % PhiBins[k]
    sel = (TargetAzimuthAllAntennas == PhiBins[k]) & (TargetEnergyAllAntennas==3.9811)
    PeakResidualAllAntennas_cut = PeakResidualAllAntennas[sel]
    plt.hist(PeakResidualAllAntennas_cut, bin_edges, alpha=0.6, edgecolor='black', label=label)
plt.legend()
plt.xlabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
plt.ylabel("counts")
#plt.savefig("RM_rel_err_vs_azimuth.pdf", bbox_inches ="tight")
plt.show()


TargetEnergyCut = get_ZenithCut(TargetEnergyAllAntennas)
N_Ebins = len(TargetEnergyCut)
for k in range(len(PhiBins)):

    MeanPeakResidualAllAntennasEbins = np.zeros(N_Ebins)
    RMS_PeakResidualAllAntennasEbins = np.zeros(N_Ebins)

    PeakResidualAllAntennas_cut = PeakResidualAllAntennas[TargetAzimuthAllAntennas == PhiBins[k]]
    TargetEnergyAllAntennas_cut = TargetEnergyAllAntennas[TargetAzimuthAllAntennas == PhiBins[k]]

    NoNanRes = PeakResidualAllAntennas_cut[~np.isnan(PeakResidualAllAntennas_cut)]
    NoNanE = TargetEnergyAllAntennas_cut[~np.isnan(PeakResidualAllAntennas_cut)]

    for i in range(N_Ebins):
        MeanPeakResidualAllAntennasEbins[i] = \
        np.mean(NoNanRes[NoNanE ==  TargetEnergyCut[i]])
        
        RMS_PeakResidualAllAntennasEbins[i] = \
        np.std(NoNanRes[NoNanE ==  TargetEnergyCut[i]])
    MeanPeakResidualsEbins_phi[k] = MeanPeakResidualAllAntennasEbins
    RMSPeakResidualsEbins_phi[k] = RMS_PeakResidualAllAntennasEbins


### PLOT for article
plt.plot(TargetEnergyCut, MeanPeakResidualsEbins_phi[0], "-s", label ="$\\varphi=%.d^{\circ}$" %PhiBins[0], color ="#1f77b4", linewidth=2)
plt.plot(TargetEnergyCut, MeanPeakResidualsEbins_phi[2], "-o", label ="$\\varphi=%.d^{\circ}$" %90, color ="#2ca02c", linewidth=2)
plt.plot(TargetEnergyCut, MeanPeakResidualsEbins_phi[1], "-*", label ="$\\varphi=%.d^{\circ}$" %PhiBins[1], color ="#d62728", markersize=8, linewidth=2)
plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
plt.xlabel("Target Azimuth [Deg.]")
plt.axhline(0, color='#B22222', linestyle='--', alpha=0.6)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
#plt.savefig("RM_reldiff_vs_azimuth_vsE.pdf", bbox_inches ="tight")
plt.show()


MeanPeakResidualAllAntennasEbins = np.zeros(N_Ebins)
RMS_PeakResidualAllAntennasEbins = np.zeros(N_Ebins)

PeakResidualAllAntennas_cut = PeakResidualAllAntennas.copy()
TargetEnergyAllAntennas_cut = TargetEnergyAllAntennas.copy()

NoNanRes = PeakResidualAllAntennas_cut[~np.isnan(PeakResidualAllAntennas_cut)]
NoNanE = TargetEnergyAllAntennas_cut[~np.isnan(PeakResidualAllAntennas_cut)]

for i in range(N_Ebins):
    MeanPeakResidualAllAntennasEbins[i] = \
    np.mean(NoNanRes[NoNanE ==  TargetEnergyCut[i]])
    
    RMS_PeakResidualAllAntennasEbins[i] = \
    np.std(NoNanRes[NoNanE ==  TargetEnergyCut[i]])
MeanPeakResidualsEbins_phi[k] = MeanPeakResidualAllAntennasEbins
RMSPeakResidualsEbins_phi[k] = RMS_PeakResidualAllAntennasEbins


### PLOT for article
plt.errorbar(TargetEnergyCut, MeanPeakResidualAllAntennasEbins, fmt = 'x', color = 'black', markersize = 8)
plt.fill_between(TargetEnergyCut, MeanPeakResidualAllAntennasEbins - RMS_PeakResidualAllAntennasEbins, MeanPeakResidualAllAntennasEbins + RMS_PeakResidualAllAntennasEbins, alpha=0.2, color="#009E73")
plt.xlabel("Target Energy [EeV]")
plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
plt.axhline(0, color="#D21717", linestyle='--', alpha=0.6)
plt.grid(True, linestyle='--', alpha=0.5)
#plt.savefig("RM_reldiff_vs_targetE.pdf", bbox_inches ="tight")
plt.show()



# =============================================================================
#                       Computation time
# =============================================================================

MeanTsim = np.zeros(len(TargetZenithCut))
StdTsim = np.zeros(len(TargetZenithCut))

for i in range(len(TargetZenithCut)):
    
    MeanTsim[i] = np.mean(TallSim[TargetZenithAllSim == TargetZenithCut[i]])
    StdTsim[i] = np.std(TallSim[TargetZenithAllSim == TargetZenithCut[i]])


### PLOT for article
plt.errorbar(TargetZenithCut, MeanTsim, yerr=StdTsim, marker ="s", color = "black", markersize = 7)
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("Mean CPU time [s]")
plt.tight_layout()
plt.grid()
#plt.savefig("/Users/chiche/Desktop/MeanCpuTime.pdf", bbox_inches ="tight")
plt.show()


#### FILTERED RESULTS #####
import pickle

filteredpath ="./Data/FilteredData/"
ZenithFiltered = np.loadtxt(filteredpath + "zenith_all_raw.txt")
EnergyFiltered = np.loadtxt(filteredpath + "energy_all.txt")

#with open(filteredpath + "error_all_correlation_mean.pkl", "rb") as f:
#    error_all_lofar = pickle.load(f)
with open(filteredpath + "error_all_correlation_grand_trigg60_peak.pkl", "rb") as f:
    error_all_lofar = pickle.load(f)

ZenithFilteredcut = np.unique(ZenithFiltered)
Meanerr_filtered = np.zeros(len(ZenithFilteredcut))
RMSerr_filtered = np.zeros(len(ZenithFilteredcut))

for i in range(len(ZenithFilteredcut)):
    indices = np.where((ZenithFiltered == ZenithFilteredcut[i]) & (EnergyFiltered>0.13))[0]
    err_zen = {k: error_all_lofar.get(k) for k in indices if k in error_all_lofar}
    err_1d = [val for arr in err_zen.values() for val in arr]
    err_1d = np.array(err_1d)
    #err_1d = err_1d[abs(err_1d)!=1.0]
    #print(max(abs(err_1d)))
    #err_1d= err_1d[err_1d!=0.0]
    #print(min(abs(err_1d)))
    plt.hist(err_1d, bins=50, color="skyblue", edgecolor='black', linewidth=0.5, alpha=0.8)
    plt.xlabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
    plt.ylabel("Number of antennas")
    plt.title("Zenith = %.1f°" %ZenithFilteredcut[i])
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    #plt.savefig("RelPeakErrorvsZenith%.1f.pdf" %ZenithFilteredcut[i], bbox_inches = "tight")
    plt.show()
    Meanerr_filtered[i] = np.mean(err_1d)
    RMSerr_filtered[i] = np.std(err_1d)


plt.scatter(ZenithFilteredcut, Meanerr_filtered, marker ="x", color = '#0072B2', s = 65)
plt.show()

#plt.scatter(np.array(TargetZenithCut)[ConditionZenith], RMSPeakResidualAll[ConditionZenith], marker ='*', s= 60, color="#4C72B0")
#plt.scatter(np.array(TargetZenithCut)[not_ConditionZenith], RMSPeakResidualAll[not_ConditionZenith], marker ='*', s= 60, color="#E69F00")
plt.scatter(ZenithFilteredcut, RMSerr_filtered, marker ='*', s= 65, color="#4C72B0")
plt.axhline(y = 0.15, color = '#C44E52', linestyle = '--')
plt.axhline(y = 0.12, color = '#C44E52', linestyle = '--')
plt.axvline(x = 80, color = 'black', linestyle = '--')
plt.axvspan(80, 90, color='orange', alpha=0.15, label="Region > 80°")
plt.xlabel("target zenith [Deg.]")
plt.ylabel("RMS($\\delta$)")
#plt.ylim(0.09,0.16)
plt.text(
    67, 0.122, "12% limit", 
    color='#C44E52', 
    fontsize=12, 
    va='bottom',  # ancre verticale (texte au-dessus de la ligne)
    ha='right'    # ancre horizontale
)
plt.text(
    67, 0.15, "15% limit", 
    color='#C44E52', 
    fontsize=12, 
    va='bottom',  # ancre verticale (texte au-dessus de la ligne)
    ha='right'    # ancre horizontale
)
plt.text(
    89.7, 0.15, "highly inclined showers", 
    color='black', 
    fontsize=10, 
    va='bottom',  # ancre verticale (texte au-dessus de la ligne)
    ha='right'    # ancre horizontale
)
plt.tight_layout()
#plt.ylim(0.08,0.22)
#plt.savefig("RMSvsThetaRelErrorPeak.pdf", bbox_inches = "tight")
plt.show()


TimingData = "./Data/Timing/"
with open(TimingData + "dt_all.pkl", "rb") as f:
    dt_all= pickle.load(f)

dt_all_1d = []
for i in range(1, len(dt_all)+1):

    mean_dt = np.mean(dt_all[i])
    dt_distrib= (dt_all[i] - mean_dt)*2
    dt_all_1d.extend(dt_distrib)

bins = np.linspace(-10, 10, 100)
plt.hist(dt_all_1d, bins=bins, edgecolor='black', color="skyblue", linewidth=1)

import seaborn as sns
# Style seaborn clair
sns.set_style("whitegrid")
sns.set_context("talk")
plt.figure()
plt.hist(dt_all_1d, bins=bins, color=sns.color_palette("muted")[0],
         edgecolor="black", linewidth=0.5, alpha=0.8)

plt.xlabel("T", fontsize=14)
plt.ylabel("Number of antennas", fontsize=14)
plt.tight_layout()
plt.xlabel("$\Delta_t = t_{\\rm peak}^{\\rm RM} - t_{\\rm peak}^{\\rm ZHS}$ [ns] ")
#plt.savefig("TimeDelayDistrib.pdf", bbox_inches = "tight")
plt.show()

#print(np.sqrt(np.std(dt_all_1d)))