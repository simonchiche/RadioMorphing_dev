
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from ModuleTestRM import get_ZenithCut, AverageOnZenith, \
AverageOnZenithOneAzimuth,AverageOnOmega, AverageOnRefPeak, AverageOnDplane,\
AverageOnDplaneRel, AverageOnOmegaZenithCut
from PlotTestRM import PlotMeanRMS, PlotMeanRMSPlaneDistance, PlotRMSDistrib, GetMeanRMSerr
import pickle

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)


filteredpath ="./Data/FilteredData/"
ZenithFiltered = np.loadtxt(filteredpath + "zenith_all_raw.txt")
EnergyFiltered = np.loadtxt(filteredpath + "energy_all.txt")

#with open(filteredpath + "error_all_correlation_mean.pkl", "rb") as f:
#    error_all_lofar = pickle.load(f)
with open(filteredpath + "error_all_correlation_grand_trigg60.pkl", "rb") as f:
    error_all = pickle.load(f)

energy_threshold = 0.13
ZenithFilteredcut, Meanerr_filtered, RMSerr_filtered =\
    GetMeanRMSerr(EnergyFiltered, ZenithFiltered, error_all, energy_threshold, filteredpath, PLOT=True)



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
