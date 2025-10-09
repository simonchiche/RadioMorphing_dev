
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from ModuleTestRM import get_ZenithCut, AverageOnZenith, \
AverageOnZenithOneAzimuth,AverageOnOmega, AverageOnRefPeak, AverageOnDplane,\
AverageOnDplaneRel, AverageOnOmegaZenithCut
from PlotTestRM import PlotMeanRMS, PlotMeanRMSPlaneDistance, PlotRMSDistrib, GetMeanRMSerr, PlotMeanErr, PlotRMSvsTheta
import pickle

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

savepath= "/Users/chiche/Desktop/RadioMorphingUptoDate/RMTest/RMTestPerformances/Figures/"
filteredpath ="./Data/FilteredData/"
ZenithFiltered = np.loadtxt(filteredpath + "zenith_all_raw.txt")
EnergyFiltered = np.loadtxt(filteredpath + "energy_all.txt")

#with open(filteredpath + "error_all_correlation_mean.pkl", "rb") as f:
#    error_all_lofar = pickle.load(f)

path1 = filteredpath + "error_all_correlation_raw_notrigg.pkl"
path2 = filteredpath + "error_all_correlation_raw_trigg60.pkl"
path3 = filteredpath + "error_all_correlation_raw_trigg110.pkl"
path4 = filteredpath + "error_all_correlation_grand_trigg60.pkl"
path5 = filteredpath + "error_all_correlation_grand_trigg110.pkl"
path6 = filteredpath + "error_all_correlation_lofar_trigg60.pkl"
path7 = filteredpath + "error_all_correlation_lofar_trigg110.pkl"

pathAll = [path1, path2, path3, path4, path5, path6, path7]
RMSerrAll= dict()
for i in range(len(pathAll)):
    with open(pathAll[i], "rb") as f:
        error_all = pickle.load(f)

    energy_threshold = 0.13
    ZenithFilteredcut, Meanerr_filtered, RMSerr_filtered =\
        GetMeanRMSerr(EnergyFiltered, ZenithFiltered, error_all, energy_threshold, filteredpath, PLOT=False)

    if(i==0):
        PlotMeanErr(ZenithFilteredcut, Meanerr_filtered, savepath)

        PlotRMSvsTheta(ZenithFilteredcut, RMSerr_filtered, savepath)

    RMSerrAll[i] = RMSerr_filtered


plt.plot(ZenithFilteredcut, RMSerrAll[1], color="black", marker="o", label ="Full band")
plt.plot(ZenithFilteredcut, RMSerrAll[3], color="#D55E00", marker="P", label ="50-200 MHz")
plt.plot(ZenithFilteredcut, RMSerrAll[5], color="#0072B2", marker="s", label ="30-80 MHz")
plt.xlabel("target zenith [Deg.]")
plt.ylabel("$\sigma{(\\delta)}$")
plt.axvline(x = 80, color = 'black', linestyle = '--')
plt.axvspan(80, 90, color='orange', alpha=0.15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(y = 0.15, color = '#C44E52', linestyle = '--')
plt.axhline(y = 0.12, color = '#C44E52', linestyle = '--')
plt.title("Trigger Threshold = 60 $\mu V/m$", fontsize=12)
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
plt.legend()
plt.ylim(0.07,0.16)
#plt.savefig(savepath + "RMSrelErr_vs_Theta_trigg60.pdf", bbox_inches = "tight")
plt.show()

plt.plot(ZenithFilteredcut, RMSerrAll[2], color="black", marker="o", label ="Full band")
plt.plot(ZenithFilteredcut, RMSerrAll[4], color="#D55E00", marker="P", label ="50-200 MHz")
plt.plot(ZenithFilteredcut, RMSerrAll[6], color="#0072B2", marker="s", label ="30-80 MHz")
plt.xlabel("target zenith [Deg.]")
plt.ylabel("$\sigma{(\\delta)}$")
plt.axvline(x = 80, color = 'black', linestyle = '--')
plt.axvspan(80, 90, color='orange', alpha=0.15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(y = 0.15, color = '#C44E52', linestyle = '--')
plt.axhline(y = 0.12, color = '#C44E52', linestyle = '--')
plt.title("Trigger Threshold = 110 $\mu V/m$", fontsize=12)
plt.legend()
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
plt.ylim(0.07,0.16)
#plt.savefig(savepath + "RMSrelErr_vs_Theta_trigg110.pdf", bbox_inches = "tight")
plt.show()