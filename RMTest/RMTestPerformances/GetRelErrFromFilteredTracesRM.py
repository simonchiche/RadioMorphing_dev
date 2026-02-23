
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

refangles = np.array([
    65.5, 67.8, 69.8, 74.8, 76.1, 77.4, 78.5, 79.5,
    80.4, 82.7, 83.4, 84.5, 85.0, 86.2, 86.5, 86.8
])
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

deltatheta= np.zeros(len(ZenithFilteredcut))
for i in range(len(ZenithFilteredcut)):
    deltatheta[i] = min(abs(refangles-ZenithFilteredcut[i]))
    #deltacostheta = -np.sin(np.radians(ZenithFilteredcut[i]))*np.radians(deltatheta[i])
    #deltatheta[i] = abs(deltacostheta)
    
plt.plot(ZenithFilteredcut, deltatheta, color="black", marker="x")
plt.xlabel("target zenith [Deg.]")
plt.ylabel("$\\delta_{\\theta}$ [Deg.]")
plt.show()
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


### delta theta plot


from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(7, 6))

gs = GridSpec(
    nrows=2,
    ncols=1,
    height_ratios=[1, 4],   # delta theta aplati / RMS normal
    hspace=0.05
)

ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

# ======================
# Plot delta theta (haut)
# ======================
ax_top.plot(
    ZenithFilteredcut,
    deltatheta,
    color="black",
    marker="x"
)
ax_top.set_ylabel(r"$\delta_{\theta}$ [deg]")
ax_top.tick_params(labelbottom=False)
ax_top.grid(True, linestyle="--", alpha=0.4)

# ======================
# Plot RMSerrAll (bas)
# ======================
ax_bot.plot(
    ZenithFilteredcut,
    RMSerrAll[1],
    color="black",
    marker="o",
    label="Full band"
)
ax_bot.plot(
    ZenithFilteredcut,
    RMSerrAll[3],
    color="#D55E00",
    marker="P",
    label="50–200 MHz"
)
ax_bot.plot(
    ZenithFilteredcut,
    RMSerrAll[5],
    color="#0072B2",
    marker="s",
    label="30–80 MHz"
)

ax_bot.set_xlabel("target zenith [deg]")
ax_bot.set_ylabel(r"$\sigma(\delta)$")
ax_bot.axvline(x=80, color="black", linestyle="--")
ax_bot.axvspan(80, 90, color="orange", alpha=0.15)
ax_bot.grid(True, linestyle="--", alpha=0.5)
ax_bot.set_ylim(0.07, 0.16)
ax_bot.legend()

plt.show()

# end deltha theta plot

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











# --- Combined figure: delta theta (top, flattened) + RMS (bottom)
fig = plt.figure(figsize=(7, 6))
gs = GridSpec(nrows=2, ncols=1, height_ratios=[1, 4], hspace=0.05)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

# Top panel: delta theta
ax_top.plot(ZenithFilteredcut, deltatheta, color="black", marker="x")
ax_top.set_ylabel("$\\delta_{\\theta}$ [Deg.]")
ax_top.tick_params(labelbottom=False)
ax_top.grid(True, linestyle='--', alpha=0.4)
ax_top.set_title("Trigger Threshold = 60 $\mu V/m$", fontsize=12)


# Bottom panel: RMS
ax_bot.plot(ZenithFilteredcut, RMSerrAll[1], color="black", marker="o", label ="Full band")
ax_bot.plot(ZenithFilteredcut, RMSerrAll[3], color="#D55E00", marker="P", label ="50-200 MHz")
ax_bot.plot(ZenithFilteredcut, RMSerrAll[5], color="#0072B2", marker="s", label ="30-80 MHz")
ax_bot.set_xlabel("target zenith [Deg.]")
ax_bot.set_ylabel("$\sigma{(\\delta)}$")
ax_bot.axvline(x = 80, color = 'black', linestyle = '--')
ax_bot.axvspan(80, 90, color='orange', alpha=0.15)
ax_bot.grid(True, linestyle='--', alpha=0.5)
ax_bot.axhline(y = 0.15, color = '#C44E52', linestyle = '--')
ax_bot.axhline(y = 0.12, color = '#C44E52', linestyle = '--')
ax_bot.text(67, 0.122, "12% limit", color='#C44E52', fontsize=12, va='bottom', ha='right')
ax_bot.text(67, 0.15, "15% limit", color='#C44E52', fontsize=12, va='bottom', ha='right')
ax_bot.legend()
ax_bot.set_ylim(0.07,0.16)

plt.savefig(savepath + "RMSrelErr_vs_Theta_trigg60.pdf", bbox_inches = "tight")
plt.show()

# --- Combined figure: delta theta (top, flattened) + RMS (bottom)
fig = plt.figure(figsize=(7, 6))
gs = GridSpec(nrows=2, ncols=1, height_ratios=[1, 4], hspace=0.05)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

# Top panel: delta theta
ax_top.plot(ZenithFilteredcut, deltatheta, color="black", marker="x")
ax_top.set_ylabel("$\\delta_{\\theta}$ [Deg.]")
ax_top.tick_params(labelbottom=False)
ax_top.grid(True, linestyle='--', alpha=0.4)
ax_top.set_title("Trigger Threshold = 110 $\mu V/m$", fontsize=12)


# Bottom panel: RMS
ax_bot.plot(ZenithFilteredcut, RMSerrAll[2], color="black", marker="o", label ="Full band")
ax_bot.plot(ZenithFilteredcut, RMSerrAll[4], color="#D55E00", marker="P", label ="50-200 MHz")
ax_bot.plot(ZenithFilteredcut, RMSerrAll[6], color="#0072B2", marker="s", label ="30-80 MHz")
ax_bot.set_xlabel("target zenith [Deg.]")
ax_bot.set_ylabel("$\sigma{(\\delta)}$")
ax_bot.axvline(x = 80, color = 'black', linestyle = '--')
ax_bot.axvspan(80, 90, color='orange', alpha=0.15)
ax_bot.grid(True, linestyle='--', alpha=0.5)
ax_bot.axhline(y = 0.15, color = '#C44E52', linestyle = '--')
ax_bot.axhline(y = 0.12, color = '#C44E52', linestyle = '--')
ax_bot.text(67, 0.122, "12% limit", color='#C44E52', fontsize=12, va='bottom', ha='right')
ax_bot.text(67, 0.15, "15% limit", color='#C44E52', fontsize=12, va='bottom', ha='right')
ax_bot.legend()
ax_bot.set_ylim(0.07,0.16)

plt.savefig(savepath + "RMSrelErr_vs_Theta_trigg110.pdf", bbox_inches = "tight")
plt.show()
