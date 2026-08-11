
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from ModuleTestRM import get_ZenithCut, AverageOnZenith, \
AverageOnZenithOneAzimuth,AverageOnOmega, AverageOnRefPeak, AverageOnDplane,\
AverageOnDplaneRel, AverageOnOmegaZenithCut
from PlotTestRM import PlotMeanRMS, PlotMeanRMSPlaneDistance, PlotRMSDistrib, GetMeanRMSerr, PlotMeanErr, PlotRMSvsTheta, GetMeanRMSerrClean, GetMeanRMSerrE
import pickle

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

savepath= "/Users/chiche/Desktop/RadioMorphing/RadioMorphingUptoDate/RMTest/RMTestPerformances/Figures/"
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

#path8 = filteredpath + "error_all_correlation_flucene_raw_notrigg.pkl"
path8 = filteredpath + "error_all_correlation_raw_trigg_peak_squared.pkl"

def PlotMeanErr(ZenithFilteredcut, Meanerr_filtered, savepath):
    ref_zen = np.array([67.8, 74.8, 77.4, 79.5, 86.5])
    argrefzen = np.where(np.isin(ZenithFilteredcut, ref_zen))[0]
    notargrefzen = np.where(~np.isin(ZenithFilteredcut, ref_zen))[0]
    print(ZenithFilteredcut[argrefzen])
    print(ZenithFilteredcut[notargrefzen])
    print(ZenithFilteredcut)
    #plt.figure()
    #plt.scatter(ZenithFilteredcut,  Meanerr_filtered)
    plt.scatter(ZenithFilteredcut[argrefzen], Meanerr_filtered[argrefzen], marker ="x", color = '#0072B2', s = 65, label = "$\\theta^{t} = \\theta^{\\rm ref}$")
    plt.scatter(ZenithFilteredcut[notargrefzen], Meanerr_filtered[notargrefzen], marker ="x", color ="#E69F00", s = 60, label = "$\\theta^{t} \\neq \\theta^{\\rm ref}$")
    #plt.ylim(-0.15, 0.15)
    plt.ylabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$")
    plt.xlabel("target zenith [Deg.]")
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.axhline(0, color='#B22222', linestyle='--', alpha=0.6)
    plt.savefig(savepath + "MeanvsThetaRelError.pdf")
    plt.show()
    return
pathAll = [path1, path2, path3, path4, path5, path6, path7, path8]
RMSerrAll= dict()
for i in range(len(pathAll)):
    with open(pathAll[i], "rb") as f:
        error_all = pickle.load(f)

    energy_threshold = 0.12
    #ZenithFilteredcut, Meanerr_filtered, RMSerr_filtered =\
    #    GetMeanRMSerr(EnergyFiltered, ZenithFiltered, error_all, energy_threshold, filteredpath, PLOT=False)
    ZenithFilteredcut, Meanerr_filtered, RMSerr_filtered =\
        GetMeanRMSerr(EnergyFiltered, ZenithFiltered, error_all, energy_threshold, filteredpath, PLOT=False)

    if(i==7):
        print(Meanerr_filtered)
        PlotMeanErr(ZenithFilteredcut, Meanerr_filtered, savepath)

        PlotRMSvsTheta(ZenithFilteredcut, RMSerr_filtered, savepath)


    RMSerrAll[i] = RMSerr_filtered
############
#referee comment fluence
RMSerrAll= dict()
for i in range(len(pathAll)):
    with open(pathAll[i], "rb") as f:
        error_all = pickle.load(f)

    energy_threshold = 0.12
    #ZenithFilteredcut, Meanerr_filtered, RMSerr_filtered =\
    #    GetMeanRMSerr(EnergyFiltered, ZenithFiltered, error_all, energy_threshold, filteredpath, PLOT=False)
    EnergyFilteredcut, Meanerr_filtered, RMSerr_filtered =\
        GetMeanRMSerrE(EnergyFiltered, ZenithFiltered, error_all, energy_threshold, filteredpath, PLOT=False)

    if(i==1):
        plt.errorbar(EnergyFilteredcut[1:], Meanerr_filtered[1:], yerr=RMSerr_filtered[1:], fmt='o', capsize=5)
    if(i==7):
        EnergyBins= EnergyFilteredcut[1:]
        MeanErrBins = Meanerr_filtered[1:]
        RMSerrBins = RMSerr_filtered[1:]

        np.savetxt(savepath + "EnergyRelErr.txt", np.column_stack((EnergyBins, MeanErrBins, RMSerrBins)))

    RMSerrAll[i] = RMSerr_filtered

plt.errorbar(EnergyBins, MeanErrBins, yerr=RMSerrBins, fmt='o', capsize=5)

#############

sys.exit()
### Plot Zenith Distributions ###
for i in range(len(pathAll)):
    with open(pathAll[i], "rb") as f:
        error_all = pickle.load(f)

    energy_threshold = 0.12

    ZenithFilteredcut = np.unique(ZenithFiltered)
    Meanerr_filtered = np.zeros(len(ZenithFilteredcut))
    RMSerr_filtered = np.zeros(len(ZenithFilteredcut))
    ZenithDistrib = [ZenithFilteredcut[0], ZenithFilteredcut[3], ZenithFilteredcut[6]]
   
    if(i==7):
        err_1d_distrib = dict()
        for i in range(len(ZenithFilteredcut)):
            indices = np.where((ZenithFiltered == ZenithFilteredcut[i]) & (EnergyFiltered>energy_threshold))[0]
            err_zen = {k: error_all.get(k) for k in indices if k in error_all}
            err_1d = [val for arr in err_zen.values() for val in arr]
            err_1d = np.array(err_1d)
            
            Meanerr_filtered[i] = np.mean(err_1d)
            print("Zenith: %.1f, Mean: %.4f" %(ZenithFilteredcut[i], Meanerr_filtered[i]*100))
            if ZenithFilteredcut[i] in ZenithDistrib:
                err_1d_distrib[ZenithFilteredcut[i]] = err_1d

            PlotRMSDistrib(err_1d, ZenithFilteredcut[i], filteredpath)

fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(err_1d_distrib)))
for (zenith, err_1d), color in zip(err_1d_distrib.items(), colors):
    ax.hist(
        err_1d ,          # if errors are fractions -> percent
        bins=30,
        histtype="step",
        linewidth=1.8,
        label=rf"$\theta = {zenith:.1f}^\circ$",
        color=color
    )
ax.set_xlabel("$\delta = (E^{ZHS} - E^{RM})/E^{ZHS}$", fontsize=14)
ax.set_ylabel("Counts", fontsize=14)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.xlim(-0.5, 0.5)
plt.legend()
#plt.savefig(savepath + "delta_Distrib_vs_zenith.pdf", bbox_inches="tight")
plt.show()


######

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
