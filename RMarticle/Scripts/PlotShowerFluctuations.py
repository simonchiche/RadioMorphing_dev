import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

RMarticlePath = "/Users/chiche/Desktop/RadioMorphingUptoDate/RMarticle"
OutputPath =  RMarticlePath + "/figures/ShowerFluctuations/"
DataPath = "/Users/chiche/Desktop/RadioMorphingUptoDate/RMarticle/Data/"
w_vxvxb, LDFvxvxb_RM1 = np.loadtxt(DataPath + "Data1RM.txt", unpack=True).T
w_vxvxb, LDFvxvxb_RM2 = np.loadtxt(DataPath + "Data2RM.txt", unpack=True).T
w_vxvxb, LDFvxvxb_RM3 = np.loadtxt(DataPath + "Data3RM.txt", unpack=True).T

w_vxvxb, LDFvxvxb_ZHS = np.loadtxt(DataPath + "DataZHS.txt", unpack=True).T


plt.plot(w_vxvxb, LDFvxvxb_RM3, marker='o',  label = "Seed 1")
plt.plot(w_vxvxb, LDFvxvxb_RM1, marker='^', label = "Seed 2")
plt.plot(w_vxvxb, LDFvxvxb_RM2, marker='+', label = "Seed 3")
plt.grid()
plt.legend()
plt.xlabel("$\omega$ [Deg.]")
plt.ylabel("Lateral distribution function $[\mu $V/m]")
plt.xlim(-2,2)
plt.savefig(OutputPath + "ShowertoShowerFluctuationsRM_zenith_E1_zenith75.pdf", bbox_inches = "tight")
plt.show()





######## 
# Functions and script initially to show shower to shower fluctuations with histos
###########

def PlotAmplitudeDistribution(Etot1, Etot2, Etot3, bin_edges, labels, scale = "linear"):


    plt.hist(Etot1, bin_edges, alpha=0.6, edgecolor='black', label=labels[0])
    plt.hist(Etot2, bin_edges, alpha=0.6, edgecolor='black', label=labels[1])
    plt.hist(Etot3 , bin_edges, alpha=0.6, edgecolor='black', label=labels[2])
    plt.xlabel('$E_{tot}\, [\mu V s/m]$')
    plt.ylabel('Nant')
    #plt.xlim(0,2000)
    plt.legend()
    if(scale=="log"): plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.title(r"In-air, $\theta =0^{\circ}$, $E=10^{17.5} eV$")
    #plt.savefig("/Users/chiche/Desktop/InAirFilteredPulseDistrib.pdf", bbox_inches="tight")
    plt.show()

#labels = ['1', '2', '3']
#bin_edges =  np.linspace(0, 5000, 20)
#PlotAmplitudeDistribution(RMdata1, RMdata2, RMdata3, bin_edges, labels, scale = "log")