import numpy as np
import glob
import matplotlib.pyplot as plt
from coreRadiomorphing_ground import extractData

def GetTargetShowerParamFromZHSsim(Simulated_path):
    Flag = False
    SimulatedShower = extractData(Simulated_path)
    np.savetxt("./../DesiredPositions/desired_pos.txt", SimulatedShower.pos)

    # We load the different target shower parameters
    energy = SimulatedShower.energy
    xmaxsim = SimulatedShower.xmaxpos
    zenith = 180 - SimulatedShower.zenith
    azimuth = float(Simulated_path.split("_")[4])

    # we check that the RM simulated shower is not already in the reference  library
    zenithfile =  float(Simulated_path.split("_")[3])
    filename = Simulated_path.split("/")[-1]
    
    path = "./Simulations/SelectedPlane/theta_%.1f/" %(zenithfile) + filename
    ref_sim =  glob.glob(path)
    if(len(ref_sim)>0): Flag = True

    return energy, zenith, azimuth, xmaxsim, SimulatedShower, Flag


# Test functions in LoopRM.py to be cleaned

"""
    plt.scatter(w_interpolated, ResidualPeak, label = "M = %.2f, R = %.2f" %(np.mean(ResidualPeak[~np.isnan(ResidualPeak)]), np.std(ResidualPeak[~np.isnan(ResidualPeak)])))
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("Relative peak error")
    plt.legend()
    plt.title("$\\theta=$%.f$\degree$" %(180-TargetShower.zenith))
    plt.tight_layout()
    #plt.savefig("/Users/chiche/Desktop/RelErrorVsOmega_theta%.f_BiasCorrected.pdf" %(180-TargetShower.zenith))
    plt.show()
    

    #print(ResidualPeak)
    #print(np.mean(abs(ResidualPeak[~np.isnan(ResidualPeak)])))

    #print(ResidualPeak[~np.isnan(ResidualPeak)])
    #print(np.std(ResidualPeak[(~np.isnan(ResidualPeak)) & (ResidualPeak<0.95)]), "HEEERE")
    #np.savetxt("error86_5.txt", ResidualPeak[~np.isnan(ResidualPeak)])
    plt.plot(ResidualPeak[~np.isnan(ResidualPeak)])
    plt.xlabel("AntennaID")
    plt.ylabel("Relative peak error")
    plt.tight_layout()
    #plt.savefig("Resdiual86_5_D150km")
    plt.show()
    Trigger = 75
    if(Trigger!=0):
        CleanRef =  np.array(RefPeak)[~np.isnan(ResidualPeak)]
        CleanRes = ResidualPeak[~np.isnan(ResidualPeak)]
        print(np.mean(CleanRes[CleanRef>Trigger]), np.std(CleanRes[CleanRef>Trigger]))
    Trigger = True
    if(Trigger):
        ErrorPeakTrigger75 = ResidualPeak[RefPeak>75]
        print(np.mean(abs(ErrorPeakTrigger75)))
"""
