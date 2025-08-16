import numpy as np
import glob
from coreRadiomorphing_ground import process
import sys
import os
from Modules.ModuleCleanRM import CleanFolders
    

def run(energy, zenith, azimuth, simxmax):
    ###simxmax only for the tests
    
    #Cleaning Output folder
    CleanFolders()

    # Settings of the radiomorphing
    # folder containing your reference shower simulations
    sim_dir = glob.glob("./../Simulations/*.hdf5")
    # folder which will contain radio morphed traces afterwards
    out_dir = glob.glob("./OutputDirectory")
    # Path to the list of antenna positions you would like to simulate, 
    #stored in DesiredPositions
    antennaPath = glob.glob("./../DesiredPositions/desired_pos.txt")[0]

    params = {}
    
    with open("./../ShowerInputs/Shower.inp") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split(":", 1)
            value = value.split("#")[0].strip()  # remove inline comment
            params[key.strip()] = value
    
    fluctuations = params["Fluctuations"] == "True"
    # definition of target shower parameters
    shower = {
        "primary" : params["Primary"], #"Iron",              # primary (Proton or Iron)
        "energy" : float(params["Energy"]), #energy,               # EeV
        "zenith" : 180 - float(params["Zenith"]), #180 - zenith,         # deg (GRAND frame)
        "azimuth" : (180 + float(params["Azimuth"])) % 360, #(180 + azimuth)%360, # deg (GRAND frame)
        "injection" : 1e6,               # m (injection height)
        "altitude" : float(params["Altitude"]),              # m (altitude above sea level)
        "fluctuations" : fluctuations,           # enable shower to shower fluctuations
        "filter" : False,                # enable the traces filtering
        "antennaDir" : antennaPath,      # path to desired antennas positions
        }   
    
    # Perform the radiomorphing
    TargetShower, RefShower, efield_interpolated, w_interpolated, IndexAll = \
    process(sim_dir, shower, out_dir, simxmax)
  
    return TargetShower, RefShower, efield_interpolated, w_interpolated, IndexAll

