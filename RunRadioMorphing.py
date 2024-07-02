import numpy as np
import glob
from coreRadiomorphing_ground import process
import sys
import os

Local = True
Lyon = not(Local)

def CleanFolders():
    if(Local):
        files = glob.glob('/Users/chiche/Desktop/Thesis/RMDesktop/RadioMorphing/RadioMorphingUptoDate/OutputDirectory/*')
    elif(Lyon):
        files = glob.glob('/sps/trend/chiche/RadioMophingUptoDate/OutputDirectory/*')
    for f in files:
        os.remove(f)
    

def run(energy, zenith, azimuth, simxmax):
    ###simxmax only for the tests
    CleanFolders()

    # Settings of the radiomorphing
    if(Local):
        # folder containing your reference shower simulations
        sim_dir = glob.glob("./Simulations/*.hdf5")
        # folder which will contain radio morphed traces afterwards
        out_dir = glob.glob("./OutputDirectory")
        # Path to the list of antenna positions you would like to simulate, 
        #stored in DesiredPositions
        antennaPath = glob.glob("./DesiredPositions/desired_pos.txt")[0]
    elif(Lyon):
        sim_dir = glob.glob("/sps/trend/chiche/RadiomorphingUptoDate/Simulations/*.hdf5")
        # folder which will contain radio morphed traces afterwards
        out_dir = glob.glob("/sps/trend/chiche/RadiomorphingUptoDate/OutputDirectory")
        # Path to the list of antenna positions you would like to simulate, 
        #stored in DesiredPositions
        antennaPath = glob.glob("/sps/trend/chiche/RadiomorphingUptoDate/DesiredPositions/desired_pos%.d.txt" %start)[0]
        
    # definition of target shower parameters
    shower = {
        "primary" : "Iron",              # primary (Proton or Iron)
        "energy" : energy,               # EeV
        "zenith" : 180 - zenith,         # deg (GRAND frame)
        "azimuth" : (180 + azimuth)%360, # deg (GRAND frame)
        "injection" : 1e5,               # m (injection height)
        "altitude" : 1000.,              # m (altitude above sea level)
        "fluctuations" : False,          # enable shower to shower fluctuations
        "filter" : False,                # enable the traces filtering
        "antennaDir" : antennaPath,      # path to desired antennas positions
        }   
    
    # Perform the radiomorphing
    TargetShower, RefShower, efield_interpolated, w_interpolated, IndexAll = \
    process(sim_dir, shower, out_dir, simxmax)
  
    return TargetShower, RefShower, efield_interpolated, w_interpolated, IndexAll
