#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:08:09 2021

@author: chiche
"""


import numpy as np
import glob
import sys

def select_zenith(target_zenith):
    
    # zenith of the target shower
    target_zenith = 180 - target_zenith # cosmic ray convention
    
    # zenith of the ref library
    # To modify if we change the ref library
    zenith_sim  = np.array([67.8, 74.8, 81.3, 83.9, 86.5])
    
    # We use as reference shower the shower with the closest zenith from the
    # targeted one
    min_index = np.argmin(abs(zenith_sim - target_zenith))
    selected_zenith =  zenith_sim[min_index]
    
    return selected_zenith


def select_azimuth(path, target_azimuth):

    sim = glob.glob(path)
    azimuth_sim = []
    n = len(sim)

    # azimuth of the target shower
    target_azimuth = abs(180  - target_azimuth)
        
    # azimuth of the ref library
    # To modify if we change the ref library
    for i in range(n):
    #azimuth_sim  = np.array([0, 90, 180])
        azRef = float(sim[i].split("_")[-3])
        azimuth_sim.append(azRef)
    
    azimuth_sim = np.array(azimuth_sim)

    # We use as reference shower the shower with the closest azimuth from the
    # targeted one
    min_index = np.argmin(abs(azimuth_sim - target_azimuth))
    selected_azimuth =  azimuth_sim[min_index]

    return selected_azimuth


def select_path(path, dplane, selected_azimuth):
    
    # Gives the path to the reference simulation
    
    sim = glob.glob(path)
    n = len(sim)
    dsim = []

    #azsim_all = np.zeros(n)    
    #for i in range(n):
        #azsimRef = float(sim[i].split("_")[-3])
        #if(azsim)
        #azsim_all[i] = (azsimRef - azsimTarget)%180    
   # argmin = np.argmin(azsim_all)
        
    for j in range(n):    
    
        azsimRef = float(sim[j].split("_")[-3])
        if(int(azsimRef) == int(selected_azimuth)):         
            dsim.append(float(sim[j].split("_")[-1][:-5]))
            
    dsim = np.array(dsim)
    index_all = np.argsort(abs(dsim - dplane))
    
    min_index = index_all[0]
    print(dsim[min_index])
    #min_index = index_all[1] si on veut tester le radiomorphing pour des 
    # plans perp en prenant plan test et plan simulé différent
        
    return sim[min_index], dsim[min_index]
        
        
def select_plane_ground(primary, energy, zenith, azimuth, \
                        injection, altitude, fluctuations, cross_check, Xmax):
    
    # We get the closest plane of antenna
    dplane = \
    get_distplane(zenith, azimuth, cross_check[:,0], cross_check[:,1], \
                  cross_check[:,2], Xmax[0], Xmax[1], Xmax[2])
    dplane = np.mean(dplane)

    # We get the closest reference zenith and and azimuth angle
    target_zenith = select_zenith(zenith)
    
    # We infer the path to the reference shower
    path = "./Simulations/SelectedPlane/theta_%.1f/*.hdf5" \
                      %(target_zenith)
    
    target_azimuth = select_azimuth(path, azimuth)
    
    selected_plane, dsim = select_path(path, dplane, target_azimuth)    
    
    return selected_plane, dplane

def get_distplane(zenith, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax):
     
    pi = np.pi
    zenith = zenith*pi/180.0
    azimuth = azimuth*pi/180.0
    
    # distance along the x-axis between the antennas postions and Xmax
    x_antenna = x - x_Xmax 
    y_antenna = y - y_Xmax
    z_antenna = z - z_Xmax
     
    # direction of the shower
    uv = np.array([np.sin(zenith)*np.cos(azimuth), \
         np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)])
    # direction of the unit vectors that goes from Xmax to 
    #the position of the antennas
    u_antenna = np.array([x_antenna, y_antenna, z_antenna]) 
    distplane = np.dot(np.transpose(u_antenna), uv)
    
    return distplane

def print_plane(RefShower, TargetShower, target_dplane):
    
    print("-----------------------")
    print("Target shower: Energy = %.2f, Zenith = %.2f,  Azimuth = %.2f, \
          Dxmax = %.2d" %(TargetShower.energy, 180 -TargetShower.zenith,\
          TargetShower.azimuth, target_dplane))
    print("")
    print("Ref shower: Energy = %.2f, Zenith = %.2f,  Azimuth = %.2f, \
          Dxmax = %.2d" %(RefShower.energy, 180 -RefShower.zenith,\
          RefShower.azimuth, RefShower.distplane))
    print("-----------------------")
    
