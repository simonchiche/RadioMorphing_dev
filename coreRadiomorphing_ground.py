#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 02:45:28 2021

@author: chiche
"""
import h5py
import numpy as np
import hdf5fileinout as hdf5io
from scaling import myscale
from interpolation_ground import do_interpolation_hdf5
import sys
import copy
from select_plane_ground import select_plane_ground, print_plane
sys.path.append("./InterpolationTest")
from module_signal_process import filter_traces
from ShowerClass import Shower, extractData, CerenkovStretch

def process(sim_dir, shower,  out_dir, simxmax):
    """Rescale and interpolate the radio traces for all antennas 
        - start the Radio Morphing procedure

    Parameters:
    ----------
        sim_dir: str
            path to the simulated traces
        shower: dict
            properties of the requested shower
        antennas: str
            path the requested antenna positions
        out_dir: str
            path where the output traces should be dumped
    """
    # Rescale the simulated showers to the requested one
    
        
# =============================================================================
#                         Shower building
# =============================================================================
    
    # We get the desired positions    
    desired_positionsAll =  np.loadtxt(shower["antennaDir"])
    
    # We select the best antenna plane for the interpolation
    IndexAll, DplaneRefAll, PathAll, DplaneTargetAll = select_plane_ground(shower["primary"], \
    shower["energy"], shower["zenith"], shower["azimuth"], \
    shower["injection"], shower["altitude"],shower["fluctuations"], \
    desired_positionsAll, simxmax, shower)
    
    print(DplaneRefAll)
    IndexAll = np.arange(0,176,1)
    #DplaneRefAll = DplaneRefAll[8]
    #PathAll = PathAll[8]     
    #sys.exit()
    
    
    EfieldAllAntennas, wAllAntennas = [], []
    
    for i in range(1):#len(PathAll)):
        
        selected_plane = PathAll#[i]
        dplane = np.mean(DplaneTargetAll[IndexAll])#[i]])
        print("yeaaah", dplane)
        #sys.exit()
        desired_positions = desired_positionsAll[IndexAll]#[i]]
        
        
        # We create the Target Shower
        RefShower = extractData(selected_plane)
        Nant = RefShower.nant
        TargetShower = copy.deepcopy(RefShower) 
        TargetShower.primary = shower['primary']
        TargetShower.energy = shower['energy']
        TargetShower.zenith = shower['zenith']
        TargetShower.azimuth = shower['azimuth']
        TargetShower.injection = shower['injection']
        TargetShower.fluctuations = shower['fluctuations']
        TargetShower.filter = shower["filter"]
        print_plane(RefShower, TargetShower, dplane)
        
    # =============================================================================
    #                              Scaling
    # =============================================================================
    
        TargetShower, krho_geo = myscale(RefShower, TargetShower, simxmax)
        
    # =============================================================================
    #                          Preparing Interpolation      
    # =============================================================================
       #test
        ### à supprimer ?
        TargetShower.pos, TargetShower.traces[:,Nant:], ks = \
        CerenkovStretch(RefShower, TargetShower)
        TargetShower.pos, TargetShower.traces = TargetShower.GetinGeographicFrame()
    
    # =============================================================================
    #                           antennas affectation
    # =============================================================================
        
        # Desired antennas positions
        x, y, z = \
        desired_positions[:,0], desired_positions[:,1], desired_positions[:,2]
        desired_shape = len(x)
        initial_shape = RefShower.InitialShape
        
        # Antennas position assigned to the target shower
        target_x, target_y, target_z = \
        np.zeros(initial_shape + desired_shape),np.zeros(initial_shape \
                + desired_shape), np.zeros(initial_shape + desired_shape)
        
        # The first antennas are the reference antennas used for the interpolation
        target_x[0:initial_shape] = TargetShower.pos[0:initial_shape,0]
        target_y[0:initial_shape] = TargetShower.pos[0:initial_shape,1]
        target_z[0:initial_shape] = TargetShower.pos[0:initial_shape,2]
        
        # The second antenna group corresponds to the desired antennas positions
        target_x[initial_shape:] = x
        target_y[initial_shape:] = y
        target_z[initial_shape:] = z
        
        # We affect the new positions to the Target Shower
        NewPos = np.transpose(np.array([target_x, target_y, target_z]))
        TargetShower.pos = NewPos
        TargetShower.nant = len(NewPos[:, 0])
    
    # =============================================================================
    #                          Traces filtering
    # =============================================================================
    
        pre_filtering = TargetShower.filter
        if pre_filtering:
            time_sample = int(len(TargetShower.traces[:,0]))
            TargetShower.traces = \
            filter_traces(TargetShower.traces, TargetShower.NantTraces, time_sample)
    
    # =============================================================================
    #                           Interpolation
    # =============================================================================
       
        # We perform the interpolation 
        efield_interpolated, w_interpolated = \
        do_interpolation_hdf5(TargetShower, VoltageTraces = None, \
        FilteredVoltageTraces = None, antennamin=0,antennamax=initial_shape-1, \
        DISPLAY=False, usetrace="efield")  
        
        TargetShower.distplane = dplane # TODO: remove later, only for testing
        
        print("we are here!")
        
        EfieldAllAntennas =  EfieldAllAntennas + efield_interpolated
        wAllAntennas = wAllAntennas + w_interpolated
        
    return TargetShower, RefShower, EfieldAllAntennas, wAllAntennas
