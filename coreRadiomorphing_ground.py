#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 02:45:28 2021

@author: chiche
"""

Local = True
Lyon = not(Local)

import h5py
import numpy as np
import hdf5fileinout as hdf5io
from scaling import myscale
from interpolation_ground import do_interpolation_hdf5
import sys
import copy
from select_plane_ground import select_plane_ground, print_plane
if(Local):
    sys.path.append("./InterpolationTest")
elif(Lyon): 
    sys.path.append("/sps/trend/chiche/RadiomorphingUptoDate/InterpolationTest")
from module_signal_process import filter_traces
from ShowerClass import Shower, extractData, CerenkovStretch
import matplotlib.pyplot as plt

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
    #sys.exit()
    TestOnePlane = False
    #print(IndexAll)
    #print(DplaneRefAll)
    
    if(TestOnePlane):
        print("Enter Plane Index")
        PlaneIndex = int(input())
        # condition pour choisir un plan qui faisait déjà partie de la 
        # selection de base
        # Condition: si l'indice du plan est plus petit que la longueur de la liste
        if(PlaneIndex<len(IndexAll)): 
            # les positions sélectionées sont les position déjà pré-sélectionnées, associées à ce plane
            IndexAllCut = IndexAll[PlaneIndex]
            #print(IndexAllCut)
        # condition pour choisir un plan qui ne faisait pas déjà partie de la 
        # selection de base. donc à priori un plan assez loin du sol.
        else:
            # Dans ce cas toutes les antennes sont sélectionnées
            IndexAllCut  = np.arange(0,176,1)
        IndexAll = np.arange(0,176,1)
        DplaneRefAll = [DplaneRefAll]  
    else: IndexAllCut = IndexAll # To check
   
    EfieldAllAntennas, wAllAntennas, IndexAllAntennas = [], [], []
    for i in range(len(DplaneRefAll)):
        
        if((len(DplaneRefAll)>=1) & (not(TestOnePlane))): #TODO: change in Lyon
            selected_plane = PathAll[i]
            dplane = np.mean(DplaneTargetAll[IndexAll[i]])
            desired_positions = desired_positionsAll[IndexAll[i]]
            #print(IndexAll[i])
        else:
            selected_plane = PathAll
            dplane = np.mean(DplaneTargetAll[IndexAll])
            desired_positions = desired_positionsAll[IndexAll]
        
        # We create the Target Shower
        #print(selected_plane, "PLANE")
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
        #sys.exit()
    # =============================================================================
    #                           Interpolation
    # =============================================================================
       
        # We perform the interpolation 
        if(TestOnePlane):
            efield_interpolated, w_interpolated = \
            do_interpolation_hdf5(TargetShower, IndexAll, VoltageTraces = None, \
            FilteredVoltageTraces = None, antennamin=0,antennamax=initial_shape-1, \
            DISPLAY=False, usetrace="efield")  
        
        else:
            efield_interpolated, w_interpolated = \
            do_interpolation_hdf5(TargetShower, IndexAll[i], VoltageTraces = None, \
            FilteredVoltageTraces = None, antennamin=0,antennamax=initial_shape-1, \
            DISPLAY=False, usetrace="efield")  
        
        
        TargetShower.distplane = dplane # TODO: remove later, only for testing
                
        EfieldAllAntennas =  EfieldAllAntennas + efield_interpolated
        wAllAntennas = wAllAntennas + w_interpolated
        IndexAllAntennas =  IndexAllAntennas + IndexAll[i]
    #print(IndexAllAntennas)
    
    Indexes = np.argsort(IndexAllAntennas)
    if(len(DplaneRefAll)>1):        
        TargetShower.nant = TargetShower.nant \
        - len(w_interpolated) + len(Indexes) 
        EfieldAllAntennas = np.array(EfieldAllAntennas, dtype=object)[Indexes]
        #EfieldAllAntennas[0:50] = EfieldAllAntennas[0:50]/1.2
        wAllAntennas = np.array(wAllAntennas, dtype=object)[Indexes]
        
    CorrectBias = False
    if(CorrectBias):
        if((RefShower.zenith - TargetShower.zenith) >0.5):
            cerangle_target = 0.75*TargetShower.get_cerenkov_angle()
            EfieldAllAntennas[wAllAntennas<cerangle_target] = \
            EfieldAllAntennas[wAllAntennas<cerangle_target]/1.25
        
        if((RefShower.zenith - TargetShower.zenith) <-0.5):
            cerangle_target = 0.75*TargetShower.get_cerenkov_angle()
            EfieldAllAntennas[wAllAntennas<cerangle_target] = \
            EfieldAllAntennas[wAllAntennas<cerangle_target]*1.25

    return TargetShower, RefShower, EfieldAllAntennas, wAllAntennas, np.array(IndexAllCut, dtype = object)
