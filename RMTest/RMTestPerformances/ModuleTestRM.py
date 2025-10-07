#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:56:26 2022

@author: chiche
"""


import numpy as np


def get_ZenithCut(ZenithAll):
    
    
    ZenithCut = []
    ZenithAllSort = np.sort(ZenithAll)
    CurrentZenith = ZenithAllSort[0]
    ZenithCut.append(CurrentZenith)
    
    for i in range(len(ZenithAll)):
        
        if(ZenithAllSort[i]!=CurrentZenith):
            CurrentZenith = ZenithAllSort[i]
            ZenithCut.append(CurrentZenith)   
    
    return ZenithCut

def AverageOnZenith(Observable, ZenithCut, ZenithAll):
    
    ZenithAll= ZenithAll[~np.isnan(Observable)]
    ObservableClean = Observable[~np.isnan(Observable)]
    
    N = len(ZenithCut)
    MeanObservable = np.zeros(N)
    RMSObservable = np.zeros(N)
    
    for i in range(N):
        
        MeanObservable[i] =  np.mean(ObservableClean[ZenithAll == ZenithCut[i]])
        RMSObservable[i] =  np.std(ObservableClean[ZenithAll == ZenithCut[i]])
    
    return MeanObservable, RMSObservable

def AverageOnZenithOneAzimuth(Observable, ZenithCut, ZenithAll, SelectedAzimuth, AzimuthAll):
    
    AzimuthAll= AzimuthAll[~np.isnan(Observable)]
    ZenithAll= ZenithAll[~np.isnan(Observable)]
    ZenithAll = ZenithAll[AzimuthAll == SelectedAzimuth]
    ObservableClean = Observable[~np.isnan(Observable)]
    ObservableClean = ObservableClean[AzimuthAll == SelectedAzimuth]
    
    N = len(ZenithCut)
    MeanObservable = np.zeros(N)
    RMSObservable = np.zeros(N)
    
    for i in range(N):
        
        MeanObservable[i] =  np.mean(ObservableClean[ZenithAll == ZenithCut[i]])
        RMSObservable[i] =  np.std(ObservableClean[ZenithAll == ZenithCut[i]])
    
    return MeanObservable, RMSObservable

def AverageOnOmega(OmegaAll, Observable):
    
    limit = max(OmegaAll)    
    increment = 0
    BinnedObservable = []
    BinnedRMS = []
    Bins = []
    
    OmegaAll = OmegaAll[~np.isnan(Observable)]
    ObservableClean = Observable[~np.isnan(Observable)]
    
    while(increment<limit):
        incrementUp = increment + 0.1
        Condition  = ((OmegaAll>increment) & (OmegaAll<incrementUp))
        if(len(OmegaAll[Condition])>0):
            BinnedObservable.append(np.mean(ObservableClean[Condition]))
            BinnedRMS.append(np.std(ObservableClean[Condition]))
            Bins.append(increment)
        increment = increment + 0.1
    
    return np.array(Bins), np.array(BinnedObservable), np.array(BinnedRMS)

def AverageOnOmegaZenithCut(OmegaAll, Observable, ZenithCut, ZenithAll):
    
    OmegaAllRaw = np.copy(OmegaAll) 
    ObservableRaw =np.copy(Observable) 
    
    OmegaBinsZenithCut, BinnedOmegaResidualZenithCut, BinnedOmegaRMSZenithCut = \
    [], [], []
    
    for i in range(len(ZenithCut)):
        
        OmegaAll = OmegaAllRaw[ZenithAll == ZenithCut[i]]
        Observable = ObservableRaw[ZenithAll == ZenithCut[i]]
        
        
        limit = max(OmegaAll)    
        increment = 0
        BinnedObservable = []
        BinnedRMS = []
        Bins = []
        
        OmegaAll = OmegaAll[~np.isnan(Observable)]
        ObservableClean = Observable[~np.isnan(Observable)]
        
        while(increment<limit):
            incrementUp = increment + 0.1
            Condition  = ((OmegaAll>increment) & (OmegaAll<incrementUp))
            if(len(OmegaAll[Condition])>0):
                BinnedObservable.append(np.mean(ObservableClean[Condition]))
                BinnedRMS.append(np.std(ObservableClean[Condition]))
                Bins.append(increment)
            increment = increment + 0.1
        Bins = np.array(Bins)
        BinnedObservable = np.array(BinnedObservable)
        BinnedRMS = np.array(BinnedRMS)
            
        OmegaBinsZenithCut.append(Bins) 
        BinnedOmegaResidualZenithCut.append(BinnedObservable)
        BinnedOmegaRMSZenithCut.append(BinnedRMS)
    
    return np.array(OmegaBinsZenithCut, dtype=object), np.array(BinnedOmegaResidualZenithCut, dtype=object)\
    ,np.array(BinnedOmegaRMSZenithCut, dtype=object)
        

def AverageOnRefPeak(RefPeakAll, Observable):
    
    limit = max(RefPeakAll)    
    increment = 0
    BinnedObservable = []
    BinnedRMS = []
    Bins = []
    
    RefPeakAll = RefPeakAll[~np.isnan(Observable)]
    ObservableClean = Observable[~np.isnan(Observable)]
    
    while(increment<limit):
        incrementUp = increment + 50
        Condition  = ((RefPeakAll>increment) & (RefPeakAll<incrementUp))
        if(len(RefPeakAll[Condition])>0):
            BinnedObservable.append(np.mean(ObservableClean[Condition]))
            BinnedRMS.append(np.std(ObservableClean[Condition]))
            Bins.append(increment)
        increment = increment + 50
    
    return np.array(Bins), np.array(BinnedObservable), np.array(BinnedRMS)
        

def AverageOnDplane(Dplane, Observable):
    
    limit = max(Dplane)    
    increment = 0
    BinnedObservable = []
    BinnedRMS = []
    Bins = []
    
    Dplane = Dplane[~np.isnan(Observable)]
    ObservableClean = Observable[~np.isnan(Observable)]
    
    while(increment<limit):
        incrementUp = increment + 20
        Condition  = ((Dplane>increment) & (Dplane<incrementUp))
        if(len(Dplane[Condition])>0):
            BinnedObservable.append(np.mean(ObservableClean[Condition]))
            BinnedRMS.append(np.std(ObservableClean[Condition]))
            Bins.append(increment)
        increment = increment + 20
    
    return np.array(Bins), np.array(BinnedObservable), np.array(BinnedRMS)
    
    
def AverageOnDplaneRel(Dplane, Observable):
    
    limit = max(Dplane)    
    increment = 0
    BinnedObservable = []
    BinnedRMS = []
    Bins = []
    
    Dplane = Dplane[~np.isnan(Observable)]
    ObservableClean = Observable[~np.isnan(Observable)]
    
    while(increment<limit):
        incrementUp = increment + 0.5
        Condition  = ((Dplane>increment) & (Dplane<incrementUp))
        if(len(Dplane[Condition])>0):
            BinnedObservable.append(np.mean(ObservableClean[Condition]))
            BinnedRMS.append(np.std(ObservableClean[Condition]))
            Bins.append(increment)
        increment = increment + 0.5
    
    return np.array(Bins), np.array(BinnedObservable), np.array(BinnedRMS)
        
    
    
    
    
    
     