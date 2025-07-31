#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import hilbert
from scipy.integrate import simps



def GetRMtraces(SimulatedShower, TargetShower, efield_interpolated, IndexAll):
        
    Traces = SimulatedShower.traces
    NantRefStarshape = SimulatedShower.InitialShape # 193 antennas
    NantRefLayout = SimulatedShower.NantTraces
    NantRefCrossCheck = NantRefLayout - NantRefStarshape
    NantTargetLayout = TargetShower.nant - TargetShower.InitialShape   
    
# =============================================================================
#                  Extracting the reference Traces
# =============================================================================

    refTime, refEx, refEy, refEz = [], [], [], []
    for i in range(4*NantRefLayout):
        if((i>=0) & (i<NantTargetLayout)): 
            refTime.append(Traces[:,i])
            print(i, NantTargetLayout, "time")
        if((i>=NantRefLayout) & (i<NantRefLayout + NantTargetLayout)): \
            refEx.append(Traces[:,i])
        if((i>=2*NantRefLayout) & (i<2*NantRefLayout + NantTargetLayout)): 
            refEy.append(Traces[:,i])
        if((i>=3*NantRefLayout) & (i<3*NantRefLayout + NantTargetLayout)): 
            refEz.append(Traces[:,i])
                
# =============================================================================
#                   Loading the Target Traces
# =============================================================================
  
    refEx, refEy, refEz = np.array(refEx),  np.array(refEy),  np.array(refEz)
    error_peak,rm_peak, zh_peak, diff_time_all  = [], [], [], []
    #print(np.shape(refTime)), print(np.shape(refEx))
    #print(NantTargetLayout)
    RMtime, RMx, RMy, RMz, index = [], [], [], [], []
    
    for i in range(NantTargetLayout):

        # We load the Target traces
        Load = True
        try:
           # targetTime, targetEx, targetEy, targetEz = np.loadtxt(\
           #"./OutputDirectory/DesiredTraces_%d.txt"%i, unpack = True)
            targetTime, targetEx, targetEy, targetEz = \
            efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
            efield_interpolated[i][:,2], efield_interpolated[i][:,3]
        except(IndexError):
            Load = False
            rm_peak.append(np.nan)
            zh_peak.append(np.nan)
            error_peak.append(np.nan)
            #print("OULAH")
        if(Load):
            targetTime, targetEx, targetEy, targetEz = \
            efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
            efield_interpolated[i][:,2], efield_interpolated[i][:,3]
            
            #print("target", np.shape(targetTime))
            RMtime.append(targetTime)
            RMx.append(targetEx)
            RMy.append(targetEy)
            RMz.append(targetEz)
            index.append(i)
            #t, targetEx, targetEy, targetEz = np.loadtxt(\
            #"./OutputDirectory/DesiredTraces_%d.txt"%i, unpack = True)

    return RMtime, RMx, RMy, RMz, index, refTime, refEx, refEy, refEz




def ScalingcheckNew(TargetShower, SimulatedShower, efield_interpolated, OutputPath):
    
    NantStsp  = 176

    ### We get the traces and postions of the simulated ZHS shower in the shower plane
    posSP_sim, SimulatedTraces = SimulatedShower.GetinShowerPlane()
    vxb_sim, vxvxb_sim = posSP_sim[:,1], posSP_sim[:,2]

    plt.scatter(vxb_sim, vxvxb_sim)
    plt.show()
    #sys.exit()
   
    
    ### We get the traces and postions of the RM shower in the shower plane
    # We keep only the positions and traces resulting from the RM
    vxb_scaled, vxvxb_scaled = vxb_sim, vxvxb_sim 

    plt.scatter(vxb_scaled, vxvxb_scaled)
    plt.show()
    
    eta_scaled  = np.arctan2(vxvxb_scaled, vxb_scaled)
    print("w_scaled, w_sim")    
    
    w_sim = SimulatedShower.get_w()
    w_scaled =  SimulatedShower.get_w()

    # ZHS traces
    Traces = SimulatedShower.traces
    NantRefLayout = SimulatedShower.NantTraces
    NantTargetLayout = TargetShower.nant - TargetShower.InitialShape
    refTime, refEx, refEy, refEz = [], [], [], []

    for i in range(4*NantRefLayout):
        if((i>=0) & (i<NantTargetLayout)): 
            refTime.append(Traces[:,i])
            #print(i, NantTargetLayout, "time")
        if((i>=NantRefLayout) & (i<NantRefLayout + NantTargetLayout)): \
            refEx.append(Traces[:,i])
        if((i>=2*NantRefLayout) & (i<2*NantRefLayout + NantTargetLayout)): 
            refEy.append(Traces[:,i])
        if((i>=3*NantRefLayout) & (i<3*NantRefLayout + NantTargetLayout)): 
            refEz.append(Traces[:,i])
    
    refEx, refEy, refEz = np.array(refEx),  np.array(refEy),  np.array(refEz)

    refEmax = np.zeros(len(refEx))
    for i in range(len(refEx)):
        refEtot =  np.sqrt(refEx[i]**2 + refEy[i]**2 + refEz[i]**2)
        refEmax[i] = max(refEtot)


    ## RM traces
    RMtime, RMx, RMy, RMz, index, RMmax = [], [], [], [], [], []
    
    print(NantTargetLayout)
    for i in range(NantTargetLayout):

        # We load the Target traces
        Load = True
        try:
           # targetTime, targetEx, targetEy, targetEz = np.loadtxt(\
           #"./OutputDirectory/DesiredTraces_%d.txt"%i, unpack = True)
            targetTime, targetEx, targetEy, targetEz = \
            efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
            efield_interpolated[i][:,2], efield_interpolated[i][:,3]
        except(IndexError):
            Load = False
            #print("OULAH")
        if(Load):
            targetTime, targetEx, targetEy, targetEz = \
            efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
            efield_interpolated[i][:,2], efield_interpolated[i][:,3]
            
            #print("target", np.shape(targetTime))
            RMtime.append(targetTime)
            RMx.append(targetEx)
            RMy.append(targetEy)
            RMz.append(targetEz)
            index.append(i)

            RMtot = np.sqrt(targetEx**2 + targetEy**2 + targetEz**2)
            RMmax.append(max(RMtot))
        else:
            RMmax.append(np.nan)

    
    RMmax = np.array(RMmax)
    
    
    ### PLOTS ###
    plt.scatter(vxb_scaled, vxvxb_scaled, c = RMmax, cmap = "jet")
    plt.show()
    RMtime, RMx, RMy, RMz = \
    np.array(RMtime), np.array(RMx), np.array(RMy), np.array(RMz)

    sel_vxb, sel_vxvxb = GetAntIdShowerPlane(176, Nstsp=160)
    w_selvxb = w_sim[sel_vxb]
    w_selvxb[1::2] *= -1
    RMmax_selvxb =  RMmax[sel_vxb]
    refEmax_selvxb = refEmax[sel_vxb]

    arg = np.argsort(w_selvxb)
    mask = ~np.isnan(RMmax_selvxb[arg])
    IntLDFRM = np.trapz(w_selvxb[arg][mask], RMmax_selvxb[arg][mask])
    IntLDFZHS = np.trapz(w_selvxb[arg], refEmax_selvxb[arg])
    print(IntLDFRM, IntLDFZHS)
    RelErr = (IntLDFZHS- IntLDFRM)/IntLDFZHS*100
    print(RelErr)
    plt.plot(w_selvxb[arg], refEmax_selvxb[arg], label = "ZHAireS", linewidth=3)
    plt.plot(w_selvxb[arg], RMmax_selvxb[arg], label = "RM", linewidth=3, linestyle='--')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xlabel("$\omega$ [Degrees]")
    plt.ylabel("E [$\mu V/m$]")
    plt.legend()
    plt.text(0.05, 0.95, fr'$\Delta I / I = {RelErr:.2f}\%$', 
         transform=plt.gca().transAxes,
         verticalalignment='top', fontsize=13)
    #plt.savefig(OutputPath + "/Escaling_Eref3.98_Etarget1.0_without_scaling.pdf", bbox_inches = "tight")
    #plt.savefig(OutputPath + "/Escaling_Eref3.98_Etarget1.0_with_scaling.pdf", bbox_inches = "tight")
    #plt.savefig(OutputPath + "/Phiscaling_Phiref90_Phitarget0_with_scaling.pdf", bbox_inches = "tight")
    #plt.savefig(OutputPath + "/Phiscaling_Phiref90_Phitarget0_without_scaling.pdf", bbox_inches = "tight")
    plt.show()


    w_selvxvxb = w_sim[sel_vxvxb]
    w_selvxvxb[1::2] *= -1
    RMmax_selvxvxb =  RMmax[sel_vxvxb]
    refEmax_selvxvxb = refEmax[sel_vxvxb]

    arg = np.argsort(w_selvxvxb)
    mask = ~np.isnan(RMmax_selvxvxb[arg])
    IntLDFRM = np.trapz(w_selvxvxb[arg][mask], RMmax_selvxvxb[arg][mask])
    IntLDFZHS = np.trapz(w_selvxvxb[arg], refEmax_selvxvxb[arg])
    print(IntLDFRM, IntLDFZHS)
    RelErr = (IntLDFZHS- IntLDFRM)/IntLDFZHS*100
    print(RelErr)
    plt.plot(w_selvxvxb[arg], refEmax_selvxvxb[arg], label = "ZHAireS", linewidth=3)
    plt.plot(w_selvxvxb[arg], RMmax_selvxvxb[arg], label = "RM", linewidth=3, linestyle='--')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xlabel("$\omega$ [Degrees]")
    plt.ylabel("E [$\mu V/m$]")
    plt.legend()
    plt.text(0.05, 0.95, fr'$\Delta I / I = {RelErr:.2f}\%$', 
         transform=plt.gca().transAxes,
         verticalalignment='top', fontsize=13)
    #plt.savefig(OutputPath + "/Escaling_Eref3.98_Etarget1.0_without_scaling.pdf", bbox_inches = "tight")
    #plt.savefig(OutputPath + "/Escaling_Eref3.98_Etarget1.0_with_scaling.pdf", bbox_inches = "tight")
    #plt.savefig(OutputPath + "/Phiscaling_Phiref90_Phitarget0_with_scaling_vxvxB.pdf", bbox_inches = "tight")
    #plt.savefig(OutputPath + "/Phiscaling_Phiref90_Phitarget0_without_scaling_vxvxB.pdf", bbox_inches = "tight")
    plt.show()

    LDFfluctRM = np.array([w_selvxvxb[arg], RMmax_selvxvxb[arg]])
    LDFfluctZHS = np.array([w_selvxvxb[arg], refEmax_selvxvxb[arg]])
    ### Saving Data for Shower to Shower fluctuations
    np.savetxt("/Users/chiche/Desktop/RadioMorphingUptoDate/RMarticle/Data/Data3RM.txt", LDFfluctRM)
    np.savetxt("/Users/chiche/Desktop/RadioMorphingUptoDate/RMarticle/Data/DataZHS.txt", LDFfluctZHS)
    ###

    print(len(index))
    sys.exit()

        
    #Ev_scaled, Evxb_scaled, Evxvxb_scaled,Etot_scaled =\
          #GetHilbertPeakfromTraces(TargetTraces, n)
    
    #Ev_sim, Evxb_sim, Evxvxb_sim, Etot_sim =\
          #GetHilbertPeakfromTraces(SimulatedTraces, n)

    
    #sel_vxb, sel_vxvxb = GetAntIdShowerPlane(n, Nstsp=160)

    #wvxb_sim, wvxvxb_sim = GetwSP(w_sim, sel_vxb, sel_vxvxb, Narm=40)
    #wvxb_scaled, wvxvxb_scaled = GetwSP(w_scaled, sel_vxb, sel_vxvxb, Narm=40)
    
# =============================================================================
#                        LDF comparison
# ============================================================================
    
    # LDF of antennas on the vxb axis
    #LDFvxb_sim, LDFvxb_scaled, ILdfvxb = plot_LDF_vxb(Etot_scaled, Etot_sim, sel_vxb,  \
    #wvxb_scaled, wvxb_sim, SimulatedShower, TargetShower, RefShower)
    
    '''
    # LDF of antennas on the vxvxb axis
    LDFvxvxb_sim, LDFvxvxb_scaled, ILdfvxvxb = plot_LDF_vxvxb(Etot_scaled, Etot_sim, \
    sel_vxvxb, wvxvxb_scaled, wvxvxb_sim, SimulatedShower, TargetShower, RefShower)
    
# =============================================================================
#                     LDF: relative deviations
# =============================================================================

    LDF_relative_deviation(wvxb_scaled, wvxb_sim, wvxvxb_scaled, wvxvxb_sim, \
    LDFvxb_scaled, LDFvxb_sim, LDFvxvxb_scaled, LDFvxvxb_sim, SimulatedShower, TargetShower, RefShower)
    
# =============================================================================
#                              All antennas
# =============================================================================
# =============================================================================
#                           Relative deviation
# =============================================================================

# TODO : use the antennas position to check if we are comparing antennas that are close
            
    w_scaledAll = w_scaled[0:160]
    w_simAll =  w_sim[0:160]      
    
    Etot_all = []
    w_diff = []

    for i in range(len(w_scaledAll)):
        
        #minimum = np.argmin(abs(r_sim[i] - r_target))
        wdiff = abs(w_scaledAll - w_simAll[i])
        weta = abs(eta_scaled - eta_sim[i])*180/np.pi
        for j in range(len(wdiff)):
            if((weta[j]<5) & (wdiff[j] <0.05)):
                #if(i == 24): print(j)
                Etot_all.append((Etot_scaled[j] - Etot_sim[i])/Etot_sim[i])
                w_diff.append(w_simAll[i])
    

# =============================================================================
#                          Total integral
# =============================================================================

    positions  = SimulatedShower.pos  
    core = SimulatedShower.get_center()
    x, y, z  = positions[:,0], positions[:,1], positions[:,2]
    x,y,z = x- core[0], y - core[1], z - core[2]
    r = np.sqrt(x**2 + y**2 + z**2) 
    
    #plt.scatter(x,y)

    
    
    arg = np.argsort(r[0:160])
    r = r[0:160][arg]
          
    Etot_sim = Etot_sim[0:160][arg]
    
 
        
      
    abscisse = r
    abscisse = abscisse
    y = Etot_sim**2
    Integral_sim = scipy.integrate.trapz(y*abs(abscisse), x = abscisse) 
    
    positions2  = TargetShower.pos  
    core2 = TargetShower.get_center()
    x2, y2, z2  = positions2[:,0], positions2[:,1], positions2[:,2]
    x2, y2, z2 = x2 - core2[0], y2 - core2[1], z2 - core2[2]
    r2 = np.sqrt(x2**2 + y2**2 + z2**2) 
    
    #plt.scatter(x2, y2)
   # plt.show()
    

    
        
    arg = np.argsort(r2[0:160])
    r2 = r2[0:160][arg]
    Etot_scaled = Etot_scaled[0:160][arg]
      
    abscisse = r2
    y = Etot_scaled**2
    plt.show()
    #y = y[0:160]
    abscisse = abscisse[0:160]
    Integral_scaled = scipy.integrate.trapz(y*abs(abscisse), x = abscisse) 
    DeltaI = (Integral_sim-Integral_scaled)/Integral_sim
    
        
    
    plt.scatter(w_diff, Etot_all)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("relative deviation $E_{tot}$")
    plt.legend(["$\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.tight_layout()
    plt.savefig(path_all_array + "Etot_reldiff_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(RefShower.energy, RefShower.zenith, RefShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))   
    plt.show()
    
    
    plt.scatter(w_scaled[0:160], Etot_scaled[0:160])
    plt.scatter(w_sim[0:160], Etot_sim[0:160])
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("E $[\mu V/m]$")
    plt.legend(["scaled", "simulated, $\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.tight_layout()
    plt.savefig(path_all_array + "Etot_comparison_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(RefShower.energy, RefShower.zenith, RefShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))    
    plt.show()
    '''
    
    return #ILdfvxb, ILdfvxvxb, DeltaI



def GetHilbertPeakfromTraces(TargetTraces, Nant):

    Ex, Ey, Ez, Etot = (np.zeros(Nant) for _ in range(4))
    for i in range(Nant):
        
        Ex[i]=(max(abs(hilbert(TargetTraces[:,Nant + i])))) 
        Ey[i]=(max(abs(hilbert(TargetTraces[:,2*Nant + i]))))
        Ez[i] =(max(abs(hilbert(TargetTraces[:,3*Nant + i]))))
        Etot[i] = np.sqrt(Ex[i]**2 + Ey[i]**2 + Ez[i]**2)

    return Ex, Ey, Ez, Etot

def GetAntIdShowerPlane(Nant, Nstsp):

    AntIds = np.arange(0,Nant,1)
    # number of antennas without the cross check antennas 
    #     
    sel_vxb = (AntIds<Nstsp) & ((AntIds)%4==0)
    sel_vxvxb = (AntIds<Nstsp) & ((AntIds-2)%4==0)

    return sel_vxb, sel_vxvxb

def GetwSP(w, sel_vxb, sel_vxvxb, Narm):

    AntIDArm = np.arange(0,40,1)

    # w of antennas on the vxb axis
    wvxb =  w[sel_vxb]
    wvxb[AntIDArm%2==0] = -wvxb[AntIDArm%2==0]
    
    # w of antennas on the vxvxb axis
    wvxvxb = w[sel_vxvxb]
    wvxvxb[AntIDArm%2==0] = -wvxvxb[AntIDArm%2==0]

    return wvxb, wvxvxb