"""
Created on Sun Apr 18 02:24:48 2021
@author: chiche

Script to make the scaling plots of the Radio Morphing article
"""

import numpy as np
from ModuleFilter import LoadData

MainDir = "RMresults_11_12_24" 
SaveDir = "E1_th81_phi0_0"
path = "/Users/chiche/Desktop/RadioMorphingUptoDate/RMFilterTests/Traces/"
ZHStime, ZHSx, ZHSy, ZHSz, RMtime, RMx, RMy, RMz, index, Nant =  LoadData(SaveDir, path)