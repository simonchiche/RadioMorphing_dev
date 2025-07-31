import numpy as np
import glob
import os

def CleanFolders():
    
    files = glob.glob('/Users/chiche/Desktop/Thesis/RMDesktop/RadioMorphing/RadioMorphingUptoDate/OutputDirectory/*')
    for f in files:
        os.remove(f)