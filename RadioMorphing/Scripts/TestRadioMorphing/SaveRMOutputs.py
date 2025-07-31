import numpy as np


def SaveRMOutputs(RefTheta,TargetTheta, RefEnergy,TargetEnergy, RefPhi,TargetPhi, DplaneRef, DplaneTarget, ResidualPeakAll, TargetPeakAll, RefPeakAll, OmegaAll, start, Local):

    Lyon = not(Local)

    # Ref and Target parameters
    if(Local):
        np.savetxt("RefTargetParameters.txt", np.transpose([RefTheta,TargetTheta, \
        RefEnergy,TargetEnergy, RefPhi,TargetPhi, DplaneRef, DplaneTarget]))
        
        # LDF error
        #np.savetxt("LDFScalingTest.txt", np.transpose(\
        #[ILDFvxbAll, ILDFvxvxbAll, ItotAll]))
        
        #peak error, TODO: inclure peaktime
        np.savetxt("PeakResidual.txt", ResidualPeakAll)
        np.savetxt("PeakTarget.txt", TargetPeakAll)
        np.savetxt("PeakRef.txt", RefPeakAll)
        np.savetxt("OmegaAngle.txt", OmegaAll)

    elif(Lyon):
        np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/RefTargetParameters%.2d.txt" %start, np.transpose([RefTheta,TargetTheta, \
        RefEnergy,TargetEnergy, RefPhi,TargetPhi, DplaneRef, DplaneTarget]))
        
        # LDF error
        #np.savetxt("LDFScalingTest.txt", np.transpose(\
        #[ILDFvxbAll, ILDFvxvxbAll, ItotAll]))
        
        #peak error, TODO: inclure peaktime
        np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/PeakResidual%.2d.txt" %start, ResidualPeakAll)
        np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/PeakTarget%.2d.txt" %start, np.array(TargetPeakAll))
        np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/PeakRef%.2d.txt" %start, np.array(RefPeakAll))
        np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/RMTestResults/OmegaAngle%.2d.txt" %start, np.array(OmegaAll))

        return