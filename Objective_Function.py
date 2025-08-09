import numpy as np
import Glob_Vars
from Model_MWFF_CNN import Model_MWFF_CNN
from Model_PWLCM import Model_PLCM


def Objfun_Cls(Soln):
    Feat1 = Glob_Vars.Feat1
    Feat2 = Glob_Vars.Feat2
    Feat3 = Glob_Vars.Feat3
    Targ = Glob_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        weight = sol  # For Proposed Only
        Feat = np.concatenate((Feat1, Feat2, Feat3), axis=1)
        Weighted_Feature_Fusion = Feat * weight
        Eval = Model_MWFF_CNN(Weighted_Feature_Fusion, Targ)
        Fitn[i] = 1 / Eval[4]
    return Fitn


def Objective_Crypto(Soln):
    Data = Glob_Vars.Data
    Target = Glob_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Eval = Model_PLCM(Data, sol.astype('int'))
        Fitn[i] = 1 / (Eval[2] + Eval[3])
    return Fitn