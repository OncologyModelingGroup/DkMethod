import pandas as pd
import numpy as np
import ProcessPatientData as ppd
import Model3Dand1D as mdl
import FitDkForEachPatient as fdk
import os
BRATS_PATH="/PATH/TO/PATIENT/NII/FOLDERS/"
OUT="/OUTPUT/DIRECTORY/"
os.makedirs(OUT, exist_ok=True)

if __name__=="__main__":
# generates a dictionary with the paths needed to pull the patient data
    ppd.GenBratsPaths(BRATS_PATH,OUT+"paths.p")
# converts the patient data into 1D density distributions
    ppd.ProcessPts(pd.read_pickle(OUT+"paths.p"),OUT+"pts.p",ppd.GetLargestCom)
# generates a grid of PDE results for grid search
    mdl.GenGrid([mdl.ModelParams(d,k=1E-4,steps=1E6,sideLen=201,dx=1,dt=1) for d in np.logspace(-5,-2,1000)],OUT+"grid.p")
# runs grid search followed on the patient data followed by LM to get optimal D/k
    fdk.GenPatientDks(pd.read_pickle(OUT+"pts.p"),pd.read_pickle(OUT+"grid.p"),OUT+"dks.p")
# generate dataframe from results for easier review
    fdk.GenPatientDkResDf(pd.read_pickle(OUT+"dks.p"),outPath=OUT+"dkDf.csv")