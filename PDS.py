# Piecewise Direct Standardization (PDS) algorithm:
import numpy as np
import glob, os
import math
import matplotlib.pyplot as plt
import spc
import scipy.signal
import csv
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize

# INPUT:   masterSpectra = Spectra acquired with the master instrument (matrix).
#         slaveSpectra = Spectra acquired with the slave instrument (matrix).
#         MWsize = Half size of the moving window (integer).
#         Ncomp = Number of latent variables used in the PLS model (integer).
#         wavelength = wavelength (numeric vector).

# OUTPUT:  P = the PDS transfer matrix.


def PDS(masterSpectra, slaveSpectra, MWsize, Ncomp, wavelength):

    # Loop Initialization:
    i = MWsize
    k = i - 1
    # Creation of an empty P matrix:
    P = np.zeros((np.size(masterSpectra, 1), np.size(masterSpectra, 1) - (2 * i) + 2), dtype=float)
    #InterceptReg = c()

    while i < (np.size(masterSpectra, 1) - k):
        start = i-k
        end = i+k
        # PLS regression:
        pls1 = PLSRegression(n_components=Ncomp, scale=False)
        pls1.fit(slaveSpectra[:, start:end], masterSpectra[:, i])
        #fit < - plsr(masterSpectra[, i] ~ as.matrix(slaveSpectra[, (i-k):(i + k)]),
        #ncomp = Ncomp, scale = F, method = "oscorespls")

        # Extraction of the regression coefficients:
        #coefReg < -as.numeric(coef(fit, ncomp=Ncomp, intercept=TRUE))
        coefReg = pls1.coef_
        print(coefReg)
        input()
        #InterceptReg < -c(InterceptReg, coefReg[1])
        #coefReg < -coefReg[2:length(coefReg)]

        # Add coefficients to the transfer matrix:
        # P[(i-k):(i+k),i-k]<-t(coefReg)
        coefReg = coefReg.T
        coefReg = P[(i - k):(i + k), i - k]

        #rm(coefReg, fit)
        del coefReg
        del pls1

        i = i + 1

        # Diplay progression:
        print(str(round(i / np.size(masterSpectra, 1), 4)))

    #P < -data.frame(matrix(0, nrow=ncol(masterSpectra), ncol=k), P,
    #                matrix(0, nrow=ncol(masterSpectra), ncol=k))

    p = np.array(0)
    InterceptReg < -c(rep(0, k), InterceptReg, rep(0, k))

    Output < -list(P=P, Intercept=InterceptReg)

    return(Output)


master = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_100ms_2021-09-13-17.02.16.593.spc")
master1 = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_100ms_2021-09-13-17.02.18.093.spc")
master2 = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_100ms_2021-09-13-17.02.19.593.spc")

slave = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_100ms_50mW_2021-09-21-14.27.09.370.spc")
slave1 = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_100ms_50mW_2021-09-21-14.27.13.070.spc")
slave2 = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_100ms_50mW_2021-09-21-14.27.14.673.spc")

masterF = np.vstack((master.sub[0].y, master1.sub[0].y))
masterF = np.vstack((masterF, master2.sub[0].y))

slaveF = np.vstack((slave.sub[0].y, slave1.sub[0].y))
slaveF = np.vstack((slaveF, slave2.sub[0].y))

PDSarray = PDS(masterF, slaveF, 2, 2, master.x)

plt.plot(master.x, PDSarray)
plt.show()