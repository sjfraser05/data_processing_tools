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
from warnings import simplefilter
from snv import snv
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
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
    InterceptReg = [0]

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

        #InterceptReg < -c(InterceptReg, coefReg[1])
        #coefReg < -coefReg[2:length(coefReg)]
        Intercept = pls1.y_mean_ - np.dot(pls1.x_mean_, pls1.coef_)
        InterceptReg = np.append(InterceptReg, Intercept)

        # Add coefficients to the transfer matrix:
        # P[(i-k):(i+k),i-k]<-t(coefReg)
        coefReg = coefReg.T

        P[(i - k):(i + k), i - k] =  coefReg


        #rm(coefReg, fit)
        del coefReg
        del pls1

        i = i + 1

        # Diplay progression:
        print(str(round(i / np.size(masterSpectra, 1), 4)))

    #P < -data.frame(matrix(0, nrow=ncol(masterSpectra), ncol=k), P,
    #                matrix(0, nrow=ncol(masterSpectra), ncol=k))
    #InterceptReg = np.delete(InterceptReg, 0)
    InterceptReg = np.append(np.zeros(k), InterceptReg)
    InterceptReg = np.append(InterceptReg, np.zeros(k))
    P = np.hstack((P, np.zeros((np.size(masterSpectra, 1), k))))
    P = np.hstack((np.zeros((np.size(masterSpectra, 1), k)), P))
    print(InterceptReg)
    #padded_array[:shape[0], :shape[1]] = P_padded
    #np.savetxt("transfermatrix.csv", P, delimiter=',', fmt='%1.3f')
    #Po = np.hstack((np.zeros(sizedArray, k), P))


    #P = np.hstack((Po, np.zeros(np.size(masterSpectra, 1), k)))
    #InterceptReg < -c(rep(0, k), InterceptReg, rep(0, k))

    return P, InterceptReg


master = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_100ms_2021-09-13-17.02.16.593.spc")
master1 = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_100ms_2021-09-13-17.02.18.093.spc")
master2 = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_100ms_2021-09-13-17.02.19.593.spc")

slave = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_100ms_50mW_2021-09-21-14.27.09.370.spc")
slave1 = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_100ms_50mW_2021-09-21-14.27.13.070.spc")
slave2 = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_100ms_50mW_2021-09-21-14.27.14.673.spc")

masterF = np.vstack((snv(master.sub[0].y), snv(master1.sub[0].y)))
masterF = np.vstack((masterF, snv(master2.sub[0].y)))

ms250160001 = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_250ms_50mW_2021-09-21-14.31.18.647.spc")
ms250160003 = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_250ms_2021-09-14-13.30.42.230.spc")

slaveF = np.vstack((snv(slave.sub[0].y), snv(slave1.sub[0].y), snv(slave2.sub[0].y)))
#slaveF = np.vstack((slaveF, )

P, InterceptReg = PDS(masterF, slaveF, 2, 2, master.x)

slaveCor = np.matmul(slave.sub[0].y, P)
slaveCor = np.add(slaveCor, InterceptReg)


plt.plot(master.x, slave.sub[0].y, label="Slave")
plt.plot(master.x, slaveCor, label="PDS corrected slave")
plt.plot(master.x, master.sub[0].y, label="Master")
plt.legend()
plt.show()

slaveCor2 = np.matmul(snv(ms250160001.sub[0].y), P)
slaveCor2 = np.add(slaveCor2, InterceptReg)

plt.plot(master.x, snv(ms250160001.sub[0].y), label="Slave")
plt.plot(master.x, slaveCor, label="PDS corrected slave")
plt.plot(master.x, snv(ms250160003.sub[0].y), label="Master")
plt.legend()
plt.show()