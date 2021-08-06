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
    P = np.zeros((len(masterSpectra[0]), len(masterSpectra[0]) - (2 * i) + 2))
    #InterceptReg = c()

    while i <= (len(masterSpectra[0]) - k):

        # PLS regression:
        fit = PLSRegression(n_components=2, scale=False)
        pls2.fit(masterSpectra[:, i], slaveSpectra[:, (i-k):(i + k)])
        #fit < - plsr(masterSpectra[, i] ~ as.matrix(slaveSpectra[, (i-k):(i + k)]),
        #ncomp = Ncomp, scale = F, method = "oscorespls")

        # Extraction of the regression coefficients:
        #coefReg < -as.numeric(coef(fit, ncomp=Ncomp, intercept=TRUE))
        fit.coef_
        #InterceptReg < -c(InterceptReg, coefReg[1])
        #coefReg < -coefReg[2:length(coefReg)]

        # Add coefficients to the transfer matrix:
        P[(i - k):(i + k), i - k] < -t(coefReg)

        rm(coefReg, fit)
        i < -i + 1

        # Diplay progression:
        cat("\r", paste(round(i / ncol(masterSpectra) * 100), " %", sep=""))}

    P < -data.frame(matrix(0, nrow=ncol(masterSpectra), ncol=k), P,
                    matrix(0, nrow=ncol(masterSpectra), ncol=k))
    InterceptReg < -c(rep(0, k), InterceptReg, rep(0, k))

    Output < -list(P=P, Intercept=InterceptReg)

    return (Output)}