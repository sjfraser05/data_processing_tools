# Piecewise Direct Standardization (PDS) algorithm translated from R to Python from https://github.com/guifh/RNIR:
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import spc
import math
import scipy.signal
import csv
from sklearn.cross_decomposition import PLSRegression
from warnings import simplefilter
from snv import snv
from scipy.signal import savgol_filter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# INPUT:   masterSpectra = Spectra acquired with the master instrument (matrix).
#         slaveSpectra = Spectra acquired with the slave instrument (matrix).
#         MWsize = Half size of the moving window (integer).
#         Ncomp = Number of latent variables used in the PLS model (integer).
#         wavelength = wavelength (numeric vector).

# OUTPUT:  P = the PDS transfer matrix.


def PDS(masterSpectra, slaveSpectra, MWsize, Ncomp, wavelength):

    i = MWsize
    k = i - 1
    # Empty P matrix:
    P = np.zeros((np.size(masterSpectra, 1), np.size(masterSpectra, 1) - (2 * i) + 2), dtype=float)
    # Empty Intercept matrix
    InterceptReg = [0]

    while i < (np.size(masterSpectra, 1) - k):
        start = i-k
        end = i+k

        # PLS regression:
        pls1 = PLSRegression(n_components=Ncomp, scale=False)
        pls1.fit(slaveSpectra[:, start:end], masterSpectra[:, i])
        # Extraction of the regression coefficients:
        coefReg = pls1.coef_
        # Extraction of Intercept
        Intercept = pls1.y_mean_ - np.dot(pls1.x_mean_, pls1.coef_)
        InterceptReg = np.append(InterceptReg, Intercept)

        coefReg = coefReg.T
        P[(i - k):(i + k), i - k] =  coefReg

        del coefReg
        del pls1
        i = i + 1

        # Diplay progression:
        print(str(round(i / np.size(masterSpectra, 1), 4)))

    InterceptReg = np.hstack((np.zeros(k), InterceptReg, np.zeros(k)))
    P = np.hstack((np.zeros((np.size(masterSpectra, 1), k)), P, np.zeros((np.size(masterSpectra, 1), k))))
    np.savetxt("transfermatrix.csv", P, delimiter=',', fmt='%1.3f')

    return P, InterceptReg

def meancet(matrix):
    matrx = matrix.T
    for col in matrix:
        meanVal = np.mean(col)
        for count, i in enumerate(col):
            col[count] = col[count] - meanVal
    matrix = matrix.T
    return matrix, meanval
# Build matrices for main slave and main master spectra

mainMasterdir = r'C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\Calibration Transfer Testing IPA and Water\Master 160001'
mainSlavedir = r'C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\Calibration Transfer Testing IPA and Water\Slave 0097'
applySlavedir =r'C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\Calibration Transfer Testing IPA and Water\Slave 0097\other spectra'
mainMasterAllSpectradir = r'C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\Calibration Transfer Testing IPA and Water\HFPP 160001'
mainSlaveAllSpectradir = r'C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\Calibration Transfer Testing IPA and Water\Slave 0097\all spectra'

mainMasterarray = np.zeros((1, 3101))

os.chdir(mainMasterdir)
dirList = glob.glob('*.spc')
for iFile in dirList:
    f = spc.File(iFile)
    x1 = f.x
    y1 = f.sub[0].y
    y1 = snv(y1)
    mainMasterarray = np.vstack((mainMasterarray, y1))

mainMasterarray = np.delete(mainMasterarray, 0, 0)

mainSlavearray = np.zeros((1, 3101))

os.chdir(mainSlavedir)
dirList = glob.glob('*.spc')
for iFile in dirList:
    f = spc.File(iFile)
    x1 = f.x
    y1 = f.sub[0].y
    y1 = snv(y1)
    mainSlavearray = np.vstack((mainSlavearray, y1))

mainSlavearray = np.delete(mainSlavearray, 0, 0)

P, InterceptReg = PDS(mainMasterarray, mainSlavearray, 2, 2, x1)


os.chdir(mainSlavedir)
dirList = glob.glob('*.spc')
for iFile in dirList:
    f = spc.File(iFile)
    x1 = f.x
    y1 = f.sub[0].y
    y1 = snv(y1)
    mainSlavearray = np.vstack((mainSlavearray, y1))

mainSlavearray = np.delete(mainSlavearray, 0, 0)
slaveCor = np.matmul(mainSlavearray[0, :], P)
slaveCor = np.add(slaveCor, InterceptReg)

plt.plot(x1, mainSlavearray[0, :], label="Slave")
plt.plot(x1, slaveCor, label="PDS corrected slave")
plt.plot(x1, mainMasterarray[0, :], label="Master")
plt.legend()
plt.show()


applySlavearray = np.zeros((1, 3101))
os.chdir(applySlavedir)
dirList = glob.glob('*.spc')
for iFile in dirList:
    f = spc.File(iFile)
    x1 = f.x
    y1 = f.sub[0].y
    y1 = snv(y1)
    applySlavearray = np.vstack((applySlavearray, y1))

applySlavearray = np.delete(applySlavearray, 0, 0)
appliedSlavearray = np.zeros((1, 3101))

for row in applySlavearray:
    slaveCorred = np.dot(row, P)
    slaveCorred = np.add(slaveCorred, InterceptReg)
    #slaveCorred = np.add(row, InterceptReg)
    appliedSlavearray = np.vstack((appliedSlavearray, slaveCorred))
appliedSlavearray = np.delete(appliedSlavearray, 0, 0)

mainMasterAllSpectraarray = np.zeros((1, 3101))
os.chdir(mainMasterAllSpectradir)
dirList = glob.glob('*.spc')
for iFile in dirList:
    f = spc.File(iFile)
    x1 = f.x
    y1 = f.sub[0].y
    y1 = snv(y1)
    mainMasterAllSpectraarray = np.vstack((mainMasterAllSpectraarray, y1))

mainMasterAllSpectraarray = np.delete(mainMasterAllSpectraarray, 0, 0)
x2 = np.reshape(x1, (1, 3101))

os.chdir(mainMasterAllSpectradir)
dirList = glob.glob('*.spc')
for i, row in enumerate(mainMasterAllSpectraarray):
    plt.plot(x1, row, label="Master Spectra", color='blue')
    filename = str(dirList[i].split('.spc')[0]) + " master.csv"
    writeData = np.vstack([x2, row])
    #np.savetxt(filename, np.transpose(writeData), delimiter=',', fmt='%1.3f')


os.chdir(applySlavedir)
dirList = glob.glob('*.spc')
for i, row in enumerate(applySlavearray):
    plt.plot(x1, row, label="Slave original", color='green')
    filename = str(dirList[i].split('.spc')[0]) + " orig_slave.csv"
    writeData = np.vstack([x2, row])
    #np.savetxt(filename, np.transpose(writeData), delimiter=',', fmt='%1.3f')

for i, row in enumerate(appliedSlavearray):
    plt.plot(x1, row, label="Slave corrected", color='orange')
    filename = str(dirList[i].split('.spc')[0]) + " app_slave.csv"
    writeData = np.vstack([x2, row])
    #np.savetxt(filename, np.transpose(writeData), delimiter=',', fmt='%1.3f')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()




r'''
master = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_100ms_2021-09-13-17.02.16.593.spc")
master1 = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_100ms_2021-09-13-17.02.18.093.spc")
master2 = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_100ms_2021-09-13-17.02.19.593.spc")

slave = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_100ms_50mW_2021-09-21-14.27.09.370.spc")
slave1 = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_100ms_50mW_2021-09-21-14.27.13.070.spc")
slave2 = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_100ms_50mW_2021-09-21-14.27.14.673.spc")

masterF = np.vstack((master.sub[0].y, master1.sub[0].y))
masterF = np.vstack((masterF, master2.sub[0].y))

ms250160001 = spc.File(r"C:\Users\Shaun.fraser\Documents\PS testing\160001_250ms_50mW_2021-09-21-14.31.18.647.spc")
ms250160003 = spc.File(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\PeaXact Integration testing\PEAXACT Baseline\PEAXACT_Baseline_250ms_2021-09-14-13.30.42.230.spc")

slaveF = np.vstack((slave.sub[0].y, slave1.sub[0].y, slave2.sub[0].y))
#slaveF = np.vstack((slaveF, )

P, InterceptReg = PDS(masterF, slaveF, 2, 2, master.x)

slaveCor = np.matmul(slave.sub[0].y, P)
slaveCor = np.add(slaveCor, InterceptReg)

HQI1 = (np.dot(slave.sub[0].y, master.sub[0].y)**2)/((np.dot(slave.sub[0].y, slave.sub[0].y)*(np.dot(master.sub[0].y, master.sub[0].y))))

HQIcor = (np.dot(slaveCor, master.sub[0].y)**2)/((np.dot(slaveCor, slaveCor)*(np.dot(master.sub[0].y, master.sub[0].y))))


plt.plot(master.x, slave.sub[0].y, label="Slave")
plt.plot(master.x, slaveCor, label="PDS corrected slave")
plt.plot(master.x, master.sub[0].y, label="Master")
plt.legend()
plt.show()

slaveCor2 = np.matmul(ms250160001.sub[0].y, P)
slaveCor2 = np.add(slaveCor2, InterceptReg)

plt.plot(master.x, ms250160001.sub[0].y, label="Slave")
plt.plot(master.x, slaveCor, label="PDS corrected slave")
plt.plot(master.x, ms250160003.sub[0].y, label="Master")
plt.legend()
plt.show()

'''