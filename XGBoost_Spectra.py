'''Reads all spc or csv spectral files in a folder and plots'''
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import spc
import easygui
from scipy.signal import savgol_filter
import csv
from RubberBandSubtract import rubberband
from snv import snv
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import math

msg = "spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)

directory = easygui.diropenbox()
print(directory)
os.chdir(directory)
dirList = glob.glob(choice)
nList = len(dirList)
aveArray = np.zeros((1, 3101))
# SIMCAray = np.arange(200, 3301, 1, int)

concentration = [0.25,0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.75,0.75,0.75,0.75,0.75,1,1,1,1,1,1.25,1.25,1.25,1.25,1.25,1.5,1.5,1.5,1.5,1.5,1.75,1.75,1.75,1.75,1.75,2,2,2,2,2,2.25,2.25,2.25,2.25,2.25]
concentration = np.transpose(concentration)
colors = plt.cm.rainbow(np.linspace(0, 1, nList))
dataMatrix = np.arange(200, 3301, 1, int)
colors = np.flip(colors, 0)
for count, iFile in enumerate(dirList):

    if choice == "*.spc":
        f = spc.File(iFile)
        x1 = f.x
        y1 = f.sub[0].y

    elif choice == "*.csv":
        with open(iFile, newline='') as csvfile:
            f = list(csv.reader(csvfile, delimiter=','))
        y1 = f[23]

        x1 = np.asarray([float(i) for i in f[21]])

        y1 = np.asarray([float(i) for i in y1[0:len(y1) - 1]])

    y2 = y1
    # plt.plot(x1, snv(y1))
    y2 = y1 - rubberband(x1, y1)
    y2 = savgol_filter(y2, 15, 3, deriv=1)
    y2 = snv(y2)
    col = float(0.5 * float(count) / float(nList))
    dataMatrix = np.vstack((dataMatrix, y2))
    plt.plot(x1, y2, label=str(iFile.split('.spc')[0]), color=colors[count])
    # SIMCAray = np.vstack([SIMCAray, y1])
    # aveArray = np.vstack([aveArray, y2])


plt.xlabel('Raman Shifts ($cm^{-1}$)', fontsize=16)
plt.ylabel('Raman Intensity', fontsize=16)
plt.title(directory.split('\\')[-1])
# plt.legend()
plt.show()
dataMatrix = np.delete(dataMatrix, (0), axis=0)

# Range Selection
idx1 = (np.abs(x1 - 800)).argmin()
idx2 = (np.abs(x1 - 1800)).argmin()
xvals = x1[idx1:idx2]

# Mean center data
dataMatrixTrans = dataMatrix.T
dataMatrix = dataMatrix[idx1:idx2]

for col in dataMatrixTrans:
    meanVal = np.mean(col)

    for count, i in enumerate(col):
        col[count] = col[count] - meanVal

# Mean center concentration data
meanconc = np.mean(concentration)
for count, i in enumerate(concentration):
    concentration[count] = concentration[count] - np.mean(concentration)

concentration =  np.reshape(concentration, (1, 45))
print(dataMatrixTrans.shape)
print(concentration.shape)
# PLS Regression
pls1 = PLSRegression(n_components=5)
pls1.fit(dataMatrixTrans.T, concentration.T)


# Cross-validation
y_cv = cross_val_predict(pls1, dataMatrixTrans.T, concentration.T, cv=10)

# Calculate scores
score = r2_score(concentration.T, y_cv)
mse = mean_squared_error(concentration.T, y_cv)
print(y_cv)
for i in y_cv:
    print(str(i+meanconc))
plt.plot(concentration.T, y_cv+meanconc)
plt.show()
print(score)
print(math.sqrt(mse))

