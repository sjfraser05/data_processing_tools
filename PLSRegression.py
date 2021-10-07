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
dataMatrix = np.arange(200, 3301, 1, float)
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
    #y2 = y1 - rubberband(x1, y1)
    y2 = savgol_filter(y2, 19, 3, deriv=1)
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
idx1 = int((np.abs(x1 - 800)).argmin())
idx2 = int((np.abs(x1 - 1800)).argmin())

xvals = x1[idx1:idx2]

# Mean center data
dataMatrix = dataMatrix[:, idx1:idx2]
dataMatrixTrans = dataMatrix.T
for count, row in enumerate(dataMatrix):
    plt.plot(xvals, row, color=colors[count])
plt.show()
for col in dataMatrixTrans:
    meanVal = np.mean(col)

    for count, i in enumerate(col):
        col[count] = col[count] - meanVal

# Mean center concentration data
meanconc = np.mean(concentration)
print(concentration)
for count, i in enumerate(concentration):
    concentration[count] = concentration[count] - meanconc
print(concentration)

concentration =  np.reshape(concentration, (1, 45))
# PLS Regression
compList = range(1, 10, 1)
rmsecvList = []
rmseList =[]
for i in compList:
    print(i)
    pls1 = PLSRegression(n_components=i)
    pls1.fit(dataMatrixTrans.T, concentration.T)
    rmse = math.sqrt(mean_squared_error(concentration.T, pls1.predict(dataMatrixTrans.T)))
    rmseList.append(rmse)
    # Cross-validation
    y_cv = cross_val_predict(pls1, dataMatrixTrans.T, concentration.T, cv=10)
    mse = mean_squared_error(concentration.T, y_cv)
    rmsecv = math.sqrt(mse)
    rmsecvList.append(rmsecv)
    print("RMSEcv = "+ str(math.sqrt(mse)))

plt.scatter(compList, rmsecvList, color="red", label="RMSECV vs LV")
plt.scatter(compList, rmseList, color="blue", label="RMSE vs LV")
plt.legend()
plt.show()

text = "Enter number of factors"
# window title
title = "Window Title GfG"
# creating a enter box
d_int = [x - y for x,y in zip(rmsecvList,rmsecvList[1:])]
print(d_int)
d_int = d_int.index(max(d_int)) +2
lower = 1
upper = max(compList)
output = easygui.integerbox(text, title, d_int, lower, upper)

# title for the message box
title = "Enter number of factors"

pls1 = PLSRegression(n_components=output)
pls1.fit(dataMatrixTrans.T, concentration.T)
y_cv = cross_val_predict(pls1, dataMatrixTrans.T, concentration.T, cv=10)
mse = mean_squared_error(concentration.T, y_cv)
plt.scatter(concentration.T+meanconc, y_cv+meanconc)
plt.plot([min(concentration.T+meanconc), max(concentration.T+meanconc)],[min(concentration.T+meanconc), max(concentration.T+meanconc)] )
print(f"PLS R^2 {pls1.score(dataMatrixTrans.T, concentration.T):.5f}")
print("RMSEcv = "+ str(math.sqrt(mse)))
plt.show()


