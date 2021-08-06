import numpy as np
import glob, os
import matplotlib.pyplot as plt
import spc
import easygui
from scipy.signal import find_peaks
import csv


msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)

directory = easygui.diropenbox()
os.chdir(directory)
dirList = glob.glob(choice)

nList = len(dirList)

#aveArray = np.zeros((1,3101), int)
SIMCAray = np.arange(200, 3301, 1, int)
#print(nList - boxcar)

np.savetxt("DirList.csv", dirList, delimiter=',', fmt='% s')

for count, iFile in enumerate(dirList):
    f = spc.File(iFile)
    x1 = f.x
    y1 = f.sub[0].y
    N2point1 = 2320
    N2point2 = 2340

    idx1 = (np.abs(x1 - N2point1)).argmin()
    idx2 = (np.abs(x1 - N2point2)).argmin()

    interpolant = np.interp([x1[idx1], x1[idx2]], x1, y1)
    coefficients = np.polyfit([x1[idx1], x1[idx2]], interpolant, 1)

    N2area = np.trapz(y1[idx1:idx2] - ((x1[idx1:idx2]*coefficients[0] + coefficients[1])), x1[idx1:idx2])

    print(coefficients)
    slope = (y1[idx2]-y1[idx1])/(N2point2-N2point1)
    b = y1[idx2] - slope*N2point1


    y2 = np.divide(y1, N2area)
    plt.plot(x1, y2)
    #plt.plot(x1[idx1:idx2], y1[idx1:idx2] - (x1[idx1:idx2]*coefficients[0] + x1[idx1:idx2]*coefficients[1]))

    SIMCAray = np.vstack([SIMCAray, y2])


    #f.plot()  # plot data
    #filename = 'N2Norm ' + iFile + '.csv'
    #x1 = np.reshape(x1, (1, 3101))
    #writeData = np.vstack([x1, aveData])

    #np.savetxt(filename, SIMCAray, delimiter=',', fmt='%d')  #np.transpose(writeData)
    #plt.plot(x1, aveData)

    #plt.plot(x2[index], y1[index], "x")
    #plt.plot(x1[index], y1[index], "x")

filename ="SIMCA Format Array_" 'N2_Norm' + '.csv'
np.savetxt(filename, SIMCAray, delimiter=',', fmt='%d')
plt.show()
