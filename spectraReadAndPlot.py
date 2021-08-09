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


msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)

directory = easygui.diropenbox()
print(directory)
os.chdir(directory)
dirList = glob.glob(choice)
nList = len(dirList)
aveArray = np.zeros((1, 3101))
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
    #plt.plot(x1, snv(y1))
    y2 = y1 - rubberband(x1, y1)
    y2 = savgol_filter(y2, 15, 3, deriv=0)
    y2 = snv(y2)
    col = float(0.5*float(count)/float(nList))

    plt.plot(x1, y2, label=str(iFile.split('.spc')[0]))

    #aveArray = np.vstack([aveArray, y2])


#aveArray = np.delete(aveArray, (0), axis=0)
#subData = aveArray[1,:] - aveArray[0,:]
#plt.plot(x1, y2, label="Subtracted")

plt.xlabel('Raman Shifts ($cm^{-1}$)', fontsize=16)
plt.ylabel('Raman Intensity', fontsize=16)
#plt.title(directory.split('\\')[-1])
plt.legend()
plt.show()

