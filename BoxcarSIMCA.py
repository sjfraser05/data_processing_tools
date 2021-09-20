import numpy as np
import glob, os
import matplotlib.pyplot as plt
import spc
import easygui
from scipy.signal import find_peaks
import csv
from RubberBandSubtract import rubberband


msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)

directory = easygui.diropenbox()
os.chdir(directory)
dirList = glob.glob(choice)
nList = len(dirList)

boxcar = 30
aveArray = np.zeros((1,3101), int)
SIMCAray = np.arange(200, 3301, 1, int)


np.savetxt("DirList.csv", dirList, delimiter=',', fmt='% s')

for count, iFile in enumerate(dirList):
    #f = spc.File(iFile)

    if count > (nList - boxcar):
        break
    elif count > boxcar:
        for numb in range(0, boxcar):
            f = spc.File(dirList[int(count) - int(numb)])
            x1 = f.x
            y1 = f.sub[0].y

            y1 = np.reshape(y1, (1, 3101))
            aveArray = np.vstack([aveArray, y1])
            #print(aveArray)
        aveArray = np.delete(aveArray, (0), axis=0)
        #aveData = np.mean(aveArray, axis=0)
        stdDevs = np.std(aveArray, axis=0)
        aveArrayTrans = aveArray.T
        aveData = np.mean(aveArray, axis=0)
        # cosmic ray filter
        for count1, row in enumerate(aveArrayTrans):
            for i in row:
                if i > aveData[count1] + stdDevs[count1] * 4 or i < aveData[count1] - stdDevs[count1] * 4:
                    i = np.nan

        aveData = np.nanmean(aveArrayTrans.T, axis=0)
        #aveArray = np.empty((0, 3101), float)
        SIMCAray = np.vstack([SIMCAray, aveData])
        aveArray = np.empty((0, 3101), int)
    else:
        continue



plt.plot(np.transpose(SIMCAray))

filename = "SIMCA Format Array_CRF_" + str(boxcar) + '_ave_' + '.csv'
np.savetxt(filename, SIMCAray, delimiter=',', fmt='%1.3f')
plt.show()
