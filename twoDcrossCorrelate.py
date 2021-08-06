import numpy as np
import glob, os
import matplotlib.pyplot as plt
import spc
import easygui
from scipy.signal import find_peaks, correlate, hilbert, savgol_filter
import csv
from RubberBandSubtract import rubberband
from snv import snv
from snvV2 import snvV2


msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)

directory = easygui.diropenbox()
os.chdir(directory)
dirList = glob.glob(choice)
nList = len(dirList)
wvNumStart = 1400
wvNumEnd = 1600
spectralRange = wvNumEnd-wvNumStart
corrArray = np.zeros((1, int(spectralRange)))

print(np.size(corrArray))


for count, iFile in enumerate(dirList):

    if choice == "*.spc":
        f = spc.File(iFile)
        x1 = f.x
        y1 = f.sub[0].y
        y1 = savgol_filter(y1, 21, 3, 2)
        #y1 = y1 - rubberband(x1, y1)
        y1 = snv(y1)

    elif choice == "*.csv":
        with open(iFile, newline='') as csvfile:
            f = list(csv.reader(csvfile, delimiter=','))
        y1 = f[23]
        x1 = np.asarray([float(i) for i in f[21]])
        y1 = np.asarray([float(i) for i in y1[0:len(y1) - 1]])


    idx1 = (np.abs(x1 - wvNumStart)).argmin()
    idx2 = (np.abs(x1 - wvNumEnd)).argmin()
    colormap = (count/nList, count/nList, 0.5)
    plt.plot(x1[idx1:idx2], y1[idx1:idx2], color = colormap)
    corrArray = np.vstack((corrArray, y1[idx1:idx2]))


corrArray = np.delete(corrArray, (0), axis=0)
#corrArray = np.delete(corrArray, np.s_[301:1500],axis=1)
plt.xlabel('Raman Shifts ($cm^{-1}$)')
plt.ylabel('Counts')
plt.title(directory.split('\\')[-1])
plt.show()

S=np.dot(corrArray.conj().transpose(),corrArray)/float(4-1)

A=np.dot(corrArray.conj().transpose(),np.dot(hilbert(corrArray),corrArray[0,:]))/float(4-1)
#plt.imshow(S, extent=[x1[idx1], x1[idx2], x1[idx1], x1[idx2]])


#CS = plt.contourf(S, extent=[x1[idx1], x1[idx2], x1[idx1], x1[idx2]])
#cbar = CS.colorbar()

fig,ax = plt.subplots()
contourf_ = ax.contourf(S, extent=[x1[idx1], x1[idx2], x1[idx1], x1[idx2]])
cbar = fig.colorbar(contourf_)
plt.show()


'''
meanSpectra = corrArray.mean(axis=0)
meanSpectra = np.vstack((x1[idx1:idx2], meanSpectra))
corrMat1 = np.vstack((x1[idx1:idx2], corrArray[0,:]))
corrMat2 = np.vstack((x1[idx1:idx2], corrArray[-1,:]))
corr = correlate(meanSpectra, corrMat2)

corrInit = correlate(meanSpectra, corrMat1)
print(corr)
plt.plot(x1[idx1:idx2], corr, 'ro')
plt.show()


y, x = np.unravel_index(np.argmax(corr), corr.shape)
fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1)

ax_orig.imshow(corrMat2, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_template.imshow(meanSpectra, cmap='gray')
ax_template.set_title('Template')
ax_template.set_axis_off()
ax_corr.plt(corr, cmap='gray')
ax_corr.set_title('Cross-correlation')
ax_corr.set_axis_off()
ax_orig.plot(x, y, 'ro')
plt.show()
'''
