import numpy as np
import glob, os
import matplotlib.pyplot as plt
import spc
import easygui
from scipy.signal import find_peaks
import csv
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from RubberBandSubtract import rubberband
from snv import snv


msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)

directory = easygui.diropenbox()
os.chdir(directory)
dirList = glob.glob(choice)
print(dirList)
nList = len(dirList)
print(nList)

peak_list =  np.empty(0)
lasertemplist = np.empty(0)
cc6peak = 1740
for iFile in dirList:
    if choice == "*.spc":

        f = spc.File(iFile)

        x1 = f.x
        y1 = f.sub[0].y

        x1 = np.asarray(x1[1300:1600])
        y1 = np.asarray(y1[1300:1600])
        y2 = y1 - rubberband(x1, y1)
        y2 = savgol_filter(y2, 43, 3, deriv=0)
        y2 = snv(y2)
        #plt.plot(x1, y2)
        #y2 = snv(y2)
        f1 = interp1d(x1, y2, kind='cubic', fill_value="extrapolate")  # kind='cubic', fill_value="extrapolate"
        try:
            xnew = np.linspace(1500, 1800, num=31000, endpoint=True)
            plt.plot(xnew, f1(xnew), label=iFile)
            peaks = find_peaks(f1(xnew), prominence=1)
            peaks = np.asarray(peaks[0])
            idx1 = (np.abs(xnew[peaks] - cc6peak)).argmin()
            #plt.plot(x1, y1)
            #plt.plot(x1[index], y1[index], "x")
            #plt.plot(x1[index], y1[index], "x")
            peak_list = np.append(peak_list, xnew[peaks[idx1]])

            lasertempraw = str(f.log_content[22])
            #print(lasertempraw)
            lasertempsplit = lasertempraw.split("= ")
            lasertempsplit = lasertempsplit[1]
            lasertempsplit = lasertempsplit[:-1]
            #print(lasertempsplit)
            #lasertempsplit = lasertempraw[1].split("'")
            #print(lasertempsplit)
            lasertemplist = np.append(lasertemplist, float(lasertempsplit))
        except:
            a=1
    else:
        with open(iFile, newline='') as csvfile:
            f = list(csv.reader(csvfile, delimiter=','))

        #x1 = f[21]
        y1 = f[23]
        lasertemp = float(f[15][1])
        x1 = [float(i) for i in f[21]]
        y1 = [float(i) for i in y1[0:len(y1)-1]]

        x1 = np.asarray(x1)
        y1 = np.asarray(y1)
        f1 = interp1d(x1, y1, kind='cubic', fill_value="extrapolate") #kind='cubic', fill_value="extrapolate"
        xnew = np.linspace(201, 3301, num=31000, endpoint=True)

        peaks = find_peaks(f1(xnew), prominence=500)
        peaks = np.asarray(peaks[0])

        #print(peaks)
        idx1 = (np.abs(xnew[peaks] - cc6peak)).argmin()
        print(xnew[peaks[idx1]])
        #print(idx1)

        #print(xnew[peaks[idx1]])
        # offset = 1321.446429 - x1[index]
        # x2 = x1 + offset
        #plt.plot(x1, y1)
        #plt.plot(xnew, f1(xnew))
        # plt.plot(x1[index], y1[index], "x")
        # plt.plot(x1[index], y1[index], "x")

        peak_list = np.append(peak_list, xnew[peaks[idx1]])
        lasertemplist = np.append(lasertemplist, lasertemp)
        #print(iFile)
        #spectraArray = np.vstack((x2,y1)).T
        #np.savetxt(iFile + ".csv", spectraArray, delimiter=",")
#print(f.__dict__)
print(np.mean(peak_list))
N = 10
cumsum, moving_aves = [0], []
for val in peaks:
    plt.vlines(xnew[val],np.min(f1(xnew)), np.max(f1(xnew)), color='red')
plt.show()
for i, x in enumerate(peak_list, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)
#plt.plot(lasertemplist)
#plt.plot(moving_aves)
plt.scatter(range(len(peak_list)), peak_list)
#plt.plot(moving_aves)
#np.savetxt("1600 band changes" + ".csv", peak_list, delimiter=",")
plt.show()





#f = spc.File('C:/Users/Shaun.fraser/PycharmProjects/read_and_plot_spectra/venv/Data Files/sample.spc')  # read file

#x-y(1)  # format string
#f.data_txt()  # output data
#f.write_file('output.txt')  # write data to file
#f.plot()  # plot data
#plt.show()




'''f = spc.File('/Desktop/sample.spc')  # read file
x-y(20)  # format string
f.data_txt()  # output data
f.write_file('output.txt')  # write data to file
f.plot()  # plot data'''