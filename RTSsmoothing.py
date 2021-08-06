import time
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import spc
import easygui
from scipy.signal import find_peaks, correlate, hilbert, savgol_filter
import csv
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

f = KalmanFilter (dim_x=2, dim_z=1)
f.x = np.array([2., 0.])
f.F = np.array([[1.,1.],
                [0.,1.]])
f.H = np.array([[1.,0.]])
f.P *= 1000
f.R = 100
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

trainSetFile = r"C:\Users\Shaun.fraser\Documents\Kalman filter tutorial\kalman filter_BL.csv"
#trainSetFile =
trainSetArray = []
kfPred = []
with open(trainSetFile, newline='') as csvfile:
    csvfile = csv.reader(csvfile, delimiter=',', quotechar='|')
    for count, row in enumerate(csvfile):

        z = float(row[0])
        trainSetArray = np.append(trainSetArray, z)
        f.predict()
        f.update(z)
        #plt.scatter(count, row[0], color='b')
        #plt.scatter(count, f.x[0], color='r')
        kfPred = np.append(kfPred, f.x[0])

xvals = range(0, count+1, 1)
plt.scatter(xvals, kfPred, color='b')
plt.plot(xvals, trainSetArray, color='r')

plt.show()

fig = plt.plot(xvals, kfPred, color='b')
plt.show()

