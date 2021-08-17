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
from scipy.linalg import pinv


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

f = KalmanFilter (dim_x=3, dim_z=3)
f.x = np.array([0., 0., 0.])
'''f.F = np.array([[1.,0.03],
                [0.,1.]])'''

f.F = np.array([[1.,0, 0.03],
                [0.,1, 0],
               [0, 0, 1]])
'''f.H = np.array([[1.,0.],
               [0, 1]])'''

f.H = np.array([[0.03,0, 0.03],
                [0.,0.03, 0.03],
               [0.03, 0.03, 0.03]])
f.P *= 1000
#f.P = np.diag([1000, 1000])
#f.R = np.asarray([[1000, 0.],
               #[0., 1000]])
f.R = 1000
f.Q = Q_discrete_white_noise(dim=3, dt=0.03, var=0.13)
f.B = 0
f.U = 0

trainSetFile = r"C:\Users\Shaun.fraser\Documents\Mar12_CytC_BSA_UV.csv"
#trainSetFile =
trainSetArray = []
kfPred = []


with open(trainSetFile, newline='') as csvfile:
    csvfile = csv.reader(csvfile, delimiter=',', quotechar='|')
    for count, row in enumerate(csvfile):
        #print(row[2])

        z = [float(row[2]), float(row[4]), float(row[6])]
        f.predict()
        #z = np.reshape(np.asarray(z), (2))
        trainSetArray = np.append(trainSetArray, z[0])
        f.update(z)
        #print(f.x[0])
        #plt.scatter(count, row[0], color='b')
        #plt.scatter(count, f.x[0], color='r')
        kfPred = np.append(kfPred, f.x[0])

print(np.shape(trainSetArray))

xvals = range(0, count+1, 1)
print(np.shape(xvals))
plt.scatter(xvals, trainSetArray, color='r')
plt.scatter(xvals, kfPred, color='b')

plt.show()

fig = plt.plot(xvals, kfPred, color='b')
plt.show()

