import time
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

extractData = r"C:\Users\Shaun.fraser\Documents\2 stage vessel extraction measurement.csv"
time = []
CBDconc = []
with open(extractData, newline='') as csvfile:
    csvfile = csv.reader(csvfile, delimiter=',', quotechar='|')
    for count, row in enumerate(csvfile):

        time = np.append(time, float(row[0]))

        CBDconc = np.append(CBDconc, float(row[1]))

time = time[6:]
CBDconc = CBDconc[6:]
#def model(t, c0, c1, c2):
   # return c0 - (c0-c1)*np.exp(c2*t)

def model(t, c0, c1, c2):
    return c0*((np.log(t))**2) + c1*(np.log(t)) + c2

#def model(t, c0, c1, c2):
#    return c0*((np.log(t))**2) + c1*(np.log(t)) + c2

g = [1000, 100, -.1]
y = np.empty(len(time))
for i in range(len(time)):
    y[i] = model(time[i], g[0], g[1], g[2])



c, cov = curve_fit(model, time, CBDconc, g)
print(c)

yfit = np.empty(len(time))
for i in range(len(time)):
    yfit[i] = model(time[i], c[0], c[1], c[2])

print('R^2: ', r2_score(yfit, CBDconc))
plt.scatter(time, CBDconc, color='b', label="Raman Predicted CBD concnetration (mg/mL)")
#plt.plot(time, y, 'r.')
plt.plot(time, yfit,'g', label="Asymptotic regression model")
plt.xlabel('Time (minutes)')
plt.ylabel('CBD Concentration (mg/mL)')
plt.show()