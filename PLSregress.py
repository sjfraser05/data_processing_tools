import numpy as np
import glob, os
import math
import pandas as pd
import matplotlib.pyplot as plt
import spc
import easygui
import scipy.signal
import csv
from tkinter import *
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize

def find_in_grid(frame, row, column):
    for children in frame.children.values():
        info = children.grid_info()
        #note that rows and column numbers are stored as string
        if info['row'] == str(row) and info['column'] == str(column):
            return children
    return None

msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)
directory = easygui.diropenbox()
os.chdir(directory)
dirList = glob.glob(choice)
nList = len(dirList)

root = Tk()

height = nList
width = 1
y = np.zeros(height)
for i in range(height): #Rows
    for j in range(width): #Columns
        b = Entry(root, text="")
        b.grid(row=i, column=j)

mainloop()

for k in range(height):
    y = np.vstack(y, float(find_in_grid(root, k+1, j).get()))
y = np.delete(y, (0), axis=0)

#df = pd.read_csv("XYData.csv", dtype={"CallGuid": np.int64}, header=None)
max_comp = 3+1
PRESS = np.zeros((1,2), int)

#Range Selection
Ytrain = df[df.columns[-1:]]
RangeBegin = 800
RangeEnd = 1800
Xtrain = df[df.columns[RangeBegin:RangeEnd]]

#Pre-processing

#Smoothing
window_length = 19
polyorder = 2

Xtrain = scipy.signal.savgol_filter(Xtrain, window_length, polyorder, deriv=0, delta=1.0, axis= 1, mode='interp', cval=0.0)



preproc = plt.plot(np.transpose(Xtrain))
for x in range(1, max_comp):
    pls1 = PLSRegression(n_components=x)

    print(Ytrain)
    print(Xtrain)
    pls1.fit(Xtrain, Ytrain)
    y_c = pls1.predict(Xtrain)

    # Cross-validation
    y_cv = cross_val_predict(pls1, Xtrain, Ytrain, cv=10)
    # Calculate mean square error for calibration and cross validation
    mse_c = mean_squared_error(Ytrain, y_c)
    mse_cv = mean_squared_error(Ytrain, y_cv)
    RMSEs = [math.sqrt(mse_c), math.sqrt(mse_cv),]
    PRESS = np.vstack([PRESS, RMSEs])



#plt.scatter(Ytrain, Y_pred)
#print(pls1.x_loadings_)

PRESS = np.delete(PRESS, (0), axis=0)
fig, ax = plt.subplots()
ax.plot(range(1, max_comp, 1), PRESS[:,0], label='RMSE')
ax.plot(range(1, max_comp, 1), PRESS[:,1], label='RMSECV' )

leg = ax.legend()
plt.show()

norm_load = normalize(pls1.x_loadings_[:, 0:max_comp-1], norm='l2', axis=1)
a = plt.plot(range(RangeBegin, RangeEnd, 1), pls1.x_loadings_)
plt.show()


'''from sklearn.cross_decomposition import PLSRegression  from sklearn.metrics import mean_squared_error, r2_score  from sklearn.model_selection import cross_val_predict    # Define PLS object  pls = PLSRegression(n_components=5)    # Fit  pls.fit(X, Y)    # Cross-validation  y_cv = cross_val_predict(pls, X, y, cv=10)    # Calculate scores  score = r2_score(y, y_cv)  mse = mean_squared_error(y, y_cv)'''