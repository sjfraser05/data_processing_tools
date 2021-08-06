from numpy import *
import glob, os
import matplotlib.pyplot as plt
import spc
import easygui
from scipy.signal import find_peaks, correlate, hilbert, savgol_filter
import csv
from RubberBandSubtract import rubberband
from snv import snv
from numpy.random import randn
from numpy.linalg import inv
'''
X : The mean state estimate of the previous step ( k −1).
P : The state covariance of previous step ( k −1).
A : The transition n n × matrix.
Q : The process noise covariance matrix.
B : The input effect matrix.
U : The control input. 
'''
def kf_predict(X, P, A, Q, B, U):
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return(X,P)


'''
K : the Kalman Gain matrix
IM : the Mean of predictive distribution of Y
IS : the Covariance or predictive mean of Y
LH : the Predictive probability (likelihood) of measurement which is
computed using the Python function gauss_pdf. '''

def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X,P,K,IM,IS,LH)

def gauss_pdf(X, M, S):
    if M.shape()[1] == 1:
        DX = X - tile(M, X.shape()[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    elif X.shape()[1] == 1:
        DX = tile(X, M.shape()[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    else:
        DX = X-M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    return (P[0],E[0])


msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)

directory = easygui.diropenbox()
os.chdir(directory)
dirList = glob.glob(choice)
nList = len(dirList)
wvNumStart = 200
wvNumEnd = 3200
spectralRange = wvNumEnd-wvNumStart
spectralArray = zeros((1, int(spectralRange)))

for count, iFile in enumerate(dirList):

    if choice == "*.spc":
        f = spc.File(iFile)
        x1 = f.x
        y1 = f.sub[0].y
        #y1 = savgol_filter(y1, 41, 5, 2)
        #y1 = y1 - rubberband(x1, y1)
        #y1 = snv(y1)

    elif choice == "*.csv":
        with open(iFile, newline='') as csvfile:
            f = list(csv.reader(csvfile, delimiter=','))
        y1 = f[23]
        x1 = asarray([float(i) for i in f[21]])
        y1 = asarray([float(i) for i in y1[0:len(y1) - 1]])


    idx1 = (abs(x1 - wvNumStart)).argmin()
    idx2 = (abs(x1 - wvNumEnd)).argmin()
    plt.plot(x1[idx1:idx2], y1[idx1:idx2])
    spectralArray = vstack((spectralArray, y1[idx1:idx2]))

spectralArray = delete(spectralArray, (0), axis=0)

#time step of mobile movement
dt = 0.1
# Initialization of state matrices
X = spectralArray[0,:]
P = zeros(3101)
A = array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
Q = eye(X.shape[0])
B = eye(X.shape[0])
U = zeros((X.shape[0],1))

# Measurement matrices
Y = array([[X[0] + abs(randn(1)[0])], [X[1] + abs(randn(1)[0])]])
H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = eye(Y.shape[0])
# Number of iterations in Kalman Filter
N_iter = 50
# Applying the Kalman Filter
for i in arange(0, N_iter):
    (X, P) = kf_predict(X, P, A, Q, B, U)
    (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
    Y = array([[X[0,0] + abs(0.1 * randn(1)[0])],[X[1, 0] + abs(0.1 * randn(1)[0])]])