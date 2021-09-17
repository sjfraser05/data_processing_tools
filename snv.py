import numpy as np

#Performs Standard Normal Variate Normalization.
def snv(y):
    average = np.mean(y)
    standardDev = np.std(y)
    for count, element in enumerate(y):
        y[count] = (element - average) / standardDev


    return y
