import numpy as np


def snv(y):
    average = np.mean(y)
    standardDev = np.std(y)
    for count, element in enumerate(y):
        y[count] = (element - average) / standardDev


    return y
