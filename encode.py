import numpy as np

#encodes text array containing categorical labels into numeric labels

def encode(x):
    uniqueClasses = set(x)

    encodedArray = np.array(x)
    for count, word in enumerate(uniqueClasses):
        indices = [i for i, y in enumerate(x) if y == word]
        encodedArray[indices] = count
    #print(encodedArray)
    return encodedArray