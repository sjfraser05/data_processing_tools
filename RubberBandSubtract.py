import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

#Baseline is corrected like this:
    #y = y - rubberband(x, y)

def rubberband(x, y, *points):



    # Find the convex hull

    #print(zip(x, y))

    v = ConvexHull(np.array(list(zip(x, y))), incremental=True).vertices
    # Rotate convex hull vertices until they start from the lowest one

    v = np.roll(v, -v.argmin())
    #v = np.sort(v)

    # Leave only the ascending part
    v = v[:v.argmax()]


    if points:

        points = np.sort(np.reshape(points, (np.size(points), 1)))

        pointarray = [0]
        for val in points:

            xIndex = (np.abs(x - val)).argmin()

            pointarray = np.append(pointarray, xIndex)
        pointarray = np.delete(pointarray, (0))


        insertLocs = [0]

        for val in pointarray:

            vIndex = (np.abs(v - val)).argmin()

            if v[vIndex] > val:
                insertLocs = np.append(insertLocs, vIndex)
            else:
                insertLocs = np.append(insertLocs, vIndex+1)

        insertLocs = np.delete(insertLocs, (0))

        for count, number in enumerate(pointarray):
            v = np.insert(v, insertLocs[count], number)

        v = np.roll(v, -v.argmin())


        # Leave only the ascending part
        v = v[:v.argmax()]

    np.insert(v, 0, 0)
    #print(v)
    #print(np.max(x))
    v = np.append(v, len(x)-1)
    #print(v)
    #attempt = np.full(len(v), 10000, dtype=np.int)
    #plt.scatter(v+200, attempt)
    #plt.plot(x, y)
    #plt.plot(x, yInterp)
    #plt.show()
    # Create baseline using linear interpolation between vertices
    #y = y - np.interp(x, x[v], y[v])
    #print(x[v])
    #print(y[v])
    #spl_i = make_interp_spline(x[v], y[v])
    #plt.plot(x, spl_i(x), label='spline')
    #plt.plot(x, y, label='original')
    #plt.legend()
    #plt.show()
    return np.interp(x, x[v], y[v])
    #return spl_i(x)