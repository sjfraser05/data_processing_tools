import numpy as np
import datetime
import math
import glob, os
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import spc
import easygui
from scipy.signal import find_peaks
import pandas as pd
import csv
import PyPDF2


directory = easygui.diropenbox()
os.chdir(directory)
dirList = glob.glob("Laser-Calibration*.pdf")
print(dirList)
nList = len(dirList)
print(nList)

calSoftwareList = []
calDateList = []
calDeviceList = []
headers = ["Exposure Time", "Camera Set Temperature", "Camera Actual Temperature",
           "Laser Hours", "Laser Power Current", "Throughput"]
calDataArray = np.zeros((nList, len(headers)))

for count, iFile in enumerate(dirList):
    with open(iFile, 'rb') as pdffile:
        pdfReader = PyPDF2.PdfFileReader(pdffile)
        if pdfReader.isEncrypted:
            pdfReader.decrypt('')
            pagehandle = pdfReader.getPage(0)
            print(pagehandle.extractText())
            #print(f"Number of page: {pdfReader.getNumPages()}")
        #pagehandle = pdfReader.getPage(0)

























    '''#Software version
    Software = data[0][1]
    calSoftwareList.insert(count, Software)
    #Date
    Date = data[1][1]
    calDateList.insert(count, Date)
    #DeviceID
    DeviceID = data[2][0].split(' ')[1]
    calDeviceList.insert(count, DeviceID)
    #ExposureTime
    ExposureTime = float(data[3][1])

    calDataArray[count, headers.index("Exposure Time")] = ExposureTime
    #CameraSetTemp
    CameraSetTemp = float(data[14][1])
    calDataArray[count, headers.index("Camera Set Temperature")] = CameraSetTemp
    #CameraActualTemp
    CameraActualtemp = float(data[15][1])
    calDataArray[count, headers.index("Camera Actual Temperature")] = CameraActualtemp
    #LaserHours
    LaserHours = data[16][2].split(': ')
    LaserHours = float(LaserHours[1])
    calDataArray[count, headers.index("Laser Hours")] = LaserHours
    #LaserPowerCurrent
    LaserPowerCurrent = float(data[17][1])
    calDataArray[count, headers.index("Laser Power Current")] = LaserPowerCurrent

    Xaxis = data[21]
    Yaxis = data[23]
    ###Plot Spectra for Calibration###
    if(len(Xaxis) == len(Yaxis)):
        pass
    else:
        Yaxis = Yaxis[:-1]

    Xaxis = np.asarray(Xaxis).astype(np.float)
    Yaxis = np.asarray(Yaxis).astype(np.float)

    ###Throughput vs Noise RMS###
    closestStartXaxis = np.abs(Xaxis - 932).argmin()
    closestEndXaxis = np.abs(Xaxis - 946).argmin()
    closestElementStart = Xaxis[closestStartXaxis]
    closestElementEnd = Xaxis[closestEndXaxis]
    ssq = np.sum(Yaxis[closestStartXaxis:closestEndXaxis]**2)
    noiseRMS = math.sqrt((1/len(Xaxis[closestStartXaxis:closestEndXaxis]))*ssq)

    Throughput = Yaxis.max()/noiseRMS
    calDataArray[count, headers.index("Throughput")] = Throughput
    ###Add values to array###
    #Exposure time




    Xmax = Xaxis.max()
    Ymax = Yaxis.max()
    plt.plot(Xaxis, Yaxis, label=iFile)
    #plt.yticks([])
    plt.xticks(np.arange(Xaxis.min(), Xmax, 50))
    plt.yticks(np.arange(0, Ymax+1000, 5000))
    #plt.xticks([])

date_time_obj = [datetime.datetime.strptime(date, '%Y-%m-%d-%H:%M:%S') for date in calDateList]

plt.plot(date_time_obj, calDataArray[:,3],  marker='o')
plt.xlabel("Date and time")
plt.xticks(rotation='vertical', fontsize=5)
plt.title("Laser Hours Over Time")

csvFileData = pd.DataFrame(
    {'Software': calSoftwareList,
     'Date': calDateList,
     'DeviceID': calDeviceList,
     "Exposure Time": calDataArray[:,0],
    "Camera Set Temperature": calDataArray[:,1],
    "Camera Actual Temperature": calDataArray[:,2],
    "Laser Hours": calDataArray[:,3],
    "Laser Power Current": calDataArray[:,4],
    "Throughput": calDataArray[:,5]
    })

csvFileData.to_csv("Calibration Records Results.csv")


timedel = max(date_time_obj) - min(date_time_obj)
print((timedel/10))
plt.xticks(np.arange(min(date_time_obj), max(date_time_obj), timedel/10))
plt.ylabel("Laser Hours")
plt.show()



###Plot Spectra
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Counts")
plt.show()
'''
