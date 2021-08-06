import xml.etree.ElementTree as ET
import numpy as np
import tkinter as tk
import easygui

class Table:

    def __init__(self, root):

        # code for creating table
        for i in range(total_rows):
            for j in range(total_columns):
                self.e = tk.Entry(root, width=35, fg='black',
                               font=('Arial', 16, 'bold'))

                self.e.grid(row=i, column=j)
                self.e.insert(tk.END, printList[i][j])
try:
    tree = ET.parse(easygui.fileopenbox("*.xml"))
except:
    output = easygui.msgbox("Could not import config file", "Import Error", "OK")
    quit()

#tree = ET.parse("C:\\Users\\Shaun.fraser\\OneDrive - Tornado Spectral Systems\\Python config script testing\\config_andor.xml")

xmlroot = tree.getroot()
elemNameList = ["Device_ID",
                'version',
                'FileStreamDisabled',
                'PeakFinding_LaserPeakThreshold',
                'PeakFinding_XaxisPeakThreshold',
                'defaultLaserCalibration1stPrecision',
                'defaultLaserCalibration2ndPrecision',
                'defaultLaserCalibration3rdPrecision',
                'laserExcitationCalSchUpperLimit',
                'laserExcitationCalSchLowerLimit',
                'ProductivityModule',
                'SafetyModule',
                'SecurityModule',
                'IndustrialControlModule',
                'PredictionEngineModule'
                ]

defaultValues = ['NA','NA','False','0.07','0.0009','1','1','2','785.5','784.5','NA','NA','NA','NA','NA']

elemList = [".//" + s for s in elemNameList]
printList = np.reshape(["CONFIG TAG", "CONFIG VALUE", "SUGGESTED VALUE"], [1,3])

#print(printList)
for count, i in enumerate(elemList):
    try:
        for element in xmlroot.iterfind(i):
            templist = [element.tag, element.text, defaultValues[count]]
            #print(np.resize(templist, [1,3]))
            printList = np.concatenate((printList, np.resize(templist, [1,3])), axis=0)
    except:
        output = easygui.msgbox("Error reading Config tag " + i, "File reading error", "Continue")
        continue


while len(printList) > 16:
   printList = np.delete(printList, (2), axis=0)

# find total number of rows and
# columns in list
total_rows = len(printList)
total_columns = len(printList[0])

# create root window
root = tk.Tk()
t = Table(root)
root.mainloop()




#for element in root.iterfind(".//PeakFinding_LaserPeakThreshold"):
    #print(element.text)

