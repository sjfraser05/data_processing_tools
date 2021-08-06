import numpy as np
import glob, os
import matplotlib.pyplot as plt
import spc
import easygui
import csv
import inspect
import pickle
import matplotlib
import pickle
matplotlib.use('tkagg')

class Analyte:
  def __init__(self, name, peakLoc, lowBound, highBound, bgWidth):
    self.name = name
    self.peakLoc = peakLoc
    self.lowBound = float(lowBound)
    self.highBound = float(highBound)
    self.bgWidth = int(bgWidth)

msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)


if choice == None:
    easygui.buttonbox(msg='Press OK to Exit', title='SNR Calc', choices=["OK"])
    exit()
else:
    blah = 0


insMsg ="Tornado HFPP or Kaiser?"
institle = "Choose instrument type"
inschoices = ["Tornado HFPP", "KOSI"]
inschoice = easygui.choicebox(insMsg, institle, inschoices)

if inschoice == None:
    easygui.buttonbox(msg='Press OK to Exit', title='SNR Calc', choices=["OK"])
    exit()
else:
    blah = 0

directory = easygui.diropenbox()
if directory == None:
    easygui.buttonbox(msg='Press OK to Exit', title='SNR Calc', choices=["OK"])
    exit()
else:
    os.chdir(directory)
    dirList = glob.glob(choice)

matMsg ="Please choose an analyte."
matTitle = "Choose analyte type"
matChoices = ["cyclohexane", "isopropyl alcohol", "polystyrene", "saved custom", "new custom"]
MatChoice = easygui.choicebox(matMsg, matTitle, matChoices)

if MatChoice == None:
    easygui.buttonbox(msg='Press OK to Exit', title='SNR Calc', choices=["OK"])
    exit()
else:
    blah = 0

if MatChoice == "cyclohexane":
    peakMsg = "Choose cyclohexane peak"
    peakTitle = "SNR Calculation"
    peakChoices = [801.3, 1028.3, "CH Stretch"]
    peakChoice = easygui.choicebox(peakMsg, peakTitle, peakChoices)
    bgWidth = 20
    try:
        MatChoice = Analyte(MatChoice, peakChoice, float(peakChoice) - 75, float(peakChoice) + 75, 20)

    except:
        blah = 0 #bad code here,

    if peakChoice == "CH Stretch":
        MatChoice = Analyte(MatChoice, peakChoice, 2780, 3080, 20)
    else:
        blah = 0


elif MatChoice == "isopropyl alcohol":
    peakMsg = "Choose isopropyl alcohol peak"
    peakTitle = "SNR Calculation"
    peakChoices = [819.9, 1029.0]
    peakChoice = easygui.choicebox(peakMsg, peakTitle, peakChoices)
    MatChoice = Analyte(MatChoice, peakChoice, float(peakChoice) - 50, float(peakChoice) + 50, 20)

elif MatChoice == "polystyrene":
    peakMsg = "Choose polystyrene peak"
    peakTitle = "SNR Calculation"
    peakChoices = [620.9, 1600.0]
    peakChoice = easygui.choicebox(peakMsg, peakTitle, peakChoices)
    MatChoice = Analyte(MatChoice, peakChoice, float(peakChoice) - 50, float(peakChoice) + 50, 20)

elif MatChoice == "saved custom":
    try:
        programPath = os.path.expanduser('~\\SNRcalc Data')
        os.chdir(programPath)
        saveCustDirList = glob.glob("*.pkl")
        peakMsg = "Choose saved custom material and peak"
        peakTitle = "SNR Calculation"
        peakChoices = saveCustDirList
        peakChoice = easygui.choicebox(peakMsg, peakTitle, peakChoices)
        with open(peakChoice, 'rb') as input:
            MatChoice = pickle.load(input)
        os.chdir(directory)
    except:
        easygui.buttonbox(msg='Press OK to exit', title='No saved custom materials found', choices=["OK"])
        exit()

elif MatChoice == "new custom":

    peakMsg = "Enter your material information"
    peakTitle = "SNR Calculation"
    fieldNames = ["Material", "Peak Location", "Peak area start","Peak area end","Background width"]
    fieldValues = []  # we start with blanks for the values
    fieldValues = easygui.multenterbox(peakMsg, peakTitle, fieldNames)
    MatChoice = Analyte(fieldValues[0], fieldValues[1], fieldValues[2], fieldValues[3], fieldValues[4])

    while 1:
        if fieldValues == None: break
        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
        if errmsg == "": break  # no problems found
        fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)

nList = len(dirList)
areaList = np.zeros(1)
fig, (ax1, ax2, ax3) = plt.subplots(3,2)

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(212)


for iFile in dirList:
    
    if choice == "*.spc":
        f = spc.File(iFile)
        x1 = f.x
        y1 = f.sub[0].y
        if inschoice == "KOSI":

            expTime = ' ' + str(f.log_content[20]).split('=')[1]
            averaging = ' ' + str(f.log_content[21]).split('=')[1]
            bgWidth2 = MatChoice.bgWidth/(x1[2]-x1[1])

        elif inschoice == "Tornado HFPP":
            expTime = str(f.log_content[7]).split('=')[1]
            averaging = str(f.log_content[10]).split('=')[1]
            bgWidth2 = MatChoice.bgWidth

    elif choice == "*.csv":
        with open(iFile, newline='') as csvfile:
            f = list(csv.reader(csvfile, delimiter=','))
        y1 = f[23]
        x1 = np.asarray([float(i) for i in f[21]])
        y1 = np.asarray([float(i) for i in y1[0:len(y1) - 1]])
        expTime = ' ' + str(f[3][1]) + ' '
        averaging = ' ' + str(f[4][0]).split(' ')[1] + ' '
        bgWidth2 = MatChoice.bgWidth

    else:

        continue

    #Finds the index for the peaklocs selected
    idx1 = (np.abs(x1 - MatChoice.lowBound)).argmin()
    idx2 = (np.abs(x1 - MatChoice.highBound)).argmin()
    #Average yvalues from 20 points before idx1 and 20 points after idx2
    bg1 = np.mean(y1[idx1-bgWidth2:idx1])
    bg2 = np.mean(y1[idx2:idx2+bgWidth2])

    #Insert the averaged value at idx1 and idx2
    y2 = np.insert(y1, idx1, bg1)
    y2 = np.delete(y2, (idx1+1), axis=0)

    y2 = np.insert(y2, idx2, bg2)
    y2 = np.delete(y2, (idx2+1), axis=0)

    #Calculate linear baseline used to subtract from each spectrum
    interpolant = np.interp([x1[idx1], x1[idx2]], x1, y2)
    coefficients = np.polyfit([x1[idx1], x1[idx2]], interpolant, 1)

    #area calculation after baseline subtraction
    area = np.sum(y1[idx1:idx2] - ((x1[idx1:idx2] * coefficients[0]) + coefficients[1]))
    #area = np.trapz(y1[idx1:idx2] - ((x1[idx1:idx2] * coefficients[0]) + coefficients[1]), x1[idx1:idx2])
    mody1 = y1[idx1:idx2] - ((x1[idx1:idx2] * coefficients[0]) + coefficients[1])
    areaList = np.vstack((areaList, area))
    ax2.plot(x1[idx1-bgWidth2:idx2+bgWidth2], (y1[idx1-bgWidth2:idx2+bgWidth2] - (x1[idx1-bgWidth2:idx2+bgWidth2]*coefficients[0] + coefficients[1]))) #include bgWidth in the plot!!!!

#remove the empty row from areaList, and calculate SNR, RSD
areaList = np.delete(areaList, (0), axis=0)
SNR = np.mean(areaList) / np.std(areaList)
RSD = np.std(areaList) / np.mean(areaList) * 100

#Add vertical plot lines designating abackground and area calculation
ax2.axvline(x=MatChoice.lowBound, color='b', linestyle='--' )
ax2.text(MatChoice.lowBound, max(mody1)/1.5, ' Area start', fontsize=7)
ax2.axvline(x=MatChoice.highBound, color='b', linestyle='--')
ax2.text(MatChoice.highBound, max(mody1)/1.5, 'Area end ', ha='right', fontsize=7)
ax2.axvline(x=MatChoice.lowBound-bgWidth2, color='r', linestyle='--')
ax2.axvline(x=MatChoice.highBound+bgWidth2, color='r', linestyle='--')
ax2.text(MatChoice.lowBound-bgWidth2, max(mody1)/1.6, ' BG', fontsize=7)
ax2.text(MatChoice.highBound+bgWidth2, max(mody1)/1.6, 'BG ', ha='right', fontsize=7)

# Add title and axis names
try:
    float(MatChoice.peakLoc)
    ax1.set_title('Peak area variance for ' + str(MatChoice.name) + ' ' + str(MatChoice.peakLoc) + ' $cm^{-1}$ band'  + '\n'
            + 'with ' + str(inschoice) + ' (' + averaging[1:-1] + 'x' + expTime[1:-1] + ' ms exposures):'
              + '\n' + r'$\bf{RSD = ' + str(round(RSD,3)) + '%, SNR = ' + str(round(SNR,3)) + '}$', fontsize = 8)
    ax3.set_title('Peak area variance trend ' + str(MatChoice.name) + ' ' + str(MatChoice.peakLoc) + ' $cm^{-1}$ band with ' + str(inschoice)
                  + '\n' + '(' + averaging[1:-1] + 'x' + expTime[1:-1] + ' ms exposures): RSD = '
                  + str(round(RSD, 3)) + '%, SNR = ' + str(round(SNR, 3)), fontsize=8)
    ax2.set_title(str(MatChoice.peakLoc) + ' $cm^{-1}$ band for ' + str(MatChoice.name) + '\n' + 'measured with ' + str(
        inschoice), fontsize=8)
except:
    ax1.set_title(
        'Peak area variance for ' + str(MatChoice.name) + ' ' + str(MatChoice.peakLoc) + ' band' + '\n'
        + 'with ' + str(inschoice) + ' (' + averaging[1:-1] + 'x' + expTime[1:-1] + ' ms exposures):'
        + '\n' + r'$\bf{RSD = ' + str(round(RSD, 3)) + '%, SNR = ' + str(round(SNR, 3)) + '}$', fontsize=8)
    ax3.set_title(
        'Peak area variance trend ' + str(MatChoice.name) + ' ' + str(MatChoice.peakLoc) + ' band with ' + str(
            inschoice) + '\n' + '(' + averaging[1:-1] + 'x' + expTime[1:-1] + ' ms exposures): RSD = '
        + str(round(RSD, 3)) + '%, SNR = ' + str(round(SNR, 3)), fontsize=8)
    ax2.set_title(str(MatChoice.peakLoc) + ' band for ' + str(MatChoice.name) + '\n' + 'measured with ' + str(
        inschoice), fontsize=8)

ax1.set(xlabel = 'Boxplot Distribution of Peak Areas', ylabel = 'Peak Area') #xlabel = 'Spectrum #',
ax2.set(xlabel = 'Raman Shifts ($cm^{-1}$)', ylabel = 'Counts')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax3.set(xlabel = 'Spectrum #', ylabel = 'Peak Area')

areaRange = range(1,len(areaList)+1)
ax3.scatter(areaRange, areaList, color ='b', marker='o')
ax1.boxplot(areaList)
fig.subplots_adjust(hspace = 0.7)
plt.show()

saveMsg ="Do you want to save your custom peak choice?"
saveTitle = "Custom analysis"
saveChoices = ["Yes", "No"]
saveChoice = easygui.choicebox(saveMsg, saveTitle, saveChoices)


if saveChoice == None or saveChoice == "No":
    easygui.buttonbox(msg='Press OK to Exit', title='SNR Calc', choices=["OK"])
    exit()
else:
    try:
        programPath = os.path.expanduser('~\\SNRcalc Data')
        os.chdir(programPath)
        pklFileName = str(MatChoice.name) + " " + str(MatChoice.peakLoc) + ".pkl"
        with open(pklFileName, 'wb') as output:
            pickle.dump(MatChoice, output, pickle.HIGHEST_PROTOCOL)
    except:
        programPath = os.path.expanduser('~')
        os.chdir(programPath)
        os.mkdir('SNRcalc Data')
        os.chdir('SNRcalc Data')
        pklFileName = str(MatChoice.name) + " " + str(MatChoice.peakLoc) + ".pkl"
        with open(pklFileName, 'wb') as output:
            pickle.dump(MatChoice, output, pickle.HIGHEST_PROTOCOL)


