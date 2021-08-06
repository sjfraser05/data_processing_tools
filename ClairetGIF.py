import numpy as np
import glob, os
import matplotlib.pyplot as plt
import spc
import csv
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import imageio
from scipy.signal import find_peaks, correlate, hilbert, savgol_filter
from RubberBandSubtract import rubberband
from PeakNorm import PeakNorm
import easygui

def animate(i):
    data = VA #select data range
    p = sns.lineplot(UA, VA, data=data, color="r")
    p.tick_params(labelsize=17)
    plt.setp(p.lines,linewidth=7)

msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)

directory = easygui.diropenbox()
print(directory)
os.chdir(directory)
dirList = glob.glob(choice)
nList = len(dirList)
aveArray = np.zeros((1, 3101))
images =[]
for count, iFile in enumerate(dirList):

    if (count) % 10 == 0:
        if choice == "*.spc":
            f = spc.File(iFile)
            x1 = f.x
            y1 = f.sub[0].y



        y2 = PeakNorm(x1, y1, 2315, 2345)

        #plt.plot(x1, snv(y1))
        y2 = y2 - rubberband(x1, y2)
        y2 = savgol_filter(y2, 15, 3, deriv=0)
        #y2 = snv(y2)
        fig = plt.plot(x1[30:100], y2[30:100], label=iFile, color='orange')
        plt.xlabel('Raman Shifts ($cm^{-1}$)')
        plt.ylabel('Counts')
        plt.ylim(-.01 , 1.5)

        stringText = str((count*10)//60) + " minutes " + str((count*10)%(60)) + " seconds"
        plt.figtext(0.5,0.8, stringText, fontsize= 15)
        plt.ion()
        plt.show()
        imageSaveFile = r'C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\Customer Data\Clairet\gifimagesclairet' + "\\" + str(count) + " - gif image.png"
        plt.savefig(imageSaveFile)
        images.append(imageio.imread(imageSaveFile))
        plt.close('all')
        time.sleep(0.01)
        os.remove(imageSaveFile)
        print(count)
imageio.mimsave(r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\Customer Data\Clairet\gifimagesclairet\mygif.gif", images, duration=0.25)
input("waiting")

spectralDataFolder = r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\Customer Data\Clairet\Spectra"

os.chdir(spectralDataFolder)
dirList = glob.glob("*.spc")
#spectraList = np.zeros((len(dirList), 3101))

UVdata = r"C:\Users\Shaun.fraser\Documents\UV data AKTA run single.csv"

with open(UVdata, newline='') as UVcsvfile:
    UVcsvfile = np.asarray(list(csv.reader(UVcsvfile, delimiter=',')))
    VA = UVcsvfile[:,0]
    VA = np.asarray([float(i) for i in VA[0:len(VA) - 1]])
    UA = UVcsvfile[:,1]
    UA = np.asarray([float(i) for i in UA[0:len(UA) - 1]])
    #UA = UVcsvfile[1]
    #volumeArray =  np.asarray([float(i) for i in VA[0:len(VA) - 1]])
    #UVarray = np.asarray([float(i) for i in UA[0:len(UA) - 1]])

#plt.plot(VA, UA)
#plt.show()

xvals = np.asarray(range(200, 3300, 1))

#with open(spectralDataFile, newline='') as csvfile:
    #f = np.asarray(list(csv.reader(csvfile, delimiter=',')))
    #yVals = f[2, 2:]
   #y1 = np.asarray([float(i) for i in y1[0:len(y1) - 1]])



    #plt.plot(xvals, y1)

#plt.show()
#sec = input('Let us wait for user input.')
#title = 'UV Chromatogram'
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
images =[]
#for file_name in sorted(os.listdir(r"C:\Users\Shaun.fraser\Documents\Aktagifs files")):
 #   if file_name.endswith('.png'):
       # file_path = os.path.join(r"C:\Users\Shaun.fraser\Documents\Aktagifs files", file_name)
       # images.append(imageio.imread(file_path))
        #print(file_name)


#plt.ylim(np.min(UA), np.max(UA))
#plt.xlabel('Column Volume (mL)',fontsize=20)
#plt.ylabel("UV Absorption (mAU)",fontsize=20)
#fig = plt.figure(figsize=(10,6))
fig = plt.ion()

VAidx1 = (np.abs(VA - 25)).argmin()
VAidx2 = (np.abs(VA - 95)).argmin()

#specCount = - 50         #100


for count, row in enumerate(VA):
    if count >= 54206:
        break
    else:
        try:
            if (count) % 357 == 0:   #715
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(25, 10, forward=True)

                specCount = specCount + 24      #48
                with open(dirList[specCount], newline='') as csvfile:
                    f = np.asarray(list(csv.reader(csvfile, delimiter=',')))
                    y1 = f[:,1]
                    y1 = np.asarray([float(i) for i in y1[0:len(y1) - 1]])
                y1 = savgol_filter(y1, 11, 5, 0)

                ax2.plot(xvals, y1, color='orange', label=str(specCount))
                ax2.set_xlabel("Raman Shifts ($cm^{-1}$)")
                #ax2.title(directory.split('\\')[-1])
                ax2.set_ylabel("Counts")
                #ax2.set_ylim(10000, 25000)
                #ax2.set_xlim(900, 1800)

                #ax2.redraw_in_frame()


                imageSaveFile = r"C:\Users\Shaun.fraser\Documents\Aktagifs files" + "\\" + str(count) + " UV chrom.png"
                plt.savefig(imageSaveFile)


                #images.append(imageio.imread(imageSaveFile))
                #plt.clf()
                ax1.clear()
                ax2.clear()
                plt.close(fig)
                #time.sleep(0.01)
                #os.remove(imageSaveFile)
                #plt.show()
                #time.sleep(5)
                print(count, specCount)
        except:
            print('buuuuuuuh')
#imageio.mimsave(r"C:\Users\Shaun.fraser\Documents\Aktagifs files\mygif.gif", images)
#for file_name in sorted(os.listdir(r"C:\Users\Shaun.fraser\Documents\Aktagifs files")):
    #if file_name.endswith('.png'):
        #file_path = os.path.join(r"C:\Users\Shaun.fraser\Documents\Aktagifs files", file_name)
        #images.append(imageio.imread(file_path))
#images.append(imageio.imread(images))
#imageio.mimsave(
    #plt.close('all')


#with imageio.get_writer('mygif.gif', mode='I') as writer:
 #   for filename in ['1.png', '2.png', '3.png', '4.png']:
  #      image = imageio.imread(filename)
   #     writer.append_data(image)
#plt.title('Heroin Overdoses per Year',fontsize=20)
#XN,YN = augment(VA,UA,10)
#augmented = pd.DataFrame(YN,XN)
#overdose.columns = {title}

#ani = matplotlib.animation.FuncAnimation(fig, animate, frames=2000, repeat=True)
