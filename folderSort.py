import numpy as np
import glob, os
import matplotlib.pyplot as plt
import spc
import easygui
import csv
import shutil, os


msg ="spc or csv?"
title = "Choose file type"
choices = ["*.spc", "*.csv"]
choice = easygui.choicebox(msg, title, choices)

directory = easygui.diropenbox()
os.chdir(directory)
dirList = glob.glob(choice)
nList = len(dirList)

for count, iFile in enumerate(dirList):

    if 370 <= count <= 1000:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\15%")

    if 1255 <= count <= 2020:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\20%")

    if 2168 <= count <= 2569:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\25%")

    if 3170 <= count <= 3800:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\30%")

    if 4000 <= count <= 4600:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\35%")

    if 4800 <= count <= 5400:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\40%")

    if 5631 <= count <= 6175:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\45%")

    if 6500 <= count <= 7130:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\50%")

    if 7355 <= count <= 7780:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\55%")

    if 8250 <= count <= 8800:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\60%")

    if 9090 <= count <= 9500:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\65%")

    if 9660 <= count <= 10125:
        shutil.copy(iFile, r"C:\Users\Shaun.fraser\OneDrive - Tornado Spectral Systems\SNR Laser Testing\100ms\70%")